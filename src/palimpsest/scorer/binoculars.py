"""A second instrument: read the *disagreement between two* language models.

Why this module exists
----------------------
``local_lm.py`` reads one observer (GPT-2 124 M) and asks "how surprising is this text?".
That question has an answer only while the observer is a decent proxy for what a competent
writer would say next. Measured against the 2026 corpus it is not: on Claude essays the
single-observer detector flags 2.6% of sentences, i.e. nothing, because frontier prose is
not *distinctively* surprising to a 2019 model -- it is merely fluent, and so is good human
prose. See docs/08-cross-vendor.md.

Worse, when the model-based features carry no signal the classifier still has 29 surface
features to lean on, and those measure register (tricolon, hedges, boosters). That is how a
detector ends up scoring "the world is full of broken systems" at 0.94 and "I didn't fix all
of them" at 0.04 -- both from the same machine-written essay.

The fix is not a bigger observer. It is a *different question*, from Binoculars (Hans et al.,
ICML 2024): stop asking how surprising the text is in absolute terms, and start asking how
surprising it is **relative to what a closely-related model expected**.

    B(s) = log PPL_observer(s) / log X-PPL(observer, performer)

The numerator is the observer's surprise at the tokens that actually appeared. The
denominator is the average cross-entropy between the two models' *predictive distributions*
-- how much they disagree about what could have come next, regardless of what did.

That ratio is the point. Simple human writing has low absolute perplexity too, which is why
raw perplexity thresholds accuse plain writers (and, in the published literature and in our
own ``esl`` set, non-native writers especially). But when a human writes simply, the two
models still disagree about the *next* word in the ordinary way, so the denominator stays
large and the ratio stays high. Machine text sits in the region where both models agree AND
the realised tokens are unsurprising, so the ratio collapses. Lower B = more machine-like.

The property we actually need
-----------------------------
Nothing here is fitted to any vendor. There is no training set, no logistic regression, no
threshold learned from Gemini essays. That is the entire reason to reach for it: the failure
in docs/08-cross-vendor.md is a *fitting* failure -- a classifier fitted on flash-lite
learned flash-lite's register -- and a statistic that fits nothing cannot fail that way.
It can fail in other ways, and this module's job is to let us measure whether it does.

Running two models on 8 GB
--------------------------
The canonical pair is Falcon-7B + Falcon-7B-Instruct: 14 GB in bf16, impossible here. We use
Qwen3-0.6B-Base + Qwen3-0.6B (~1.2 GB each in bf16). The published ablations show accuracy
falls with pair size, so this is a weaker instrument than the paper's -- an honest caveat,
not a hidden one, and ``scripts/bench_binoculars.py`` measures the difference it makes
rather than assuming it.

Both models MUST share a tokenizer. The denominator is a cross-entropy between two
distributions over the same vocabulary; with different vocabularies it is not a
cross-entropy at all, just two unrelated numbers divided. This is asserted at construction.

Memory is the binding constraint, so the two full logit tensors are the only large things
alive at once (~311 MB each at a 1024 window over Qwen's 151 k vocab) and every softmax is
taken over a 128-position slice cast to float32. Peak stays under ~1 GB above the weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .local_lm import select_device

log = logging.getLogger(__name__)

__all__ = ["BinocularsScores", "BinocularsScorer", "OBSERVER", "PERFORMER"]

#: Base and instruct checkpoints of one family. Same tokenizer, different post-training --
#: which is what makes their disagreement informative rather than arbitrary.
OBSERVER = "Qwen/Qwen3-0.6B-Base"
PERFORMER = "Qwen/Qwen3-0.6B"

#: Softmax chunk. A [1024 x 151936] float32 tensor is 622 MB; a [128 x 151936] one is 78 MB.
_CHUNK = 128

#: Below this many scored tokens the ratio is dominated by noise in a handful of positions.
#: We return NaN rather than a confident-looking number, matching MIN_TOKENS in model_based.
MIN_TOKENS = 5


@dataclass(frozen=True, slots=True)
class BinocularsScores:
    """Per-token quantities needed to form the ratio over any span.

    Deliberately stores the two *sums'* ingredients per token rather than a per-token ratio.
    The Binoculars score of a span is a ratio of means, not the mean of per-token ratios;
    those differ, and only the first matches the published statistic.
    """

    tokens: list[str]
    char_start: np.ndarray  # int32
    char_end: np.ndarray  # int32
    logprob: np.ndarray  # float32, log P_observer(token | context)
    xent: np.ndarray  # float32, H(P_observer, P_performer) at this position
    observer: str
    performer: str
    device: str

    def __len__(self) -> int:
        return len(self.tokens)

    def select(self, start: int, end: int) -> "BinocularsScores":
        """Sub-view of tokens whose *centre* lies in ``[start, end)``.

        Centre-based, identically to ``TokenScores.select``, so a token straddling a
        sentence boundary is attributed to exactly one span in both instruments.
        """
        centre = (self.char_start + self.char_end) / 2.0
        keep = np.flatnonzero((centre >= start) & (centre < end))
        return BinocularsScores(
            tokens=[self.tokens[i] for i in keep],
            char_start=self.char_start[keep],
            char_end=self.char_end[keep],
            logprob=self.logprob[keep],
            xent=self.xent[keep],
            observer=self.observer,
            performer=self.performer,
            device=self.device,
        )

    @property
    def score(self) -> float:
        """The Binoculars ratio for this span. Lower means more machine-like.

        NaN when the span is too short to estimate, or when the denominator is degenerate
        (two models in perfect agreement everywhere), which we report rather than clamp.
        """
        if len(self.logprob) < MIN_TOKENS:
            return float("nan")
        ppl = float(-np.mean(self.logprob))  # observer's surprise at what appeared
        xppl = float(np.mean(self.xent))  # the two models' disagreement about what could
        if not np.isfinite(ppl) or not np.isfinite(xppl) or xppl <= 1e-6:
            return float("nan")
        return ppl / xppl


class BinocularsScorer:
    """Scores text under two models sharing one tokenizer."""

    def __init__(
        self,
        observer: str = OBSERVER,
        performer: str = PERFORMER,
        device: str = "auto",
        max_length: int = 1024,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.observer_name = observer
        self.performer_name = performer
        self.device = select_device(device)
        # bf16 halves the weights and the logit tensors, which is what makes the pair fit.
        # Every softmax below is still taken in float32: bf16 has ~3 decimal digits, and the
        # ratio is a quotient of two averaged logs where that error would not cancel.
        self.dtype = dtype or (torch.float32 if self.device == "cpu" else torch.bfloat16)

        self._tok = AutoTokenizer.from_pretrained(observer)
        perf_tok = AutoTokenizer.from_pretrained(performer)
        if self._tok.get_vocab() != perf_tok.get_vocab():
            raise ValueError(
                f"{observer} and {performer} do not share a vocabulary; their "
                "cross-entropy would be meaningless. Use a base/instruct pair."
            )

        self._obs = self._load(observer)
        self._perf = self._load(performer)
        self.max_length = int(max_length)
        self.stride = self.max_length // 2
        log.info(
            "binoculars: %s vs %s on %s (%s, context %d)",
            observer, performer, self.device, self.dtype, self.max_length,
        )

    def _load(self, name: str) -> AutoModelForCausalLM:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=self.dtype)
        m.eval()
        m.to(self.device)
        return m

    # -- public API ---------------------------------------------------------------

    def score(self, text: str) -> BinocularsScores:
        """Score every token with at least one token of left context."""
        if not text.strip():
            return _empty(self.observer_name, self.performer_name, self.device)

        enc = self._tok(text, return_offsets_mapping=True, return_tensors="np")
        ids = enc["input_ids"][0].astype(np.int64)
        offsets = np.asarray(enc["offset_mapping"][0], dtype=np.int32)
        n = len(ids)
        if n < 2:
            return _empty(self.observer_name, self.performer_name, self.device)

        logprob = np.zeros(n, dtype=np.float32)
        xent = np.zeros(n, dtype=np.float32)
        scored = np.zeros(n, dtype=bool)

        ids_t = torch.from_numpy(ids)
        prev_end = 1  # token 0 has no left context and is never scored.

        for begin in range(0, n, self.stride):
            stop = min(begin + self.max_length, n)
            chunk = ids_t[begin:stop].unsqueeze(0).to(self.device)
            with torch.no_grad():
                obs_logits = self._obs(chunk).logits[0]
                perf_logits = self._perf(chunk).logits[0]

            lo = max(prev_end, begin + 1)
            if lo < stop:
                targets = ids_t[lo:stop].to(self.device)
                rows = slice(lo - begin - 1, stop - begin - 1)  # predict position i from i-1
                lp, xe = _pair_stats(obs_logits[rows], perf_logits[rows], targets)
                logprob[lo:stop], xent[lo:stop] = lp, xe
                scored[lo:stop] = True
                prev_end = stop

            del obs_logits, perf_logits
            if stop == n:
                break

        keep = np.flatnonzero(scored)
        return BinocularsScores(
            tokens=[self._tok.decode([int(i)]) for i in ids[keep]],
            char_start=offsets[keep, 0],
            char_end=offsets[keep, 1],
            logprob=logprob[keep],
            xent=xent[keep],
            observer=self.observer_name,
            performer=self.performer_name,
            device=self.device,
        )


def _pair_stats(
    obs_logits: torch.Tensor, perf_logits: torch.Tensor, targets: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position observer log-prob of the realised token, and observer/performer x-entropy.

    Chunked so that no [n_positions x vocab] float32 tensor is ever materialised whole.
    """
    n = targets.shape[0]
    out_lp = np.empty(n, dtype=np.float32)
    out_xe = np.empty(n, dtype=np.float32)

    for i in range(0, n, _CHUNK):
        j = min(i + _CHUNK, n)
        o = obs_logits[i:j].float()
        p = perf_logits[i:j].float()
        o_logp = torch.log_softmax(o, dim=-1)
        p_logp = torch.log_softmax(p, dim=-1)

        out_lp[i:j] = (
            o_logp.gather(-1, targets[i:j].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        )
        # H(P_obs, P_perf) = -sum_v P_obs(v) log P_perf(v). The expectation is taken under
        # the OBSERVER, matching the paper: we are asking how well the performer's
        # predictions cover the observer's beliefs about this position.
        out_xe[i:j] = (-(o_logp.exp() * p_logp).sum(-1)).cpu().numpy()
        del o, p, o_logp, p_logp

    return out_lp, out_xe


def _empty(observer: str, performer: str, device: str) -> BinocularsScores:
    z32 = np.zeros(0, dtype=np.float32)
    return BinocularsScores(
        tokens=[],
        char_start=np.zeros(0, dtype=np.int32),
        char_end=np.zeros(0, dtype=np.int32),
        logprob=z32,
        xent=z32,
        observer=observer,
        performer=performer,
        device=device,
    )

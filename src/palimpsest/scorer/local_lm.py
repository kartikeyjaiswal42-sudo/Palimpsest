"""The instrument: read a small local causal LM for per-token probability statistics.

This module is the one place in Palimpsest that touches a language model, and it is
deliberately incapable of asking one for an opinion. It calls the model in a single
forward pass and reads the logits; there is no ``generate()`` call, no prompt, no chat
template, and no place for a verdict to enter. Everything downstream is arithmetic on the
numbers produced here.

Four statistics are extracted per token, and they capture genuinely different things:

``logprob``
    log P(token | left context). Low absolute surprisal is the classic "too fluent" signal.
``rank``
    Where the observed token sat in the model's ranked next-token distribution. Robust to
    the calibration of the probabilities themselves, which is why GLTR used it.
``entropy``
    H of the predictive distribution. Separates "the model was confident and right" from
    "the model had no idea" -- the same logprob means different things in each case.
``mu`` / ``sigma2``
    Mean and variance of log P under the model's *own* next-token distribution. These are
    what make the Fast-DetectGPT curvature computable analytically in one pass, instead of
    the ~100 perturbation passes the original DetectGPT needed.

Long essays exceed the model's context window, so tokens are scored with a sliding window
that always gives each token the maximum left context available. The first token of a
document has no left context and is therefore never scored -- it is excluded, not imputed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

__all__ = ["TokenScores", "LocalLMScorer", "select_device"]

# Positions are reduced to summary statistics in chunks so that a [1024 x 50257] logit
# tensor never has two full-size float copies alive at once. Matters on an 8 GB laptop.
_STATS_CHUNK = 128


@dataclass(frozen=True, slots=True)
class TokenScores:
    """Per-token statistics for one document, aligned to character offsets.

    All arrays have length ``n_tokens`` and are index-aligned with each other. ``char_start``
    and ``char_end`` index the *original* text, so a caller can select the tokens belonging
    to any character span without re-tokenising.
    """

    tokens: list[str]
    char_start: np.ndarray  # int32
    char_end: np.ndarray  # int32
    logprob: np.ndarray  # float32, log P(token | context)
    rank: np.ndarray  # int32, 1-based
    entropy: np.ndarray  # float32, H of the predictive distribution
    mu: np.ndarray  # float32, E[log P] under the model's own distribution
    sigma2: np.ndarray  # float32, Var[log P] under the model's own distribution
    model_name: str
    device: str
    #: True when the observer did not see the whole document.
    #:
    #: Always False for this scorer -- a long essay is scored with a sliding window, so every
    #: token gets read. It lives here rather than on the remote scorer's own dataclass because
    #: this is the type `Analyzer` consumes, and a fact that stops at the adapter is a fact the
    #: interface cannot report. The remote observer has a hard 6,000-character window, and
    #: whether it was hit changes what the verdict is a verdict ABOUT.
    clipped: bool = False

    def __len__(self) -> int:
        return len(self.tokens)

    def select(self, start: int, end: int) -> "TokenScores":
        """Return the sub-view of tokens whose *centre* falls inside ``[start, end)``.

        Centre-based selection avoids double-counting a token that straddles a sentence
        boundary: each token is attributed to exactly one span.
        """
        centre = (self.char_start + self.char_end) / 2.0
        keep = np.flatnonzero((centre >= start) & (centre < end))
        return TokenScores(
            tokens=[self.tokens[i] for i in keep],
            char_start=self.char_start[keep],
            char_end=self.char_end[keep],
            logprob=self.logprob[keep],
            rank=self.rank[keep],
            entropy=self.entropy[keep],
            mu=self.mu[keep],
            sigma2=self.sigma2[keep],
            model_name=self.model_name,
            device=self.device,
            clipped=self.clipped,
        )


def select_device(preference: str = "cpu") -> str:
    """Resolve a device string.

    Defaults to ``cpu``. That is a deliberate choice: every number reported in
    ``docs/03-evaluation.md`` was produced on CPU, and Metal kernels do not always agree
    with CPU in the last decimal place. Speed is available via ``preference="auto"``, at
    the cost of exact reproducibility of the published figures.
    """
    if preference == "cpu":
        return "cpu"
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


class LocalLMScorer:
    """Wraps one local causal LM and turns text into :class:`TokenScores`.

    The default observer is GPT-2 (124 M parameters, ~500 MB on disk). It is small enough
    to run on a laptop CPU in well under a second for an essay, and its training data
    predates every model we are trying to detect -- so it cannot have memorised their
    outputs. Its job is not to know what AI text looks like; it is to provide a fixed,
    neutral yardstick of "how predictable is this word here".
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = select_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self.device)

        ctx = getattr(self._model.config, "n_positions", None) or getattr(
            self._model.config, "max_position_embeddings", 1024
        )
        self.max_length = int(max_length or ctx)
        self.stride = self.max_length // 2
        log.info(
            "loaded observer %s on %s (context %d)", model_name, self.device, self.max_length
        )

    # -- public API ---------------------------------------------------------------

    def score(self, text: str) -> TokenScores:
        """Score every token of ``text`` that has at least one token of left context."""
        if not text.strip():
            return _empty_scores(self.model_name, self.device)

        enc = self._tokenizer(text, return_offsets_mapping=True, return_tensors="np")
        ids = enc["input_ids"][0].astype(np.int64)
        offsets = np.asarray(enc["offset_mapping"][0], dtype=np.int32)
        n = len(ids)
        if n < 2:
            return _empty_scores(self.model_name, self.device)

        logprob = np.zeros(n, dtype=np.float32)
        rank = np.zeros(n, dtype=np.int32)
        entropy = np.zeros(n, dtype=np.float32)
        mu = np.zeros(n, dtype=np.float32)
        sigma2 = np.zeros(n, dtype=np.float32)
        scored = np.zeros(n, dtype=bool)

        ids_t = torch.from_numpy(ids)
        window, stride = self.max_length, self.stride
        prev_end = 1  # token 0 can never be scored: nothing precedes it.

        for begin in range(0, n, stride):
            stop = min(begin + window, n)
            chunk = ids_t[begin:stop].unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._model(chunk).logits[0]  # [stop-begin, vocab]

            lo = max(prev_end, begin + 1)
            if lo < stop:
                targets = ids_t[lo:stop].to(self.device)
                pred_rows = logits[lo - begin - 1 : stop - begin - 1]
                stats = _summarise(pred_rows, targets)
                sl = slice(lo, stop)
                logprob[sl], rank[sl], entropy[sl], mu[sl], sigma2[sl] = stats
                scored[sl] = True
                prev_end = stop

            del logits
            if stop == n:
                break

        keep = np.flatnonzero(scored)
        return TokenScores(
            tokens=[self._tokenizer.decode([int(i)]) for i in ids[keep]],
            char_start=offsets[keep, 0],
            char_end=offsets[keep, 1],
            logprob=logprob[keep],
            rank=rank[keep],
            entropy=entropy[keep],
            mu=mu[keep],
            sigma2=sigma2[keep],
            model_name=self.model_name,
            device=self.device,
        )


def _summarise(
    pred_rows: torch.Tensor, targets: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce raw logits to the five per-token statistics, in memory-bounded chunks."""
    n = pred_rows.shape[0]
    out_lp = torch.empty(n, dtype=torch.float32)
    out_rank = torch.empty(n, dtype=torch.int32)
    out_ent = torch.empty(n, dtype=torch.float32)
    out_mu = torch.empty(n, dtype=torch.float32)
    out_var = torch.empty(n, dtype=torch.float32)

    for a in range(0, n, _STATS_CHUNK):
        b = min(a + _STATS_CHUNK, n)
        lp = torch.log_softmax(pred_rows[a:b].float(), dim=-1)
        tgt = targets[a:b].unsqueeze(-1)

        token_lp = lp.gather(-1, tgt).squeeze(-1)
        # Rank is the count of strictly-better tokens, plus one. Ties resolve optimistically,
        # which only matters for the degenerate case of a perfectly uniform distribution.
        out_rank[a:b] = ((lp > token_lp.unsqueeze(-1)).sum(-1) + 1).to(torch.int32).cpu()

        probs = lp.exp()
        out_ent[a:b] = (-(probs * lp).sum(-1)).cpu()
        m = (probs * lp).sum(-1)
        out_mu[a:b] = m.cpu()
        out_var[a:b] = ((probs * lp.pow(2)).sum(-1) - m.pow(2)).clamp_min(0).cpu()
        out_lp[a:b] = token_lp.cpu()
        del lp, probs

    return (
        out_lp.numpy(),
        out_rank.numpy(),
        out_ent.numpy(),
        out_mu.numpy(),
        out_var.numpy(),
    )


def _empty_scores(model_name: str, device: str) -> TokenScores:
    z_f = np.zeros(0, dtype=np.float32)
    z_i = np.zeros(0, dtype=np.int32)
    return TokenScores([], z_i, z_i, z_f, z_i, z_f, z_f, z_f, model_name, device)


@lru_cache(maxsize=4)
def get_scorer(model_name: str = "gpt2", device: str = "cpu") -> LocalLMScorer:
    """Process-wide cached scorer. Loading GPT-2 takes seconds; do it once."""
    return LocalLMScorer(model_name=model_name, device=device)

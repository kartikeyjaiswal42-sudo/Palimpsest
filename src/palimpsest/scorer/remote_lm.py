"""Read a *large* observer's per-token statistics over the network.

Why a remote observer
---------------------
Every model-based feature in Palimpsest answers one question -- "how predictable was this
token here?" -- and the answer is only as good as the model asked. ``local_lm.py`` asks
GPT-2 (124 M, 2019). docs/08-cross-vendor.md records the cost: on Claude Opus essays the
detector flags 2.6% of sentences, because frontier prose is not *distinctively* surprising
to a 2019 model. It is merely fluent, and so is good human prose. The instrument is blind,
and no amount of reweighting downstream recovers a signal that was never measured.

The obvious repair is a bigger observer. A 30 B model does not fit on the 8 GB laptop this
project is developed on -- and a two-model 0.6 B pair was tried and abandoned when the
machine was found sitting at 12.4 GB of swap before loading anything. So the observer runs
on Cloudflare Workers AI instead, behind ``observer-worker/``, and this module is its client.

What comes back, and what does not
----------------------------------
Workers AI scores a *given* text with ``prompt_logprobs``, returning each position's
log-probability and its true rank in the model's ranked next-token distribution. That is
enough for the log-probability and rank families of features -- GLTR's rank buckets, mean
log-rank, DetectLLM's LRR, surprisal dispersion.

``entropy`` and Fast-DetectGPT ``curvature`` need more: the model's predictive distribution
at each position, not just the token that was realised. With ``top_k=0`` the API returns one
candidate per position, those two are NaN, and the pipeline reports them as "not measured".

``top_k=20`` changes that, and the change was measured rather than assumed:

* the API **does** return the top 20 candidates with log-probabilities and ranks;
* it costs **exactly the same neurons** as ``top_k=0`` (0.1738 either way on a 31-token
  probe) -- it is the same forward pass, only more of the output is reported;
* the top 20 hold a **median 88% / mean 82%** of the probability mass, not ~100%.

That last number is the honest caveat and it is why ``mass`` travels with every token. What
is computed here is the conditional mean and variance of ``log p`` over the observer's top
20 candidates, renormalised -- a well-defined statistic about the head of the distribution,
and the head is where a language model doing the writing actually samples from. It is not
the full-vocabulary quantity in the Fast-DetectGPT paper, and a document whose ``mass`` runs
low is one where the difference is largest.

The alignment was verified before any of this was trusted: at ``top_k=20`` the worker
returns byte-identical tokens, offsets, ranks and log-probabilities to ``top_k=0``, with a
``misaligned`` counter that reports any position where no candidate continued the text.

Caching
-------
Every response is cached to disk keyed by (model, text). Scoring the corpus costs real
neurons against a 10,000/day free allowance, and iterating on features must not re-spend
them. The cache is keyed on the text itself, so editing an essay correctly misses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["RemoteScores", "RemoteLMScorer", "DEFAULT_ENDPOINT", "DEFAULT_MODEL"]

DEFAULT_ENDPOINT = "https://palimpsest-observer.amitynoidalibrary.workers.dev/probe"
DEFAULT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
CROSS_MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"

#: Matches MAX_CHARS in the Worker. Longer texts are clipped there; we record that it
#: happened rather than silently reporting statistics over a truncated essay.
MAX_CHARS = 6000

MIN_TOKENS = 5


def _ssl_context() -> ssl.SSLContext:
    """A context that can actually verify Cloudflare's certificate.

    python.org builds on macOS ship without a usable CA bundle -- they rely on a separate
    ``Install Certificates.command`` step -- so ``urlopen`` fails with CERTIFICATE_VERIFY_
    FAILED on a perfectly valid host. ``certifi`` is already a transitive dependency here,
    so we point at its bundle explicitly. Verification stays ON: disabling it would be the
    easy fix and would silently accept any interception of the corpus we are scoring.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with requests/transformers
        return ssl.create_default_context()


@dataclass(frozen=True, slots=True)
class RemoteScores:
    """Per-token statistics from the remote observer, aligned to character offsets.

    Deliberately mirrors :class:`palimpsest.scorer.local_lm.TokenScores` so the same feature
    code can consume either instrument, with ``entropy``/``mu``/``sigma2`` absent.
    """

    tokens: list[str]
    char_start: np.ndarray  # int32
    char_end: np.ndarray  # int32
    logprob: np.ndarray  # float32
    rank: np.ndarray  # int32, 1-based
    model_name: str
    clipped: bool
    neurons: float | None
    #: Present only when the scorer asked for ``top_k > 0``; all-NaN otherwise. See the
    #: module docstring for exactly what they do and do not measure.
    entropy: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma2: np.ndarray | None = None
    top_mass: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def select(self, start: int, end: int) -> "RemoteScores":
        """Sub-view of tokens whose *centre* lies in ``[start, end)``.

        Centre-based, identically to the local scorer, so a token straddling a sentence
        boundary is attributed to exactly one span under both instruments.
        """
        centre = (self.char_start + self.char_end) / 2.0
        keep = np.flatnonzero((centre >= start) & (centre < end))
        sub = lambda a: None if a is None else a[keep]  # noqa: E731
        return RemoteScores(
            tokens=[self.tokens[i] for i in keep],
            char_start=self.char_start[keep],
            char_end=self.char_end[keep],
            logprob=self.logprob[keep],
            rank=self.rank[keep],
            model_name=self.model_name,
            clipped=self.clipped,
            neurons=self.neurons,
            entropy=sub(self.entropy),
            mu=sub(self.mu),
            sigma2=sub(self.sigma2),
            top_mass=sub(self.top_mass),
        )


class RemoteLMScorer:
    """Client for ``observer-worker``. Caches every response to disk."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        secret: str | None = None,
        cache_dir: Path | None = None,
        timeout: float = 120.0,
        retries: int = 4,
        top_k: int = 0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.top_k = int(top_k)
        self.secret = secret or os.environ.get("PALIMPSEST_PROBE_SECRET") or _secret_from_env_file()
        if not self.secret:
            raise ValueError(
                "no probe secret; set PALIMPSEST_PROBE_SECRET or put PROBE_SECRET=... in "
                "observer-worker/.env"
            )
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[3] / "data" / "cache" / "remote_scores"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ssl = _ssl_context()
        self.spent_neurons = 0.0
        self.n_calls = 0
        self.n_cached = 0
        self.n_misaligned = 0

    # -- public API ---------------------------------------------------------------

    def score(self, text: str) -> RemoteScores:
        payload = self._fetch(text)
        toks = payload["tokens"]
        if not toks:
            return _empty(self.model)
        self._check_alignment(text, payload, toks)
        return RemoteScores(
            tokens=[t["token"] for t in toks],
            char_start=np.asarray([t["start"] for t in toks], dtype=np.int32),
            char_end=np.asarray([t["end"] for t in toks], dtype=np.int32),
            logprob=np.asarray([t["logprob"] for t in toks], dtype=np.float32),
            rank=np.asarray([t["rank"] for t in toks], dtype=np.int32),
            entropy=_col(toks, "entropy"),
            mu=_col(toks, "mu"),
            sigma2=_col(toks, "sigma2"),
            top_mass=_col(toks, "mass"),
            model_name=payload.get("model", self.model),
            clipped=bool(payload.get("clipped")),
            neurons=payload.get("neurons"),
        )

    # -- internals ----------------------------------------------------------------

    def _check_alignment(self, text: str, payload: dict, toks: list[dict]) -> None:
        """Verify the token stream really describes THIS text, and say so if it does not.

        Every sentence-level feature is computed by slicing the token stream on character
        offsets, so offsets that drift do not raise -- they quietly attribute the wrong
        tokens to the wrong sentence, and the resulting features look entirely plausible.
        That happened: the first top-k aligner picked the longest candidate that matched at
        the cursor, which chose " blueprint" (rank 2) over the realised " blue" (rank 3) and
        shifted every later offset in the essay by five characters.

        Three things are checked, none of which need the tokenizer: offsets must be
        non-decreasing, each token must be the text it claims to span, and the worker's own
        `misaligned` count must be zero. Cheap, and it turns a silent corruption into a line
        of output.
        """
        bad = 0
        prev_end = -1
        for t in toks:
            start, end = int(t["start"]), int(t["end"])
            if start < prev_end or text[start:end] != t["token"]:
                bad += 1
            prev_end = end
        reported = int(payload.get("misaligned") or 0)
        if bad or reported:
            self.n_misaligned += 1
            log.warning(
                "observer token stream does not align with the text: %d/%d tokens "
                "mismatched, worker reported %d unresolved. Features from this document "
                "are NOT trustworthy. (top_k=%d, %d chars)",
                bad, len(toks), reported, self.top_k, len(text),
            )

    def _cache_path(self, text: str) -> Path:
        # top_k is part of the key: a k=0 reply and a k=20 reply for the same text are
        # different payloads (the second carries the distribution statistics), and silently
        # serving one for the other would make entropy and curvature vanish or appear
        # depending on which run happened to populate the cache first.
        suffix = "" if self.top_k <= 0 else f"\x00k{self.top_k}"
        key = hashlib.sha256(f"{self.model}\x00{text}{suffix}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _fetch(self, text: str) -> dict:
        path = self._cache_path(text)
        if path.exists():
            self.n_cached += 1
            return json.loads(path.read_text(encoding="utf-8"))

        req_body = {"text": text[:MAX_CHARS], "model": self.model}
        if self.top_k > 0:
            req_body["top_k"] = self.top_k
        body = json.dumps(req_body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.endpoint,
                    data=body,
                    headers={
                        "authorization": f"Bearer {self.secret}",
                        "content-type": "application/json",
                        # Cloudflare's edge rejects urllib's default agent string with a
                        # 403 before the Worker ever runs -- which looks exactly like an
                        # auth failure but is not one. Identify ourselves honestly.
                        "user-agent": "palimpsest-observer-client/1.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                # Workers AI rate-limits under load; back off rather than hammering an
                # allowance we are trying to conserve.
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"probe failed after {self.retries} attempts: {last}")

        if "error" in payload:
            raise RuntimeError(f"probe error: {payload['error']}")

        path.write_text(json.dumps(payload), encoding="utf-8")
        self.n_calls += 1
        if isinstance(payload.get("neurons"), (int, float)):
            self.spent_neurons += float(payload["neurons"])
        return payload


#: Features that cannot be computed from this instrument, because the API returns only the
#: realised token's log-probability and rank -- never the full predictive distribution.
#: ``mean_entropy``/``entropy_sd`` need H of that distribution; ``curvature`` needs its mean
#: and variance. They are reported as NaN, which the pipeline already treats as "not
#: measured", rather than approximated from a truncated top-k and labelled as though they
#: were the statistic their name claims.
REMOTE_UNAVAILABLE: tuple[str, ...] = ("mean_entropy", "entropy_sd", "curvature")

#: With ``top_k > 0`` the observer returns its k most likely candidates at every position,
#: so those three become computable -- from the head of the distribution rather than all of
#: it. Nothing is unavailable then, and ``remote_unavailable()`` says so.
def remote_unavailable(top_k: int = 0) -> tuple[str, ...]:
    """Which features this instrument cannot measure, given how it was asked."""
    return () if top_k > 0 else REMOTE_UNAVAILABLE


class RemoteObserverScorer:
    """Presents :class:`RemoteLMScorer` with the interface :class:`Analyzer` expects.

    ``Analyzer`` and ``build_features.py`` are written against ``LocalLMScorer.score`` ->
    ``TokenScores``. Rather than fork the feature pipeline for a second observer -- which is
    exactly how training and serving drift apart -- this adapter fills the three
    unavailable arrays with NaN and hands back the same dataclass. Every feature that only
    needs log-probability or rank (11 of the 14 model features, and all 29 surface, corpus
    and context features) is then computed by the identical code path.
    """

    def __init__(self, **kwargs) -> None:
        self._remote = RemoteLMScorer(**kwargs)
        self.model_name = self._remote.model
        #: Mirrors LocalLMScorer.device so callers that report what is loaded -- /api/health
        #: among them -- work against either scorer. "remote" is the honest value: no
        #: device on this machine is doing the work.
        self.device = "remote"
        self.max_length = MAX_CHARS

    @property
    def spent_neurons(self) -> float:
        return self._remote.spent_neurons

    @property
    def n_calls(self) -> int:
        return self._remote.n_calls

    @property
    def n_cached(self) -> int:
        return self._remote.n_cached

    @property
    def n_misaligned(self) -> int:
        return self._remote.n_misaligned

    def score(self, text: str):
        from .local_lm import TokenScores

        r = self._remote.score(text)
        nan = np.full(len(r), np.nan, dtype=np.float32)
        return TokenScores(
            tokens=r.tokens,
            char_start=r.char_start,
            char_end=r.char_end,
            logprob=r.logprob,
            rank=r.rank,
            # NaN unless the observer was asked for top_k candidates. `is None` and not
            # `or nan`: an all-NaN array returned by a real top_k call would be a
            # measurement that happened to fail, and must not be indistinguishable from
            # never having asked.
            entropy=nan if r.entropy is None else r.entropy,
            mu=nan if r.mu is None else r.mu,
            sigma2=nan if r.sigma2 is None else r.sigma2,
            model_name=r.model_name,
            device="remote",
        )


def _col(toks: list[dict], name: str) -> np.ndarray | None:
    """Pull one optional per-token column, or None if the observer did not report it.

    None rather than a NaN array so the caller can tell "asked for top_k and the model
    answered" from "did not ask": the first is a measurement, the second is an absence, and
    conflating them is how an unmeasured feature ends up with a coefficient.
    """
    if not toks or name not in toks[0]:
        return None
    return np.asarray([t.get(name, float("nan")) for t in toks], dtype=np.float32)


def _secret_from_env_file() -> str | None:
    p = Path(__file__).resolve().parents[3] / "observer-worker" / ".env"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("PROBE_SECRET="):
            return line.split("=", 1)[1].strip()
    return None


def _empty(model: str) -> RemoteScores:
    return RemoteScores(
        tokens=[],
        char_start=np.zeros(0, dtype=np.int32),
        char_end=np.zeros(0, dtype=np.int32),
        logprob=np.zeros(0, dtype=np.float32),
        rank=np.zeros(0, dtype=np.int32),
        model_name=model,
        clipped=False,
        neurons=None,
    )

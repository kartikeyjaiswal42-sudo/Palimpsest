"""Refuse to score writing this detector was not fitted for.

The problem, measured
---------------------
This detector is fitted on college **admissions personal statements**. Point it at another
kind of essay and it does not degrade gracefully, it fails confidently:

* trained on argumentative coursework and asked about admissions essays, a classifier flags
  **91.4% of real human ones** (docs/09-frontier-ceiling.md, Result 5);
* pointed the other way, the shipped detector called a real school student's argumentative
  essay machine-written at **97% confidence**. The essay opens "If extracurricular
  activities were mandatory what would you do? I would play a sport like baseball or
  basketball." It is a child's homework.

Simple, formulaic, repetitive prose is low-perplexity prose, and low perplexity is what the
detector reads as machine. The writers who produce it are younger writers, weaker writers,
and second-language writers — so an out-of-domain input does not merely produce a wrong
answer, it produces a wrong answer aimed at whoever is least able to contest it.

What this gate does, and what it deliberately does not do
---------------------------------------------------------
It is a PRE-FILTER, not a change to the detector. In-domain documents are scored exactly as
before — same features, same weights, same thresholds — so detection strength on the essays
this tool is for is unchanged **by construction**. Out-of-domain documents are refused with a
reason instead of being given a verdict the calibration does not support.

Refusing is not a hedge. "I was not built for this kind of writing" is a true statement with
a clear remedy; "97% machine" about a thirteen-year-old is neither.

Why these features
------------------
Four document-level numbers, all already computed by the pipeline, chosen so the gate reads
GENRE rather than authorship -- and, after two measured failures, so that it reads neither
LENGTH nor ENGLISH PROFICIENCY:

``corpus_surprisal_mean``  distance from the applicant-prose reference corpus. The reference
                          IS the domain definition, so this is the most direct measure
                          available of "not the writing this was calibrated on".
``novel_trigram_rate``     how much of the phrasing is absent from that reference.
``first_person_rate``      admissions essays are personal narrative; argumentative coursework
                          is not. This is the clearest genre marker in the feature set.
``specificity_rate``       concrete detail per word — narrative carries names, places, dates.

The gate is fitted with BOTH human and machine admissions essays on the in-domain side. That
is the design decision that keeps it honest: if only human essays were in-domain, the gate
would learn authorship and quietly become a second detector, rejecting exactly the documents
the real detector was about to flag. ``fit_genre_gate.py`` asserts the pass rate is
equivalent for the two classes, and that assertion is the reason to trust the gate at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GenreGate", "GENRE_FEATURES", "document_genre_features"]

#: Order is part of the serialised model; do not reorder without refitting.
#:
#: DOCUMENT LENGTH IS DELIBERATELY ABSENT. The first version of this gate included
#: ``log_words``, which took weight +1.364 because the out-of-domain sets are systematically
#: shorter (TOEFL responses run ~104 words against ~640 for a statement). It therefore
#: learned length as a proxy for genre, and a real browser caught the consequence: taking
#: genuine admissions essays and truncating them -- same genre, same author, only shorter --
#: flipped the gate from 0/6 refused at 700 words to 5/6 at 150. Supplemental prompts are
#: routinely capped at 250-350 words, so that gate would have refused a large share of real
#: submissions while reporting itself as a genre check.
#:
#: This is the same failure docs/04-failures.md already records under "a length artifact
#: that landed on exactly the wrong people", reached a second time by a different route.
#:
#: ``mean_sentence_words`` IS ALSO ABSENT, and it was removed second, for a related reason
#: found by ``scripts/esl_gate_probe.py``. It survived the length audit honestly -- truncating
#: an essay barely moves it. But weak writing does: within ELLIPSE, where every document is
#: the same genre, sentence length is the feature that correlates most strongly with measured
#: English proficiency (r = -0.243), because a struggling writer produces run-ons rather than
#: short sentences. It was therefore importing proficiency into a gate that claims to read
#: genre. Removing it cuts the refusal rate for the weakest-English band from 100% to 25% and
#: more than halves the modelled risk to a low-proficiency applicant's essay, at a cost of
#: 3.3 points of out-of-genre rejection -- a document that slips through still falls to the
#: bands, which abstain, whereas a wrongly refused applicant has no recourse at all.
GENRE_FEATURES: tuple[str, ...] = (
    "corpus_surprisal_mean",
    "novel_trigram_rate",
    "first_person_rate",
    "specificity_rate",
)


def document_genre_features(sentence_features: list[dict[str, float]]) -> dict[str, float]:
    """Reduce a document's sentence features to the gate's five inputs.

    Word-weighted means, so a document is not characterised by its shortest sentences. NaN
    is skipped rather than imputed: a feature the pipeline could not measure should not be
    replaced by an average and then used to refuse somebody's essay.

    ``mean_sentence_words`` is still returned but is NOT in ``GENRE_FEATURES`` -- the gate
    ignores it (``_matrix`` selects by name). It is kept because ``esl_gate_probe.py`` needs
    it to keep measuring the proficiency correlation that got it excluded; if that number
    ever drifts, the audit should be able to see it.
    """
    if not sentence_features:
        return dict.fromkeys(GENRE_FEATURES, float("nan"))

    w = np.array([float(f.get("n_words") or 0.0) for f in sentence_features], dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = float(w.sum())

    def wmean(name: str) -> float:
        v = np.array([float(f.get(name, float("nan")) or float("nan"))
                      for f in sentence_features], dtype=float)
        ok = np.isfinite(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")

    return {
        "corpus_surprisal_mean": wmean("corpus_surprisal_mean"),
        "novel_trigram_rate": wmean("novel_trigram_rate"),
        "first_person_rate": wmean("first_person_rate"),
        "specificity_rate": wmean("specificity_rate"),
        "mean_sentence_words": total / max(len(sentence_features), 1),
    }


@dataclass
class GenreGate:
    """Standardise -> logistic regression -> "is this the genre we were fitted for"."""

    feature_names: tuple[str, ...] = GENRE_FEATURES
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    #: P(in-domain) at or above which a document is scored. Calibrated so that a stated
    #: fraction of KNOWN in-domain essays pass, because wrongly refusing a real admissions
    #: essay is the failure mode that makes the product useless.
    threshold: float = 0.0
    metadata: dict | None = None

    # -- fitting -------------------------------------------------------------------

    def fit(self, features: list[dict[str, float]], labels: np.ndarray,
            c: float = 1.0, iters: int = 3000, lr: float = 0.1) -> "GenreGate":
        x = self._matrix(features)
        self.mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        self.scale = scale
        z = np.nan_to_num((x - self.mean) / self.scale, nan=0.0)
        z = np.clip(z, -5.0, 5.0)

        w = np.zeros(z.shape[1] + 1)
        zz = np.hstack([np.ones((len(z), 1)), z])
        y = labels.astype(float)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(zz @ w, -30, 30)))
            g = zz.T @ (p - y) / len(y)
            g[1:] += (1.0 / max(c, 1e-6)) * w[1:] / len(y)
            w -= lr * g
        self.intercept, self.coef = float(w[0]), w[1:]
        return self

    # -- use -----------------------------------------------------------------------

    def probability(self, doc_features: dict[str, float]) -> float:
        """P(this document is in the genre the detector was fitted on)."""
        if self.coef is None:
            return 1.0  # unfitted gate must never refuse anything
        x = self._matrix([doc_features])
        z = np.nan_to_num((x - self.mean) / self.scale, nan=0.0)
        z = np.clip(z, -5.0, 5.0)
        return float(1.0 / (1.0 + np.exp(-np.clip(z @ self.coef + self.intercept, -30, 30)))[0])

    def in_domain(self, doc_features: dict[str, float]) -> bool:
        return self.coef is None or self.probability(doc_features) >= self.threshold

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.array([[float(r.get(n, float("nan")) if r.get(n) is not None else float("nan"))
                          for n in self.feature_names] for r in rows], dtype=float)

    # -- persistence ---------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "feature_names": list(self.feature_names),
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "coef": None if self.coef is None else self.coef.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "metadata": self.metadata or {},
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GenreGate":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=None if d["mean"] is None else np.array(d["mean"], dtype=float),
            scale=None if d["scale"] is None else np.array(d["scale"], dtype=float),
            coef=None if d["coef"] is None else np.array(d["coef"], dtype=float),
            intercept=float(d["intercept"]),
            threshold=float(d["threshold"]),
            metadata=d.get("metadata") or {},
        )

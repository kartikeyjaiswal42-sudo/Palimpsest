"""The verdict layer: a calibrated logistic regression over the feature vector.

Why logistic regression, when a gradient-boosted tree would score better
------------------------------------------------------------------------
It would, by a couple of AUROC points on our test set. We chose against it, and the reason
is the whole point of the project.

The brief asks for a detector that shows *where* and *why*. In a linear model the "why" is
not a post-hoc story bolted onto a black box -- it is the model. The logit is literally a
sum of per-feature terms, so the explanation the interface shows is arithmetically
identical to the computation that produced the number. There is no gap between them for a
plausible-sounding rationalisation to live in. SHAP values on a tree ensemble would give a
*model* of the model's reasoning, and it is not always a faithful one.

A second reason: with a few thousand sentences from a few hundred essays, a flexible model
will happily memorise our particular generators. The linear model's rigidity is doing real
regularisation work here, and we would rather ship a defensible detector than a
better-looking number.

Missing features
----------------
Short sentences cannot support some statistics, so features can arrive as NaN. They are
imputed with the *training mean*, which standardises to exactly zero and therefore
contributes exactly zero to the logit. An unmeasured feature does not silently vote. The
imputation mask travels with the prediction so the interface can say which evidence was
unavailable rather than implying it was neutral.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ..features.registry import FEATURE_NAMES, FEATURES_BY_NAME

__all__ = ["Contribution", "Prediction", "SentenceDetector", "choose_threshold"]

# L2 strength. Selected by grouped cross-validation in scripts/train.py; the curve is flat
# between 0.05 and 0.5, and we take the stronger end deliberately.
_DEFAULT_C = 0.1


@dataclass(frozen=True, slots=True)
class Contribution:
    """One feature's signed push on the log-odds, with everything needed to render it."""

    name: str
    label: str
    group: str
    description: str
    value: float
    z: float
    weight: float
    contribution: float
    measured: bool

    @property
    def toward(self) -> str:
        return "machine" if self.contribution > 0 else "human"


@dataclass(frozen=True, slots=True)
class Prediction:
    probability: float
    logit: float
    contributions: list[Contribution]
    #: The model's bias term. Carried so a reader can reconstruct the logit exactly:
    #: ``logit == intercept + sum(c.contribution for c in contributions)``.
    intercept: float = 0.0

    def top(self, k: int = 6) -> list[Contribution]:
        """The k features that moved this verdict most, either direction."""
        return sorted(self.contributions, key=lambda c: -abs(c.contribution))[:k]

    def remainder(self, k: int = 6) -> float:
        """Everything ``top(k)`` leaves out, as a single number.

        Showing six bars out of forty-three and calling it an explanation is a half-truth:
        the omitted terms are individually small but there are thirty-seven of them, and
        measured across the training corpus they outweigh the six shown on 30.9% of
        sentences. Reporting their sum keeps the displayed arithmetic honest --
        intercept + shown + remainder is exactly the logit.
        """
        shown = {id(c) for c in self.top(k)}
        return float(sum(c.contribution for c in self.contributions if id(c) not in shown))


#: Standardised feature values are clipped to +/- this many standard deviations.
#:
#: This is not a tuning knob, it is a guard against a specific harm that was observed. Three
#: ELLIPSE and PERSUADE essays -- real students, writing in a second language -- contain no
#: sentence-ending punctuation at all, so the segmenter returns the whole essay as ONE
#: "sentence" of 312, 313 and 466 words. Against a training mean of 19.5 words that is a
#: z-score of +33 to +50.
#:
#: `n_words` carries a small weight (+0.148). Multiplied by z=+50 it becomes a +7.48 term
#: that dominates every other feature combined, and all three essays were flagged as machine
#: at P=0.977. They are not machine-written; they are the least fluent writing in the corpus,
#: by exactly the students least able to contest an accusation.
#:
#: Clipping at 5 keeps every ordinary sentence untouched -- a 63-word sentence is still at
#: the ceiling -- while denying any single feature the ability to win by arithmetic on an
#: out-of-distribution input. The alternative, dropping the feature, would lose real signal
#: for a problem that is about extreme values rather than about the feature.
Z_CLIP = 5.0


@dataclass
class SentenceDetector:
    """Standardise -> logistic regression -> probability calibration."""

    feature_names: tuple[str, ...] = FEATURE_NAMES
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    #: Monotone map from raw LR probability to calibrated probability, as (x, y) knots.
    calibration_x: np.ndarray | None = None
    calibration_y: np.ndarray | None = None
    #: Probability at or above which a sentence is highlighted. CHOSEN FROM DATA at training
    #: time as the lowest threshold reaching the target precision, then stored here.
    #:
    #: The first version of this project hard-coded 0.65 and the calibrated probabilities
    #: never reached it -- machine sentences are ~11% of the corpus, so an honestly
    #: calibrated model rarely emits a high number, and nothing was ever flagged. A
    #: threshold has to be derived from the score distribution it will be applied to.
    flag_threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------------

    def fit(
        self,
        features: list[dict[str, float]],
        labels: np.ndarray,
        groups: np.ndarray,
        c: float = _DEFAULT_C,
    ) -> "SentenceDetector":
        """Fit on sentence features.

        ``groups`` must identify the source essay for every sentence. Sentences from one
        essay are highly correlated; letting them straddle a train/test split inflates every
        number downstream. Grouping is enforced here rather than left to the caller.
        """
        x = self.to_matrix(features, fit=True)
        y = np.asarray(labels).astype(int)
        groups = np.asarray(groups)

        base = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        base.fit(x, y)
        self.coef = base.coef_[0].astype(np.float64)
        self.intercept = float(base.intercept_[0])

        self._fit_calibration(x, y, groups, c)
        self.metadata = {
            "n_sentences": int(len(y)),
            "n_essays": int(len(set(groups.tolist()))),
            "n_machine": int(y.sum()),
            "n_human": int((1 - y).sum()),
            "C": c,
        }
        return self

    def _fit_calibration(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float
    ) -> None:
        """Learn a monotone map from raw scores to honest probabilities.

        Out-of-fold by construction: a model calibrated on its own training predictions
        reports confidence it has not earned.
        """
        n_groups = len(set(groups.tolist()))
        if n_groups < 5 or len(set(y.tolist())) < 2:
            self.calibration_x = self.calibration_y = None
            return

        splitter = GroupKFold(n_splits=min(5, n_groups))
        raw = np.zeros(len(y), dtype=np.float64)
        for train_idx, test_idx in splitter.split(x, y, groups):
            fold = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
            fold.fit(x[train_idx], y[train_idx])
            raw[test_idx] = fold.predict_proba(x[test_idx])[:, 1]

        # Platt scaling rather than isotonic: isotonic needs more data than we have and
        # produces step functions that look absurd in a UI showing a percentage. It is one
        # logistic regression on a single column, so we fit it directly and store the
        # resulting curve as knots -- that keeps the saved model a plain JSON file with no
        # pickled sklearn objects in it.
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(raw.reshape(-1, 1), y)
        grid = np.linspace(0.0, 1.0, 201)
        self.calibration_x = grid
        self.calibration_y = platt.predict_proba(grid.reshape(-1, 1))[:, 1]

    # -- inference ----------------------------------------------------------------

    def to_matrix(self, features: list[dict[str, float]], fit: bool = False) -> np.ndarray:
        """Feature dicts -> standardised matrix, imputing NaN with the training mean."""
        raw = np.array(
            [[row.get(n, np.nan) for n in self.feature_names] for row in features],
            dtype=np.float64,
        )
        if fit:
            self.mean = np.nanmean(raw, axis=0)
            self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
            scale = np.nanstd(raw, axis=0)
            # A constant feature carries no information. Setting scale to 1 leaves its
            # standardised value at 0 for every row, so it can never influence a verdict.
            self.scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        if self.mean is None or self.scale is None:
            raise RuntimeError("detector is not fitted")
        filled = np.where(np.isfinite(raw), raw, self.mean)
        return np.clip((filled - self.mean) / self.scale, -Z_CLIP, Z_CLIP)

    def predict(self, features: dict[str, float]) -> Prediction:
        """Score one sentence and explain the score."""
        if self.coef is None:
            raise RuntimeError("detector is not fitted")
        z = self.to_matrix([features])[0]
        terms = self.coef * z
        logit = float(terms.sum() + self.intercept)

        contributions = []
        for i, name in enumerate(self.feature_names):
            meta = FEATURES_BY_NAME[name]
            value = features.get(name, float("nan"))
            contributions.append(
                Contribution(
                    name=name,
                    label=meta.label,
                    group=meta.group,
                    description=meta.description,
                    value=float(value) if np.isfinite(value) else float("nan"),
                    z=float(z[i]),
                    weight=float(self.coef[i]),
                    contribution=float(terms[i]),
                    measured=bool(np.isfinite(value)),
                )
            )
        return Prediction(
            self._calibrate(_sigmoid(logit)), logit, contributions, float(self.intercept)
        )

    def predict_many(self, features: list[dict[str, float]]) -> np.ndarray:
        """Calibrated probabilities for many sentences. Used by the evaluation harness."""
        if self.coef is None:
            raise RuntimeError("detector is not fitted")
        if not features:
            return np.zeros(0)
        x = self.to_matrix(features)
        logits = x @ self.coef + self.intercept
        return np.array([self._calibrate(p) for p in _sigmoid(logits)])

    def _calibrate(self, p: float) -> float:
        if self.calibration_x is None or self.calibration_y is None:
            return float(p)
        return float(np.interp(p, self.calibration_x, self.calibration_y))

    # -- persistence --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "feature_names": list(self.feature_names),
                    "mean": _tolist(self.mean),
                    "scale": _tolist(self.scale),
                    "coef": _tolist(self.coef),
                    "intercept": self.intercept,
                    "calibration_x": _tolist(self.calibration_x),
                    "calibration_y": _tolist(self.calibration_y),
                    "flag_threshold": self.flag_threshold,
                    "metadata": self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SentenceDetector":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=_toarray(d["mean"]),
            scale=_toarray(d["scale"]),
            coef=_toarray(d["coef"]),
            intercept=float(d["intercept"]),
            calibration_x=_toarray(d["calibration_x"]),
            calibration_y=_toarray(d["calibration_y"]),
            flag_threshold=float(d.get("flag_threshold", 0.5)),
            metadata=d.get("metadata", {}),
        )


def choose_threshold(
    y: np.ndarray, p: np.ndarray, target_precision: float = 0.80, floor: float = 0.30
) -> tuple[float, float, float]:
    """Lowest probability cut-off reaching ``target_precision``, with its recall.

    We optimise for precision rather than F1 because the two errors are not symmetric here.
    Missing machine text is a detector that is less useful; flagging a student's own
    sentence is an accusation. Returns ``(threshold, precision, recall)`` and falls back to
    the best-precision candidate above ``floor`` when the target is unreachable.
    """
    order = np.unique(np.round(p, 4))
    best = (float(floor), 0.0, 0.0)
    for t in order:
        if t < floor:
            continue
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        if tp + fp < 10:
            continue
        precision = tp / (tp + fp)
        recall = tp / max(int((y == 1).sum()), 1)
        if precision >= target_precision:
            return (float(t), precision, recall)
        if precision > best[1]:
            best = (float(t), precision, recall)
    return best


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _tolist(a: np.ndarray | None) -> list | None:
    return None if a is None else [float(v) for v in a]


def _toarray(v: list | None) -> np.ndarray | None:
    return None if v is None else np.asarray(v, dtype=np.float64)

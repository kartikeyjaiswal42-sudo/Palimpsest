"""Features that describe a sentence *relative to the rest of its own essay*.

Why this family exists
----------------------
The brief names the realistic case: "a paragraph a person wrote and a model later polished".
That case is nearly invisible to absolute thresholds. A polished paragraph is not
remarkable in isolation -- plenty of humans write smooth prose -- but it is remarkable
sitting inside four paragraphs by the same person that read nothing like it. The signal is
the *discontinuity*, not the level.

Two consequences follow, and both matter.

First, this is what makes passage-level detection work at all. An essay written entirely by
a model has no internal discontinuity, so these features stay silent and the absolute
features carry the verdict; a part-machine essay lights these up exactly at the seam.

Second -- and this is the reason we prioritised the family -- self-relative features are
structurally fairer to writers whose English is not native. Absolute likelihood features
punish anyone whose prose sits far from a native-speaker norm, which is precisely the
documented failure mode of every detector in this space. Comparing an author to *themselves*
removes the norm from the comparison. It does not eliminate the bias, and
``docs/05-esl.md`` reports how much it actually removed, measured rather than asserted.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["extract_context_features", "CONTEXT_FEATURE_NAMES", "MIN_SENTENCES"]

# Below this many sentences an essay has no stable internal baseline to compare against.
MIN_SENTENCES = 5

#: Cap on any in-document z-score. The quantity is unbounded when the baseline has no
#: spread, and a single feature must not be able to dominate the logit.
_DEGENERATE_Z = 6.0

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "logprob_z_in_doc",
    "curvature_z_in_doc",
    "len_z_in_doc",
    "style_gap_from_doc",
    "local_len_burstiness",
    "rel_position",
)

# The features whose joint deviation defines "this sentence does not sound like the rest of
# this essay". Chosen to span rhythm, register and construction rather than to be the
# individually strongest -- a style gap should not just restate the likelihood features.
_STYLE_KEYS: tuple[str, ...] = (
    "mean_logprob",
    "logprob_sd",
    "mean_log_rank",
    "curvature",
    "mean_word_len",
    "comma_rate",
    "subordination_rate",
    "function_word_rate",
    "discourse_marker_rate",
    "root_ttr",
    "n_words",
)


def extract_context_features(
    base: list[dict[str, float]], n_sentences: int
) -> list[dict[str, float]]:
    """Given every sentence's base features, derive each sentence's in-document features.

    Returns a list parallel to ``base``. All statistics are leave-one-out: a sentence is
    never compared against a baseline it is itself part of, which would shrink its own
    deviation and hide short essays' anomalies.
    """
    if n_sentences < MIN_SENTENCES:
        return [dict.fromkeys(CONTEXT_FEATURE_NAMES, float("nan")) for _ in base]

    matrix = np.array(
        [[row.get(k, np.nan) for k in _STYLE_KEYS] for row in base], dtype=np.float64
    )
    scaled = _robust_scale_columns(matrix)
    lengths = np.array([row.get("n_words", np.nan) for row in base], dtype=np.float64)

    out: list[dict[str, float]] = []
    for i in range(n_sentences):
        others = np.delete(scaled, i, axis=0)
        # A column can be entirely NaN -- a style statistic that no other sentence in this
        # essay was long enough to measure. numpy warns and returns NaN, which is exactly
        # the answer we want ("no baseline to compare against"), so the warning is noise.
        # It is silenced here rather than globally so a genuine all-NaN elsewhere still warns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmedian(others, axis=0)
        here = scaled[i]

        gap_terms = np.abs(here - centre)
        style_gap = float(np.nanmean(gap_terms)) if np.isfinite(gap_terms).any() else np.nan

        out.append(
            {
                # Signed: positive means this sentence is MORE predictable than the author's
                # own baseline, which is the direction that indicates polishing.
                "logprob_z_in_doc": _loo_z(base, i, "mean_logprob"),
                "curvature_z_in_doc": _loo_z(base, i, "curvature"),
                "len_z_in_doc": _loo_z(base, i, "n_words"),
                "style_gap_from_doc": style_gap,
                "local_len_burstiness": _local_burstiness(lengths, i),
                "rel_position": i / max(n_sentences - 1, 1),
            }
        )
    return out


def _loo_z(base: list[dict[str, float]], i: int, key: str) -> float:
    """Leave-one-out robust z-score of ``base[i][key]`` against the other sentences."""
    values = np.array([row.get(key, np.nan) for row in base], dtype=np.float64)
    here = values[i]
    if not np.isfinite(here):
        return float("nan")
    others = np.delete(values, i)
    others = others[np.isfinite(others)]
    if len(others) < 3:
        return float("nan")
    centre = float(np.median(others))
    # Median absolute deviation, scaled to be a consistent estimator of sigma for normal
    # data. Robust matters here: one inserted machine paragraph must not move the baseline
    # it is being measured against.
    mad = float(np.median(np.abs(others - centre))) * 1.4826
    if mad < 1e-9:
        # More than half the other sentences share one value, so the MAD is degenerate.
        # Fall back to the standard deviation; if that is zero too, every other sentence is
        # identical. A sentence differing from a perfectly uniform baseline is maximally
        # unusual, NOT typical -- returning 0.0 here (as an earlier version did) reported
        # the exact opposite of what the data says. Capped, because the true value is
        # unbounded and one feature should not be able to dominate the logit.
        spread = float(others.std(ddof=1))
        if spread < 1e-9:
            delta = here - centre
            if abs(delta) < 1e-9:
                return 0.0
            return float(np.sign(delta) * _DEGENERATE_Z)
        mad = spread
    return float(np.clip((here - centre) / mad, -_DEGENERATE_Z, _DEGENERATE_Z))


def _local_burstiness(lengths: np.ndarray, i: int, half_window: int = 2) -> float:
    """Coefficient of variation of sentence length in a window centred on ``i``."""
    lo = max(0, i - half_window)
    hi = min(len(lengths), i + half_window + 1)
    window = lengths[lo:hi]
    window = window[np.isfinite(window)]
    if len(window) < 3 or window.mean() < 1e-9:
        return float("nan")
    return float(window.std(ddof=1) / window.mean())


def _robust_scale_columns(matrix: np.ndarray) -> np.ndarray:
    """Median/MAD-scale each column so features with different units contribute equally."""
    out = np.full_like(matrix, np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < 3:
            continue
        centre = np.median(finite)
        mad = np.median(np.abs(finite - centre)) * 1.4826
        if mad < 1e-9:
            mad = finite.std(ddof=1) if finite.std(ddof=1) > 1e-9 else 1.0
        out[:, j] = (col - centre) / mad
    return out

#!/usr/bin/env python
"""Does COMBINING the weak per-statistic signals recover frontier detection?

    python scripts/fusion_probe.py

``bench_observer.py`` measured each statistic alone. Under the 30 B observer the best single
statistic reaches AUROC 0.955 on Claude Haiku but only 0.695 on Claude Opus. Two readings
survive that:

  (a) the Opus signal is REAL but SPREAD across several weak statistics, so a fitted
      combination recovers it, or
  (b) there is almost nothing there, and any apparent gain from fitting is the fit learning
      this particular sample.

The distinction decides whether frontier detection is an engineering problem or a wall, so
it must be tested in the only way that can tell them apart: the generator being scored is
held out ENTIRELY. The model is fitted on human essays plus machine essays from OTHER
generators, and then asked about a generator it has never seen. Fitting on Opus and scoring
Opus would answer a question nobody is asking -- in production the next model is always one
you did not fit on.

Two guards make the negative result trustworthy rather than merely disappointing:

  * a PERMUTATION test on the test labels. With ~30 machine and ~25 human documents the
    sampling error on an AUROC is large, so "0.64" needs a p-value before it means anything.
    The null holds the model's scores fixed and shuffles which documents are machine.
    (An earlier version of this script shuffled the TRAINING labels instead. That is the
    wrong null and it was discarded: shuffling training labels does not remove the feature
    differences between the two test sets, so a random weight vector still separates them,
    producing a null so wide that every result looked insignificant.)
  * leave-one-generator-out, repeated for every generator, so the reported number is not the
    best of several attempts.

Read TPR@5%FPR, not AUROC. AUROC answers "does the score rank machine above human on
average", which can be comfortably above chance while the detector still catches nothing at
a false-accusation rate anyone would accept. The two columns disagree here, and the second
one is the product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "artifacts" / "observer_bench_raw.json"
HUMAN = ("human_native", "human_esl")
MACHINE = ("gemini_flash_lite", "gemini_flash", "claude_haiku", "claude_sonnet", "claude_opus")
FEATS = ("mean_logprob", "logprob_sd", "mean_log_rank", "frac_rank_top1",
         "frac_rank_top10", "frac_rank_tail", "lrr")
SEED = 20260811


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[r.get(f, np.nan) for f in FEATS] for r in rows], dtype=np.float64)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos >= float(np.quantile(neg, 1.0 - fpr))).mean())


def fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.05) -> np.ndarray:
    """Plain gradient-descent logistic regression with L2. No sklearn dependency here."""
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y)
        g[1:] += 1e-3 * w[1:]
        w -= lr * g
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def standardise(train: np.ndarray, *others: np.ndarray):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd[sd < 1e-9] = 1.0
    out = [np.nan_to_num((a - mu) / sd, nan=0.0) for a in (train, *others)]
    return out


def main() -> int:
    if not RAW.exists():
        print("run scripts/bench_observer.py first")
        return 1
    data = json.loads(RAW.read_text())["remote"]

    human = matrix([r for s in HUMAN for r in data.get(s, [])])
    machine = {m: matrix(data.get(m, [])) for m in MACHINE if data.get(m)}
    print(f"human {len(human)} docs | " + " ".join(f"{k} {len(v)}" for k, v in machine.items()))
    print(f"features: {', '.join(FEATS)}\n")

    rng = np.random.default_rng(SEED)
    print(f"{'held-out generator':22s} {'n':>4s} {'AUROC':>7s} {'TPR@5%':>7s} "
          f"{'null p95':>11s} {'p-value':>9s} {'signif':>7s}")
    print("-" * 78)

    results = {}
    for target in machine:
        # Fit on humans + every OTHER generator. The target is unseen, as it would be
        # in production when a new model ships.
        others = [v for k, v in machine.items() if k != target]
        Xm = np.vstack(others)
        # Half the humans train, half act as the reference the threshold is read off, so
        # the operating point is never calibrated on the documents it is scored against.
        idx = rng.permutation(len(human))
        h_tr, h_te = human[idx[: len(human) // 2]], human[idx[len(human) // 2:]]

        X = np.vstack([h_tr, Xm])
        y = np.concatenate([np.zeros(len(h_tr)), np.ones(len(Xm))])
        Xs, h_te_s, tgt_s = standardise(X, h_te, machine[target])
        w = fit_logreg(Xs, y)
        p_pos, p_neg = predict(w, tgt_s), predict(w, h_te_s)
        a, t = auroc(p_pos, p_neg), tpr_at_fpr(p_pos, p_neg, 0.05)

        # Null: hold the SCORES fixed and shuffle which documents are labelled machine.
        # This is the right null for "does this score rank the true labels better than
        # chance". An earlier version shuffled the TRAINING labels instead, which does not
        # destroy the feature differences between the two test sets -- a random weight
        # vector over features that genuinely differ still separates them -- so it produced
        # a null with enormous variance and declared everything insignificant.
        scores = np.concatenate([p_pos, p_neg])
        n_pos = len(p_pos)
        nulls = np.empty(2000)
        for i in range(2000):
            sh = rng.permutation(len(scores))
            nulls[i] = auroc(scores[sh[:n_pos]], scores[sh[n_pos:]])
        p_value = float((nulls >= a).mean())
        null_hi = float(np.nanquantile(nulls, 0.95))

        verdict = "yes" if p_value < 0.05 else "NO"
        results[target] = {"auroc": a, "tpr_at_5fpr": t, "null_p95": null_hi,
                           "p_value": p_value, "beats_chance": verdict == "yes"}
        print(f"{target:22s} {len(machine[target]):4d} {a:7.3f} {t:7.3f} "
              f"{null_hi:11.3f} {p_value:9.4f} {verdict:>7s}")

    (ROOT / "artifacts" / "fusion_probe.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print("\nwrote artifacts/fusion_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

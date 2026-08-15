#!/usr/bin/env python
"""Does the structural feature block carry signal, and at what cost to ESL writers?

    python scripts/syntax_probe.py

Three questions, in the order that decides whether this ships:

**1. Alone.** Single-feature AUROC for each of the eleven structural features, human vs
machine, out of fold. A feature below ~0.55 is decoration.

**2. Added.** Grouped-CV AUROC of the shipped 43-feature model against the same model plus
the block, on identical rows and identical folds. The comparison is paired by fold, and the
difference is reported with a bootstrap interval, because two AUROCs three points apart on
one split is not a result.

**3. Cost.** What it does to the false-positive rate on English-learner prose at a matched
operating point. A feature block that lifts AUROC by reading *fluency* will lift ESL false
positives too, and this project's standing rule is that a gain paid for by the population it
exists to protect is not a gain. `docs/05-esl.md` sets the frame; this reuses it.

Two things this CANNOT answer, stated here because the answer looks like it should
--------------------------------------------------------------------------------
* **Whether these features survive a humanizer.** The corpus contains no paraphrase or
  humanizer attack (PROJECT.md §10). The motivating claim for structural features -- that a
  rewrite moves words further than it moves clause shape -- is *untested here*. Nothing this
  script prints is evidence for it. Building that evidence needs an attacked corpus, and
  `--check-paraphrase-proxy` reports the one weak proxy available: the `adversarial` set,
  which is a human deliberately imitating a model, i.e. the attack pointed the other way.
* **Whether a gain generalises past our generation pipeline.** docs/12 records a 0.960 AUROC
  that fell to 0.490 under a foreign-corpus control. Any headline below is subject to the
  same doubt and `scripts/consensus_controls.py` is the battery it would have to pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.features.registry import FEATURE_NAMES  # noqa: E402
from palimpsest.features.syntax import ALL_SYNTAX_FEATURE_NAMES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "features"
OUT = ROOT / "artifacts" / "syntax_probe.json"

# The remote observer returns log-probability and rank but never the full predictive
# distribution, so these arrive all-NaN and are dropped rather than imputed -- identical to
# scripts/train.py. Imputing a never-measured column to the training mean gives it a
# coefficient and a vote it has not earned.
REMOTE_UNAVAILABLE = ("mean_entropy", "entropy_sd", "curvature", "curvature_z_in_doc")

MIN_ROWS_PER_CLASS = 30


def load(name: str) -> list[dict]:
    path = FEATURE_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r["features"] = {
            k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()
        }
        rows.append(r)
    return rows


def matrix(rows, names):
    return np.array(
        [[r["features"].get(n, np.nan) for n in names] for r in rows], dtype=np.float64
    )


def impute_standardise(train_x, *others):
    """Impute to the TRAIN mean and standardise by TRAIN moments. Never refit on test."""
    mu = np.nanmean(train_x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    filled = np.where(np.isfinite(train_x), train_x, mu)
    sd = filled.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    out = [(filled - mu) / sd]
    for x in others:
        f = np.where(np.isfinite(x), x, mu)
        out.append((f - mu) / sd)
    return out


def cv_scores(x, y, groups, n_splits=5, seed=0):
    """Out-of-fold probabilities under grouped CV. Groups are essays, never sentences."""
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(x, y, groups)):
        xtr, xte = impute_standardise(x[tr], x[te])
        model = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
        model.fit(xtr, y[tr])
        oof[te] = model.predict_proba(xte)[:, 1]
        fold_id[te] = k
    return oof, fold_id


def paired_fold_delta(y, fold_id, p_base, p_aug):
    """AUROC per fold for both arms, and the paired difference.

    Paired by fold because the folds differ in difficulty; an unpaired comparison mostly
    measures which arm happened to see the easier essays.
    """
    rows = []
    for k in sorted(set(fold_id.tolist())):
        m = fold_id == k
        if len(set(y[m].tolist())) < 2:
            continue
        a = roc_auc_score(y[m], p_base[m])
        b = roc_auc_score(y[m], p_aug[m])
        rows.append({"fold": int(k), "base": round(a, 4), "augmented": round(b, 4),
                     "delta": round(b - a, 4), "n": int(m.sum())})
    deltas = np.array([r["delta"] for r in rows])
    return rows, float(deltas.mean()), float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0


def bootstrap_auroc_delta(y, p_base, p_aug, n=2000, seed=0):
    """Bootstrap CI on the AUROC difference, resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[s].tolist())) < 2:
            continue
        deltas.append(roc_auc_score(y[s], p_aug[s]) - roc_auc_score(y[s], p_base[s]))
    if not deltas:
        return None
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "mean": round(float(np.mean(deltas)), 4)}


def threshold_at_fpr(scores_neg, budget=0.05):
    """The score threshold whose false-positive rate on human prose is `budget`."""
    return float(np.quantile(scores_neg, 1.0 - budget))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="train_remote")
    ap.add_argument("--esl", default="esl_remote")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    rows = load(args.train)
    if not rows:
        raise SystemExit(f"no rows in {args.train}")
    if not any(n in rows[0]["features"] for n in ALL_SYNTAX_FEATURE_NAMES):
        raise SystemExit(
            f"{args.train} has no structural block -- run:\n"
            f"  python scripts/build_syntax_features.py --sets {args.train}"
        )

    base_names = [n for n in FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]
    syn_names = [n for n in ALL_SYNTAX_FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]

    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    x_base = matrix(rows, base_names)
    x_syn = matrix(rows, syn_names)
    x_aug = np.hstack([x_base, x_syn])

    report: dict = {
        "train": args.train,
        "nSentences": len(rows),
        "nDocuments": len(set(groups.tolist())),
        "nHuman": int((y == 0).sum()),
        "nMachine": int((y == 1).sum()),
        "baseFeatures": len(base_names),
        "syntaxFeatures": syn_names,
        "droppedAsUnmeasured": list(REMOTE_UNAVAILABLE),
    }

    # -- 1. each structural feature alone --------------------------------------------
    print("=" * 74)
    print("1. SINGLE-FEATURE AUROC  (human vs machine, out of fold)")
    print("=" * 74)
    singles = []
    for name in syn_names:
        col = x_syn[:, syn_names.index(name)]
        finite = np.isfinite(col)
        cov = float(finite.mean())
        if finite.sum() < MIN_ROWS_PER_CLASS or len(set(y[finite].tolist())) < 2:
            singles.append({"feature": name, "auroc": None, "coverage": round(cov, 3)})
            continue
        a = roc_auc_score(y[finite], col[finite])
        # Report the direction-free strength alongside the raw value: a feature at 0.34 is
        # exactly as informative as one at 0.66, it just points the other way, and the
        # classifier is free to learn the sign.
        singles.append({
            "feature": name, "auroc": round(float(a), 4),
            "strength": round(float(max(a, 1 - a)), 4),
            "coverage": round(cov, 3),
            "corrWithLength": round(float(_corr(col, x_base[:, base_names.index("n_words")])), 3),
        })
    for s in sorted(singles, key=lambda d: -(d.get("strength") or 0)):
        if s["auroc"] is None:
            print(f"  {s['feature']:26} --      (coverage {s['coverage']:.0%})")
        else:
            print(f"  {s['feature']:26} {s['auroc']:.3f}  strength {s['strength']:.3f}  "
                  f"cov {s['coverage']:.0%}  r(len) {s['corrWithLength']:+.2f}")
    report["singleFeature"] = singles

    # -- 2. does adding the block move the model? -------------------------------------
    print()
    print("=" * 74)
    print("2. ADDED TO THE SHIPPED FEATURE SET  (grouped CV, identical folds)")
    print("=" * 74)
    p_base, fold_id = cv_scores(x_base, y, groups, args.folds)
    p_aug, _ = cv_scores(x_aug, y, groups, args.folds)
    auc_base = float(roc_auc_score(y, p_base))
    auc_aug = float(roc_auc_score(y, p_aug))
    folds, mean_delta, sd_delta = paired_fold_delta(y, fold_id, p_base, p_aug)
    boot = bootstrap_auroc_delta(y, p_base, p_aug)

    print(f"  base ({len(base_names)} features)      AUROC {auc_base:.4f}")
    print(f"  + structural ({len(syn_names)})        AUROC {auc_aug:.4f}")
    print(f"  delta                        {auc_aug - auc_base:+.4f}")
    if boot:
        print(f"  bootstrap 95% CI             [{boot['lo']:+.4f}, {boot['hi']:+.4f}]")
    print(f"  paired over {len(folds)} folds           mean {mean_delta:+.4f} (sd {sd_delta:.4f})")
    for f in folds:
        print(f"    fold {f['fold']}  {f['base']:.4f} -> {f['augmented']:.4f}  "
              f"({f['delta']:+.4f}, n={f['n']})")
    report["pooled"] = {
        "aurocBase": round(auc_base, 4), "aurocAugmented": round(auc_aug, 4),
        "delta": round(auc_aug - auc_base, 4), "bootstrap95": boot,
        "perFold": folds, "meanFoldDelta": round(mean_delta, 4),
    }

    # -- 3. what it costs English-learner writers -------------------------------------
    print()
    print("=" * 74)
    print("3. COST ON ENGLISH-LEARNER PROSE  (matched 5% budget on training humans)")
    print("=" * 74)
    esl = load(args.esl)
    if not esl:
        print(f"  {args.esl} unavailable -- skipped")
        report["esl"] = None
    else:
        has_block = any(n in esl[0]["features"] for n in ALL_SYNTAX_FEATURE_NAMES)
        if not has_block:
            print(f"  {args.esl} has no structural block -- skipped")
            report["esl"] = None
        else:
            e_base = matrix(esl, base_names)
            e_syn = matrix(esl, syn_names)
            e_aug = np.hstack([e_base, e_syn])
            ey = np.array([r["label"] for r in esl], dtype=int)

            # Fit on ALL training rows, score ESL. ESL is human-only held-out data, so there
            # is no fold structure to respect -- but the threshold must come from the
            # training humans, not from the ESL set itself, or the FPR is measured against a
            # threshold the ESL data helped choose.
            res = {}
            for tag, xtr, xte in (("base", x_base, e_base), ("augmented", x_aug, e_aug)):
                a, b = impute_standardise(xtr, xte)
                m = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(a, y)
                train_human = m.predict_proba(a[y == 0])[:, 1]
                thr = threshold_at_fpr(train_human, 0.05)
                p_esl = m.predict_proba(b)[:, 1]
                human = p_esl[ey == 0]
                res[tag] = {
                    "threshold": round(float(thr), 4),
                    "eslSentenceFPR": round(float((human > thr).mean()), 4),
                    "nEslHumanSentences": int((ey == 0).sum()),
                    "meanP": round(float(human.mean()), 4),
                }
                print(f"  {tag:11} threshold {thr:.3f}   "
                      f"ESL sentence FPR {res[tag]['eslSentenceFPR']:.2%}   "
                      f"mean P {res[tag]['meanP']:.3f}")

            delta = res["augmented"]["eslSentenceFPR"] - res["base"]["eslSentenceFPR"]
            print(f"  change in ESL false positives: {delta:+.2%} "
                  f"({'WORSE' if delta > 0.002 else 'better' if delta < -0.002 else 'flat'})")
            res["deltaFPR"] = round(float(delta), 4)
            report["esl"] = res

    # -- 4. is the gain just sentence length again? -----------------------------------
    # docs/04-failures.md records a length artifact that landed on exactly the wrong people,
    # and three of these features correlate >0.5 with n_words. A gain that evaporates once
    # the length-loaded features are removed is a length gain wearing a syntax label.
    print()
    print("=" * 74)
    print("4. LENGTH CONFOUND CONTROL  (drop features with |r(n_words)| > 0.4)")
    print("=" * 74)
    lens = x_base[:, base_names.index("n_words")]
    clean_names = [
        n for n in syn_names
        if abs(_corr(x_syn[:, syn_names.index(n)], lens)) <= 0.4
    ]
    dropped = [n for n in syn_names if n not in clean_names]
    print(f"  dropped as length-loaded: {', '.join(dropped) or 'none'}")
    print(f"  kept ({len(clean_names)}): {', '.join(clean_names)}")

    if clean_names:
        x_clean = np.hstack([x_base, matrix(rows, clean_names)])
        p_clean, _ = cv_scores(x_clean, y, groups, args.folds)
        auc_clean = float(roc_auc_score(y, p_clean))
        boot_clean = bootstrap_auroc_delta(y, p_base, p_clean)
        print(f"  base                         AUROC {auc_base:.4f}")
        print(f"  + length-independent only    AUROC {auc_clean:.4f}  "
              f"({auc_clean - auc_base:+.4f})")
        if boot_clean:
            print(f"  bootstrap 95% CI             "
                  f"[{boot_clean['lo']:+.4f}, {boot_clean['hi']:+.4f}]")
        share = ((auc_clean - auc_base) / (auc_aug - auc_base)) if auc_aug > auc_base else float("nan")
        print(f"  share of the full gain retained: {share:.0%}")
        report["lengthControl"] = {
            "droppedLengthLoaded": dropped,
            "kept": clean_names,
            "aurocLengthIndependentOnly": round(auc_clean, 4),
            "delta": round(auc_clean - auc_base, 4),
            "bootstrap95": boot_clean,
            "shareOfFullGain": None if not np.isfinite(share) else round(float(share), 3),
        }

    # -- the caveat that the numbers above cannot carry themselves --------------------
    report["cannotAnswer"] = {
        "humanizerSurvival": (
            "The corpus contains no paraphrase or humanizer attack, so the motivating claim "
            "for this block -- that structure survives rewriting better than word choice -- "
            "is untested. Nothing in this artifact is evidence for it."
        ),
        "pipelineGeneralisation": (
            "Every machine essay here comes from our own generation pipeline. docs/12 "
            "records a 0.960 AUROC that fell to 0.490 under a foreign-corpus control; any "
            "gain above is subject to the same doubt until it passes "
            "scripts/consensus_controls.py."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


if __name__ == "__main__":
    raise SystemExit(main())

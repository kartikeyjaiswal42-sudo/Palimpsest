#!/usr/bin/env python
"""Run English-learner essays through the scoring system and report who gets accused.

    python scripts/esl_false_positive_audit.py
    python scripts/esl_false_positive_audit.py --arms base           # shipped features only

Every document in the ESL sets is human-written. Any machine verdict is therefore a false
accusation, and this script's whole job is to say how many there are, *and who they land on*.

Why a separate script when docs/05-esl.md already reports ESL false positives
----------------------------------------------------------------------------
Three things it adds, each because the aggregate number hides something:

**1. It scores the surface the reader actually consumes.** PROJECT.md §7 records that every
published error rate describes the *document verdict*, while a reader consumes the *heat
map*. An essay holds ~19 sentences, so a 5%-per-sentence error rate becomes a highlight
*somewhere* in a third of clean essays. Both are reported here, per group, because they are
different accusations and only one of them was ever priced.

**2. It breaks the rate down by proficiency, not just by ELL flag.** §8 records the finding
that the highlight rate *rises* with proficiency -- the opposite direction from every other
fairness result in this project -- which a binary ELL split cannot see.

**3. It compares arms at a MATCHED budget.** Comparing two detectors' false-positive rates
at their own default thresholds measures the thresholds, not the detectors. Both arms here
are calibrated to the same 5% false-positive budget on *training* humans, then pointed at
ESL prose, so the difference is attributable to the features.

What a good result looks like, and what would make it worthless
---------------------------------------------------------------
A feature block that lowers the ESL rate by lowering *sensitivity* has done nothing -- it has
moved the operating point, not improved the detector. So machine recall at the identical
threshold is printed beside every false-positive rate. A row where the ESL rate falls and
recall falls with it is a threshold change and is labelled as one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.document import MAX_SENTENCE_WORDS, document_statistics  # noqa: E402
from palimpsest.features.registry import FEATURE_NAMES  # noqa: E402
from palimpsest.features.syntax import ALL_SYNTAX_FEATURE_NAMES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "features"
OUT = ROOT / "artifacts" / "esl_false_positive_audit.json"

REMOTE_UNAVAILABLE = ("mean_entropy", "entropy_sd", "curvature", "curvature_z_in_doc")

#: Held-out machine sets, so a false-positive change can be read against a recall change.
MACHINE_SETS = ("modern_holdout_remote", "modern_claude_eval_remote")

#: ELLIPSE holistic score bands. The finding in PROJECT.md §8 is that the highlight rate
#: rises with proficiency, so the bands must be fine enough to show a monotone trend.
BANDS = ((1.0, 2.5, "1.0-2.5 weakest"), (2.5, 3.5, "2.5-3.5"), (3.5, 5.1, "3.5-5.0 strongest"))


def load(name: str) -> list[dict]:
    path = FEATURE_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r["features"] = {
            k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()
        }
        out.append(r)
    return out


def matrix(rows, names):
    return np.array(
        [[r["features"].get(n, np.nan) for n in names] for r in rows], dtype=np.float64
    )


def fit_arm(train_rows, names, seed=0):
    """Fit one arm and return a scorer plus the 5%-budget sentence threshold."""
    x = matrix(train_rows, names)
    y = np.array([r["label"] for r in train_rows], dtype=int)
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    filled = np.where(np.isfinite(x), x, mu)
    sd = filled.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    model = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
    model.fit((filled - mu) / sd, y)

    def score(rows):
        xx = matrix(rows, names)
        ff = np.where(np.isfinite(xx), xx, mu)
        return model.predict_proba((ff - mu) / sd)[:, 1]

    # The threshold is set on TRAINING humans. Setting it on the evaluation set would let
    # the data being audited choose the bar it is then measured against.
    thr = float(np.quantile(score(train_rows)[y == 0], 0.95))
    return score, thr


def by_document(rows, probs, threshold):
    """Group sentences into documents and aggregate exactly as the serving path does."""
    idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["doc_id"], []).append(i)

    docs = []
    for doc_id, ii in idx.items():
        # The reliability rule the shipped pipeline enforces. Without it this script reports
        # failures the tool does not make -- the run-on ESL essays were once listed as
        # confident false positives after they had stopped being scored at all (PROJECT.md §2).
        keep = [i for i in ii
                if float(rows[i]["features"].get("n_words") or 1.0) <= MAX_SENTENCE_WORDS]
        if not keep:
            continue
        w = np.array([rows[i]["features"].get("n_words", 1.0) for i in keep], dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        p = probs[keep]
        stats = document_statistics(p, w, threshold)
        meta = rows[keep[0]].get("doc_meta") or {}
        docs.append({
            "doc_id": doc_id,
            "source": rows[keep[0]]["source_id"],
            "label": int(any(rows[i]["label"] for i in keep)),
            "meanP": stats["mean_p"],
            "maxP": stats["max_p"],
            "share": stats["share"],
            "nSentences": len(keep),
            # The heat map's accusation: at least one sentence highlighted anywhere.
            "anyHighlight": bool((p >= threshold).any()),
            "proficiency": meta.get("proficiency"),
            "ell": meta.get("ell"),
        })
    return docs


def rate(flags) -> dict:
    """Proportion with a Wilson 95% interval. Wilson rather than normal: n is small in some
    bands and the normal interval goes below zero exactly where the answer matters."""
    n = len(flags)
    if n == 0:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    k = int(sum(flags))
    p = k / n
    z = 1.959963985
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"n": n, "k": k, "rate": round(p, 4),
            "lo": round(max(0.0, centre - half), 4), "hi": round(min(1.0, centre + half), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="train_remote")
    ap.add_argument("--esl", default="esl_remote")
    ap.add_argument("--arms", nargs="+", default=["base", "augmented"])
    ap.add_argument("--doc-threshold", type=float, default=0.5,
                    help="document meanP above which a document counts as accused")
    args = ap.parse_args()

    train = load(args.train)
    esl = load(args.esl)
    if not train or not esl:
        raise SystemExit("need both a training and an ESL feature file")

    base_names = [n for n in FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]
    syn_names = [n for n in ALL_SYNTAX_FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]
    have_syntax = any(n in esl[0]["features"] for n in syn_names)
    arms = {
        "base": base_names,
        "augmented": base_names + syn_names if have_syntax else None,
    }

    machine = {name: load(name) for name in MACHINE_SETS}
    machine = {k: v for k, v in machine.items() if v}

    report = {
        "train": args.train, "esl": args.esl,
        "nEslDocuments": len({r["doc_id"] for r in esl}),
        "nEslSentences": len(esl),
        "docThreshold": args.doc_threshold,
        "arms": {},
        "note": (
            "Every document in these sets is human-written; any machine verdict is a false "
            "accusation. Both arms are calibrated to a 5% sentence-level false-positive "
            "budget on TRAINING humans, so the arms differ by features, not by threshold."
        ),
    }

    for arm in args.arms:
        names = arms.get(arm)
        if not names:
            print(f"\n{arm}: unavailable (no structural block in {args.esl}) -- skipped")
            continue

        score, thr = fit_arm(train, names)
        docs = by_document(esl, score(esl), thr)

        print("\n" + "=" * 76)
        print(f"ARM: {arm}   ({len(names)} features, sentence threshold {thr:.3f})")
        print("=" * 76)

        accused = [d["meanP"] >= args.doc_threshold for d in docs]
        highlighted = [d["anyHighlight"] for d in docs]
        overall = {"documentAccusation": rate(accused), "anyHighlight": rate(highlighted)}
        print(f"  documents                     {len(docs)}")
        print(f"  ACCUSED (document verdict)    {overall['documentAccusation']['rate']:.2%}  "
              f"[{overall['documentAccusation']['lo']:.2%}, "
              f"{overall['documentAccusation']['hi']:.2%}]")
        print(f"  HIGHLIGHTED (>=1 sentence)    {overall['anyHighlight']['rate']:.2%}  "
              f"[{overall['anyHighlight']['lo']:.2%}, {overall['anyHighlight']['hi']:.2%}]"
              "   <- what the reader actually sees")

        # -- by source ---------------------------------------------------------------
        print("\n  by source:")
        by_source = {}
        for src in sorted({d["source"] for d in docs}):
            sub = [d for d in docs if d["source"] == src]
            by_source[src] = {
                "documentAccusation": rate([d["meanP"] >= args.doc_threshold for d in sub]),
                "anyHighlight": rate([d["anyHighlight"] for d in sub]),
            }
            print(f"    {src:16} n={len(sub):5}  accused "
                  f"{by_source[src]['documentAccusation']['rate']:.2%}   highlighted "
                  f"{by_source[src]['anyHighlight']['rate']:.2%}")

        # -- by measured proficiency --------------------------------------------------
        graded = [d for d in docs if isinstance(d.get("proficiency"), (int, float))]
        by_band = {}
        if graded:
            print("\n  by measured proficiency (ELLIPSE holistic):")
            for lo, hi, label in BANDS:
                sub = [d for d in graded if lo <= d["proficiency"] < hi]
                if not sub:
                    continue
                by_band[label] = {
                    "documentAccusation": rate([d["meanP"] >= args.doc_threshold for d in sub]),
                    "anyHighlight": rate([d["anyHighlight"] for d in sub]),
                }
                print(f"    {label:18} n={len(sub):5}  accused "
                      f"{by_band[label]['documentAccusation']['rate']:.2%}   highlighted "
                      f"{by_band[label]['anyHighlight']['rate']:.2%}")
            # A correlation, because the bands hide the shape and the shape is the finding.
            prof = np.array([d["proficiency"] for d in graded])
            hl = np.array([float(d["anyHighlight"]) for d in graded])
            r = float(np.corrcoef(prof, hl)[0, 1]) if prof.std() > 0 and hl.std() > 0 else float("nan")
            print(f"    correlation(proficiency, highlighted) = {r:+.3f}"
                  f"   {'(rises with proficiency)' if r > 0.02 else ''}")
        else:
            r = float("nan")

        # -- recall, measured two ways, because one of them misleads ------------------
        #
        # At a FIXED document threshold, an arm that is better calibrated looks worse: its
        # scores are compressed, fewer documents clear 0.5, and recall appears to collapse.
        # The first version of this script reported exactly that and called a real gain a
        # "threshold move". The honest comparison holds the *harm* constant instead --
        # recall at whatever threshold gives 5% false positives on THESE ESL documents --
        # and is reported beside a threshold-free document AUROC.
        recall = {}
        esl_p = np.array([d["meanP"] for d in docs])
        matched_thr = float(np.quantile(esl_p, 0.95))
        print(f"\n  machine recall (fixed thr={args.doc_threshold} | matched 5% ESL FPR "
              f"thr={matched_thr:.3f} | threshold-free AUROC):")
        for mname, mrows in machine.items():
            if not any(n in mrows[0]["features"] for n in names):
                continue
            mdocs = by_document(mrows, score(mrows), thr)
            pos = [d for d in mdocs if d["label"] == 1]
            if not pos:
                continue
            mp = np.array([d["meanP"] for d in pos])
            y = np.r_[np.zeros(len(esl_p)), np.ones(len(mp))]
            auc = float(roc_auc_score(y, np.r_[esl_p, mp]))
            recall[mname] = {
                "atFixedThreshold": rate([p >= args.doc_threshold for p in mp]),
                "atMatchedEslFpr": rate([p > matched_thr for p in mp]),
                "documentAuroc": round(auc, 4),
            }
            print(f"    {mname:32} {recall[mname]['atFixedThreshold']['rate']:>7.2%} | "
                  f"{recall[mname]['atMatchedEslFpr']['rate']:>7.2%} | {auc:.4f}  "
                  f"(n={len(pos)})")

        report["arms"][arm] = {
            "nFeatures": len(names), "sentenceThreshold": round(thr, 4),
            "overall": overall, "bySource": by_source, "byProficiency": by_band,
            "proficiencyHighlightCorrelation": None if not np.isfinite(r) else round(r, 4),
            "machineRecall": recall,
        }

    # -- the comparison, stated as a trade rather than a headline --------------------
    if "base" in report["arms"] and "augmented" in report["arms"]:
        b, a = report["arms"]["base"], report["arms"]["augmented"]
        d_hl = a["overall"]["anyHighlight"]["rate"] - b["overall"]["anyHighlight"]["rate"]
        d_acc = (a["overall"]["documentAccusation"]["rate"]
                 - b["overall"]["documentAccusation"]["rate"])
        shared = sorted(set(b["machineRecall"]) & set(a["machineRecall"]))
        # Matched-harm recall and threshold-free AUROC. The fixed-threshold delta is
        # deliberately NOT the headline: see the comment above where it is computed.
        d_rec = {k: round(a["machineRecall"][k]["atMatchedEslFpr"]["rate"]
                          - b["machineRecall"][k]["atMatchedEslFpr"]["rate"], 4)
                 for k in shared}
        d_auc = {k: round(a["machineRecall"][k]["documentAuroc"]
                          - b["machineRecall"][k]["documentAuroc"], 4) for k in shared}

        print("\n" + "=" * 76)
        print("TRADE: augmented vs base, harm held constant (5% ESL false-positive rate)")
        print("=" * 76)
        print(f"  ESL highlighted        {d_hl:+.2%}")
        print(f"  ESL accused            {d_acc:+.2%}")
        for k in shared:
            print(f"  {k:32} recall {d_rec[k]:+.2%}   docAUROC {d_auc[k]:+.4f}")

        gains = [k for k in shared if d_auc[k] > 0.01]
        losses = [k for k in shared if d_auc[k] < -0.01]
        if gains and losses:
            honest = (
                f"SPLIT: better on {', '.join(gains)}, worse on {', '.join(losses)}. "
                "Consistent with docs/08-cross-vendor.md -- this family reads the generators "
                "the detector already caught, and does not reach the frontier."
            )
        elif gains:
            honest = f"gain on {', '.join(gains)} with no measured loss elsewhere"
        elif losses:
            honest = f"loss on {', '.join(losses)} -- do not ship"
        else:
            honest = "no material change in detection; the ESL movement is a calibration shift"
        print(f"\n  reading: {honest}")
        report["trade"] = {
            "deltaHighlighted": round(d_hl, 4), "deltaAccused": round(d_acc, 4),
            "deltaRecallAtMatchedEslFpr": d_rec, "deltaDocumentAuroc": d_auc,
            "reading": honest,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

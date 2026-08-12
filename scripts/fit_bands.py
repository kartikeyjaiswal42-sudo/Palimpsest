#!/usr/bin/env python
"""Fit the three-band operating point: machine / insufficient evidence / human.

    python scripts/fit_bands.py --suffix _remote

A single threshold forces one number to do two incompatible jobs. Calibrating it so that at
most 5% of English-learner essays are flagged makes it strict, and mid-tier Gemini document
recall falls from 0.546 to 0.091 -- the tool becomes safe by becoming useless. Loosening it
buys recall by accusing students. There is no setting of one number that avoids both.

The way out is to stop pretending every document can be classified. Two thresholds, three
bands:

  score >= T_machine   "likely machine"        -- chosen so the false-positive rate on
                                                 at-risk human writing has a 95% upper
                                                 bound within budget
  score <= T_human     "no evidence of machine" -- chosen so the MISS rate on known machine
                                                 text has a 95% upper bound within budget
  in between           "insufficient evidence"  -- the tool declines to answer

The middle band is the honest product. Both live failures that started this work landed
there: a Gemini essay scored 35% and an Opus statement of purpose 0%, and both were read as
acquittals because the interface only had two words to say. "This document is in the range
where this tool cannot tell you anything" is a correct answer to those inputs; "0% machine"
is a wrong one.

Both bounds are Clopper-Pearson, for the reason recorded in train.py: a threshold picked to
minimise an observed rate makes that rate optimistically biased, and the previous point-
estimate rule shipped a detector that exceeded its own budget by half again on held-out data.

The abstention RATE is the price and it is printed, not hidden. A tool that abstains on
everything satisfies both bounds and is worthless, so the rate is the number to argue about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402

#: Human sets whose false positives the product must bound. Deliberately the at-risk ones:
#: English-learner writing and out-of-domain human prose, not the essays most like training.
HUMAN_SETS = ("esl", "domain_shift")
#: Machine sets whose misses the "no evidence" verdict must bound.
MACHINE_SETS = ("modern_holdout", "modern_unseen_family", "modern_claude")
#: With frontier text in TRAINING, the claude eval half is the honest machine set.
MACHINE_SETS_FRONTIER = ("modern_holdout", "modern_unseen_family", "modern_claude_eval")


def upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson upper bound on a rate observed as k/n."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def doc_scores(path: Path, det: SentenceDetector, dm: DocumentDetector,
               want_machine: bool) -> list[float]:
    if not path.exists():
        return []
    by_doc: dict[str, list[dict]] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        by_doc.setdefault(r["doc_id"], []).append(r)
    out = []
    for _doc, rs in sorted(by_doc.items()):
        is_machine = any(r["label"] for r in rs)
        if is_machine != want_machine:
            continue
        p = np.asarray(det.predict_many([r["features"] for r in rs]), dtype=float)
        w = np.array([float(r["features"].get("n_words") or 1.0) for r in rs], dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        out.append(float(dm.predict(document_statistics(p, w, det.flag_threshold))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_remote")
    # Model artifacts and feature files do not have to share a suffix: an ablation
    # detector (_frontier) is evaluated against the SAME held-out features (_remote).
    ap.add_argument("--features-suffix", default=None,
                    help="suffix of the feature files; defaults to --suffix")
    ap.add_argument("--fpr-budget", type=float, default=0.05,
                    help="max false-accusation rate on at-risk human writing")
    ap.add_argument("--miss-budget", type=float, default=0.10,
                    help="max rate at which machine text is actively cleared")
    args = ap.parse_args()

    det = SentenceDetector.load(ROOT / "artifacts" / f"detector{args.suffix}.json")
    dm = DocumentDetector.load(ROOT / "artifacts" / f"document_detector{args.suffix}.json")
    fsuf = args.features_suffix or args.suffix
    feats = ROOT / "data" / "features"

    # Calibrate on the EVEN half of each set; evaluate.py holds the odd half out.
    human, machine = [], []
    for name in HUMAN_SETS:
        s = doc_scores(feats / f"{name}{fsuf}.jsonl", det, dm, want_machine=False)
        human += s[::2]
    machine_sets = MACHINE_SETS_FRONTIER if "frontier" in args.suffix else MACHINE_SETS
    for name in machine_sets:
        s = doc_scores(feats / f"{name}{fsuf}.jsonl", det, dm, want_machine=True)
        machine += s[::2]

    h, m = np.array(human), np.array(machine)
    print(f"calibration: {len(h)} at-risk human documents, {len(m)} machine documents\n")
    if not len(h) or not len(m):
        print("! need both classes")
        return 1

    grid = np.unique(np.round(np.concatenate([h, m, [0.0, 1.0]]), 4))

    # T_machine: lowest cut whose human false-positive UPPER BOUND is within budget.
    t_machine = 1.0
    for cut in grid:
        if upper_bound(int((h >= cut).sum()), len(h)) <= args.fpr_budget:
            t_machine = float(cut)
            break

    # T_human: highest cut whose machine MISS upper bound is within budget. Clearing a
    # document is a claim too, and it is the claim a cheating student wants; it gets a
    # bounded error rate for the same reason the accusation does.
    t_human = 0.0
    for cut in grid[::-1]:
        if upper_bound(int((m <= cut).sum()), len(m)) <= args.miss_budget:
            t_human = float(cut)
            break

    if t_human > t_machine:  # bands would overlap; collapse to the safe ordering
        t_human = t_machine

    fp = int((h >= t_machine).sum())
    miss = int((m <= t_human).sum())
    abstain_h = float(((h > t_human) & (h < t_machine)).mean())
    abstain_m = float(((m > t_human) & (m < t_machine)).mean())

    print(f"T_human   = {t_human:.4f}   'no evidence of machine'")
    print(f"T_machine = {t_machine:.4f}   'likely machine'\n")
    print(f"  false accusations : {fp}/{len(h)} = {fp / len(h):.3f} "
          f"(95% upper {upper_bound(fp, len(h)):.3f}, budget {args.fpr_budget})")
    print(f"  machine cleared   : {miss}/{len(m)} = {miss / len(m):.3f} "
          f"(95% upper {upper_bound(miss, len(m)):.3f}, budget {args.miss_budget})")
    print(f"  abstain on human  : {abstain_h:.1%}")
    print(f"  abstain on machine: {abstain_m:.1%}   <- the price of the other two bounds")

    out = {"tHuman": t_human, "tMachine": t_machine,
           "fprBudget": args.fpr_budget, "missBudget": args.miss_budget,
           "observedFpr": fp / len(h), "observedMiss": miss / len(m),
           "abstainHuman": abstain_h, "abstainMachine": abstain_m,
           "nHuman": len(h), "nMachine": len(m)}
    (ROOT / "artifacts" / f"bands{args.suffix}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote artifacts/bands{args.suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

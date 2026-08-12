#!/usr/bin/env python
"""Score the same held-out sets with two detectors and report the difference.

    python scripts/compare_generators.py --baseline <dir-with-detector.json>

The question this answers is not "is the retrained detector good" -- `evaluate.py` reports
that. It is "what did adding modern generators to the training pool actually buy, and what
did it cost", measured on identical documents with identical code so nothing but the fitted
weights differs between the two columns.

Both halves matter and they pull in opposite directions:

  recall on machine text   should go UP. That is the point of the exercise.
  false positives on human should NOT go up. If it does, the detector did not learn to
                           recognise modern prose, it learned to accuse everybody, and the
                           people it accuses are real students.

A gain in the first bought with a loss in the second is not an improvement, and this script
prints them side by side so that trade cannot be reported as a win by quoting one column.

**These are not the published rates and must not be quoted as them.** `evaluate.py` reports
false positives on the held-out half of each human set, because the other half calibrates
the operating point. This script scores every document in the set with both detectors, so
its ESL figure sits near 6.3% where the published one is 7.3%. That is the right basis for a
difference -- the same documents on both sides -- and the wrong basis for a headline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.detect.document import (  # noqa: E402
    MAX_SENTENCE_WORDS,
    DocumentDetector,
    document_statistics,
)

FEATURES = ROOT / "data" / "features"

#: (set name, what it is, whether flagging a document there is a WIN or a HARM)
SETS = [
    ("modern_unseen_family", "modern model FAMILY withheld entirely", "recall"),
    ("modern_unseen", "modern checkpoint withheld entirely", "recall"),
    ("modern_holdout", "unseen essays, generator seen in training", "recall"),
    ("modern_control", "same generator, NO subject steering (topic control)", "recall"),
    ("unseen_prompting", "GPT-3.5 prompted to evade", "recall"),
    ("adversarial", "prose composed by hand to imitate a model", "recall"),
    ("esl", "essays by English-language learners", "false positive"),
    ("domain_shift", "human writing from another domain", "false positive"),
]


def load(name: str) -> list[dict]:
    path = FEATURES / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def document_rate(rows, det: SentenceDetector, doc: DocumentDetector) -> tuple[float, int]:
    """Share of documents this detector calls machine, and how many documents there were."""
    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)
    probs = np.array(det.predict_many([r["features"] for r in rows]))
    flagged = 0
    for idx in by_doc.values():
        # `n_words` arrives as None when the feature builder could not compute it, so the
        # array must be built as float with a default rather than coerced afterwards --
        # np.isfinite refuses an object array and the failure is a TypeError, not a NaN.
        w = np.array([float(rows[i]["features"].get("n_words") or 1.0) for i in idx],
                     dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        stats = document_statistics(probs[idx], w, det.flag_threshold)
        flagged += int(doc.predict(stats) >= doc.threshold)
    return flagged / max(len(by_doc), 1), len(by_doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="directory holding the PREVIOUS detector.json + document_detector.json")
    ap.add_argument("--out", default="artifacts/generator_comparison.json")
    args = ap.parse_args()

    base_dir = Path(args.baseline)
    pairs = {
        "before (GPT-3.5 only)": (
            SentenceDetector.load(base_dir / "detector.json"),
            DocumentDetector.load(base_dir / "document_detector.json"),
        ),
        "after (+ modern)": (
            SentenceDetector.load(ROOT / "artifacts" / "detector.json"),
            DocumentDetector.load(ROOT / "artifacts" / "document_detector.json"),
        ),
    }

    print(f"{'set':<34}{'n':>5}{'before':>9}{'after':>9}{'change':>9}")
    print("-" * 66)
    report: dict = {}
    for name, label, kind in SETS:
        rows = load(name)
        if not rows:
            print(f"{name:<34}{'--':>5}   (no features built)")
            continue
        rates = {}
        for key, (det, doc) in pairs.items():
            rate, n = document_rate(rows, det, doc)
            rates[key] = rate
        before = rates["before (GPT-3.5 only)"]
        after = rates["after (+ modern)"]
        delta = after - before
        arrow = "+" if delta > 0 else ""
        good = (delta > 0) if kind == "recall" else (delta <= 0)
        print(f"{name:<34}{n:>5}{100 * before:>8.1f}%{100 * after:>8.1f}%"
              f"{arrow}{100 * delta:>7.1f}  {'' if good else '<-- WRONG DIRECTION'}")
        report[name] = {"what": label, "kind": kind, "nDocuments": n,
                        "before": round(before, 4), "after": round(after, 4),
                        "change": round(delta, 4)}

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")

    harms = [k for k, v in report.items() if v["kind"] == "false positive" and v["change"] > 0]
    if harms:
        print("\nFALSE POSITIVES ROSE on: " + ", ".join(harms))
        print("Read that before quoting the recall column. Recall bought by accusing more")
        print("human writers is not an improvement to this tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

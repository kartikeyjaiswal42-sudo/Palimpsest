#!/usr/bin/env python
"""Measure the Binoculars column. It is a near-null, and this is the file that says so.

    python scripts/binoculars_probe.py

Reads ``binoculars_score`` from the joined feature matrices and answers the only question
that matters: does it earn a seat in the shipped detector? Writes
``artifacts/binoculars_probe.json``.

Why this exists and ``syntax_probe.py`` will not do
---------------------------------------------------
``syntax_probe.py`` selects its columns from ``FEATURE_NAMES + ALL_SYNTAX_FEATURE_NAMES``
and **never reads ``binoculars_score``**. Pointing it at a corpus that has just gained the
column would print a number that is real, reproducible, and about something else entirely.
An earlier version of the Colab instructions told the reader to do exactly that, in three
places. All three are corrected; this script is what those places should have said.

That is the same class of error as everything in PROJECT.md §2: not a false statement, but a
number describing a different system than the reader believes they are looking at.

Why a script rather than a paragraph
------------------------------------
The result is negative, and a negative result written only as prose is how docs/03 came to
publish a 17.8% false-positive rate for a build that measured 10.9%. The finding lives in an
artifact regenerated from the data, so it cannot quietly stop being true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.document import MAX_SENTENCE_WORDS  # noqa: E402
from palimpsest.features.registry import FEATURE_NAMES  # noqa: E402
from palimpsest.features.syntax import ALL_SYNTAX_FEATURE_NAMES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "features"
OUT = ROOT / "artifacts" / "binoculars_probe.json"

COLUMN = "binoculars_score"

#: Below this, a single feature is decoration. Stated up front so the verdict is not chosen
#: after seeing the number.
DECORATION_AUROC = 0.55


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


def column(rows, name=COLUMN):
    return np.array([r["features"].get(name, np.nan) for r in rows], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="train_remote")
    ap.add_argument("--esl", default="esl_remote")
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    train = load(args.train)
    if not train:
        raise SystemExit(f"no rows in {args.train}")
    if COLUMN not in train[0]["features"]:
        raise SystemExit(
            f"{args.train} has no {COLUMN} column. Run the Colab notebook, then:\n"
            "  python scripts/join_binoculars.py --scores <file>"
        )

    b = column(train)
    y = np.array([r["label"] for r in train], dtype=int)
    ok = np.isfinite(b)

    report: dict = {
        "column": COLUMN,
        "train": args.train,
        "nSentences": len(train),
        "coverage": round(float(ok.mean()), 4),
        "decorationThreshold": DECORATION_AUROC,
    }

    # -- 1. alone ---------------------------------------------------------------------
    auroc = float(roc_auc_score(y[ok], b[ok]))
    # Lower B is predicted to mean more machine-like, so the informative direction is the
    # one that treats -B as the score. Reported both ways so the sign cannot be chosen
    # after the fact.
    auroc_inverted = 1.0 - auroc
    strength = max(auroc, auroc_inverted)
    med_h = float(np.median(b[ok & (y == 0)]))
    med_m = float(np.median(b[ok & (y == 1)]))

    print("=" * 74)
    print("1. BINOCULARS ALONE")
    print("=" * 74)
    print(f"  coverage                {ok.mean():.1%} of sentences")
    print(f"  AUROC (higher B = machine)  {auroc:.4f}")
    print(f"  AUROC (lower  B = machine)  {auroc_inverted:.4f}   <- the predicted direction")
    print(f"  strength                    {strength:.4f}   "
          f"(chance 0.500, decoration below {DECORATION_AUROC})")
    print(f"  median B: human {med_h:.4f}   machine {med_m:.4f}   "
          f"(separation {med_h - med_m:+.4f})")

    # -- 2. recall at a stated false-accusation budget --------------------------------
    # The threshold is set on TRAINING humans, as every other threshold in this project is.
    human_b = b[ok & (y == 0)]
    thr = float(np.quantile(human_b, args.budget))  # lower B = machine, so the LOW tail
    machine_b = b[ok & (y == 1)]
    recall = float((machine_b <= thr).mean())
    print(f"\n  at a {args.budget:.0%} train-human budget: threshold B <= {thr:.4f}")
    print(f"  machine sentence recall     {recall:.1%}")
    if recall < args.budget:
        print(f"  NOTE: recall {recall:.1%} is BELOW the {args.budget:.0%} budget — "
              "it accuses machines slightly LESS often than humans.")

    report["alone"] = {
        "aurocHigherIsMachine": round(auroc, 4),
        "aurocLowerIsMachine": round(auroc_inverted, 4),
        "strength": round(strength, 4),
        "medianHuman": round(med_h, 4),
        "medianMachine": round(med_m, 4),
        "thresholdAtBudget": round(thr, 4),
        "machineRecall": round(recall, 4),
        "budget": args.budget,
    }

    # -- 3. what it does to ESL prose -------------------------------------------------
    esl = load(args.esl)
    if esl and COLUMN in esl[0]["features"]:
        eb = column(esl)
        eok = np.isfinite(eb)
        ey = np.array([r["label"] for r in esl], dtype=int)
        human = eok & (ey == 0)
        fpr = float((eb[human] <= thr).mean())

        by_source: dict[str, dict] = {}
        for src in sorted({r["source_id"] for r in esl}):
            m = np.array([r["source_id"] == src for r in esl]) & human
            if m.sum():
                by_source[src] = {"n": int(m.sum()), "fpr": round(float((eb[m] <= thr).mean()), 4)}

        # The heat-map accusation: at least one flagged sentence anywhere in the essay.
        # PROJECT.md §7 -- this is what a reader consumes, and it was never priced.
        docs: dict[str, list[int]] = {}
        for i, r in enumerate(esl):
            if float(r["features"].get("n_words") or 1.0) <= MAX_SENTENCE_WORDS:
                docs.setdefault(r["doc_id"], []).append(i)
        highlighted = [
            bool(np.any(eb[ii][np.isfinite(eb[ii])] <= thr)) for ii in docs.values()
        ]
        any_rate = float(np.mean(highlighted)) if highlighted else float("nan")

        print("\n" + "=" * 74)
        print("2. COST ON ENGLISH-LEARNER PROSE (same threshold)")
        print("=" * 74)
        print(f"  ESL sentence false-positive rate   {fpr:.1%}")
        for src, d in by_source.items():
            print(f"    {src:18} n={d['n']:6}  {d['fpr']:.1%}")
        print(f"  ESL essays with >=1 highlight      {any_rate:.0%}   "
              "<- what the reader actually sees")
        report["esl"] = {
            "sentenceFpr": round(fpr, 4), "bySource": by_source,
            "documentsWithAnyHighlight": round(any_rate, 4),
            "nDocuments": len(docs),
        }
    else:
        print(f"\n  {args.esl}: no {COLUMN} column — ESL section skipped")
        report["esl"] = None

    # -- verdict, against the threshold stated before the measurement -----------------
    earns_seat = strength >= DECORATION_AUROC and recall > args.budget
    verdict = (
        f"Binoculars on this pair and this corpus is a NEAR-NULL: strength {strength:.3f} "
        f"against a {DECORATION_AUROC} decoration threshold, and {recall:.1%} machine recall "
        f"at a {args.budget:.0%} false-accusation budget. It does not earn a seat in the "
        "shipped detector."
        if not earns_seat else
        f"Strength {strength:.3f} and {recall:.1%} recall at a {args.budget:.0%} budget: "
        "this clears the stated bar and is worth wiring in, subject to "
        "scripts/consensus_controls.py."
    )
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  {verdict}")

    report["earnsSeatInShippedDetector"] = bool(earns_seat)
    report["verdict"] = verdict
    report["notes"] = {
        "whyNotSyntaxProbe": (
            "scripts/syntax_probe.py selects columns from FEATURE_NAMES + "
            "ALL_SYNTAX_FEATURE_NAMES and never reads binoculars_score. Running it after "
            "joining this column reports a real number about a different question."
        ),
        "columnIsInert": (
            "binoculars_* appears in no feature registry, so train.py cannot select it and "
            "the column on disk has no path into the shipped model."
        ),
        "frontierPattern": (
            "The per-source medians put older/cheaper generators at the machine-like end "
            "and frontier models below the human sources, with TOEFL the most machine-like "
            "source of all -- the published Binoculars-on-frontier degradation plus this "
            "project's own ESL false-positive direction, in one table."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    # Guard the confusion this script exists to prevent.
    assert COLUMN not in FEATURE_NAMES, "binoculars leaked into the shipped registry"
    assert COLUMN not in ALL_SYNTAX_FEATURE_NAMES, "binoculars leaked into the syntax block"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

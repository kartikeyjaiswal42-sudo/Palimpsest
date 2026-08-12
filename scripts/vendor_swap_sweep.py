#!/usr/bin/env python
"""Does spending part of a FIXED training budget on a second vendor buy cross-vendor recall?

    python scripts/vendor_swap_sweep.py

docs/08-cross-vendor.md left one question open: the detector scored 0.0% document recall on
Claude essays while scoring 94.8% on the Gemini it was fitted to, so it was reading vendor
rather than machine-ness. The obvious repair -- add the Claude training split -- was tried
first and it does not work. It is recorded here because the failure is instructive:

    adding all 238 Claude essays moves the sentence-level machine prior from 0.498 to 0.773,
    out-of-fold sentence AUROC falls 0.925 -> 0.726, and the ESL false-positive rate rises
    0.174 -> 0.791. The flag rate on held-out Claude does go 0.066 -> 0.930, but a detector
    that flags 79% of genuine English-learner sentences is not detecting anything. It is the
    "willing accuser" failure docs/06-decisions.md warned about, reached literally.

The training set was balanced (0.498) and there are only 3,544 human sentences, so there is
no headroom to add a machine vendor and stay balanced. The budget is fixed by the human half.

So this sweep SWAPS instead of adding. The machine class stays the size it already was; a
fraction `f` of its documents is drawn from Claude instead of Gemini/GPT-3.5. That holds the
prior constant and isolates the one variable that matters -- how the machine budget is spent
across vendors.

Swapping is done by DOCUMENT, never by sentence: a document is the unit that must not
straddle the boundary, and the in-document context features read the rest of their own essay.

Reported per condition:
  * out-of-fold sentence AUROC        -- did the detector stay a detector
  * ESL sentence false-positive rate  -- measured on the half evaluate.py holds out, i.e.
                                         the half the operating point never saw
  * flag rate on held-out Claude      -- the cross-vendor question
  * flag rate on held-out Gemini      -- what the swap COSTS on the original vendor

The last two are the trade. A swap that buys Claude by losing Gemini has moved the blind
spot, not removed it, and that is the outcome this file exists to be able to report.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector  # noqa: E402

FEATURES = ROOT / "data" / "features"
TMP = Path("/tmp/palimpsest_sweep")
TMP.mkdir(exist_ok=True)

#: Fraction of the machine budget spent on Claude. 0.0 reproduces the shipped detector.
CONDITIONS = [0.0, 0.25, 0.50, 0.75]
SEED = 20260811


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["doc_id"], []).append(r)
    return out


def flag_rate(det: SentenceDetector, rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    p = det.predict_many([r["features"] for r in rows])
    return float((np.asarray(p) >= det.flag_threshold).mean())


def main() -> int:
    # Built by: build_features.py --sets train_with_claude. Kept separate from train.jsonl so
    # the default pipeline still reproduces the shipped (and better) 0%-Claude detector.
    train_rows = load(FEATURES / "train_with_claude.jsonl")
    claude = {d: rs for d, rs in by_doc(train_rows).items() if d.startswith("claude_")}
    other = {d: rs for d, rs in by_doc(train_rows).items() if not d.startswith("claude_")}

    human_docs = {d: rs for d, rs in other.items() if rs[0]["label"] == 0}
    machine_docs = {d: rs for d, rs in other.items() if rs[0]["label"] == 1}
    budget = sum(len(rs) for rs in machine_docs.values())   # machine sentences to keep fixed

    print(f"human {sum(len(r) for r in human_docs.values())} sentences / {len(human_docs)} docs")
    print(f"machine budget {budget} sentences / {len(machine_docs)} docs "
          f"(claude pool: {sum(len(r) for r in claude.values())} / {len(claude)} docs)\n")

    # Held-out probes. ESL uses the ODD half -- evaluate.py's half, never seen by the
    # operating point, which is calibrated on the even half.
    esl = [r for r in load(FEATURES / "esl.jsonl") if r["label"] == 0]
    esl_odd = [r for i, (_, rs) in enumerate(sorted(by_doc(esl).items())) if i % 2 for r in rs]
    claude_ho = [r for r in load(FEATURES / "modern_claude.jsonl") if r["label"] == 1]
    gemini_ho = [r for r in load(FEATURES / "modern_unseen.jsonl") if r["label"] == 1]

    results = []
    for f in CONDITIONS:
        rng = random.Random(SEED)
        want_claude = int(round(budget * f))

        # Draw whole Claude documents until the claude share of the budget is filled.
        picked_claude, got = [], 0
        for d in rng.sample(sorted(claude), len(claude)):
            if got >= want_claude:
                break
            picked_claude.append(d)
            got += len(claude[d])

        # Fill the rest of the budget from the original machine documents.
        picked_other, got_other = [], 0
        for d in rng.sample(sorted(machine_docs), len(machine_docs)):
            if got_other >= budget - got:
                break
            picked_other.append(d)
            got_other += len(machine_docs[d])

        rows = [r for rs in human_docs.values() for r in rs]
        rows += [r for d in picked_claude for r in claude[d]]
        rows += [r for d in picked_other for r in machine_docs[d]]

        path = TMP / f"train_f{int(f*100):03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        n_mach = got + got_other
        prior = n_mach / len(rows)
        print(f"=== claude share {f:.0%}: {len(picked_claude)} claude docs + "
              f"{len(picked_other)} original docs, machine prior {prior:.3f}")

        proc = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "train.py"),
             "--features", str(path)],
            capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            return 1
        # train.py prints the sentence block first ("  AUROC x | average precision ...")
        # and the document block second; the first match is the sentence-level number.
        auroc = next((ln.split("AUROC")[1].split("|")[0].strip()
                      for ln in proc.stdout.splitlines()
                      if ln.strip().startswith("AUROC")), "nan")

        det = SentenceDetector.load(ROOT / "artifacts" / "detector.json")
        results.append((f, prior, float(auroc), flag_rate(det, esl_odd),
                        flag_rate(det, claude_ho), flag_rate(det, gemini_ho)))
        print(f"    AUROC {auroc}  eslFPR {results[-1][3]:.3f}  "
              f"claude {results[-1][4]:.3f}  gemini {results[-1][5]:.3f}\n")

    print("\n" + "=" * 78)
    print("FIXED MACHINE BUDGET, VARYING VENDOR MIX")
    print("=" * 78)
    print(f"{'claude share':>13} {'prior':>7} {'AUROC':>7} {'ESL FPR':>8} "
          f"{'claude flag':>12} {'gemini flag':>12}")
    for f, prior, auroc, esl_fpr, cl, gem in results:
        print(f"{f:>12.0%} {prior:>7.3f} {auroc:>7.3f} {esl_fpr:>8.3f} "
              f"{cl:>12.3f} {gem:>12.3f}")
    (ROOT / "artifacts" / "vendor_swap.json").write_text(json.dumps(
        [{"claude_share": f, "machine_prior": p, "sentence_auroc": a, "esl_sent_fpr": e,
          "claude_flag_rate": c, "gemini_flag_rate": g}
         for f, p, a, e, c, g in results], indent=1), encoding="utf-8")
    print("\nwrote artifacts/vendor_swap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

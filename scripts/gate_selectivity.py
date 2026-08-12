#!/usr/bin/env python
"""Is the gate's refusal SELECTIVE, or does it just refuse a lot of people?

    python scripts/gate_selectivity.py --suffix _remote --gates _remote,_remote_5feat

THE QUESTION THIS SETTLES. Dropping ``mean_sentence_words`` from the gate cut the refusal
rate for the weakest-English writers from 100% to 25% -- but it also moved the headline
false-accusation rate from 1.15% to 1.43%, because documents the gate used to refuse now
reach the detector and some are flagged. Trading refusals for accusations inside the exact
population the project exists to protect is not obviously an improvement, and preferring the
variant that flatters a newly written script would be the wrong way to decide it.

So measure the thing that actually distinguishes them: REFUSAL PRECISION -- of the documents
a gate refuses, what fraction would have been falsely accused had they been scored?

  * precision >> base rate  ->  refusal is aimed. It is spending abstentions on the people who
    were about to be harmed, which is exactly what a scope check is for.
  * precision ~= base rate  ->  refusal is indiscriminate. It is denying answers to many
    people to prevent a few accusations that a wider net would catch anyway, and the users
    who pay for it are disproportionately the weakest writers.

The companion number is COST: refusals spent per accusation avoided. A gate that prevents one
false accusation by refusing forty essays is not protecting those forty, it is failing them
more quietly.

Both are computed on the same held-out half that ``report_product.py`` uses, so the numbers
compose with the published table rather than sitting beside it.
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
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402
from palimpsest.detect.genre import GenreGate, document_genre_features  # noqa: E402

HUMAN_SETS = ("esl", "domain_shift")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_remote")
    ap.add_argument("--gates", default="_remote",
                    help="comma-separated gate artifact suffixes to compare")
    args = ap.parse_args()

    det = SentenceDetector.load(ROOT / "artifacts" / f"detector{args.suffix}.json")
    dm = DocumentDetector.load(ROOT / "artifacts" / f"document_detector{args.suffix}.json")
    bands = json.loads((ROOT / "artifacts" / f"bands{args.suffix}.json").read_text())
    t_m = bands["tMachine"]

    # Score every human document ONCE, with no gate. Each gate variant is then just a
    # different mask over the same scores, so the comparison cannot be contaminated by
    # rescoring differences.
    feats_dir = ROOT / "data" / "features"
    docs: list[tuple[dict, float]] = []          # (genre features, machine score)
    for name in HUMAN_SETS:
        path = feats_dir / f"{name}{args.suffix}.jsonl"
        if not path.exists():
            continue
        by_doc: dict[str, list[dict]] = {}
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            by_doc.setdefault(r["doc_id"], []).append(r)
        for i, (_d, rs) in enumerate(sorted(by_doc.items())):
            if any(r["label"] for r in rs):
                continue                          # human only
            if i % 2 == 0:
                continue                          # held-out half, matching report_product
            feats = [r["features"] for r in rs]
            p = np.asarray(det.predict_many(feats), dtype=float)
            w = np.array([float(f.get("n_words") or 1.0) for f in feats])
            w = np.where(np.isfinite(w), w, 1.0)
            s = float(dm.predict(document_statistics(p, w, det.flag_threshold)))
            docs.append((document_genre_features(feats), s))

    if not docs:
        print("! no human documents found")
        return 1

    would_flag = np.array([s >= t_m for _f, s in docs])
    n = len(docs)
    base = float(would_flag.mean())
    print(f"{n} held-out human documents")
    print(f"with NO gate at all, {int(would_flag.sum())} would be called machine "
          f"= {base:.2%}   <- the base rate every gate is judged against\n")

    hdr = (f"{'gate':22s} {'feats':>5s} {'refused':>8s} {'accused':>8s} "
           f"{'refusal precision':>18s} {'refusals/save':>14s}")
    print(hdr)
    print("-" * len(hdr))

    for suf in [g.strip() for g in args.gates.split(",") if g.strip()]:
        gpath = ROOT / "artifacts" / f"genre_gate{suf}.json"
        if not gpath.exists():
            print(f"{suf:22s} ! missing {gpath.name}")
            continue
        gate = GenreGate.load(gpath)
        refused = np.array([not gate.in_domain(f) for f, _s in docs])

        n_ref = int(refused.sum())
        accused = float((would_flag & ~refused).mean())
        # Of those refused, how many were actually heading for a false accusation?
        prec = float(would_flag[refused].mean()) if n_ref else float("nan")
        saved = int((would_flag & refused).sum())
        cost = n_ref / saved if saved else float("inf")
        print(f"{suf:22s} {len(gate.feature_names):5d} {n_ref / n:7.1%} {accused:8.2%} "
              f"{prec:17.1%} {cost:13.1f}")

    print(f"\nrefusal precision is meaningful only against the {base:.1%} base rate:")
    print("  at the base rate, refusal is random with respect to harm;")
    print("  above it, the gate is finding the documents that were about to be misjudged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

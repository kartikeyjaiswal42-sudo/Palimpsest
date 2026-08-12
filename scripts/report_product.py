#!/usr/bin/env python
"""What the PRODUCT does, end to end: genre gate, then bands, per evaluation set.

    python scripts/report_product.py --suffix _remote

``evaluate.py`` reports the classifier: sentence recall, document recall at one threshold.
Those are the right numbers for judging the model and the wrong ones for judging the tool,
because the tool no longer emits a binary verdict. It refuses out-of-genre writing, and
within its genre it answers in three bands, one of which is "I cannot tell".

So this is the table a user should be shown: for every held-out set, where do documents
actually land. Read the machine rows for recall and the human rows for harm.

The one number that decides whether the product is honest is the human "likely machine"
column. Everything else is a trade.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402
from palimpsest.detect.genre import GenreGate, document_genre_features  # noqa: E402

MACHINE_SETS = ("modern_holdout", "modern_unseen_family", "modern_claude", "modern_claude_eval")
HUMAN_SETS = ("esl", "domain_shift")
BANDS = ("likely_machine", "insufficient_evidence", "no_evidence", "out_of_scope")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_remote")
    # Model artifacts and feature files do not have to share a suffix: an ablation
    # detector (_frontier) is evaluated against the SAME held-out features (_remote).
    ap.add_argument("--features-suffix", default=None,
                    help="suffix of the feature files; defaults to --suffix")
    args = ap.parse_args()

    det = SentenceDetector.load(ROOT / "artifacts" / f"detector{args.suffix}.json")
    dm = DocumentDetector.load(ROOT / "artifacts" / f"document_detector{args.suffix}.json")
    gpath = ROOT / "artifacts" / f"genre_gate{args.suffix}.json"
    gate = GenreGate.load(gpath) if gpath.exists() else GenreGate()
    bands = json.loads((ROOT / "artifacts" / f"bands{args.suffix}.json").read_text())
    t_h, t_m = bands["tHuman"], bands["tMachine"]

    def classify(path: Path, want_machine: bool) -> Counter:
        by_doc: dict[str, list[dict]] = {}
        if not path.exists():
            return Counter()
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            by_doc.setdefault(r["doc_id"], []).append(r)
        c: Counter = Counter()
        for i, (_d, rs) in enumerate(sorted(by_doc.items())):
            if any(r["label"] for r in rs) != want_machine:
                continue
            # Odd half only: the even half calibrated the thresholds.
            if i % 2 == 0:
                continue
            feats = [r["features"] for r in rs]
            if not gate.in_domain(document_genre_features(feats)):
                c["out_of_scope"] += 1
                continue
            p = np.asarray(det.predict_many(feats), dtype=float)
            w = np.array([float(r["features"].get("n_words") or 1.0) for r in rs])
            w = np.where(np.isfinite(w), w, 1.0)
            s = float(dm.predict(document_statistics(p, w, det.flag_threshold)))
            c["likely_machine" if s >= t_m else
              "no_evidence" if s <= t_h else "insufficient_evidence"] += 1
        return c

    fsuf = args.features_suffix or args.suffix
    feats_dir = ROOT / "data" / "features"
    print(f"bands: no-evidence <= {t_h:.3f} < insufficient < {t_m:.3f} <= likely-machine")
    print(f"genre gate: P(in-domain) >= {gate.threshold:.3f}\n")
    hdr = f"{'set':26s} {'n':>4s} " + " ".join(f"{b.split('_')[0][:9]:>10s}" for b in BANDS)
    print(hdr); print("-" * len(hdr))

    out = {}
    for group, sets, want in (("MACHINE", MACHINE_SETS, True), ("HUMAN", HUMAN_SETS, False)):
        print(f"{group}")
        for name in sets:
            c = classify(feats_dir / f"{name}{fsuf}.jsonl", want)
            n = sum(c.values())
            if not n:
                continue
            out[name] = {b: c[b] / n for b in BANDS} | {"n": n}
            print(f"  {name:24s} {n:4d} " +
                  " ".join(f"{c[b] / n:9.1%}" for b in BANDS))
        print()

    fp = sum(out[s]["likely_machine"] * out[s]["n"] for s in HUMAN_SETS if s in out)
    nh = sum(out[s]["n"] for s in HUMAN_SETS if s in out)
    print(f"THE NUMBER THAT MATTERS -- human essays called machine: "
          f"{int(fp)}/{nh} = {fp / max(nh, 1):.2%}")
    (ROOT / "artifacts" / f"product_report{args.suffix}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

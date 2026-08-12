#!/usr/bin/env python
"""Can two specialists beat one generalist?

    python scripts/ensemble_probe.py --a _remote --b _frontier

THE PROBLEM THIS TESTS. Adding Claude text to training buys 18.4% recall on held-out Claude
and costs half the recall on Gemini (80.7% -> 40.4%), and a regularisation sweep does not
recover it. The reason is visible in the sentence model: out-of-fold AUROC falls 0.945 ->
0.901. Frontier prose sits so close to human prose that fitting it drags the decision
boundary, and the easier generator is what pays.

So do not make one model learn both. Score with both and take the higher signal -- the
standard answer when specialists disagree in only one direction.

THE CATCH, AND IT IS THE WHOLE MEASUREMENT. A maximum over two scores is a maximum over two
chances to be wrong. If each detector falsely accuses a different 2% of human essays, the
ensemble accuses close to 4%. Recall is free to combine; false positives combine too, and in
this project that is the number that decides whether a design is acceptable. So the bands are
re-fitted on the COMBINED score under the same 5% budget, and the honest comparison is what
recall survives at that budget -- not what recall the raw maximum shows.
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
MACHINE_SETS = ("modern_holdout", "modern_unseen_family", "modern_claude_eval")


def load_pair(suffix: str):
    return (SentenceDetector.load(ROOT / "artifacts" / f"detector{suffix}.json"),
            DocumentDetector.load(ROOT / "artifacts" / f"document_detector{suffix}.json"))


def score_set(path: Path, models, gate, want_machine: bool):
    """-> (list of per-model score arrays, refused mask), held-out half only."""
    if not path.exists():
        return None
    by_doc: dict[str, list[dict]] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        by_doc.setdefault(r["doc_id"], []).append(r)
    per_model: list[list[float]] = [[] for _ in models]
    refused: list[bool] = []
    for i, (_d, rs) in enumerate(sorted(by_doc.items())):
        if any(r["label"] for r in rs) != want_machine or i % 2 == 0:
            continue
        feats = [r["features"] for r in rs]
        refused.append(not gate.in_domain(document_genre_features(feats)))
        w = np.array([float(f.get("n_words") or 1.0) for f in feats])
        w = np.where(np.isfinite(w), w, 1.0)
        for k, (det, dm) in enumerate(models):
            p = np.asarray(det.predict_many(feats), dtype=float)
            per_model[k].append(float(dm.predict(document_statistics(p, w, det.flag_threshold))))
    return [np.array(v) for v in per_model], np.array(refused, dtype=bool)


def upper(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta
    return 1.0 if n == 0 or k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="_remote", help="generalist (no frontier text in training)")
    ap.add_argument("--b", default="_frontier", help="specialist (frontier text in training)")
    ap.add_argument("--features-suffix", default="_remote")
    ap.add_argument("--fpr-budget", type=float, default=0.05)
    args = ap.parse_args()

    models = [load_pair(args.a), load_pair(args.b)]
    gate = GenreGate.load(ROOT / "artifacts" / f"genre_gate{args.features_suffix}.json")
    feats = ROOT / "data" / "features"

    human, machine = {}, {}
    for name in HUMAN_SETS:
        r = score_set(feats / f"{name}{args.features_suffix}.jsonl", models, gate, False)
        if r:
            human[name] = r
    for name in MACHINE_SETS:
        r = score_set(feats / f"{name}{args.features_suffix}.jsonl", models, gate, True)
        if r:
            machine[name] = r

    # Scored documents only -- a refused document is not an accusation either way.
    def stack(d, k):
        s = [v[0][k][~v[1]] for v in d.values()]
        return np.concatenate(s) if s else np.array([])

    variants = {
        f"A  generalist {args.a}": lambda k: k == 0,
        f"B  specialist {args.b}": lambda k: k == 1,
    }
    print(f"{'variant':34s} {'T_machine':>10s} {'false acc':>10s} " +
          " ".join(f"{n[:15]:>16s}" for n in MACHINE_SETS))
    print("-" * 110)

    rows = []
    for label, pick in list(variants.items()) + [("A|B  max of both", None)]:
        if pick is None:
            h = np.maximum(stack(human, 0), stack(human, 1))
            m = {n: np.maximum(v[0][0][~v[1]], v[0][1][~v[1]]) for n, v in machine.items()}
        else:
            k = 0 if pick(0) else 1
            h = stack(human, k)
            m = {n: v[0][k][~v[1]] for n, v in machine.items()}

        # Highest threshold whose 95% upper bound on false accusation stays in budget.
        t = 1.0
        for cand in np.unique(np.round(np.sort(h)[::-1], 4)):
            if upper(int((h >= cand).sum()), len(h)) <= args.fpr_budget:
                t = float(cand)
                break
        fa = float((h >= t).mean())
        rec = {n: float((v >= t).mean()) for n, v in m.items()}
        rows.append((label, t, fa, rec))
        print(f"{label:34s} {t:10.4f} {fa:9.2%} " +
              " ".join(f"{rec.get(n, float('nan')):15.1%} " for n in MACHINE_SETS))

    print(f"\nall thresholds set to the SAME {args.fpr_budget:.0%} false-accusation budget "
          f"({len(stack(human, 0))} held-out human documents), so the recall columns are "
          "directly comparable.")
    (ROOT / "artifacts" / "ensemble_probe.json").write_text(json.dumps(
        [{"variant": a, "threshold": b, "falseAccusation": c, "recall": d} for a, b, c, d in rows],
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

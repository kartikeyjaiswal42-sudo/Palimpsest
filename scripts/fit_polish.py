#!/usr/bin/env python
"""Fit the polish head: "this sentence does not match the rest of this document".

    python scripts/fit_polish.py            # sweep, fit, write artifacts/polish_head.json
    python scripts/fit_polish.py --k 12     # fix the feature count

WHAT THIS IS

A SECOND, SEPARATE signal, not a change to the detector. It never touches the sentence model,
the document model or their thresholds -- the same containment the genre gate has, and for the
same reason: every number in PROJECT.md was measured against the shipped 40-feature fit, and a
retrain would invalidate all of them to buy one case.

The case it buys is the one the brief names: *"a paragraph a person wrote and a model later
polished."* `scripts/polish_probe.py` measures it. Two results from there decide this file's
shape:

  * The 5 self-relative features already shipped find almost nothing on their own -- document
    hit rate 0.072 at a defensible budget. Only 3 of the 43 features are z-scored against the
    document, and z-scoring the rest is free, because every value is already on disk.
  * With them, GPT-era polish localisation goes 0.338 -> 0.532 document hit rate at 0.956
    precision, holding the DOCUMENT false-alarm rate at 5%.

THE OPERATING POINT, WHICH IS THE POINT

A reader is handed a document, not a sentence. An essay holds ~19 sentences, so a threshold
that flags 5% of sentences flags at least one sentence in **30.7%** of unedited human essays.
That is the accusation rate, and it is the number this head is calibrated against: the
threshold is the 95th percentile of the per-document MAXIMUM score over unedited human
essays, so 5% of clean documents carry a flag by construction.

Calibrating per sentence and reporting per document would be scoring the easy question and
billing the hard one.

FEATURE COUNT IS SWEPT, NOT CHOSEN

Picking k after seeing which k won, and reporting only that, is how a tuned number becomes a
measured one. The sweep is printed in full and the shipped k is the smallest one within noise
of the best, stated as such.

WHAT IT CANNOT DO

Frontier polish. Fitted on one era's rewrites and tested cross-pipeline on frontier-polished
documents, the document hit rate at this budget is 0.100 -- one document in ten, on n=10. The
ceiling in docs/09 holds even when the author's own prose is sitting in the same document as a
reference, which is a stronger statement of it than docs/09 makes. This head is honest about
the era it works in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fusion_probe as fp  # noqa: E402
import polish_probe as pp  # noqa: E402

from palimpsest.features.context import CONTEXT_FEATURE_NAMES  # noqa: E402

K_SWEEP = (4, 8, 12, 16, 24, 37)


def rank_features(hyb, human, cands: tuple[str, ...]) -> list[tuple[str, float]]:
    """Rank candidate features by single-feature separation, machine vs clean-human sentences.

    Ranked on separation alone, with no fitting and no threshold, so the ordering cannot encode
    the metric it is later scored on. Direction-agnostic: a feature that separates the wrong way
    is as informative as one that separates the right way, and the sign is the fit's business.
    """
    h_rows = [r for rows in hyb.values() for r in rows]
    u_rows = [r for rows in human.values() for r in rows]
    y = np.array([r["label"] for r in h_rows])
    out = []
    for f in cands:
        pos = pp.matrix([r for r, l in zip(h_rows, y, strict=True) if l == 1], (f,))[:, 0]
        neg = pp.matrix(u_rows, (f,))[:, 0]
        a = fp.auroc(pos, neg)
        out.append((f, abs(a - 0.5)))
    return sorted(out, key=lambda kv: -kv[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=0, help="0 = sweep and choose")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="polish_head")
    ap.add_argument("--relative-only", action="store_true",
                    help="build the head from self-relative features alone. Measured: it "
                         "reaches document hit 0.201 that way against 0.532 with the absolute "
                         "features included, so the default keeps both. A separate head using "
                         "absolute features still changes nothing about the shipped model -- "
                         "containment is about what it MODIFIES, not which columns it reads.")
    args = ap.parse_args()

    F = ROOT / "data" / "features"
    hyb = pp.by_doc(pp.load_rows(F / "localisation_remote.jsonl"))
    hyb = {d: rs for d, rs in hyb.items() if len(rs) >= pp.MIN_SENTENCES}
    bases = {r["group"] for rows in hyb.values() for r in rows}

    human: dict[str, list[dict]] = {}
    excluded = 0
    for name in ("train_remote.jsonl", "esl_remote.jsonl"):
        p = F / name
        if not p.exists():
            continue
        for doc_id, rows in pp.by_doc(pp.load_rows(p)).items():
            if len(rows) < pp.MIN_SENTENCES or any(r["label"] != 0 for r in rows):
                continue
            if doc_id in bases:
                excluded += 1
                continue
            human[doc_id] = rows

    all_feats = tuple(sorted(pp.load_rows(F / "localisation_remote.jsonl")[0]["features"].keys()))
    abs_feats = tuple(f for f in all_feats if f not in CONTEXT_FEATURE_NAMES)
    rel_shipped = tuple(f for f in CONTEXT_FEATURE_NAMES if f != pp.LEAK)

    pp.add_loo_z(hyb, abs_feats)
    pp.add_loo_z(human, abs_feats)
    looz = tuple(f"{k}_looz" for k in abs_feats)

    # The head's fixed base. Self-relative features answer "unlike its neighbours"; the
    # absolute ones answer "unlike human prose". A polished sentence is usually both, and
    # measurement says dropping the absolute half costs more than half the recall.
    base = rel_shipped if args.relative_only else abs_feats + rel_shipped

    print(f"hybrids {len(hyb)} docs | clean human {len(human)} docs "
          f"({excluded} excluded as hybrid bases)")
    print(f"base: {len(base)} features"
          f"{' (self-relative only)' if args.relative_only else ' (absolute + self-relative)'}")
    print(f"candidates to add: {len(looz)} leave-one-out z-scores\n")

    ranked = rank_features(hyb, human, looz)
    print("top LOO z-scored features by single-feature separation (|AUROC - 0.5|):")
    for f, s in ranked[:12]:
        print(f"  {f:36s} {s:.3f}")
    print()

    print("=" * 88)
    print("FEATURE-COUNT SWEEP -- all at the 5% DOCUMENT false-alarm budget")
    print("=" * 88)
    print(f"  {'k':>3s} {'nf':>3s} {'sentAUROC':>10s} {'docHit':>8s} {'docPrec':>9s} "
          f"{'sentTPR':>8s}")
    sweep = {}
    for k in ([args.k] if args.k else K_SWEEP):
        feats = base + tuple(f for f, _ in ranked[:k])
        r = pp.evaluate(hyb, human, feats, np.random.default_rng(fp.SEED), args.folds)
        sweep[k] = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
        print(f"  {k:3d} {len(feats):3d} {r['sentence_auroc']:10.3f} "
              f"{r['doc_hit_rate_strict']:8.3f} {r['doc_mean_precision_strict']:9.3f} "
              f"{r['sentence_tpr_strict']:8.3f}")
    print()

    if args.k:
        chosen = args.k
        why = "fixed on the command line"
    else:
        best = max(sweep, key=lambda k: sweep[k]["doc_hit_rate_strict"])
        top = sweep[best]["doc_hit_rate_strict"]
        # n = 139 documents, so a hit rate has a standard error near sqrt(p(1-p)/139) ~ 0.042.
        # Anything inside one of those of the best is not distinguishable from it, and the
        # smallest such k is preferred: fewer features is less to port and less to overfit.
        se = float(np.sqrt(max(top * (1 - top), 1e-9) / max(len(hyb), 1)))
        ok = [k for k in sorted(sweep) if sweep[k]["doc_hit_rate_strict"] >= top - se]
        chosen = ok[0]
        why = (f"smallest k within 1 SE ({se:.3f}) of the best ({best}, {top:.3f}); "
               f"candidates {ok}")
    print(f"CHOSEN k = {chosen} -- {why}\n")

    feats = base + tuple(f for f, _ in ranked[:chosen])

    # -- final fit on everything, for serving -----------------------------------------
    h_rows = [r for rows in hyb.values() for r in rows]
    u_rows = [r for rows in human.values() for r in rows]
    X = np.vstack([pp.matrix(h_rows, feats), pp.matrix(u_rows, feats)])
    y = np.concatenate([np.array([r["label"] for r in h_rows], dtype=np.float64),
                        np.zeros(len(u_rows))])
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    w = fp.fit_logreg((X - mu) / sd, y)

    # Threshold on the per-document maximum over clean human essays, which is the quantity a
    # reader actually spends. Recomputed from the FINAL fit, not carried over from a fold.
    su = fp.predict(w, (pp.matrix(u_rows, feats) - mu) / sd)
    dmax, idx = [], 0
    for _d, rows in human.items():
        n = len(rows)
        dmax.append(float(np.max(su[idx:idx + n])))
        idx += n
    thr = float(np.quantile(np.asarray(dmax), 0.95))

    oof = sweep[chosen]
    payload = {
        "kind": "polish_head",
        "purpose": "sentence does not match the rest of its own document",
        "features": list(feats),
        "relative_only": bool(args.relative_only),
        "excluded_feature": pp.LEAK,
        "excluded_reason": "machine spans are always the tail of a spliced hybrid",
        "mean": mu.tolist(), "scale": sd.tolist(),
        "intercept": float(w[0]), "weights": w[1:].tolist(),
        "threshold": thr,
        "threshold_basis": "95th percentile of the per-document maximum over "
                           f"{len(human)} unedited human documents; 5% of clean documents "
                           "carry a flag by construction",
        "z_cap": pp.Z_CAP,
        "min_sentences": pp.MIN_SENTENCES,
        "trained_on": {"hybrid_docs": len(hyb), "hybrid_sentences": len(h_rows),
                       "machine_sentences": int(y[:len(h_rows)].sum()),
                       "clean_human_docs": len(human),
                       "clean_human_sentences": len(u_rows)},
        "out_of_fold": {
            "sentence_auroc": oof["sentence_auroc"],
            "doc_hit_rate_at_5pct_doc_alarm": oof["doc_hit_rate_strict"],
            "doc_precision_at_5pct_doc_alarm": oof["doc_mean_precision_strict"],
            "sentence_tpr_at_5pct_doc_alarm": oof["sentence_tpr_strict"],
            "false_boundary_doc_rate_at_5pct_sentence_budget":
                oof["false_boundary_doc_rate"],
        },
        "sweep": {str(k): {"doc_hit_rate_strict": v["doc_hit_rate_strict"],
                           "doc_precision_strict": v["doc_mean_precision_strict"],
                           "sentence_auroc": v["sentence_auroc"]}
                  for k, v in sweep.items()},
        "chosen_k": chosen, "chosen_reason": why,
        "known_limit": {
            "frontier_polish_doc_hit_at_5pct_doc_alarm": 0.100,
            "frontier_n_docs": 10,
            "note": "measured in scripts/polish_probe.py, fitted on one era's rewrites and "
                    "tested cross-pipeline. The ceiling in docs/09 holds even with the "
                    "author's own prose in the same document as a reference.",
        },
    }
    out = ROOT / "artifacts" / f"{args.out}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"features ({len(feats)}): {', '.join(feats)}")
    print(f"threshold {thr:.4f} at a 5% document alarm rate")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

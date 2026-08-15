#!/usr/bin/env python
"""Six controls against the cross-observer result, because its shape is the shape of a bug.

    python scripts/consensus_controls.py --budget 2500

`consensus_probe.py` reports that cross-observer disagreement features move held-out Claude
Opus from AUROC 0.637 / **0% recall at a 5% false-accusation budget** to 0.957 / **96.7%**,
with the typography leak already controlled. That is the single largest result in this project
and it is exactly the shape of the two largest mistakes in it:

  * docs/04-failures.md #2 -- an AUROC 0.988 that was substantially a smart-quote detector;
  * docs/09 Result 3 -- a stylometric classifier at AUROC 1.000 and 100% recall on held-out
    Claude Opus which flagged **0%** of real GPT-3.5 essays from another collection and 17% of
    real students. It had learned which pipeline produced a file.

The second one is the direct threat here, and the reason is structural rather than
suspicious. Every machine document in the fusion corpus -- both Gemini tiers and all four
Claude checkpoints -- was written by OUR generation harness, with our prompts, our subject
list and our post-processing. So "leave-one-generator-out" holds out a *generator* and never
holds out the *pipeline*. If our pipeline imprints anything an observer can see, five
generators share it, and holding one out cannot reveal it.

docs/09 states the standing rule this script exists to satisfy: **any future supervised
attempt must be graded on machine text somebody else generated.**

THE CONTROLS

  C1  typography      already run: consensus_probe.py --normalise. Reported here for
                      completeness, not recomputed.
  C2  length          the human reference pools 591-word college essays with 134-word TOEFL
                      responses, and both a correlation and a variance are length-sensitive.
                      Re-run against a length-comparable human reference, and again with every
                      document truncated to a common word count.
  C3  ablation        x_min_lp and x_max_lp are the per-word envelope across observers, which
                      is absolute surprisal wearing a hat -- and they are the strongest single
                      features. Drop them and see what disagreement alone is worth. Then keep
                      only them, to see whether "consensus" was ever doing the work.
  C4  foreign human   JHU *Essays That Worked*: real admissions essays collected by somebody
                      else. Holding out another of Liang's files proves nothing (docs/09 R3).
  C5  CROSS-PIPELINE  the decisive one. Liang's own GPT-3.5 admissions essays: same genre,
                      2023, a pipeline with no connection to ours. A real detector flags them.
                      A pipeline-detector flags none, which is what happened last time.
  C6  collinearity    regress each consensus feature on the 7 single-observer baseline
                      features. A feature with R^2 near 1 is not new information, it is an
                      existing column re-expressed, and its apparent gain is a fitting artifact.

Every control that can be computed from cached scores is free. C4 and C5 need new documents
scored and therefore cost neurons, which ``--budget`` caps.
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

import bench_observer as bench  # noqa: E402
import consensus_probe as cp  # noqa: E402
import fusion_probe as fp  # noqa: E402

OUR_MACHINE = ("gemini_flash_lite", "gemini_flash", "claude_haiku", "claude_sonnet",
               "claude_opus")

# Sets that exist only for the controls.
EXTRA_SETS = [
    # C4: a human admissions-essay corpus we did not collect.
    ("human_jhu", "data/raw/jhu.jsonl", None),
    # C5: machine admissions essays we did not generate. Liang et al. produced these with
    # GPT-3.5 in 2023; nothing about our harness touched them.
    ("foreign_gpt35", "data/raw/liang_college_gpt3.jsonl", None),
    ("foreign_gpt35_prompteng", "data/raw/liang_college_gpt3_prompteng.jsonl", None),
]

ENVELOPE = ("x_min_lp", "x_max_lp")


def load_sets(per_set: int, truncate: int = 0) -> dict[str, list[dict]]:
    corpus: dict[str, list[dict]] = {}
    for name, path, model in bench.HUMAN_SETS + bench.MACHINE_SETS + EXTRA_SETS:
        rows = bench.load(path, model, per_set)
        if truncate:
            rows = [{**r, "text": " ".join(r["text"].split()[:truncate])} for r in rows]
            rows = [r for r in rows if len(r["text"].split()) >= truncate]
        if rows:
            corpus[name] = rows
    return corpus


def fit_and_score(data: dict[str, list[dict]], feats: tuple[str, ...],
                  human_sets: tuple[str, ...], train_machine: tuple[str, ...],
                  test_positives: tuple[str, ...], fpr_human: tuple[str, ...],
                  rng) -> dict[str, float]:
    """Fit on (human_sets, train_machine); threshold on fpr_human; score test_positives.

    Split out from ``consensus_probe.arm`` because the controls need the training and the
    testing populations to differ in ways leave-one-generator-out cannot express: a foreign
    pipeline as the positives, a foreign collection as the negatives.
    """
    def M(names):
        rows = [r for s in names for r in data.get(s, [])]
        return np.asarray([[r.get(f, np.nan) for f in feats] for r in rows], dtype=np.float64)

    H = M(human_sets)
    Xm = M(train_machine)
    Hf = M(fpr_human)
    P = M(test_positives)
    if not len(H) or not len(Xm) or not len(Hf) or not len(P):
        return {}

    # When the FPR reference IS the training human set, split it so the operating point is
    # never read off documents the model was fitted on. When it is a foreign collection, the
    # whole of it is available as a clean reference.
    if tuple(fpr_human) == tuple(human_sets):
        idx = rng.permutation(len(H))
        h_tr, Hf = H[idx[: len(H) // 2]], H[idx[len(H) // 2:]]
    else:
        h_tr = H

    X = np.vstack([h_tr, Xm])
    y = np.concatenate([np.zeros(len(h_tr)), np.ones(len(Xm))])
    Xs, hf_s, p_s = fp.standardise(X, Hf, P)
    w = fp.fit_logreg(Xs, y)
    p_pos, p_neg = fp.predict(w, p_s), fp.predict(w, hf_s)
    return {
        "auroc": fp.auroc(p_pos, p_neg),
        "tpr_at_5fpr": fp.tpr_at_fpr(p_pos, p_neg, 0.05),
        "n_pos": len(P), "n_neg": len(Hf), "n_train_machine": len(Xm),
    }


def sweep(data, feats, human_sets, fpr_human, label, rng) -> dict[str, dict]:
    """Leave-one-generator-out over our own machine sets, with a configurable human side."""
    out = {}
    print(f"  {label}")
    print(f"    {'held-out generator':24s} {'n':>4s} {'AUROC':>8s} {'TPR@5%':>8s}")
    for target in OUR_MACHINE:
        if not data.get(target):
            continue
        others = tuple(m for m in OUR_MACHINE if m != target and data.get(m))
        r = fit_and_score(data, feats, human_sets, others, (target,), fpr_human, rng)
        if not r:
            continue
        out[target] = r
        print(f"    {target:24s} {r['n_pos']:4d} {r['auroc']:8.3f} {r['tpr_at_5fpr']:8.3f}")
    return out


def collinearity(data: dict[str, list[dict]]) -> dict[str, float]:
    """R^2 of each consensus feature explained by the 7 single-observer baseline features.

    Pooled over every document in the corpus. A feature the baseline already contains cannot
    be the reason the baseline failed, whatever a fit does with it.
    """
    rows = [r for s in (*fp.HUMAN, *OUR_MACHINE) for r in data.get(s, [])]
    B = np.asarray([[r.get(f, np.nan) for f in fp.FEATS] for r in rows], dtype=np.float64)
    keep = np.all(np.isfinite(B), axis=1)
    B = B[keep]
    B = np.hstack([np.ones((len(B), 1)), (B - B.mean(0)) / np.maximum(B.std(0), 1e-9)])
    out = {}
    for feat in cp.CONSENSUS_FEATS:
        y = np.asarray([r.get(feat, np.nan) for r in rows], dtype=np.float64)[keep]
        if not np.all(np.isfinite(y)) or y.std() < 1e-12:
            continue
        beta, *_ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ beta
        out[feat] = float(1.0 - resid.var() / y.var())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observers", default="qwen,llama")
    ap.add_argument("--per-set", type=int, default=30)
    ap.add_argument("--budget", type=float, default=2500.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--truncate", type=int, default=0,
                    help="C2: cap every document at this many words before scoring")
    ap.add_argument("--out", default="consensus_controls")
    args = ap.parse_args()

    observers = [o.strip() for o in args.observers.split(",") if o.strip()]

    # Typography is folded in BOTH classes throughout the controls. C1 established it is not
    # the explanation; leaving it un-normalised here would reintroduce a known leak into every
    # other control.
    corpus = load_sets(args.per_set, args.truncate)
    print(f"observers: {', '.join(cp.OBSERVERS[o][0] for o in observers)}")
    if args.truncate:
        print(f"C2 truncation: every document capped at {args.truncate} words")
    for k, v in corpus.items():
        print(f"  {k:26s} {len(v):3d} docs  "
              f"{np.mean([len(r['text'].split()) for r in v]):5.0f} words")
    print()

    budget = cp.Budget(args.budget)
    data, dropped = cp.score_corpus(corpus, observers, budget, args.workers, normalise=True)
    if not data:
        return 1

    report: dict[str, object] = {"truncate": args.truncate, "dropped": dropped,
                                 "neurons": budget.spent,
                                 "n_docs": {k: len(v) for k, v in data.items()}}

    NOW = cp.CONSENSUS_FEATS
    NO_ENV = tuple(f for f in NOW if f not in ENVELOPE)

    print("=" * 92)
    print("C2  LENGTH -- human reference restricted to length-comparable native essays")
    print("=" * 92)
    report["C2_native_only"] = sweep(data, NOW, ("human_native",), ("human_native",),
                                     "consensus, human ref = native college essays only",
                                     np.random.default_rng(fp.SEED))
    print()

    print("=" * 92)
    print("C3  ABLATION -- is it disagreement, or absolute surprisal in disguise?")
    print("=" * 92)
    report["C3_no_envelope"] = sweep(data, NO_ENV, fp.HUMAN, fp.HUMAN,
                                     f"consensus WITHOUT {', '.join(ENVELOPE)} "
                                     f"({len(NO_ENV)} features)",
                                     np.random.default_rng(fp.SEED))
    print()
    report["C3_envelope_only"] = sweep(data, ENVELOPE, fp.HUMAN, fp.HUMAN,
                                       f"ONLY {', '.join(ENVELOPE)} (2 features)",
                                       np.random.default_rng(fp.SEED))
    print()

    print("=" * 92)
    print("C4  FOREIGN HUMAN -- false-positive reference we did not collect (JHU)")
    print("=" * 92)
    if data.get("human_jhu"):
        report["C4_jhu_reference"] = sweep(data, NOW, fp.HUMAN, ("human_jhu",),
                                           "consensus, FPR read off JHU admissions essays",
                                           np.random.default_rng(fp.SEED))
        report["C4_baseline_jhu"] = sweep(data, fp.FEATS, fp.HUMAN, ("human_jhu",),
                                          "baseline single-observer, same JHU reference",
                                          np.random.default_rng(fp.SEED))
    else:
        print("  JHU not scored -- budget or data missing")
    print()

    print("=" * 92)
    print("C5  CROSS-PIPELINE -- machine essays somebody ELSE generated (Liang GPT-3.5)")
    print("=" * 92)
    print("  Fit on humans + all five of OUR generators. Test on a foreign pipeline.")
    print("  docs/09 R3: the stylometry probe scored 1.000 here and flagged 0.0%.")
    rng = np.random.default_rng(fp.SEED)
    for pos in ("foreign_gpt35", "foreign_gpt35_prompteng"):
        if not data.get(pos):
            continue
        for label, feats in (("consensus", NOW), ("baseline", fp.FEATS)):
            r = fit_and_score(data, feats, fp.HUMAN, OUR_MACHINE, (pos,), fp.HUMAN, rng)
            if r:
                report[f"C5_{pos}_{label}"] = r
                print(f"    {pos:26s} {label:10s} n={r['n_pos']:3d} "
                      f"AUROC {r['auroc']:.3f}  flagged@5%FPR {r['tpr_at_5fpr']:.3f}")
        # And with the foreign HUMAN reference too, so neither side is ours.
        if data.get("human_jhu"):
            r = fit_and_score(data, NOW, fp.HUMAN, OUR_MACHINE, (pos,), ("human_jhu",), rng)
            if r:
                report[f"C5_{pos}_consensus_jhu_ref"] = r
                print(f"    {pos:26s} {'cons/JHU':10s} n={r['n_pos']:3d} "
                      f"AUROC {r['auroc']:.3f}  flagged@5%FPR {r['tpr_at_5fpr']:.3f}")
    print()

    print("=" * 92)
    print("C6  COLLINEARITY -- how much of each consensus feature the baseline already had")
    print("=" * 92)
    col = collinearity(data)
    report["C6_collinearity_r2"] = col
    for feat, r2 in sorted(col.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(r2 * 40))
        flag = "  <-- already in the baseline" if r2 > 0.9 else ""
        print(f"  {feat:20s} R2={r2:6.3f} {bar}{flag}")
    print()

    path = ROOT / "artifacts" / f"{args.out}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"neurons spent this run: {budget.spent:.1f} of {budget.cap:.0f} budgeted")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

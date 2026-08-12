#!/usr/bin/env python
"""Is the observer the bottleneck? Score the same essays under GPT-2 and under a 30 B model.

    python scripts/bench_observer.py --per-set 25

docs/08-cross-vendor.md established that the shipped detector scores 0.0% document recall on
Claude essays and 45.5% on mid-tier Gemini, against 94.8% on the cheap Gemini it was fitted
to. Two explanations survive that evidence and they call for opposite responses:

  (a) the CLASSIFIER overfitted one vendor's register, or
  (b) the INSTRUMENT is blind -- GPT-2 (124 M, 2019) simply does not find frontier prose
      distinctively surprising, so there was never a signal for any classifier to fit.

This script separates them, and it does so WITHOUT TRAINING ANYTHING. It takes raw per-token
statistics from each observer and asks how well a single number -- no weights, no fitting,
no threshold learned from any corpus -- ranks machine essays above human ones. If (a) were
the whole story, the raw statistics would already separate under GPT-2 and only the fitted
weights would be wrong. If (b) is right, GPT-2's statistics do not separate at frontier tier
and the 30 B observer's do.

Reported per (observer, machine set): AUROC against a pooled human reference, and TPR at the
threshold that costs 5% false positives on that reference. AUROC alone is a poor guide for
this product -- what matters commercially is recall at a false-accusation rate you can defend
-- so both are printed and the FPR budget is explicit.

The human reference deliberately pools NATIVE college essays with TOEFL/ESL writing. Every
published detector and our own docs/05-fairness.md show the false positives land on
non-native writers; a benchmark that measures separation only against native prose reports a
number the product cannot honour.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.scorer.local_lm import LocalLMScorer  # noqa: E402
from palimpsest.scorer.remote_lm import DEFAULT_MODEL, RemoteLMScorer  # noqa: E402

SEED = 20260811

# (label, path, filter) -- filters pick one generator tier out of a mixed file so the
# capability gradient is visible rather than averaged away.
HUMAN_SETS = [
    ("human_native", "data/raw/liang_college_human.jsonl", None),
    ("human_esl", "data/raw/liang_toefl.jsonl", None),
]
MACHINE_SETS = [
    ("gemini_flash_lite", "data/generated/modern_holdout.jsonl", None),
    ("gemini_flash", "data/generated/modern_unseen_family.jsonl", None),
    ("claude_haiku", "data/generated/claude_modern_heldout.jsonl", "haiku"),
    ("claude_sonnet", "data/generated/claude_modern_heldout.jsonl", "sonnet"),
    ("claude_opus", "data/generated/claude_modern_heldout.jsonl", "opus"),
]


def load(path: str, model: str | None, n: int) -> list[dict]:
    rows = []
    p = ROOT / path
    if not p.exists():
        return rows
    for line in p.open(encoding="utf-8"):
        d = json.loads(line)
        if model:
            meta = d.get("meta") or {}
            if meta.get("model") != model:
                continue
        text = " ".join((d.get("text") or "").split())
        if len(text.split()) < 120:  # too short to estimate anything stably
            continue
        rows.append({"id": d.get("id"), "text": text})
    random.Random(SEED).shuffle(rows)
    return rows[:n]


def stats_from(logprob: np.ndarray, rank: np.ndarray) -> dict[str, float]:
    """Candidate detection statistics computable from log-probability and rank alone.

    These are the published families that do not need a full predictive distribution:
    absolute surprisal, its dispersion, GLTR's rank buckets, and DetectLLM's LRR. Entropy
    and Fast-DetectGPT curvature are deliberately absent -- the remote API returns only the
    realised token, and approximating them from a truncated top-k would not be the statistic
    the name claims.
    """
    if len(logprob) < 5:
        return {}
    lr = np.log(np.maximum(rank, 1).astype(np.float64))
    return {
        "mean_logprob": float(np.mean(logprob)),
        "logprob_sd": float(np.std(logprob)),
        "mean_log_rank": float(np.mean(lr)),
        "frac_rank_top1": float(np.mean(rank <= 1)),
        "frac_rank_top10": float(np.mean(rank <= 10)),
        "frac_rank_tail": float(np.mean(rank > 100)),
        # DetectLLM's LRR. Machine text is both unsurprising AND highly ranked; dividing
        # removes much of the "this writer is simply plain" confound that sinks bare
        # perplexity thresholds on ESL prose.
        "lrr": float(-np.mean(logprob) / max(np.mean(lr), 1e-6)),
    }


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a random positive ranks above a random negative). Ties count a half."""
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(a) + len(b), float)
    ranks[order] = np.arange(1, len(a) + len(b) + 1)
    # average ranks within ties
    vals = np.concatenate([a, b])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def tpr_at_fpr(pos: list[float], neg: list[float], fpr: float, higher_is_machine: bool) -> float:
    """Recall at the threshold that costs exactly ``fpr`` on the human reference."""
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    if higher_is_machine:
        thr = float(np.quantile(b, 1.0 - fpr))
        return float((a >= thr).mean())
    thr = float(np.quantile(b, fpr))
    return float((a <= thr).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-set", type=int, default=25)
    ap.add_argument("--remote-model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args()

    corpus: dict[str, list[dict]] = {}
    for name, path, model in HUMAN_SETS + MACHINE_SETS:
        rows = load(path, model, args.per_set)
        if rows:
            corpus[name] = rows
        print(f"  {name:20s} {len(rows):3d} docs")
    print()

    results: dict[str, dict[str, list[dict]]] = defaultdict(dict)

    # -- remote 30 B observer -----------------------------------------------------
    remote = RemoteLMScorer(model=args.remote_model)
    for name, rows in corpus.items():
        def one(r):
            s = remote.score(r["text"])
            return stats_from(s.logprob, s.rank)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            got = [s for s in ex.map(one, rows) if s]
        results["remote"][name] = got
        print(f"  remote {name:20s} {len(got):3d} scored "
              f"(calls {remote.n_calls}, cached {remote.n_cached}, "
              f"neurons {remote.spent_neurons:.1f})")
    print()

    # -- local GPT-2 observer ------------------------------------------------------
    if not args.skip_local:
        local = LocalLMScorer("gpt2", device="cpu")
        for name, rows in corpus.items():
            got = []
            for r in rows:
                s = local.score(r["text"])
                st = stats_from(s.logprob, s.rank)
                if st:
                    got.append(st)
            results["local"][name] = got
            print(f"  local  {name:20s} {len(got):3d} scored")
        print()

    # -- report --------------------------------------------------------------------
    # Higher value means more machine-like for these; the rest are inverted.
    HIGHER = {"mean_logprob", "frac_rank_top1", "frac_rank_top10", "lrr"}
    out = {}
    for obs, per_set in results.items():
        human = [s for n in ("human_native", "human_esl") for s in per_set.get(n, [])]
        if not human:
            continue
        print("=" * 86)
        print(f"OBSERVER: {'GPT-2 124M (local)' if obs == 'local' else args.remote_model}")
        print(f"human reference: {len(human)} docs (native + ESL pooled)")
        print("=" * 86)
        for stat in ("lrr", "mean_logprob", "frac_rank_top10", "mean_log_rank", "logprob_sd"):
            hi = stat in HIGHER
            neg = [s[stat] for s in human if stat in s]
            print(f"\n  {stat}   ({'higher' if hi else 'lower'} = machine)")
            print(f"    {'set':22s} {'n':>4s} {'AUROC':>7s} {'TPR@5%FPR':>10s}")
            for name, _, _ in MACHINE_SETS:
                rows = per_set.get(name) or []
                pos = [s[stat] for s in rows if stat in s]
                if not pos:
                    continue
                a = auroc(pos, neg) if hi else auroc([-x for x in pos], [-x for x in neg])
                t = tpr_at_fpr(pos, neg, 0.05, hi)
                out[f"{obs}|{stat}|{name}"] = {"auroc": a, "tpr_at_5fpr": t, "n": len(pos)}
                print(f"    {name:22s} {len(pos):4d} {a:7.3f} {t:10.3f}")
        print()

    (ROOT / "artifacts" / "observer_bench.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("wrote artifacts/observer_bench.json")

    # Per-document statistics, so downstream experiments (fusion, length sweeps) can reuse
    # them without re-spending neurons on text that has already been scored.
    dump = {obs: {name: rows for name, rows in per_set.items()}
            for obs, per_set in results.items()}
    (ROOT / "artifacts" / "observer_bench_raw.json").write_text(
        json.dumps(dump), encoding="utf-8")
    print("wrote artifacts/observer_bench_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

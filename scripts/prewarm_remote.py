#!/usr/bin/env python
"""Fill the remote-observer cache concurrently, so feature extraction can run off disk.

    python scripts/prewarm_remote.py --sets train daigt esl --per-set 400

``build_features.py`` scores documents one at a time, which is right for a local model and
wrong for a network call: a 570-word essay takes ~4 s round-trip, so a 1,700-document corpus
would take two hours of waiting on latency. ``RemoteLMScorer`` caches every response keyed by
(model, text), so this script simply issues the same requests in parallel and lets the later
feature build hit the cache.

It is a cache warmer and nothing else. It computes no features and makes no decisions, which
is why it can be interrupted and re-run freely -- anything already fetched is skipped.

Neurons are Cloudflare's billing unit against a 10,000/day free allowance (~3.1 for a
570-word essay). The running total is printed so a large run can be stopped before it eats
the day's budget, and `--budget` stops it automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402
from palimpsest.scorer.remote_lm import RemoteLMScorer  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_features import GENERATED, SETS, _STEMS, flatten  # noqa: E402


def documents(set_name: str, limit: int | None) -> list[str]:
    texts: list[str] = []
    for source_id in SETS[set_name]:
        folder = "generated" if source_id in GENERATED else "raw"
        path = ROOT / "data" / folder / f"{_STEMS.get(source_id, source_id)}.jsonl"
        if not path.exists():
            print(f"  ! missing {path.relative_to(ROOT)}")
            continue
        rows = read_jsonl(path)
        if limit:
            rows = rows[:limit]
        texts.extend(flatten(d.text) for d in rows)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--per-set", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=8000.0,
                    help="stop after this many neurons; the free allowance is 10,000/day")
    args = ap.parse_args()

    scorer = RemoteLMScorer()
    lock = threading.Lock()
    stop = threading.Event()
    done = {"n": 0, "err": 0}

    def one(text: str) -> None:
        if stop.is_set():
            return
        try:
            scorer.score(text)
        except Exception as exc:  # a single bad document must not kill a long run
            with lock:
                done["err"] += 1
            if done["err"] <= 3:
                print(f"    ! {str(exc)[:110]}")
            return
        with lock:
            done["n"] += 1
            if scorer.spent_neurons >= args.budget:
                stop.set()
            if done["n"] % 50 == 0:
                print(f"    {done['n']} scored | {scorer.n_calls} calls "
                      f"| {scorer.n_cached} cached | {scorer.spent_neurons:.0f} neurons")

    for name in args.sets:
        if name not in SETS:
            print(f"! unknown set {name}; known: {', '.join(sorted(SETS))}")
            continue
        texts = documents(name, args.per_set)
        print(f"\n=== {name}: {len(texts)} documents")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, texts))
        if stop.is_set():
            print(f"  ! budget of {args.budget:.0f} neurons reached; stopping")
            break

    print(f"\n{done['n']} documents cached ({done['err']} errors)")
    print(f"{scorer.n_calls} live calls, {scorer.n_cached} already cached, "
          f"{scorer.spent_neurons:.1f} neurons spent")
    (ROOT / "artifacts" / "prewarm_last.json").write_text(json.dumps({
        "documents": done["n"], "errors": done["err"], "calls": scorer.n_calls,
        "cached": scorer.n_cached, "neurons": scorer.spent_neurons,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

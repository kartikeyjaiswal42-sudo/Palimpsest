#!/usr/bin/env python
"""Put frontier-model text INTO the training set, and measure what that buys.

    python scripts/train_with_frontier.py --suffix _remote

WHY THIS EXISTS. The shipped detector's `trainSources` are GPT-3.5 (2023) and
gemini-3.1-flash-lite. It has never seen a Claude sentence. Every "0% recall on Claude"
number in this project is therefore measuring CROSS-GENERATOR GENERALISATION -- can a
detector fitted on one generation of models catch the next -- which is a scientifically
interesting question and NOT the same claim as "frontier prose is undetectable". Those two
were conflated, and a user's screenshot of a commercial detector confidently flagging a
Claude essay is what exposed it.

The held-out design was deliberate and it was right for the question it answered. It is the
wrong design for a shipped product, where you want every generator you can get.

WHAT THIS DOES. Splits the Claude corpus by TOPIC -- not by document -- so no subject appears
on both sides, then trains on half and evaluates on the other half. Topic-wise splitting is
load-bearing: the corpus was generated from pre-registered subjects, and splitting by
document would let the same topic teach and test, which reads as generalisation and is
memorisation.

WHAT TO WATCH. Two numbers decide whether this ships, and the second one is the one that
matters:

  1. Recall on held-out Claude essays. Currently 0%.
  2. The false-positive rate on ESL and out-of-domain human writing. If detecting Claude
     costs accusations of real students, it is not an improvement -- it is the trade this
     project has refused everywhere else, made quietly. The FPR budget is unchanged, so a
     model that cannot hit it will simply threshold itself into uselessness, which is the
     honest failure mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def topic_side(topic: str, salt: str = "frontier-split-v1") -> int:
    """Stable 50/50 split keyed on the topic string."""
    h = hashlib.sha256(f"{salt}:{topic}".encode()).hexdigest()
    return int(h[:8], 16) % 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_remote")
    ap.add_argument("--out-suffix", default="_frontier")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    feats = ROOT / "data" / "features"
    base = feats / f"train{args.suffix}.jsonl"
    claude = feats / f"modern_claude{args.suffix}.jsonl"
    for p in (base, claude):
        if not p.exists():
            print(f"! missing {p.relative_to(ROOT)}")
            return 1

    rows = [json.loads(line) for line in claude.open(encoding="utf-8")]
    topics = {r["doc_id"]: (r.get("doc_meta") or {}).get("topic", r["doc_id"]) for r in rows}
    side = {d: topic_side(str(t)) for d, t in topics.items()}
    n_docs = len(side)
    n_tr = sum(1 for v in side.values() if v == 0)
    n_topics = len({str(t) for t in topics.values()})
    print(f"claude corpus: {n_docs} documents across {n_topics} topics")
    print(f"  train side {n_tr}   eval side {n_docs - n_tr}   (split by topic, not document)")

    train_out = feats / f"train_frontier{args.suffix}.jsonl"
    eval_out = feats / f"modern_claude_eval{args.suffix}.jsonl"

    n_base = 0
    with train_out.open("w", encoding="utf-8") as fh:
        for line in base.open(encoding="utf-8"):
            fh.write(line)
            n_base += 1
        added = 0
        for r in rows:
            if side[r["doc_id"]] == 0:
                fh.write(json.dumps(r) + "\n")
                added += 1
    with eval_out.open("w", encoding="utf-8") as fh:
        for r in rows:
            if side[r["doc_id"]] == 1:
                fh.write(json.dumps(r) + "\n")
    print(f"wrote {train_out.name}: {n_base} base + {added} claude sentences")
    print(f"wrote {eval_out.name}\n")

    if args.skip_train:
        return 0

    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
           "--features", str(train_out), "--out-suffix", args.out_suffix,
           # Both of these were previously left at their defaults, and both defaults are
           # wrong for a remote build. Missing --drop-features fitted a mean and a weight
           # onto three columns that are entirely NaN under this observer (mean_entropy got
           # -0.94); the default --mixed-features pulled in GPT-2-scored sentences, which the
           # guard in train.py now refuses. Deriving both from --suffix means the two cannot
           # drift apart again.
           "--mixed-features", str(feats / f"localisation{args.suffix}.jsonl")]
    if args.suffix == "_remote":
        cmd += ["--drop-features", "mean_entropy", "entropy_sd", "curvature",
                "curvature_z_in_doc"]
    print("$", " ".join(cmd[1:]), "\n")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

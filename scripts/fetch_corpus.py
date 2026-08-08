#!/usr/bin/env python
"""Rebuild the human corpus from its sources.

    python scripts/fetch_corpus.py                 # every source
    python scripts/fetch_corpus.py --only liang_college_human hamilton
    python scripts/fetch_corpus.py --limit 50      # cap per source, for a quick check

Writes one JSONL per source into ``data/raw/`` (gitignored -- see docs/02-dataset.md for
why the human text is not committed) and a manifest recording exactly what was fetched,
including a SHA-256 of every document so a rebuild can be proved identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import fetch_source, write_jsonl  # noqa: E402
from palimpsest.data.sources import SOURCES, SOURCES_BY_ID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="source ids to fetch (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="cap documents per source")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest" / "fetched.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out)
    selected = [SOURCES_BY_ID[i] for i in args.only] if args.only else list(SOURCES)

    manifest: dict = {"fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": []}
    total = 0
    failures = 0

    for source in selected:
        print(f"\n=== {source.id} :: {source.label}")
        started = time.perf_counter()
        try:
            docs = fetch_source(source, limit=args.limit)
        except Exception as exc:  # a dead source must not kill the whole rebuild
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            manifest["sources"].append({"id": source.id, "ok": False, "error": str(exc)[:300]})
            failures += 1
            continue

        path = out_dir / f"{source.id}.jsonl"
        write_jsonl(docs, path)
        words = sorted(d.n_words for d in docs)
        elapsed = time.perf_counter() - started
        median = words[len(words) // 2] if words else 0
        print(
            f"    {len(docs)} docs (expected ~{source.expected_n}) | "
            f"words min/med/max {words[0] if words else 0}/{median}/{words[-1] if words else 0} | "
            f"{elapsed:.1f}s -> {path.relative_to(ROOT)}"
        )
        manifest["sources"].append(
            {
                "id": source.id,
                "ok": True,
                "label": source.label,
                "authorship": source.authorship,
                "role": source.role,
                "url": source.url,
                "licence": source.licence,
                "citation": source.citation,
                "preChatGPT": source.pre_chatgpt,
                "limitations": source.limitations,
                "redistributable": source.redistributable,
                "nDocuments": len(docs),
                "expected": source.expected_n,
                "medianWords": median,
                "documentHashes": [d.sha256 for d in docs],
            }
        )
        total += len(docs)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{total} documents from {len(selected) - failures}/{len(selected)} sources")
    print(f"manifest -> {Path(args.manifest).relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

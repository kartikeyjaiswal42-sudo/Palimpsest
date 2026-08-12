#!/usr/bin/env python3
"""Create a 500-essay synthetic college-admissions corpus for PalamiAssist.

    python scripts/create_palamiassist_essay_corpus.py

This script does not call a model. It samples from the generated essays already present in
``data/generated/`` and writes two exports:

* ``palamiassist_college_essays_500.jsonl`` uses Palimpsest's Document shape.
* ``palamiassist_college_essays_500_chat.jsonl`` uses chat fine-tuning pairs.

The default split keeps the set diverse without letting one generator dominate:

* 250 Claude-family essays
* 239 Gemini-family essays
* 11 GPT-3.5-era essays

All examples are synthetic machine-written essays. They are useful for detector training,
assistant behavior tests, or synthetic drafting workflows, but they should not be presented
as real applicant writing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import Document, read_jsonl, write_jsonl  # noqa: E402


SEED = 20260810
DEFAULT_COUNT = 500
MIN_WORDS = 400

OUT_STEM = "palamiassist_college_essays_500"
SOURCE_ID = "palamiassist_college_essays_500"
ROLE = "palamiassist-train"

FAMILY_SOURCES = {
    "claude": [
        "data/generated/claude_modern.jsonl",
        "data/generated/claude_modern_heldout.jsonl",
    ],
    "gemini": [
        "data/generated/modern_train.jsonl",
        "data/generated/modern_holdout.jsonl",
        "data/generated/modern_unseen.jsonl",
        "data/generated/modern_unseen_family.jsonl",
        "data/generated/modern_control.jsonl",
    ],
    "gpt3_5": [
        "data/generated/machine_essays.jsonl",
    ],
}

DEFAULT_QUOTAS = {
    "claude": 250,
    "gemini": 239,
    "gpt3_5": 11,
}

SYSTEM_MESSAGE = (
    "You are PalamiAssist, a college-admissions essay assistant used for synthetic training "
    "data. Write fictional, polished first-person admissions essays. Do not claim the essay "
    "belongs to a real applicant."
)


def load_docs(paths: Iterable[str]) -> list[tuple[Document, str]]:
    """Load unique, machine-authored essays from the requested files."""
    docs: list[tuple[Document, str]] = []
    seen: set[str] = set()
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing {rel}")
        for doc in read_jsonl(path):
            if doc.authorship != "machine":
                continue
            if doc.n_words < MIN_WORDS:
                continue
            if doc.sha256 in seen:
                continue
            seen.add(doc.sha256)
            docs.append((doc, rel))
    return docs


def clean_prompt(prompt: str) -> str:
    """Remove source-specific assistant naming from prompts."""
    prompt = prompt.strip()
    replacements = {
        "Hi GPT, I'd like you to write a college application essay. ": "",
        "Hi GPT, I'd like you to write a college application essay.": "",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    if not prompt:
        return "Write a college application essay."
    if not prompt.lower().startswith(("write", "describe", "reflect", "recount", "discuss")):
        return f"Write a college application essay. {prompt}"
    return prompt


def user_message(doc: Document) -> str:
    """Build the supervised prompt paired with one essay."""
    meta = doc.meta or {}
    prompt = clean_prompt(str(meta.get("prompt") or "Write a college application essay."))
    subject = meta.get("subject") or meta.get("topic")
    target_words = meta.get("target_words") or doc.n_words

    parts = [prompt]
    if subject:
        parts.append(f"Use this subject or life material: {subject}.")
    parts.append(f"Aim for about {target_words} words.")
    parts.append("Return only the essay text, with no title or commentary.")
    return "\n\n".join(parts)


def choose_family(pool: list[tuple[Document, str]], family: str, quota: int,
                  rng: random.Random) -> list[tuple[Document, str]]:
    """Deterministically sample a family pool."""
    if len(pool) < quota:
        raise SystemExit(f"{family} has only {len(pool)} usable essays; need {quota}")
    shuffled = sorted(pool, key=lambda item: (item[1], item[0].id))
    rng.shuffle(shuffled)
    return shuffled[:quota]


def build(count: int, seed: int) -> list[Document]:
    if count != DEFAULT_COUNT:
        raise SystemExit(
            "this curated builder is intentionally fixed at 500 essays. "
            "Change DEFAULT_QUOTAS in the script if you want a different balance."
        )

    rng = random.Random(seed)
    selected: list[tuple[str, Document, str]] = []
    for family, quota in DEFAULT_QUOTAS.items():
        pool = load_docs(FAMILY_SOURCES[family])
        for doc, source_file in choose_family(pool, family, quota, rng):
            selected.append((family, doc, source_file))

    rng.shuffle(selected)

    out: list[Document] = []
    seen: set[str] = set()
    for i, (family, doc, source_file) in enumerate(selected, start=1):
        if doc.sha256 in seen:
            raise SystemExit(f"duplicate selected: {doc.id}")
        seen.add(doc.sha256)
        meta = {
            "family": family,
            "original_id": doc.id,
            "original_source_id": doc.source_id,
            "original_role": doc.role,
            "source_file": source_file,
            "synthetic": True,
            **(doc.meta or {}),
        }
        out.append(
            Document(
                id=f"{SOURCE_ID}:{i:04d}",
                source_id=SOURCE_ID,
                text=doc.text,
                authorship="machine",
                role=ROLE,
                meta=meta,
            )
        )
    return out


def write_chat_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message(doc)},
                    {"role": "assistant", "content": doc.text},
                ]
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(docs: list[Document], path: Path, seed: int) -> None:
    families = Counter(d.meta.get("family") for d in docs)
    source_files = Counter(d.meta.get("source_file") for d in docs)
    models = Counter(
        d.meta.get("model") or d.meta.get("generator") or d.meta.get("original_source_id")
        for d in docs
    )
    words = sorted(d.n_words for d in docs)
    payload = {
        "name": SOURCE_ID,
        "description": "500 synthetic machine-written college admissions essays for PalamiAssist.",
        "count": len(docs),
        "seed": seed,
        "min_words_filter": MIN_WORDS,
        "authorship": "machine",
        "role": ROLE,
        "synthetic": True,
        "family_counts": dict(sorted(families.items())),
        "source_file_counts": dict(sorted(source_files.items())),
        "model_counts": dict(sorted(models.items())),
        "word_counts": {
            "min": words[0],
            "p25": words[len(words) // 4],
            "median": words[len(words) // 2],
            "p75": words[(len(words) * 3) // 4],
            "max": words[-1],
        },
        "outputs": {
            "document_jsonl": f"data/generated/{OUT_STEM}.jsonl",
            "chat_jsonl": f"data/generated/{OUT_STEM}_chat.jsonl",
        },
        "source_note": (
            "Built only from generated corpora already present in this repository; no model "
            "or network call was made by this builder."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    docs = build(args.count, args.seed)
    doc_path = ROOT / "data" / "generated" / f"{OUT_STEM}.jsonl"
    chat_path = ROOT / "data" / "generated" / f"{OUT_STEM}_chat.jsonl"
    manifest_path = ROOT / "artifacts" / f"{OUT_STEM}_manifest.json"

    write_jsonl(docs, doc_path)
    write_chat_jsonl(docs, chat_path)
    write_manifest(docs, manifest_path, args.seed)

    families = Counter(d.meta.get("family") for d in docs)
    words = sorted(d.n_words for d in docs)
    print(f"wrote {len(docs)} essays")
    print(f"  documents: {doc_path.relative_to(ROOT)}")
    print(f"  chat:      {chat_path.relative_to(ROOT)}")
    print(f"  manifest:  {manifest_path.relative_to(ROOT)}")
    print(f"  families:  {dict(sorted(families.items()))}")
    print(f"  words:     min {words[0]}, median {words[len(words)//2]}, max {words[-1]}")
    # Keep a tiny preview on stdout so the user can sanity-check the shape without opening
    # a large file.
    print("\nfirst row preview:")
    print(json.dumps(asdict(docs[0]) | {"text": docs[0].text[:220] + "..."}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

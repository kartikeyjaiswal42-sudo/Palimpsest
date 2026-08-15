#!/usr/bin/env python
"""Ingest the corpus on disk and write down what is actually in it.

    python scripts/dataset_report.py
    python scripts/dataset_report.py --check     # CI mode: fail if the report is stale

Writes ``artifacts/dataset_report.json`` and ``docs/DATASET-REPORT.md``, both generated from
the files that are really present rather than from prose describing what was intended.

Why generated rather than written
---------------------------------
``docs/02-dataset.md`` is the hand-written dataset document and it stays -- it carries
licences, provenance and the reasoning behind each source, which no script can infer. What a
script *can* do is keep the counts honest. PROJECT.md §2 records the failure this exists to
prevent: the application rendered its limitations panel from a stale artifact and published a
17.8% false-positive rate where the served build measured 10.9%. A number that is typed once
and re-read forever will eventually describe a system that no longer exists.

So every count here is recomputed from the corpus, and ``--check`` makes a stale report a
failing build rather than a thing someone notices later.

The four categories the brief asks for
-------------------------------------
``human``, ``machine`` (raw AI), ``hybrid`` (human text a model later edited -- the brief's
realistic case), and ``esl``, which is a *cross-cutting* tag rather than a fifth authorship
class: every ESL document is also a human document. They are reported both ways because
adding them as a fifth column would double-count the human total, and a composition table
whose columns do not sum is worse than one that needs a sentence of explanation.

Coverage gaps are EDITED BY HAND AND PRESERVED
----------------------------------------------
The "what this dataset does not cover" section is the one part of this file a person must
write; a script cannot know that a corpus is missing ESL-authored admissions essays. It
lives in ``docs/dataset-gaps.md``. This script reads it, embeds it in both outputs, and
**never overwrites it** -- it creates the file with a template on first run and leaves it
alone after that. Regenerating the report must not silently delete somebody's honest account
of their own blind spots.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402

JSON_OUT = ROOT / "artifacts" / "dataset_report.json"
MD_OUT = ROOT / "docs" / "DATASET-REPORT.md"
GAPS = ROOT / "docs" / "dataset-gaps.md"

#: Roles whose documents are English-learner writing. `authorship` cannot carry this: an
#: ESL essay is human-written, so the tag is orthogonal to the authorship label.
ESL_ROLES = frozenset({"esl-eval"})

GAPS_TEMPLATE = """\
<!-- Written by hand. scripts/dataset_report.py reads this file and never overwrites it. -->

# What this dataset does not cover

Each heading is a real limit on what any number measured on this corpus can mean. Delete a
line only when the gap is actually closed, not when it becomes inconvenient.

## Subject matter

The machine essays answer subjects chosen in `scripts/generate_modern.py` and pre-registered
in `scripts/plan_claude_corpus.py`. Human essays were written to whatever prompt their
applicant faced. "Machine" and "those subjects" are therefore correlated across most of the
modern corpus, which is why `modern_control` exists — same generator, no subject steering.
A reader should assume subject-matter bias anywhere that control was not run.

## Authors

No ESL-authored *admissions essays*. We hold native-authored personal statements and we hold
English-learner writing in other genres; the cell where they meet is empty. PROJECT.md §8
bounds the risk by transplanting a measured proficiency effect. It does not measure it.

## Attacks

No paraphrase or "humanizer" output. Published benchmarks find these hurt detectors badly,
so every accuracy number here is an **upper** bound on adversarial performance.

## Generators

All machine text comes from our own generation pipeline (plus one foreign GPT-3.5 set used
only as a control). docs/12 records a signal that scored 0.960 AUROC on our pipeline and
0.490 on somebody else's.

## Genre and language

Admissions essays, English only. The genre gate is fitted against *student* writing, so
"out of scope" means "unlike the student writing we tested", not "not an admissions essay".
"""


def collect() -> dict:
    """Read every corpus file and count what is in it."""
    files, unreadable = [], []
    by_authorship: Counter = Counter()
    by_role: Counter = Counter()
    by_source: dict[str, dict] = {}
    esl_docs = 0
    total_words = 0
    generators: Counter = Counter()
    seen_hashes: dict[str, str] = {}
    duplicates = 0

    for folder in ("raw", "generated"):
        base = ROOT / "data" / folder
        if not base.exists():
            continue
        for path in sorted(base.glob("*.jsonl")):
            try:
                docs = read_jsonl(path)
            except Exception as exc:
                # A file we cannot parse is reported, never skipped silently: an
                # unreadable source is a hole in the corpus and the report must show it.
                unreadable.append({"file": f"{folder}/{path.name}", "error": str(exc)[:160]})
                continue

            if not docs:
                files.append({"file": f"{folder}/{path.name}", "documents": 0})
                continue

            auth = Counter(d.authorship for d in docs)
            roles = Counter(d.role for d in docs)
            words = sum(d.n_words for d in docs)
            is_esl = any(r in ESL_ROLES for r in roles)

            for d in docs:
                if d.sha256 in seen_hashes and seen_hashes[d.sha256] != d.id:
                    duplicates += 1
                seen_hashes.setdefault(d.sha256, d.id)
                model = (d.meta or {}).get("model") or (d.meta or {}).get("generator")
                if model and d.authorship != "human":
                    generators[str(model)] += 1

            by_authorship.update(auth)
            by_role.update(roles)
            total_words += words
            if is_esl:
                esl_docs += sum(v for k, v in roles.items() if k in ESL_ROLES)

            by_source[f"{folder}/{path.name}"] = {
                "documents": len(docs),
                "authorship": dict(auth),
                "roles": dict(roles),
                "words": words,
                "medianWords": sorted(d.n_words for d in docs)[len(docs) // 2],
                "isEsl": is_esl,
            }
            files.append({"file": f"{folder}/{path.name}", "documents": len(docs)})

    total = sum(by_authorship.values())
    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "totals": {
            "documents": total,
            "words": total_words,
            "human": by_authorship.get("human", 0),
            "machineRaw": by_authorship.get("machine", 0),
            "humanEditedMachine": by_authorship.get("hybrid", 0),
            # Cross-cutting: these are ALSO counted in `human`, deliberately.
            "eslCrossCut": esl_docs,
        },
        "byAuthorship": dict(by_authorship),
        "byRole": dict(by_role),
        "bySource": by_source,
        "generators": dict(generators.most_common()),
        "unreadable": unreadable,
        "duplicateTexts": duplicates,
        "notes": {
            "eslIsCrossCutting": (
                "Every ESL document is also a human document. `eslCrossCut` is a tag on the "
                "human total, not a fourth authorship class, so the four category counts "
                "deliberately do not sum to the document total."
            ),
            "hybridMeaning": (
                "`humanEditedMachine` counts documents with known machine spans inside human "
                "prose -- the brief's realistic case, and the reason detection is per sentence."
            ),
        },
    }


def render_markdown(data: dict, gaps: str) -> str:
    t = data["totals"]
    lines = [
        "# Dataset composition",
        "",
        "<!-- GENERATED by scripts/dataset_report.py. Do not edit by hand: run the script. -->",
        f"<!-- generated {data['generatedAt']} -->",
        "",
        "Counts are recomputed from the corpus on disk every run. The hand-written source "
        "document, with licences and provenance, is [02-dataset.md](02-dataset.md).",
        "",
        "## Composition",
        "",
        "| category | documents |",
        "|---|---|",
        f"| Human | {t['human']:,} |",
        f"| Raw AI | {t['machineRaw']:,} |",
        f"| Human-edited AI (hybrid) | {t['humanEditedMachine']:,} |",
        f"| ESL *(a tag on the human total, not a fourth class)* | {t['eslCrossCut']:,} |",
        f"| **Total documents** | **{t['documents']:,}** |",
        f"| Total words | {t['words']:,} |",
        "",
        f"> {data['notes']['eslIsCrossCutting']}",
        "",
    ]

    if data["unreadable"]:
        lines += ["## Unreadable sources", "",
                  "These files are on disk and could not be parsed. They are listed rather "
                  "than skipped, because a source that silently fails to load is a hole in "
                  "every number measured downstream.", "",
                  "| file | error |", "|---|---|"]
        lines += [f"| `{u['file']}` | {u['error']} |" for u in data["unreadable"]]
        lines.append("")

    lines += ["## By source", "",
              "| source | documents | median words | authorship |", "|---|---|---|---|"]
    for name, s in sorted(data["bySource"].items()):
        auth = ", ".join(f"{k} {v}" for k, v in sorted(s["authorship"].items()))
        lines.append(f"| `{name}` | {s['documents']:,} | {s['medianWords']:,} | {auth} |")
    lines.append("")

    if data["generators"]:
        lines += ["## Machine text by generator", "",
                  "| generator | documents |", "|---|---|"]
        lines += [f"| `{k}` | {v:,} |" for k, v in data["generators"].items()]
        lines.append("")

    if data["duplicateTexts"]:
        lines += [f"> **{data['duplicateTexts']} duplicate texts** (identical SHA-256 under "
                  "different ids) are present. Duplicates across a train/test split inflate "
                  "every measured number.", ""]

    lines += ["---", "", gaps.strip(), ""]
    return "\n".join(lines)


def _without_timestamp(md: str) -> str:
    """The report text minus its generated-at comment, for staleness comparison."""
    return "\n".join(ln for ln in md.splitlines() if not ln.startswith("<!-- generated "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the written report differs from a fresh one")
    args = ap.parse_args()

    if not GAPS.exists():
        GAPS.parent.mkdir(parents=True, exist_ok=True)
        GAPS.write_text(GAPS_TEMPLATE, encoding="utf-8")
        print(f"created {GAPS.relative_to(ROOT)} (edit it; this script will never overwrite it)")
    gaps = GAPS.read_text(encoding="utf-8")

    data = collect()
    md = render_markdown(data, gaps)

    if args.check:
        stale = []
        if not JSON_OUT.exists() or not MD_OUT.exists():
            stale.append("report missing")
        else:
            old = json.loads(JSON_OUT.read_text())
            # `generatedAt` changes every run and is not a staleness signal.
            if {k: v for k, v in old.items() if k != "generatedAt"} != \
               {k: v for k, v in data.items() if k != "generatedAt"}:
                stale.append("counts differ from the corpus on disk")
            # The generated-at line changes every run and is not a staleness signal, so it
            # is stripped from both sides before comparing -- otherwise --check fails
            # immediately after a successful write, which trains people to ignore it.
            if _without_timestamp(MD_OUT.read_text(encoding="utf-8")) != _without_timestamp(md):
                stale.append("markdown differs (corpus or gaps document changed)")
        if stale:
            print("STALE: " + "; ".join(stale))
            print("  run: python scripts/dataset_report.py")
            return 1
        print("dataset report is current")
        return 0

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(md, encoding="utf-8")

    t = data["totals"]
    print(f"documents {t['documents']:,}  |  human {t['human']:,}  "
          f"raw AI {t['machineRaw']:,}  hybrid {t['humanEditedMachine']:,}  "
          f"ESL {t['eslCrossCut']:,} (of human)")
    if data["unreadable"]:
        print(f"  ! {len(data['unreadable'])} unreadable source(s) -- listed in the report")
    if data["duplicateTexts"]:
        print(f"  ! {data['duplicateTexts']} duplicate texts")
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"wrote {MD_OUT.relative_to(ROOT)}")
    print(f"gaps  {GAPS.relative_to(ROOT)} (hand-written, preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

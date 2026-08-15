#!/usr/bin/env python
"""Bundle everything Colab needs to run Binoculars over the corpus.

    python scripts/export_for_colab.py
    python scripts/export_for_colab.py --sets train_remote esl_remote --out /tmp/bundle

Writes ``colab_bundle/`` containing:

* ``documents.jsonl``  -- one row per document: id, text, and the sentence spans to score
* ``palimpsest_binoculars.py`` -- the real scorer, mechanically de-relativised
* ``README.txt``       -- what to do with it

Why the spans travel with the text
----------------------------------
The point of this run is to add a column to feature matrices that already exist, so the
scores must land on **exactly** the sentences those matrices describe. Re-segmenting in the
notebook would re-derive the spans from the text and any difference in segmentation -- a
different spaCy version, a different newline rule -- would silently misalign every score by
one sentence. So the spans are exported from the matrices themselves and the notebook only
slices what it is given. ``scripts/join_binoculars.py`` then re-checks the alignment on the
way back in.

Why the scorer is shipped rather than pasted into the notebook
--------------------------------------------------------------
This project's recurring failure is a fix applied to one copy of two (PROJECT.md §2). A
notebook with the scoring maths pasted into a cell is a second copy by construction, and it
would drift the first time either side changed. So the bundle carries
``src/palimpsest/scorer/binoculars.py`` itself, with one mechanical edit: the single
relative import (``from .local_lm import select_device``) is replaced by the body of that
function, because a package-relative import cannot resolve in a standalone notebook. The
edit is asserted rather than assumed -- if the import line ever changes shape, this script
fails instead of shipping a half-rewritten module.

BEFORE YOU RUN THIS, a data question that is not this script's to decide
-----------------------------------------------------------------------
The bundle contains essay text, and uploading it to Colab sends it to Google. That is the
same class of decision ``/api/health`` reports as ``textLeavesMachine``, and it deserves the
same explicitness:

* **PERSUADE is CC BY-NC-SA 4.0** -- non-commercial. Research use is fine; be aware of it.
* **DAIGT declares no licence** and docs/02 records that it is *not redistributed* by this
  repository. It is excluded here by default and ``--allow-unlicensed`` is required to
  override, which you should not do.
* The ELLIPSE and TOEFL sets are real student writing.

``--no-text`` produces a bundle with hashes and spans but no essay text, which is useless
for scoring and exists only so you can inspect what would be sent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402

FEATURE_DIR = ROOT / "data" / "features"
SCORER = ROOT / "src" / "palimpsest" / "scorer" / "binoculars.py"

DEFAULT_SETS = (
    "train_remote",
    "esl_remote",
    "domain_shift_remote",
    "modern_holdout_remote",
    "modern_claude_eval_remote",
    "modern_unseen_family_remote",
    "localisation_remote",
    "adversarial_remote",
)

#: Sources that must not be redistributed. docs/02-dataset.md; DAIGT's mirror declares no
#: licence at all, so it is a diagnostic here and never leaves the machine.
UNLICENSED = ("daigt",)

_RELATIVE_IMPORT = "from .local_lm import select_device"

_SELECT_DEVICE = '''
def select_device(preference: str = "cpu") -> str:
    """Resolve a device string. Inlined by scripts/export_for_colab.py from
    palimpsest.scorer.local_lm, because a package-relative import cannot resolve in a
    standalone notebook. Behaviour is identical."""
    if preference == "cpu":
        return "cpu"
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference
'''


def derelativise(source: str) -> str:
    """Replace the one relative import with the function it imports.

    Asserted, not assumed: if the import line changes, this raises instead of silently
    shipping a module whose device selection has quietly become a no-op.
    """
    if _RELATIVE_IMPORT not in source:
        raise SystemExit(
            f"expected to find {_RELATIVE_IMPORT!r} in binoculars.py and did not. "
            "The scorer's imports have changed; update export_for_colab.py rather than "
            "shipping a module that will not import."
        )
    if source.count(_RELATIVE_IMPORT) != 1:
        raise SystemExit("more than one relative import found; refusing to guess.")
    out = source.replace(_RELATIVE_IMPORT, _SELECT_DEVICE.strip())
    # Nothing else may be package-relative.
    leftover = re.findall(r"^from \.\S+ import .*$", out, flags=re.MULTILINE)
    if leftover:
        raise SystemExit(f"unhandled relative imports remain: {leftover}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--out", default=str(ROOT / "colab_bundle"))
    ap.add_argument("--no-text", action="store_true",
                    help="omit essay text: shows what would be sent without sending it")
    ap.add_argument("--allow-unlicensed", action="store_true",
                    help="include sources this repository does not redistribute (do not)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- the spans, taken from the matrices they must align with ---------------------
    wanted: dict[str, dict[str, list]] = {}
    for name in args.sets:
        path = FEATURE_DIR / f"{name}.jsonl"
        if not path.exists():
            print(f"  {name}: no such file, skipped")
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            entry = wanted.setdefault(
                r["doc_id"], {"source": r["source_id"], "spans": []}
            )
            entry["spans"].append(
                {"i": r["sentence_index"], "start": r["start"], "end": r["end"]}
            )

    texts: dict[str, str] = {}
    for folder in ("raw", "generated"):
        for path in sorted((ROOT / "data" / folder).glob("*.jsonl")):
            try:
                for d in read_jsonl(path):
                    texts[d.id] = d.text
            except Exception:
                continue

    skipped_licence, missing_text, written, spans = 0, 0, 0, 0
    with (out / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, entry in sorted(wanted.items()):
            if any(u in entry["source"] for u in UNLICENSED) and not args.allow_unlicensed:
                skipped_licence += 1
                continue
            text = texts.get(doc_id)
            if text is None:
                missing_text += 1
                continue
            entry["spans"].sort(key=lambda s: s["i"])
            row = {"doc_id": doc_id, "source": entry["source"], "spans": entry["spans"]}
            row["text"] = "" if args.no_text else text
            fh.write(json.dumps(row) + "\n")
            written += 1
            spans += len(entry["spans"])

    (out / "palimpsest_binoculars.py").write_text(
        derelativise(SCORER.read_text(encoding="utf-8")), encoding="utf-8"
    )

    (out / "README.txt").write_text(
        "Palimpsest -> Colab bundle\n"
        "==========================\n\n"
        f"documents.jsonl            {written:,} documents, {spans:,} sentence spans\n"
        "palimpsest_binoculars.py   the repository's scorer, de-relativised\n\n"
        "1. Open notebooks/binoculars_colab.ipynb in Google Colab.\n"
        "2. Runtime -> Change runtime type -> T4 GPU.\n"
        "3. Upload BOTH files from this folder when the notebook asks.\n"
        "4. Run all cells. Download binoculars_scores.jsonl at the end.\n"
        "5. Back here:  python scripts/join_binoculars.py "
        "--scores ~/Downloads/binoculars_scores.jsonl\n\n"
        "This bundle contains real student essay text. Uploading it sends that text to\n"
        "Google. PERSUADE is CC BY-NC-SA 4.0 (non-commercial). DAIGT is excluded because\n"
        "this repository does not redistribute it.\n",
        encoding="utf-8",
    )

    print(f"bundle -> {out}")
    print(f"  documents      {written:,}")
    print(f"  sentence spans {spans:,}")
    if skipped_licence:
        print(f"  excluded (not redistributable): {skipped_licence:,} documents")
    if missing_text:
        print(f"  ! {missing_text:,} documents had no source text and were dropped")
    if args.no_text:
        print("  NOTE: --no-text, so this bundle cannot score anything.")
    size = (out / "documents.jsonl").stat().st_size / 1e6
    print(f"  documents.jsonl  {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

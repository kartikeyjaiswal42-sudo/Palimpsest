#!/usr/bin/env python
"""Merge Colab's Binoculars scores back into the feature matrices.

    python scripts/join_binoculars.py --scores ~/Downloads/binoculars_scores.jsonl
    python scripts/join_binoculars.py --scores ... --sets train_remote esl_remote

The mirror of ``build_syntax_features.py``: a column computed elsewhere is joined onto rows
that already exist, by ``(doc_id, sentence_index)``, so the 43 existing values that produced
every published number are untouched and any measured difference is attributable to the new
column alone.

The alignment is CHECKED, and that is the whole point of this file
------------------------------------------------------------------
Scores computed in one process and joined in another are the ideal conditions for a silent
off-by-one: every row still gets a number, every number is plausible, and the detector is
quietly describing the sentence next door. PROJECT.md §2 records the version of this that
already happened here -- a token stream misattributed to the wrong sentence produces
"entirely plausible" features.

So four things must hold before anything is written, and any failure refuses the whole set
rather than writing a partly-correct matrix:

1. every scored document that appears in the set must be present in the scores file;
2. its sentence indices must match exactly -- same count, same values;
3. the span the notebook scored must be the span the row describes;
4. the ratio must reconstruct: ``logppl / xppl == score`` to floating-point tolerance.

(4) is cheap and catches an entire class of corruption -- a truncated download, a partial
checkpoint, a resumed run that interleaved two model pairs -- that the other three cannot
see, because those only check that numbers arrived, not that they mean anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "data" / "features"

BINOCULARS_FEATURES = ("binoculars_score", "binoculars_logppl", "binoculars_xppl")

#: Tolerance on score == logppl / xppl. The notebook rounds all three to 6 decimals, so the
#: reconstruction cannot be exact; this is loose enough for that and far tighter than any
#: real corruption.
_RATIO_TOL = 1e-4


def load_scores(path: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["doc_id"]] = {s["i"]: s for s in r["sentences"]}
    return out


def check_ratio(scores: dict[str, dict[int, dict]]) -> list[str]:
    """Verify score == logppl / xppl wherever all three are present."""
    bad = []
    for doc_id, sents in scores.items():
        for i, s in sents.items():
            b, lp, xp = (s.get("binoculars_score"), s.get("binoculars_logppl"),
                         s.get("binoculars_xppl"))
            if b is None or lp is None or xp is None or not xp:
                continue
            if not math.isclose(b, lp / xp, rel_tol=_RATIO_TOL, abs_tol=_RATIO_TOL):
                bad.append(f"{doc_id}[{i}]: {b} != {lp}/{xp} = {lp / xp:.6f}")
                if len(bad) >= 10:
                    return bad
    return bad


def augment(name: str, scores, spans_by_doc, dry_run: bool) -> bool:
    path = FEATURE_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"{name}: no such file, skipped")
        return False

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        print(f"{name}: empty, skipped")
        return False
    if any(k in rows[0]["features"] for k in BINOCULARS_FEATURES):
        print(f"{name}: already carries the Binoculars column, skipped")
        return False

    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)

    missing_docs, index_mismatch, span_mismatch = [], [], []
    for doc_id, idx in by_doc.items():
        sents = scores.get(doc_id)
        if sents is None:
            missing_docs.append(doc_id)
            continue
        want = {rows[i]["sentence_index"] for i in idx}
        got = set(sents)
        if want != got:
            # Say WHICH indices differ, not just how many there are. An off-by-one shift
            # leaves the counts equal, so a count-only message reads "26 vs 26" and sends
            # the reader looking for a missing sentence that does not exist.
            only_matrix = sorted(want - got)[:4]
            only_scores = sorted(got - want)[:4]
            if len(want) != len(got):
                detail = f"matrix has {len(want)} sentences, scores have {len(got)}"
            else:
                detail = (f"same count ({len(want)}) but different indices — "
                          f"missing from scores {only_matrix}, "
                          f"unexpected in scores {only_scores}; this is an off-by-one")
            index_mismatch.append(f"{doc_id}: {detail}")
            continue
        exported = spans_by_doc.get(doc_id, {})
        for i in idx:
            si = rows[i]["sentence_index"]
            sp = exported.get(si)
            if sp and (sp[0] != rows[i]["start"] or sp[1] != rows[i]["end"]):
                span_mismatch.append(
                    f"{doc_id}[{si}]: scored {sp}, matrix says "
                    f"({rows[i]['start']}, {rows[i]['end']})"
                )

    problems = []
    if missing_docs:
        problems.append(f"{len(missing_docs)} documents absent from the scores file "
                        f"(e.g. {missing_docs[:3]})")
    if index_mismatch:
        problems.append(f"{len(index_mismatch)} documents have mismatched sentence indices "
                        f"(e.g. {index_mismatch[:2]})")
    if span_mismatch:
        problems.append(f"{len(span_mismatch)} spans do not match "
                        f"(e.g. {span_mismatch[:2]})")
    if problems:
        print(f"{name}: REFUSED —")
        for p in problems:
            print(f"    {p}")
        print("    A partly-joined matrix silently describes the wrong sentences. "
              "Re-export and re-run rather than forcing this.")
        return False

    filled = 0
    for doc_id, idx in by_doc.items():
        sents = scores[doc_id]
        for i in idx:
            s = sents[rows[i]["sentence_index"]]
            for k in BINOCULARS_FEATURES:
                rows[i]["features"][k] = s.get(k)  # None -> NaN downstream, never 0.0
            filled += 1

    measured = sum(1 for r in rows if r["features"].get("binoculars_score") is not None)
    if dry_run:
        print(f"{name}: would write {filled:,} rows "
              f"({measured:,} measured, {filled - measured:,} unmeasurable)")
        return True

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"{name}: {filled:,} rows joined "
          f"({measured:,} measured, {filled - measured:,} unmeasurable) -> "
          f"{path.relative_to(ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="binoculars_scores.jsonl from Colab")
    ap.add_argument("--bundle", default=str(ROOT / "colab_bundle" / "documents.jsonl"),
                    help="the bundle that was uploaded, for the span cross-check")
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    score_path = Path(args.scores).expanduser()
    if not score_path.exists():
        raise SystemExit(f"no such file: {score_path}")

    scores = load_scores(score_path)
    n_spans = sum(len(v) for v in scores.values())
    print(f"scores: {len(scores):,} documents, {n_spans:,} spans")

    bad = check_ratio(scores)
    if bad:
        raise SystemExit(
            "REFUSED: the Binoculars ratio does not reconstruct from its own components.\n"
            + "\n".join(f"  {b}" for b in bad)
            + "\nThis means the file is corrupt or mixes two runs. Re-download it."
        )
    print("ratio check: score == logppl / xppl everywhere ✓")

    spans_by_doc: dict[str, dict[int, tuple]] = {}
    bundle = Path(args.bundle).expanduser()
    if bundle.exists():
        for line in bundle.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                spans_by_doc[d["doc_id"]] = {s["i"]: (s["start"], s["end"])
                                             for s in d["spans"]}
        print(f"bundle: {len(spans_by_doc):,} documents available for the span cross-check")
    else:
        print(f"bundle: {bundle} not found — span cross-check SKIPPED "
              "(indices are still checked)")

    names = args.sets or sorted(p.stem for p in FEATURE_DIR.glob("*.jsonl"))
    print()
    wrote = sum(augment(n, scores, spans_by_doc, args.dry_run) for n in names)
    print(f"\n{wrote} set(s) {'would be ' if args.dry_run else ''}updated")
    if wrote and not args.dry_run:
        # NOT syntax_probe.py: that script selects its columns from FEATURE_NAMES +
        # ALL_SYNTAX_FEATURE_NAMES and never reads binoculars_score, so it would print
        # a real number about a different question.
        print("next: python scripts/binoculars_probe.py   # measures THIS column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

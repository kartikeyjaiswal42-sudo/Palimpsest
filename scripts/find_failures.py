#!/usr/bin/env python
"""Find the documents the detector gets confidently wrong, and write them up.

    python scripts/find_failures.py

The brief asks for three essays the detector gets confidently wrong and a theory about
why. This finds them mechanically -- highest-scoring human essays and lowest-scoring
machine essays across every held-out set -- so the selection cannot be quietly flattering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402
from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402
from palimpsest.features.registry import FEATURES_BY_NAME  # noqa: E402

SETS = ("esl", "domain_shift", "unseen_prompting", "adversarial", "localisation")


def load(name):
    path = ROOT / "data" / "features" / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            r["features"] = {
                k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()
            }
            r["_set"] = name
            rows.append(r)
    return rows


def main() -> int:
    det = SentenceDetector.load(ROOT / "artifacts" / "detector.json")
    doc_model = DocumentDetector.load(ROOT / "artifacts" / "document_detector.json")

    texts: dict[str, str] = {}
    for f in (ROOT / "data" / "raw").glob("*.jsonl"):
        for d in read_jsonl(f):
            texts[d.id] = d.text
    for f in (ROOT / "data" / "generated").glob("*.jsonl"):
        for d in read_jsonl(f):
            texts[d.id] = d.text

    records = []
    for name in SETS:
        rows = load(name)
        if not rows:
            continue
        p = det.predict_many([r["features"] for r in rows])
        by_doc: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_doc.setdefault(r["doc_id"], []).append(i)
        for doc_id, ii in by_doc.items():
            w = np.array([rows[i]["features"].get("n_words", 1.0) for i in ii])
            w = np.where(np.isfinite(w), w, 1.0)
            stats = document_statistics(p[ii], w, det.flag_threshold)
            label = int(any(rows[i]["label"] for i in ii))
            records.append({
                "doc_id": doc_id, "set": name, "source": rows[ii[0]]["source_id"],
                "label": label, "docP": float(doc_model.predict(stats)),
                "share": stats["share"], "nSentences": len(ii),
                "topSentences": _top_sentences(rows, ii, p, det, 2),
            })

    fp = sorted([r for r in records if r["label"] == 0], key=lambda r: -r["docP"])[:5]
    fn = sorted([r for r in records if r["label"] == 1], key=lambda r: r["docP"])[:5]

    print("=" * 78)
    print("CONFIDENT FALSE POSITIVES -- human essays the detector calls machine-written")
    print("=" * 78)
    for r in fp:
        _show(r, texts)

    print("\n" + "=" * 78)
    print("CONFIDENT MISSES -- machine text the detector calls human")
    print("=" * 78)
    for r in fn:
        _show(r, texts)

    out = ROOT / "artifacts" / "failures.json"
    out.write_text(json.dumps({"falsePositives": fp, "misses": fn}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def _top_sentences(rows, ii, p, det, k):
    order = sorted(ii, key=lambda i: -p[i])[:k]
    out = []
    for i in order:
        pred = det.predict(rows[i]["features"])
        out.append({
            "p": round(float(p[i]), 4),
            "text": None,  # filled from the source text by the caller when available
            "start": rows[i]["start"], "end": rows[i]["end"],
            "label": rows[i]["label"],
            "evidence": [
                {"label": c.label, "contribution": round(c.contribution, 3),
                 "value": None if not c.measured else round(c.value, 3)}
                for c in pred.top(4)
            ],
        })
    return out


def _show(r, texts):
    text = texts.get(r["doc_id"], "")
    print(f"\n{r['doc_id']}  [{r['set']}/{r['source']}]")
    print(f"  P(machine)={r['docP']:.3f}  share={r['share']:.2f}  "
          f"{r['nSentences']} sentences  true label={'machine' if r['label'] else 'human'}")
    if text:
        print(f"  opening: {text[:150]!r}")
    for s in r["topSentences"]:
        snippet = text[s["start"]:s["end"]][:110] if text else ""
        s["text"] = snippet
        print(f"    [{s['p']:.3f}] {snippet}")
        print("      " + " | ".join(
            f"{e['label']} {e['contribution']:+.2f}" for e in s["evidence"]))


if __name__ == "__main__":
    raise SystemExit(main())

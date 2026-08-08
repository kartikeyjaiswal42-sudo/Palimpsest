#!/usr/bin/env python
"""Evaluate the fitted detector on every held-out set and write the honest report.

    python scripts/evaluate.py

Nothing here is fitted. Every set was excluded from training, and each one probes a
different way the detector can be wrong:

  unseen_prompting  the same generator, prompted to evade detection
  domain_shift      human writing from another domain -- false positives off-domain
  esl               human writing by English-language learners -- the documented failure
                    mode of every detector in this space
  localisation      part-human/part-machine documents -- can we find the seam?
  adversarial       prose deliberately composed to imitate a model
  ablation          identical content, one version rewritten by a model

Results land in ``artifacts/evaluation.json`` and drive docs/03-evaluation.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402

FEATURES_DIR = ROOT / "data" / "features"
DOC_T = 0.5  # replaced at runtime by the fitted operating point


def load(name: str) -> list[dict]:
    path = FEATURES_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            r["features"] = {
                k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()
            }
            rows.append(r)
    return rows


def score(rows: list[dict], det: SentenceDetector) -> np.ndarray:
    return det.predict_many([r["features"] for r in rows]) if rows else np.zeros(0)


def by_document(rows, probs, det, doc_model):
    """Aggregate to documents, returning (doc_ids, doc_probs, doc_labels, machine_shares)."""
    idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["doc_id"], []).append(i)
    ids, dp, dl, shares = [], [], [], []
    for doc_id, ii in idx.items():
        p = probs[ii]
        w = np.array([rows[i]["features"].get("n_words", 1.0) for i in ii])
        w = np.where(np.isfinite(w), w, 1.0)
        st = document_statistics(p, w, det.flag_threshold)
        ids.append(doc_id)
        dp.append(doc_model.predict(st))
        dl.append(int(any(rows[i]["label"] for i in ii)))
        shares.append(st["share"])
    return ids, np.array(dp), np.array(dl), np.array(shares)


def main() -> int:
    det = SentenceDetector.load(ROOT / "artifacts" / "detector.json")
    doc_model = DocumentDetector.load(ROOT / "artifacts" / "document_detector.json")
    global DOC_T
    DOC_T = doc_model.threshold
    t = det.flag_threshold
    report: dict = {"flagThreshold": round(t, 4), "sets": {}}

    print(f"detector threshold {t:.3f} | trained on {det.metadata.get('trainSources')}")
    print(f"out-of-fold sentence AUROC (from training) {det.metadata.get('oofSentenceAuroc')}")

    # ---------------------------------------------------------------- machine detection
    print("\n" + "=" * 74)
    print("DETECTION -- held-out machine text")
    print("=" * 74)
    for name in ("unseen_prompting", "adversarial"):
        rows = load(name)
        if not rows:
            continue
        p = score(rows, det)
        machine = np.array([r["label"] for r in rows]) == 1
        _, dp, dl, shares = by_document(rows, p, det, doc_model)
        caught = float((dp >= DOC_T).mean())
        print(f"\n{name}: {len(rows)} sentences, {len(dp)} documents")
        print(f"  sentence flag rate on machine sentences: {float((p[machine] >= t).mean()):.3f}")
        print(f"  documents called machine (P>={DOC_T:.2f}):      {caught:.3f}")
        print(f"  median machine-share reported:           {float(np.median(shares)):.3f}")
        report["sets"][name] = {
            "nSentences": len(rows), "nDocuments": int(len(dp)),
            "sentenceRecall": round(float((p[machine] >= t).mean()), 4),
            "documentRecall": round(caught, 4),
            "medianShare": round(float(np.median(shares)), 4),
        }
        if name == "adversarial":
            _adversarial_breakdown(rows, p, t, report)

    # ---------------------------------------------------------------- false positives
    print("\n" + "=" * 74)
    print("FALSE POSITIVES -- held-out human text")
    print("=" * 74)
    for name in ("domain_shift", "esl"):
        rows = load(name)
        if not rows:
            continue
        # The document operating point was calibrated on the even-indexed half of these
        # documents. Report on the odd half only, which it has never seen.
        human_all = [r for r in rows if r["label"] == 0]
        by_doc: dict[str, list[dict]] = {}
        for r in human_all:
            by_doc.setdefault(r["doc_id"], []).append(r)
        held = {d for i, d in enumerate(sorted(by_doc)) if i % 2}
        human = [r for r in human_all if r["doc_id"] in held]
        p = score(human, det)
        _, dp, _, shares = by_document(human, p, det, doc_model)
        print(f"\n{name}: {len(human)} human sentences, {len(dp)} documents")
        print(f"  sentence false-positive rate: {float((p >= t).mean()):.3f}")
        print(f"  document false-positive rate: {float((dp >= DOC_T).mean()):.3f}")
        report["sets"][name] = {
            "nSentences": len(human), "nDocuments": int(len(dp)),
            "sentenceFPR": round(float((p >= t).mean()), 4),
            "documentFPR": round(float((dp >= DOC_T).mean()), 4),
        }
        if name == "esl":
            _esl_breakdown(human, p, dp, det, doc_model, report)

    # ---------------------------------------------------------------- localisation
    # Mixed documents are split by base essay; the half used for training is excluded here
    # so nothing is scored on a document it was fitted on.
    rows = load("localisation")
    split_path = ROOT / "artifacts" / "mixed_split.json"
    if rows and split_path.exists():
        train_groups = set(json.loads(split_path.read_text())["trainGroups"])
        before = len({r["doc_id"] for r in rows})
        rows = [r for r in rows if r["group"] not in train_groups]
        print(f"\n(localisation: {before - len({r['doc_id'] for r in rows})} documents "
              f"excluded as training data; {len({r['doc_id'] for r in rows})} held out)")
    if rows:
        print("\n" + "=" * 74)
        print("LOCALISATION -- documents that are part human, part machine")
        print("=" * 74)
        p = score(rows, det)
        y = np.array([r["label"] for r in rows])
        auroc = roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")
        flagged = p >= t
        print(f"  {len(rows)} sentences in {len({r['doc_id'] for r in rows})} documents, "
              f"{int(y.sum())} machine ({100*y.mean():.1f}%)")
        print(f"  sentence AUROC within mixed documents: {auroc:.3f}")
        print(f"  precision {float((y[flagged] == 1).mean()) if flagged.any() else float('nan'):.3f} "
              f"| recall {float(flagged[y == 1].mean()):.3f}")
        report["sets"]["localisation"] = {
            "nSentences": len(rows), "sentenceAuroc": round(float(auroc), 4),
            "precision": round(float((y[flagged] == 1).mean()) if flagged.any() else 0.0, 4),
            "recall": round(float(flagged[y == 1].mean()), 4),
        }
        _seam_accuracy(rows, p, t, report)
        for pair in ("hewlett", "toefl"):
            sel = [i for i, r in enumerate(rows) if (r["doc_meta"] or {}).get("pair") == pair]
            if len(sel) < 30:
                continue
            ys, ps = y[sel], p[sel]
            if len(set(ys.tolist())) < 2:
                continue
            fl = ps >= t
            print(f"    {pair:<9} AUROC {roc_auc_score(ys, ps):.3f} | "
                  f"recall {float(fl[ys == 1].mean()):.3f} | "
                  f"precision {float((ys[fl] == 1).mean()) if fl.any() else float('nan'):.3f}")
            report["sets"]["localisation"][pair] = {
                "auroc": round(float(roc_auc_score(ys, ps)), 4),
                "recall": round(float(fl[ys == 1].mean()), 4)}

    # ---------------------------------------------------------------- ablation
    _ablation(det, doc_model, report)

    out = ROOT / "artifacts" / "evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n\nwrote {out.relative_to(ROOT)}")
    return 0


def _adversarial_breakdown(rows, p, t, report):
    styles = {}
    for r, pi in zip(rows, p, strict=True):
        if r["label"] != 1:
            continue
        s = (r["doc_meta"] or {}).get("style", "?")
        styles.setdefault(s, []).append(pi)
    print("  by prompt style (machine sentences only):")
    out = {}
    for s, v in sorted(styles.items()):
        arr = np.array(v)
        print(f"    {s:<10} n={len(arr):<4} flagged {float((arr >= t).mean()):.3f} "
              f"| median p {float(np.median(arr)):.3f}")
        out[s] = {"n": len(arr), "flagRate": round(float((arr >= t).mean()), 4),
                  "medianP": round(float(np.median(arr)), 4)}
    report["sets"]["adversarial"]["byStyle"] = out


def _esl_breakdown(rows, p, dp, det, doc_model, report):
    """The measurement this project exists to get right."""
    t = det.flag_threshold
    print("\n  BY SOURCE:")
    out = {}
    for src in sorted({r["source_id"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["source_id"] == src]
        sub = [rows[i] for i in sel]
        ps = p[sel]
        _, sdp, _, _ = by_document(sub, ps, det, doc_model)
        nw = np.array([r["features"].get("n_words", np.nan) for r in sub])
        print(f"    {src:<12} {len(sel):>5} sentences  sentFPR {float((ps >= t).mean()):.3f}  "
              f"docFPR {float((sdp >= DOC_T).mean()):.3f}  mean sentence {np.nanmean(nw):.1f}w")
        out[src] = {"nSentences": len(sel),
                    "sentenceFPR": round(float((ps >= t).mean()), 4),
                    "documentFPR": round(float((sdp >= DOC_T).mean()), 4)}

    # PERSUADE carries a matched ELL flag: same prompts, same graders, same cohort.
    per = [(i, r) for i, r in enumerate(rows) if r["source_id"] == "persuade"]
    if per:
        ell = [i for i, r in per if (r["doc_meta"] or {}).get("ell")]
        non = [i for i, r in per if not (r["doc_meta"] or {}).get("ell")]
        if ell and non:
            print(f"\n  MATCHED CONTROL (PERSUADE: same prompts, ELL flag is the only difference)")
            for nm, sel in (("ELL", ell), ("non-ELL", non)):
                sub = [rows[i] for i in sel]
                _, sdp, _, _ = by_document(sub, p[sel], det, doc_model)
                print(f"    {nm:<9} {len(sel):>5} sentences  sentFPR {float((p[sel] >= t).mean()):.3f}"
                      f"  docFPR {float((sdp >= DOC_T).mean()):.3f}")
                out[f"persuade_{nm}"] = {
                    "nSentences": len(sel),
                    "sentenceFPR": round(float((p[sel] >= t).mean()), 4),
                    "documentFPR": round(float((sdp >= DOC_T).mean()), 4)}

    # ELLIPSE grades proficiency 1-5, so we can ask whether WEAKER English is punished more.
    ell_rows = [(i, r) for i, r in enumerate(rows) if r["source_id"] == "ellipse"]
    if ell_rows:
        print(f"\n  BY MEASURED ENGLISH PROFICIENCY (ELLIPSE holistic score):")
        bands = {}
        for i, r in ell_rows:
            prof = (r["doc_meta"] or {}).get("proficiency")
            if prof is None or prof != prof:
                continue
            bands.setdefault(round(float(prof) * 2) / 2, []).append(i)
        for band in sorted(bands):
            sel = bands[band]
            if len(sel) < 30:
                continue
            print(f"    proficiency {band:<4} n={len(sel):>5}  sentFPR {float((p[sel] >= t).mean()):.3f}")
            out[f"ellipse_prof_{band}"] = {
                "nSentences": len(sel),
                "sentenceFPR": round(float((p[sel] >= t).mean()), 4)}
    report["sets"]["esl"]["breakdown"] = out


def _seam_accuracy(rows, p, t, report):
    """In a document with one switch point, how close is our boundary to the real one?"""
    docs: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        docs.setdefault(r["doc_id"], []).append(i)
    errors, found = [], 0
    for _, ii in docs.items():
        ii = sorted(ii, key=lambda i: rows[i]["sentence_index"])
        y = np.array([rows[i]["label"] for i in ii])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        true_seam = int(np.argmax(y))          # first machine sentence
        pred = p[ii] >= t
        if not pred.any():
            continue
        found += 1
        errors.append(abs(int(np.argmax(pred)) - true_seam))
    if errors:
        e = np.array(errors)
        print(f"  seam located in {found}/{len(docs)} mixed documents; "
              f"median offset {int(np.median(e))} sentences, "
              f"within 2 sentences in {float((e <= 2).mean()):.0%} of them")
        report["sets"]["localisation"]["seam"] = {
            "documentsWithSeamFound": found, "totalDocuments": len(docs),
            "medianOffsetSentences": int(np.median(e)),
            "withinTwoSentences": round(float((e <= 2).mean()), 4)}


def _ablation(det, doc_model, report):
    """Same content, one version rewritten by a model. Isolates what we respond to."""
    rows = load("ablation")
    base = load("esl") + load("domain_shift")
    if not rows:
        return
    print("\n" + "=" * 74)
    print("ABLATION -- identical content, machine-rewritten surface")
    print("=" * 74)
    pairs = [("liang_toefl", "liang_toefl_gpt4polished", "TOEFL essays, GPT-4 'polished'"),
             ("liang_hewlett_human", "liang_hewlett_gptsimplify", "ASAP essays, GPT-simplified")]
    out = {}
    for human_src, machine_src, label in pairs:
        h = [r for r in base if r["source_id"] == human_src]
        m = [r for r in rows if r["source_id"] == machine_src]
        if not h or not m:
            continue
        ph, pm = score(h, det), score(m, det)
        _, dph, _, _ = by_document(h, ph, det, doc_model)
        _, dpm, _, _ = by_document(m, pm, det, doc_model)
        print(f"\n  {label}")
        print(f"    original   : {len(dph):>4} docs, {float((dph >= DOC_T).mean()):.3f} called machine")
        print(f"    rewritten  : {len(dpm):>4} docs, {float((dpm >= DOC_T).mean()):.3f} called machine")
        out[label] = {"originalFlagged": round(float((dph >= DOC_T).mean()), 4),
                      "rewrittenFlagged": round(float((dpm >= DOC_T).mean()), 4)}
    report["sets"]["ablation"] = out


if __name__ == "__main__":
    raise SystemExit(main())

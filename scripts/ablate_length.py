#!/usr/bin/env python
"""Re-run the document-length ablation on the CURRENT build.

    python scripts/ablate_length.py

docs/04-failures.md claims that giving the document model `log_sentences` produces a large
false-positive rate on short essays by non-native speakers. That claim was first measured on
an earlier build, and a number measured on an earlier build is a number that quietly goes
wrong. This script reproduces the experiment against whatever is in data/features today, so
the figure in the documentation is reproducible rather than remembered.

It changes nothing on disk except artifacts/ablation_length.json. The shipped model is not
touched.

The comparison is like-for-like: both arms get out-of-fold sentence probabilities from the
same folds, the same document model class, and the same threshold rule (lowest cut-off whose
false-positive rate on the *calibration* half of the at-risk population is inside the
budget). Rates are reported on the other half, which neither arm's threshold has seen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import palimpsest.detect.document as docmod  # noqa: E402
from palimpsest.detect.classifier import SentenceDetector, choose_threshold  # noqa: E402
from train import _mixed_training_rows, load_rows  # noqa: E402

BUDGET = 0.05
FOLDS = 5
C = 0.1


def _doc_stats(rows, probs, threshold):
    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)
    stats, labels = [], []
    for idx in by_doc.values():
        w = np.array([rows[i]["features"].get("n_words", 1.0) for i in idx])
        w = np.where(np.isfinite(w), w, 1.0)
        stats.append(docmod.document_statistics(probs[idx], w, threshold))
        labels.append(int(any(rows[i]["label"] for i in idx)))
    return stats, np.array(labels)


def _at_risk_docs(detector, threshold):
    """Score every held-out human document once. Returns (stats, set_name, parity)."""
    out = []
    for name in ("esl", "domain_shift"):
        path = ROOT / "data" / "features" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in load_rows(path) if r["label"] == 0]
        by_doc: dict[str, list[dict]] = {}
        for r in rows:
            by_doc.setdefault(r["doc_id"], []).append(r)
        for i, (doc_id, rs) in enumerate(sorted(by_doc.items())):
            p = detector.predict_many([r["features"] for r in rs])
            w = np.array([r["features"].get("n_words", 1.0) for r in rs])
            w = np.where(np.isfinite(w), w, 1.0)
            out.append({
                "stats": docmod.document_statistics(p, w, threshold),
                "set": name,
                "subset": rs[0].get("source_id") or name,
                # train.py calibrates on the even indices; evaluate.py reports on the odd.
                "calibration": i % 2 == 0,
            })
    return out


def run_arm(features: tuple[str, ...], rows, y, groups, x) -> dict:
    """Fit the whole document stack with `features` as the document model's inputs."""
    original = docmod.DOC_FEATURES
    docmod.DOC_FEATURES = features
    try:
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=FOLDS).split(x, y, groups):
            det = SentenceDetector().fit([x[i] for i in tr], y[tr], groups[tr], c=C)
            oof[te] = det.predict_many([x[i] for i in te])
        sent_threshold, _, _ = choose_threshold(y, oof, target_precision=0.80)

        stats, labels = _doc_stats(rows, oof, sent_threshold)
        model = docmod.DocumentDetector().fit(stats, labels)
        in_domain_auroc = roc_auc_score(labels, [model.predict(s) for s in stats])

        detector = SentenceDetector().fit(x, y, groups, c=C)
        detector.flag_threshold = float(sent_threshold)
        docs = _at_risk_docs(detector, sent_threshold)
        for d in docs:
            d["p"] = model.predict(d["stats"])

        cal = np.array([d["p"] for d in docs if d["calibration"]])
        cut = 1.0
        for c in np.unique(np.round(np.append(cal, 1.0), 4)):
            if float((cal >= c).mean()) <= BUDGET:
                cut = float(c)
                break

        report = [d for d in docs if not d["calibration"]]
        counts: dict[str, tuple[int, int]] = {}

        def fpr(key: str, pred) -> float:
            sel = [d for d in report if pred(d)]
            if not sel:
                return float("nan")
            hits = int(sum(d["p"] >= cut for d in sel))
            counts[key] = (hits, len(sel))
            return hits / len(sel)

        return {
            "documentFeatures": list(features),
            "weights": dict(zip(features, [round(float(w), 3) for w in model.coef], strict=True)),
            "sentenceThreshold": round(float(sent_threshold), 4),
            "documentThreshold": round(cut, 4),
            "inDomainDocumentAuroc": round(float(in_domain_auroc), 4),
            "toeflDocumentFPR": round(fpr("toefl", lambda d: d["subset"] == "liang_toefl"), 4),
            "eslDocumentFPR": round(fpr("esl", lambda d: d["set"] == "esl"), 4),
            "domainShiftDocumentFPR": round(
                fpr("domain_shift", lambda d: d["set"] == "domain_shift"), 4),
            "nReportDocuments": len(report),
            "counts": {k: {"flagged": h, "of": n} for k, (h, n) in counts.items()},
            # Kept so the two arms can be compared document-by-document, which is what the
            # paired test needs -- the same essays are scored twice, so an unpaired
            # comparison of two rates would overstate the evidence.
            "_perDocument": {
                f"{d['subset']}::{i}": bool(d["p"] >= cut) for i, d in enumerate(report)
            },
        }
    finally:
        docmod.DOC_FEATURES = original


def main() -> int:
    rows = load_rows(ROOT / "data" / "features" / "train.jsonl")
    rows += _mixed_training_rows(0.5)
    x = [r["features"] for r in rows]
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    shipped = ("mean_p", "max_p", "q90_p", "share")
    with_length = shipped + ("log_sentences",)

    print(f"{len(rows)} sentences | {len(set(groups))} essays\n")
    results = {}
    for label, feats in (("with log_sentences", with_length), ("shipped (without)", shipped)):
        print(f"fitting arm: {label} ...")
        results[label] = run_arm(feats, rows, y, groups, x)

    a, b = results["with log_sentences"], results["shipped (without)"]
    print(f"\n{'':<34}{'with length':>14}{'without':>12}")
    for key, name in (
        ("toeflDocumentFPR", "TOEFL document FPR"),
        ("eslDocumentFPR", "ESL overall document FPR"),
        ("domainShiftDocumentFPR", "domain-shift document FPR"),
        ("inDomainDocumentAuroc", "in-domain document AUROC"),
    ):
        ck = key.replace("DocumentFPR", "").replace("domainShift", "domain_shift") or key
        ca, cb = a["counts"].get(ck), b["counts"].get(ck)
        detail = f"   {ca['flagged']}/{ca['of']} vs {cb['flagged']}/{cb['of']}" if ca and cb else ""
        print(f"{name:<34}{a[key]:>14.4f}{b[key]:>12.4f}{detail}")
    # Paired test. The same TOEFL essays are scored by both arms, so the question is not
    # "are 15/45 and 11/45 different rates" but "how many essays changed verdict, and in
    # which direction". McNemar's exact test is the right one and it is far less flattering.
    keys = [k for k in a["_perDocument"] if k.startswith("liang_toefl::")]
    b01 = sum(1 for k in keys if a["_perDocument"][k] and not b["_perDocument"][k])
    b10 = sum(1 for k in keys if b["_perDocument"][k] and not a["_perDocument"][k])
    from math import comb
    n_disc = b01 + b10
    p_exact = (
        min(1.0, 2 * sum(comb(n_disc, i) for i in range(min(b01, b10) + 1)) / 2**n_disc)
        if n_disc else 1.0
    )
    print(f"\nTOEFL, paired (McNemar exact): {b01} essays fixed by removing the feature, "
          f"{b10} broken; p = {p_exact:.3f}")
    if p_exact > 0.05:
        print("  -> NOT significant. The TOEFL improvement is within noise; the case for "
              "removing the feature is principled, not empirical.")

    print(f"\nlog_sentences weight when included: "
          f"{a['weights'].get('log_sentences'):+.3f}")
    print(f"reporting half: {b['nReportDocuments']} held-out human documents "
          f"(the calibration half is never reported on)")

    out = ROOT / "artifacts" / "ablation_length.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

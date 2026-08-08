#!/usr/bin/env python
"""Fit the sentence detector and the document model, and report cross-validated performance.

    python scripts/train.py

Every number printed here is out-of-fold. Folds are split by GROUP, never by sentence:
sentences from one essay resemble each other far more than they resemble sentences from
another essay, so a random sentence split puts near-duplicates on both sides and reports an
accuracy the detector does not have.

Training data is real GPT-3.5 output against real human admissions essays. An earlier
version trained against machine-style prose we composed ourselves; that failed, informatively,
and the story is in docs/04-failures.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.classifier import SentenceDetector, choose_threshold  # noqa: E402
from palimpsest.detect.document import DocumentDetector, document_statistics  # noqa: E402
from palimpsest.features.registry import FEATURES_BY_NAME  # noqa: E402


def load_rows(path: Path) -> list[dict]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(ROOT / "data" / "features" / "train.jsonl"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--target-precision", type=float, default=0.80)
    ap.add_argument("--fpr-budget", type=float, default=0.05,
                    help="max false-positive rate on human documents at the operating point")
    ap.add_argument("--mixed-fraction", type=float, default=0.5,
                    help="share of real mixed documents added to training (0 disables)")
    args = ap.parse_args()

    rows = load_rows(Path(args.features))
    rows += _mixed_training_rows(args.mixed_fraction)
    x = [r["features"] for r in rows]
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    print(f"{len(rows)} sentences | {len(set(groups))} essays | "
          f"{int(y.sum())} machine ({100 * y.mean():.1f}%) / {int((1 - y).sum())} human")
    print("by source:", dict(Counter(r["source_id"] for r in rows)))
    _length_check(rows, y)

    # ---------------------------------------------------------------- out-of-fold
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(
        GroupKFold(n_splits=args.folds).split(x, y, groups), 1
    ):
        det = SentenceDetector().fit([x[i] for i in tr], y[tr], groups[tr], c=args.c)
        oof[te] = det.predict_many([x[i] for i in te])
        print(f"  fold {fold}: {len(tr)} train / {len(te)} test sentences, "
              f"{len(set(groups[te]))} held-out essays, "
              f"{int(y[te].sum())} machine")

    auroc = roc_auc_score(y, oof)
    ap_score = average_precision_score(y, oof)
    print(f"\nSENTENCE LEVEL (out-of-fold)")
    print(f"  AUROC {auroc:.3f} | average precision {ap_score:.3f} (baseline {y.mean():.3f})"
          f" | Brier {brier_score_loss(y, oof):.4f}")

    threshold, prec, rec = choose_threshold(y, oof, target_precision=args.target_precision)
    print(f"  threshold {threshold:.3f} -> precision {prec:.3f}, recall {rec:.3f} "
          f"(target precision {args.target_precision})")
    _threshold_table(y, oof, threshold)
    _calibration_table(y, oof)

    # ---------------------------------------------------------------- document level
    # Persist the sentence model first: the document operating point is calibrated by
    # scoring at-risk human essays with it.
    _interim = SentenceDetector().fit(x, y, groups, c=args.c)
    _interim.flag_threshold = float(threshold)
    _interim.save(ROOT / "artifacts" / "detector.json")

    stats, labels, doc_ids = _document_stats(rows, oof, threshold)
    doc_model = DocumentDetector().fit(stats, labels)
    doc_p = np.array([doc_model.predict(s) for s in stats])
    doc_t = _document_threshold_on_at_risk(doc_model, budget=args.fpr_budget)
    doc_model.threshold = float(doc_t)
    print(f"\nDOCUMENT LEVEL ({len(labels)} essays, {int(sum(labels))} containing machine text)")
    print(f"  AUROC {roc_auc_score(labels, doc_p):.3f} | Brier {brier_score_loss(labels, doc_p):.4f}")
    from palimpsest.detect.document import DOC_FEATURES
    _fp = float((doc_p[labels == 0] >= doc_t).mean())
    _tp = float((doc_p[labels == 1] >= doc_t).mean())
    print(f"  operating point P>={doc_t:.3f}: false-positive rate {_fp:.3f} "
          f"(budget {args.fpr_budget}), recall {_tp:.3f}")
    print("  document-model weights: " + ", ".join(
        f"{n}={w:+.2f}" for n, w in zip(DOC_FEATURES, doc_model.coef, strict=True)))

    # ---------------------------------------------------------------- final artifacts
    final = SentenceDetector().fit(x, y, groups, c=args.c)
    final.flag_threshold = float(threshold)
    final.metadata.update({
        "oofSentenceAuroc": round(float(auroc), 4),
        "oofAveragePrecision": round(float(ap_score), 4),
        "flagPrecision": round(float(prec), 4),
        "flagRecall": round(float(rec), 4),
        "trainSources": sorted({r["source_id"] for r in rows}),
    })
    final.save(ROOT / "artifacts" / "detector.json")
    doc_model.metadata = {"nDocuments": len(labels), "sentenceThreshold": float(threshold),
                          "documentThreshold": float(doc_t), "fprBudget": args.fpr_budget,
                          "fprAtOperatingPoint": round(_fp, 4), "recallAtOperatingPoint": round(_tp, 4),
                          "auroc": round(float(roc_auc_score(labels, doc_p)), 4)}
    doc_model.save(ROOT / "artifacts" / "document_detector.json")
    print(f"\nsaved artifacts/detector.json + artifacts/document_detector.json")

    _weight_report(final)
    return 0


def _document_threshold_on_at_risk(doc_model, budget: float) -> float:
    """Choose the document cut-off on the human population most at risk of false accusation.

    Calibrating this on our in-domain training essays gave a 5% false-positive rate there
    and 26-52% on essays by English-language learners -- the operating point simply did not
    transfer. So it is set on ESL and out-of-domain human writing instead, using HALF of
    that data; scripts/evaluate.py reports the rate on the other half, which the threshold
    never saw.

    Choosing the operating point on the group the tool can hurt is the whole point. It costs
    recall, and docs/03-evaluation.md states how much.
    """
    from palimpsest.detect.document import document_statistics

    det = SentenceDetector.load(ROOT / "artifacts" / "detector.json")
    probs: list[float] = []
    for name in ("esl", "domain_shift"):
        path = ROOT / "data" / "features" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in load_rows(path) if r["label"] == 0]
        by_doc: dict[str, list[dict]] = {}
        for r in rows:
            by_doc.setdefault(r["doc_id"], []).append(r)
        # Deterministic half for calibration; evaluate.py takes the complement.
        for i, (doc_id, rs) in enumerate(sorted(by_doc.items())):
            if i % 2:
                continue
            p = det.predict_many([r["features"] for r in rs])
            w = np.array([r["features"].get("n_words", 1.0) for r in rs])
            w = np.where(np.isfinite(w), w, 1.0)
            probs.append(doc_model.predict(document_statistics(p, w, det.flag_threshold)))
    if not probs:
        return 0.5
    arr = np.array(probs)
    for cut in np.unique(np.round(np.append(arr, 1.0), 4)):
        if float((arr >= cut).mean()) <= budget:
            print(f"  operating point calibrated on {len(arr)} at-risk human documents")
            return float(cut)
    return 1.0


def _mixed_training_rows(fraction: float) -> list[dict]:
    """Add part-human/part-machine documents to the training pool.

    Without these, every training document is entirely one class, so the in-document
    context features -- which measure how far a sentence sits from the rest of ITS OWN
    essay -- have nothing to detect. They were designed for the mixed case and never saw
    one, so the fit learned to ignore them, and localisation inside a real mixed essay was
    correspondingly poor. See docs/03-evaluation.md for the before/after.

    The split is by base document and deterministic, and scripts/evaluate.py reads the same
    split file so a document used for training is never also used to report a score.
    """
    path = ROOT / "data" / "features" / "localisation.jsonl"
    if fraction <= 0 or not path.exists():
        return []
    rows = load_rows(path)
    groups = sorted({r["group"] for r in rows})
    n_train = int(len(groups) * fraction)
    # Deterministic, and interleaved rather than a prefix so both source pairs are present.
    train_groups = set(groups[::2][: max(n_train, 1)])
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "mixed_split.json").write_text(
        json.dumps({"trainGroups": sorted(train_groups)}, indent=2), encoding="utf-8")
    picked = [r for r in rows if r["group"] in train_groups]
    print(f"mixed documents added to training: {len(picked)} sentences from "
          f"{len(train_groups)} of {len(groups)} documents "
          f"({sum(r['label'] for r in picked)} machine)")
    return picked


def _length_check(rows: list[dict], y: np.ndarray) -> None:
    """Guard against the classifier being able to win by measuring sentence length."""
    nw = np.array([r["features"].get("n_words", np.nan) for r in rows])
    k = np.isfinite(nw)
    h, m = nw[k & (y == 0)], nw[k & (y == 1)]
    auc = roc_auc_score(y[k], nw[k])
    print(f"length check: human {h.mean():.1f}w vs machine {m.mean():.1f}w | "
          f"AUROC of length alone {auc:.3f}"
          f"{'  <-- WARNING: length is doing the work' if abs(auc - 0.5) > 0.15 else ''}")


def _threshold_table(y: np.ndarray, p: np.ndarray, chosen: float) -> None:
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>8} {'flagged':>8}")
    cands = sorted({round(v, 2) for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, round(chosen, 2))})
    for t in cands:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        mark = "  <- chosen" if abs(t - round(chosen, 2)) < 1e-9 else ""
        print(f"  {t:>10.2f} {prec:>10.3f} {tp / max(int(y.sum()),1):>8.3f} "
              f"{int(pred.sum()):>8}{mark}")


def _calibration_table(y: np.ndarray, p: np.ndarray) -> None:
    """Reliability: in each score band, how often is the sentence actually machine-written?"""
    print(f"  calibration  {'band':>12} {'n':>6} {'predicted':>10} {'actual':>8}")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        print(f"  {'':>12} {f'{lo:.1f}-{hi:.1f}':>12} {int(m.sum()):>6} "
              f"{p[m].mean():>10.3f} {y[m].mean():>8.3f}")


def _document_stats(rows, oof, threshold):
    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)
    stats, labels, ids = [], [], []
    for doc_id, idx in by_doc.items():
        p = oof[idx]
        w = np.array([rows[i]["features"].get("n_words", 1.0) for i in idx])
        w = np.where(np.isfinite(w), w, 1.0)
        stats.append(document_statistics(p, w, threshold))
        labels.append(int(any(rows[i]["label"] for i in idx)))
        ids.append(doc_id)
    return stats, np.array(labels), ids


def _weight_report(det: SentenceDetector) -> None:
    order = np.argsort(-np.abs(det.coef))
    agree = disagree = 0
    print(f"\n{'feature':<26}{'group':<12}{'weight':>9}  expected   verdict")
    for i in order[:16]:
        name = det.feature_names[i]
        f = FEATURES_BY_NAME[name]
        w = det.coef[i]
        if f.expected_direction == 0:
            verdict = "(no prior)"
        elif np.sign(w) == f.expected_direction:
            verdict, agree = "as predicted", agree + 1
        else:
            verdict, disagree = "OPPOSITE", disagree + 1
        exp = {1: "machine", -1: "human", 0: "-"}[f.expected_direction]
        print(f"{name:<26}{f.group:<12}{w:>+9.3f}  {exp:<9} {verdict}")
    for i in order[16:]:
        f = FEATURES_BY_NAME[det.feature_names[i]]
        if f.expected_direction == 0:
            continue
        if np.sign(det.coef[i]) == f.expected_direction:
            agree += 1
        else:
            disagree += 1
    print(f"\nsigns across all features with a prior: {agree} as predicted, {disagree} opposite")
    print("NOTE: individual signs are unreliable when features are correlated -- the fit")
    print("splits a shared signal across collinear columns. Group contributions and the")
    print("single-feature AUROCs in docs/03-evaluation.md are the trustworthy readings.")


if __name__ == "__main__":
    raise SystemExit(main())

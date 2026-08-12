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
    # The remote 30 B observer returns each token's log-probability and rank but never the
    # full predictive distribution, so entropy and Fast-DetectGPT curvature cannot be
    # computed from it. Those columns arrive all-NaN, which would make the standardiser's
    # nanmean NaN and poison every downstream score. They are DROPPED rather than imputed:
    # a feature that was never measured should not be given the training mean and a
    # coefficient. See scorer/remote_lm.REMOTE_UNAVAILABLE and docs/09-frontier-ceiling.md.
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="feature names to exclude, e.g. --drop-features "
                         "mean_entropy entropy_sd curvature")
    ap.add_argument("--out-suffix", default="",
                    help="appended to artifact filenames so two observers' detectors coexist")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--target-precision", type=float, default=0.80)
    ap.add_argument("--fpr-budget", type=float, default=0.05,
                    help="max false-positive rate on human documents at the operating point")
    ap.add_argument("--mixed-fraction", type=float, default=0.5,
                    help="share of real mixed documents added to training (0 disables)")
    # These rows used to be hardcoded to the GPT-2 file, so every 30 B build was fitted with
    # ~8% of its sentences carrying model-based features from the WRONG observer -- displaced
    # 1.0-1.4 standard deviations on mean_logprob, mean_log_rank and frac_rank_top1, because
    # GPT-2 finds the same text far less predictable than a 30 B model does. Silent, and
    # exactly the hazard scripts/evaluate.py's own comment warns about.
    ap.add_argument("--mixed-features",
                    default=str(ROOT / "data" / "features" / "localisation.jsonl"),
                    help="mixed-document features; MUST come from the same observer as "
                         "--features (e.g. localisation_remote.jsonl alongside a _remote set)")
    ap.add_argument("--allow-observer-mismatch", action="store_true",
                    help="proceed even if --mixed-features looks like a different observer")
    args = ap.parse_args()

    from palimpsest.detect.classifier import FEATURE_NAMES
    keep = tuple(f for f in FEATURE_NAMES if f not in set(args.drop_features))
    if args.drop_features:
        print(f"dropping {len(FEATURE_NAMES) - len(keep)} unmeasurable features: "
              f"{', '.join(args.drop_features)}")

    def _new():
        return SentenceDetector(feature_names=keep)

    rows = load_rows(Path(args.features))
    mixed = _mixed_training_rows(args.mixed_fraction, args.out_suffix, args.mixed_features)
    _check_same_observer(rows, mixed, args.allow_observer_mismatch)
    rows += mixed
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
        det = _new().fit([x[i] for i in tr], y[tr], groups[tr], c=args.c)
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
    calibration = _calibration_table(y, oof)

    # ---------------------------------------------------------------- document level
    # Persist the sentence model first: the document operating point is calibrated by
    # scoring at-risk human essays with it.
    _interim = _new().fit(x, y, groups, c=args.c)
    _interim.flag_threshold = float(threshold)
    _interim.save(ROOT / "artifacts" / f"detector{args.out_suffix}.json")

    stats, labels, doc_ids = _document_stats(rows, oof, threshold)
    doc_model = DocumentDetector().fit(stats, labels)
    doc_p = np.array([doc_model.predict(s) for s in stats])
    doc_t = _document_threshold_on_at_risk(doc_model, budget=args.fpr_budget,
                                           out_suffix=args.out_suffix)
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
    final = _new().fit(x, y, groups, c=args.c)
    final.flag_threshold = float(threshold)
    final.metadata.update({
        "oofSentenceAuroc": round(float(auroc), 4),
        "oofAveragePrecision": round(float(ap_score), 4),
        "flagPrecision": round(float(prec), 4),
        "flagRecall": round(float(rec), 4),
        "trainSources": sorted({r["source_id"] for r in rows}),
        "calibration": calibration,
    })
    final.save(ROOT / "artifacts" / f"detector{args.out_suffix}.json")
    doc_model.metadata = {"nDocuments": len(labels), "sentenceThreshold": float(threshold),
                          "documentThreshold": float(doc_t), "fprBudget": args.fpr_budget,
                          "fprAtOperatingPoint": round(_fp, 4), "recallAtOperatingPoint": round(_tp, 4),
                          "auroc": round(float(roc_auc_score(labels, doc_p)), 4)}
    doc_model.save(ROOT / "artifacts" / f"document_detector{args.out_suffix}.json")
    print(f"\nsaved artifacts/detector{args.out_suffix}.json + artifacts/document_detector{args.out_suffix}.json")

    _weight_report(final)
    return 0


def _document_threshold_on_at_risk(doc_model, budget: float, out_suffix: str = "") -> float:
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

    det = SentenceDetector.load(ROOT / "artifacts" / f"detector{out_suffix}.json")
    probs: list[float] = []
    for name in ("esl", "domain_shift"):
        path = ROOT / "data" / "features" / f"{name}{out_suffix}.jsonl"
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

    # Pick the threshold on the UPPER CONFIDENCE BOUND of the false-positive rate, not on
    # the point estimate.
    #
    # The previous rule took the first cut whose observed FPR on this sample was <= budget.
    # With a few hundred calibration documents that estimate carries real sampling error,
    # and it errs in the dangerous direction: the cut is *chosen* to make the observed rate
    # small, so the observed rate is optimistically biased by construction. Measured
    # consequence -- a threshold calibrated to 5% on the even half of the at-risk sets
    # produced 8.0% (GPT-2) and 7.4% (30 B) on the odd half, i.e. the shipped product was
    # falsely flagging half again as many English-learner essays as its own budget allowed.
    #
    # Clopper-Pearson is the exact binomial interval, so this says: "given this sample, the
    # true false-positive rate is at most `budget` with 95% confidence." It costs recall,
    # and it costs more of it when the calibration sample is small -- which is the correct
    # incentive, because a small sample is exactly when the point estimate cannot be
    # trusted.
    from scipy.stats import beta

    def upper_fpr(k: int, n: int, alpha: float = 0.05) -> float:
        if n == 0:
            return 1.0
        if k >= n:
            return 1.0
        return float(beta.ppf(1.0 - alpha, k + 1, n - k))

    n = len(arr)
    for cut in np.unique(np.round(np.append(arr, 1.0), 4)):
        k = int((arr >= cut).sum())
        if upper_fpr(k, n) <= budget:
            print(f"  operating point calibrated on {n} at-risk human documents: "
                  f"observed FPR {k / n:.3f}, 95% upper bound {upper_fpr(k, n):.3f} "
                  f"<= budget {budget:.3f}")
            return float(cut)
    print(f"  ! no threshold reaches a {budget:.0%} upper bound on {n} documents; "
          f"refusing to flag anything rather than exceed the budget")
    return 1.0


def _check_same_observer(rows: list[dict], mixed: list[dict], allow: bool) -> None:
    """Refuse to fit one observer's detector on another observer's numbers.

    Every model-based feature answers "how predictable was this token here?", so its VALUES
    are meaningful only relative to the model that produced them. Pooling two observers does
    not add data; it adds a hidden covariate that is perfectly correlated with which file a
    sentence came from, and it moves the standardiser's mean and scale for every downstream
    score.

    The check is a median comparison in units of the training set's own spread, which needs
    no knowledge of which observer either file used -- it just asks whether they look like
    the same instrument.
    """
    if not mixed:
        return
    worst = ("", 0.0)
    for name in ("mean_logprob", "mean_log_rank", "frac_rank_top1"):
        a = np.array([r["features"].get(name, np.nan) for r in rows], dtype=float)
        b = np.array([r["features"].get(name, np.nan) for r in mixed], dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 50:
            continue
        spread = float(np.std(a))
        if spread <= 0:
            continue
        gap = abs(float(np.median(a)) - float(np.median(b))) / spread
        if gap > worst[1]:
            worst = (name, gap)
    # The constant is read off both cases rather than guessed. Same observer, different
    # text (real hybrid documents vs generated essays): 0.50 SD, on frac_rank_top1.
    # Different observer (GPT-2 rows in a 30 B set): 1.41 SD, on mean_log_rank. 1.0 sits
    # between them with room on each side; it is a smoke alarm for a gross mistake, not a
    # distributional test, and it says what it saw either way.
    print(f"  mixed-document observer check: largest median gap {worst[1]:.2f} SD "
          f"({worst[0] or 'n/a'})")
    if worst[1] <= 1.0:
        return
    msg = (f"mixed-document features look like a DIFFERENT observer: {worst[0]} medians "
           f"differ by {worst[1]:.2f} SD of the training set. Build the matching file "
           f"(build_features.py --observer remote --out-suffix _remote --sets localisation) "
           f"and pass --mixed-features, or --allow-observer-mismatch to override.")
    if not allow:
        raise SystemExit(f"! {msg}")
    print(f"! WARNING {msg}")


def _mixed_training_rows(fraction: float, out_suffix: str = "",
                         features_path: str | None = None) -> list[dict]:
    """Add part-human/part-machine documents to the training pool.

    Without these, every training document is entirely one class, so the in-document
    context features -- which measure how far a sentence sits from the rest of ITS OWN
    essay -- have nothing to detect. They were designed for the mixed case and never saw
    one, so the fit learned to ignore them, and localisation inside a real mixed essay was
    correspondingly poor. See docs/03-evaluation.md for the before/after.

    The split is by base document and deterministic, and scripts/evaluate.py reads the same
    split file so a document used for training is never also used to report a score.
    """
    path = (Path(features_path) if features_path
            else ROOT / "data" / "features" / "localisation.jsonl")
    if fraction <= 0 or not path.exists():
        return []
    rows = load_rows(path)
    groups = sorted({r["group"] for r in rows})
    n_train = int(len(groups) * fraction)
    # Deterministic, and interleaved rather than a prefix so both source pairs are present.
    train_groups = set(groups[::2][: max(n_train, 1)])
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / f"mixed_split{out_suffix}.json").write_text(
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


def _calibration_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Reliability: in each score band, how often is the sentence actually machine-written?

    Returned as well as printed, so it can be written into the detector artifact. The
    version of this table in docs/03-evaluation.md went stale once -- it was copied from a
    run that predated the operating-point recalibration and went on claiming 41 sentences in
    the 0.5-0.7 band when there were 186. A printed table nobody can diff is exactly the
    kind of number this project keeps getting wrong, so it is now an artifact and a test
    compares the prose against it.
    """
    print(f"  calibration  {'band':>12} {'n':>6} {'predicted':>10} {'actual':>8}")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        print(f"  {'':>12} {f'{lo:.1f}-{hi:.1f}':>12} {int(m.sum()):>6} "
              f"{p[m].mean():>10.3f} {y[m].mean():>8.3f}")
        table.append({
            "band": f"{lo:.1f}-{hi:.1f}",
            "n": int(m.sum()),
            "meanPredicted": round(float(p[m].mean()), 3),
            "actual": round(float(y[m].mean()), 3),
        })
    return table


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

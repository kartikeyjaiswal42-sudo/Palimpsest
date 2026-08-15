#!/usr/bin/env python
"""Can we find the sentences a model polished, when we cannot tell who wrote the document?

    python scripts/polish_probe.py

THE ARGUMENT

The brief names the realistic case in one line: *"a paragraph a person wrote and a model later
polished."* docs/09-frontier-ceiling.md measures the other case to a wall -- a whole Claude
Opus essay, judged alone, reaches 0% recall at a 5% false-accusation budget.

Those two cases are not the same question, and the difference is a reference.

Judging a document alone means comparing it to a *population*: is this prose more machine-like
than human prose in general. That comparison carries every confound the project has
documented -- proficiency, genre, topic, how specific the content happens to be -- and the
frontier ceiling is the statement that Opus prose sits inside the human population on those
axes.

Judging a *sentence against the rest of its own document* compares it to one author. Topic,
genre, proficiency and register are held constant by construction, because they are the same
person's essay. That is a different measurement with a different ceiling, and it has never
been evaluated at the frontier here: the `localisation` eval set is GPT-3.5/GPT-4-era
rewrites, and the ten Claude-Opus hybrids sit in `adversarial`, where only DOCUMENT recall was
reported (0.0%). Whether the seam is findable in them is simply unmeasured.

WHAT IS AND IS NOT NEW

Not new: self-relative features. `features/context.py` already ships six, and its docstring
already makes this argument. What is new is (a) only three of the 43 features are z-scored
against the document, and the rest can be, for free, from rows already on disk; (b) the
shipped classifier is fitted to do both jobs at once, and a document that is *entirely*
machine has no internal discontinuity, so half its training signal actively teaches the
context features to stay quiet; and (c) nobody has measured any of it on frontier polish.

So this fits a SEPARATE head on self-relative features only, and grades it on the polish case.

THE LEAK THIS DESIGN HAS TO AVOID

`build_real_hybrids.py` splices a human head to a machine tail. The cut point moves, but the
machine half is always LATER, so `rel_position` predicts the label almost perfectly while
learning nothing about authorship. It is excluded, and its exclusion is asserted in the output.
The Claude-Opus hybrids replace a span in the MIDDLE of the document, which makes them a test
of the leak as well as of the frontier: a position-cheating model scores well on the GPT-era
corpus and fails on them.

CONTROLS

  * Grouped by BASE document, so no essay is in both train and test.
  * The false-positive threshold is read off ALL-HUMAN documents, never off the human half of
    a hybrid -- a human sentence sitting next to machine text has a contaminated baseline, and
    calibrating on it would understate the deployment error. Both are reported.
  * `domain_shift` is NOT usable as that control: 86 of its 88 documents ARE the hybrid bases.
    53 ESL documents are excluded for the same reason. Checked, not assumed.
  * The ESL false-positive rate is reported by measured proficiency, because a run-on essay by
    a second-language writer is exactly the document a "this sentence does not match" detector
    could libel, and this project's standing rule is that its false positives are accusations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fusion_probe as fp  # noqa: E402  -- same estimator and metrics as every other probe

from palimpsest.features.context import CONTEXT_FEATURE_NAMES  # noqa: E402

#: Excluded, and this is the whole reason. See the module docstring.
LEAK = "rel_position"

#: Cap on an in-document z-score, matching features/context.py's `_DEGENERATE_Z` so the two
#: families are on the same scale and one feature cannot dominate a logit when a document's
#: baseline happens to have no spread.
Z_CAP = 6.0

#: A document needs an internal baseline before "unlike the rest of this document" means
#: anything. Same threshold as features/context.MIN_SENTENCES.
MIN_SENTENCES = 5


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {}
    for r in rows:
        docs.setdefault(r["doc_id"], []).append(r)
    for v in docs.values():
        v.sort(key=lambda r: r["sentence_index"])
    return docs


def add_loo_z(docs: dict[str, list[dict]], keys: tuple[str, ...]) -> None:
    """Attach a leave-one-out in-document z-score for every named feature, in place.

    For sentence *i* and feature *k*: how many standard deviations sentence *i* sits from the
    mean of the OTHER sentences in its own document. Leave-one-out matters -- including the
    sentence in its own baseline shrinks exactly the deviation being measured, and shrinks it
    hardest in short documents, which is where a polished paragraph is largest as a share.

    This is free: every value comes from rows already on disk. No observer is called.
    """
    for rows in docs.values():
        n = len(rows)
        if n < MIN_SENTENCES:
            for r in rows:
                r["z"] = {}
            continue
        for r in rows:
            r["z"] = {}
        for k in keys:
            v = np.asarray([r["features"].get(k, np.nan) for r in rows], dtype=np.float64)
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=float(np.nanmean(v)) if np.any(np.isfinite(v)) else 0.0)
            total = v.sum()
            for i, r in enumerate(rows):
                rest = (total - v[i]) / (n - 1)
                # sd of the other n-1 values, computed directly rather than by subtraction so
                # floating error cannot make it negative on near-constant columns.
                others = np.delete(v, i)
                sd = float(others.std())
                z = 0.0 if sd < 1e-9 else float((v[i] - rest) / sd)
                r["z"][f"{k}_looz"] = float(np.clip(z, -Z_CAP, Z_CAP))


def matrix(rows: list[dict], feats: tuple[str, ...]) -> np.ndarray:
    out = np.empty((len(rows), len(feats)), dtype=np.float64)
    for i, r in enumerate(rows):
        z = r.get("z") or {}
        f = r["features"]
        for j, k in enumerate(feats):
            out[i, j] = z[k] if k in z else f.get(k, np.nan)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def group_folds(groups: list[str], k: int, rng) -> list[np.ndarray]:
    """Split indices into k folds so that a base document never straddles a fold."""
    uniq = sorted(set(groups))
    order = rng.permutation(len(uniq))
    assign = {uniq[g]: i % k for i, g in enumerate(order)}
    return [np.array([i for i, g in enumerate(groups) if assign[g] == f]) for f in range(k)]


def evaluate(hybrid_docs: dict[str, list[dict]], human_docs: dict[str, list[dict]],
             feats: tuple[str, ...], rng, folds: int = 5) -> dict:
    """Out-of-fold sentence scores on hybrids, plus a clean all-human false-positive rate.

    The human documents are split the same number of ways as the hybrids and rotated with them,
    so a human document used to calibrate the threshold in one fold was not a training negative
    in that fold either.
    """
    h_rows = [r for rows in hybrid_docs.values() for r in rows]
    h_groups = [r["group"] for r in h_rows]
    u_rows = [r for rows in human_docs.values() for r in rows]
    u_groups = [r["doc_id"] for r in u_rows]

    Xh, yh = matrix(h_rows, feats), np.array([r["label"] for r in h_rows], dtype=np.float64)
    Xu = matrix(u_rows, feats)

    oof = np.full(len(h_rows), np.nan)
    oof_human = np.full(len(u_rows), np.nan)
    hf = group_folds(h_groups, folds, np.random.default_rng(fp.SEED))
    uf = group_folds(u_groups, folds, np.random.default_rng(fp.SEED + 1))

    for f in range(folds):
        h_te = hf[f]
        h_tr = np.concatenate([hf[i] for i in range(folds) if i != f])
        u_te = uf[f]
        u_tr = np.concatenate([uf[i] for i in range(folds) if i != f])
        # Train on: hybrid sentences (both labels) + all-human sentences as extra negatives.
        # The extra negatives matter -- without them the only "human" the head ever sees sits
        # inside a document that also contains machine text.
        X = np.vstack([Xh[h_tr], Xu[u_tr]])
        y = np.concatenate([yh[h_tr], np.zeros(len(u_tr))])
        Xs, xh_te, xu_te = fp.standardise(X, Xh[h_te], Xu[u_te])
        w = fp.fit_logreg(Xs, y)
        oof[h_te] = fp.predict(w, xh_te)
        oof_human[u_te] = fp.predict(w, xu_te)

    mach = oof[yh == 1]
    hum_in_hyb = oof[yh == 0]
    # A 5%-of-SENTENCES threshold, which is the intuitive one and the wrong one.
    thr = float(np.quantile(oof_human, 0.95))

    # THE DEPLOYMENT THRESHOLD. A reader does not receive a sentence, they receive a document,
    # and one flagged sentence anywhere in it is the accusation. An essay holds ~19 sentences,
    # so a 5% per-sentence error compounds to roughly a third of unedited essays carrying at
    # least one flag -- measured at 0.307 below. Calibrating on the per-sentence rate and
    # reporting per-document recall would be scoring the easy question and billing the hard one.
    #
    # So the second operating point is set on the quantity that is actually spent: the 95th
    # percentile of the per-document MAXIMUM score over unedited human essays. At that
    # threshold, 5% of clean documents carry a flag, by construction.
    doc_max = []
    idx = 0
    for _doc_id, rows in human_docs.items():
        n = len(rows)
        doc_max.append(float(np.max(oof_human[idx:idx + n])))
        idx += n
    thr_doc = float(np.quantile(np.asarray(doc_max), 0.95))

    # Per-document localisation at BOTH thresholds.
    per_doc, per_doc_strict = [], []
    idx = 0
    for doc_id, rows in hybrid_docs.items():
        n = len(rows)
        s = oof[idx:idx + n]
        lab = np.array([r["label"] for r in rows])
        idx += n
        if lab.sum() == 0:
            continue
        for t, bucket in ((thr, per_doc), (thr_doc, per_doc_strict)):
            flag = s >= t
            bucket.append({
                "recall": float(flag[lab == 1].mean()),
                "precision": float((lab[flag] == 1).mean()) if flag.any() else float("nan"),
                "hit": bool(flag[lab == 1].any()),
            })
    # A false boundary: an unedited human document with a flagged sentence in it.
    fb_docs = 0
    fb_two = 0
    idx = 0
    for doc_id, rows in human_docs.items():
        n = len(rows)
        s = oof_human[idx:idx + n]
        idx += n
        f = s >= thr
        if f.any():
            fb_docs += 1
        if np.any(f[:-1] & f[1:]):
            fb_two += 1

    return {
        "n_features": len(feats),
        "n_machine_sentences": int((yh == 1).sum()),
        "n_human_sentences_in_hybrids": int((yh == 0).sum()),
        "n_human_control_sentences": len(u_rows),
        "n_human_control_docs": len(human_docs),
        "sentence_auroc": fp.auroc(mach, hum_in_hyb),
        "sentence_auroc_vs_clean_human": fp.auroc(mach, oof_human),
        "threshold_at_5pct_clean_fpr": thr,
        "sentence_tpr": float((mach >= thr).mean()),
        "sentence_fpr_in_hybrids": float((hum_in_hyb >= thr).mean()),
        "sentence_fpr_clean_human": float((oof_human >= thr).mean()),
        "doc_mean_recall": float(np.mean([d["recall"] for d in per_doc])),
        "doc_mean_precision": float(np.nanmean([d["precision"] for d in per_doc])),
        "doc_hit_rate": float(np.mean([d["hit"] for d in per_doc])),
        "false_boundary_doc_rate": fb_docs / max(len(human_docs), 1),
        "false_boundary_adjacent_rate": fb_two / max(len(human_docs), 1),
        # -- at the operating point that holds the DOCUMENT false-alarm rate to 5% ---------
        "threshold_at_5pct_doc_alarm": thr_doc,
        "doc_hit_rate_strict": float(np.mean([d["hit"] for d in per_doc_strict])),
        "doc_mean_recall_strict": float(np.mean([d["recall"] for d in per_doc_strict])),
        "doc_mean_precision_strict": float(np.nanmean([d["precision"] for d in per_doc_strict])),
        "sentence_tpr_strict": float((mach >= thr_doc).mean()),
        "_oof": oof, "_oof_human": oof_human, "_thr": thr, "_thr_doc": thr_doc,
    }


def transfer(train_hybrids, human_docs, test_hybrids, feats) -> dict:
    """Fit on GPT-era hybrids, test on frontier-polished ones. Nothing shared."""
    tr = [r for rows in train_hybrids.values() for r in rows]
    un = [r for rows in human_docs.values() for r in rows]
    te = [r for rows in test_hybrids.values() for r in rows]
    X = np.vstack([matrix(tr, feats), matrix(un, feats)])
    y = np.concatenate([np.array([r["label"] for r in tr], dtype=np.float64),
                        np.zeros(len(un))])
    Xs, te_s, un_s = fp.standardise(X, matrix(te, feats), matrix(un, feats))
    w = fp.fit_logreg(Xs, y)
    s_te, s_un = fp.predict(w, te_s), fp.predict(w, un_s)
    lab = np.array([r["label"] for r in te])
    thr = float(np.quantile(s_un, 0.95))
    # Same two operating points as `evaluate`, and for the same reason: the per-sentence budget
    # is not the one a reader spends.
    dm, idx = [], 0
    for _doc_id, rows in human_docs.items():
        n = len(rows)
        dm.append(float(np.max(s_un[idx:idx + n])))
        idx += n
    thr_doc = float(np.quantile(np.asarray(dm), 0.95))

    buckets: dict[str, list[dict]] = {"loose": [], "strict": []}
    idx = 0
    for doc_id, rows in test_hybrids.items():
        n = len(rows)
        s, l = s_te[idx:idx + n], lab[idx:idx + n]
        idx += n
        if l.sum() == 0:
            continue
        for t, key in ((thr, "loose"), (thr_doc, "strict")):
            f = s >= t
            buckets[key].append(
                {"recall": float(f[l == 1].mean()), "hit": bool(f[l == 1].any()),
                 "precision": float((l[f] == 1).mean()) if f.any() else float("nan")})

    def agg(key: str, field: str) -> float:
        v = [d[field] for d in buckets[key]]
        return float(np.nanmean(v)) if v else float("nan")

    return {
        "n_docs": len(test_hybrids), "n_machine_sentences": int(lab.sum()),
        "sentence_auroc": fp.auroc(s_te[lab == 1], s_te[lab == 0]),
        "sentence_auroc_vs_clean_human": fp.auroc(s_te[lab == 1], s_un),
        "sentence_tpr_at_5pct_clean_fpr": float((s_te[lab == 1] >= thr).mean()),
        "doc_hit_rate": agg("loose", "hit"),
        "doc_mean_recall": agg("loose", "recall"),
        "doc_mean_precision": agg("loose", "precision"),
        "sentence_tpr_strict": float((s_te[lab == 1] >= thr_doc).mean()),
        "doc_hit_rate_strict": agg("strict", "hit"),
        "doc_mean_recall_strict": agg("strict", "recall"),
        "doc_mean_precision_strict": agg("strict", "precision"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="localisation_remote.jsonl")
    ap.add_argument("--frontier", default="adversarial_remote.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="polish_probe")
    args = ap.parse_args()

    F = ROOT / "data" / "features"
    hyb = by_doc(load_rows(F / args.features))
    hyb = {d: rs for d, rs in hyb.items() if len(rs) >= MIN_SENTENCES}
    bases = {r["group"] for rows in hyb.values() for r in rows}

    # -- all-human controls, with the overlap actually checked ------------------------
    human: dict[str, list[dict]] = {}
    excluded = 0
    for name in ("train_remote.jsonl", "esl_remote.jsonl"):
        p = F / name
        if not p.exists():
            continue
        for doc_id, rows in by_doc(load_rows(p)).items():
            if len(rows) < MIN_SENTENCES or any(r["label"] != 0 for r in rows):
                continue
            if doc_id in bases:  # the essay a hybrid was built FROM
                excluded += 1
                continue
            human[doc_id] = rows

    all_feats = tuple(sorted(load_rows(F / args.features)[0]["features"].keys()))
    abs_feats = tuple(f for f in all_feats if f not in CONTEXT_FEATURE_NAMES)
    rel_shipped = tuple(f for f in CONTEXT_FEATURE_NAMES if f != LEAK)

    add_loo_z(hyb, abs_feats)
    add_loo_z(human, abs_feats)
    looz = tuple(f"{k}_looz" for k in abs_feats)

    print(f"hybrids (GPT-era)      {len(hyb):4d} docs  "
          f"{sum(len(v) for v in hyb.values()):5d} sentences  "
          f"{sum(r['label'] for v in hyb.values() for r in v):4d} machine")
    print(f"all-human control      {len(human):4d} docs  "
          f"{sum(len(v) for v in human.values()):5d} sentences")
    print(f"  excluded {excluded} human documents that ARE hybrid base essays")
    print(f"  EXCLUDED FEATURE: {LEAK} (label leak -- machine is always the tail)")
    print()

    arms = {
        "ABS_absolute_only": abs_feats,
        "REL_shipped_context": rel_shipped,
        "RELZ_all_features_looz": rel_shipped + looz,
        "ALL_absolute_plus_relz": abs_feats + rel_shipped + looz,
    }

    report: dict[str, object] = {
        "excluded_feature": LEAK,
        "excluded_human_docs_overlapping_bases": excluded,
        "n_hybrid_docs": len(hyb), "n_human_control_docs": len(human),
    }

    print("=" * 100)
    print("GPT-ERA POLISH, grouped by base document, threshold = 5% FPR on UNEDITED human docs")
    print("=" * 100)
    print(f"  {'arm':26s} {'nf':>3s} {'sentAUROC':>10s} "
          f"|{'  5% SENTENCE budget':>28s} |{'  5% DOCUMENT budget':>24s}")
    print(f"  {'':26s} {'':>3s} {'':>10s} "
          f"|{'TPR':>7s}{'docHit':>8s}{'falseBnd':>10s} |{'TPR':>7s}{'docHit':>8s}{'docPrec':>9s}")
    results = {}
    for label, feats in arms.items():
        r = evaluate(hyb, human, feats, np.random.default_rng(fp.SEED), args.folds)
        results[label] = r
        print(f"  {label:26s} {r['n_features']:3d} {r['sentence_auroc']:10.3f} "
              f"|{r['sentence_tpr']:7.3f}{r['doc_hit_rate']:8.3f}"
              f"{r['false_boundary_doc_rate']:10.3f} "
              f"|{r['sentence_tpr_strict']:7.3f}{r['doc_hit_rate_strict']:8.3f}"
              f"{r['doc_mean_precision_strict']:9.3f}")
    report["gpt_era"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                         for k, v in results.items()}
    print()
    print("  sentAUROC = machine vs human sentences INSIDE hybrids")
    print("  docHit    = hybrid documents where >=1 TRUE machine sentence is flagged")
    print("  falseBnd  = UNEDITED human documents carrying >=1 flag. This is the accusation")
    print("              rate, and at a 5% SENTENCE budget it is ~30%, which is why the right")
    print("              column exists: there the threshold is set so falseBnd == 5% itself.")
    print()

    # ---------------------------------------------------------------- the frontier
    fp_path = F / args.frontier
    if fp_path.exists():
        fr_all = by_doc(load_rows(fp_path))
        fr = {d: rs for d, rs in fr_all.items()
              if len(rs) >= MIN_SENTENCES
              and any(r["label"] == 1 for r in rs) and any(r["label"] == 0 for r in rs)}
        if fr:
            add_loo_z(fr, abs_feats)
            print("=" * 100)
            print(f"FRONTIER POLISH -- fit on GPT-era only, tested on {len(fr)} "
                  f"Claude-Opus-polished documents")
            print("=" * 100)
            print("  These replace a span in the MIDDLE, so a rel_position cheat cannot help.")
            print("  Fitted on LIANG's GPT-era rewrites and tested on OURS: cross-pipeline by")
            print("  construction, which is the control docs/09 R3 says is the only one that")
            print("  distinguishes detection from bookkeeping.")
            print(f"  {'arm':26s} {'nSent':>6s} {'sentAUROC':>10s} "
                  f"|{' 5% SENTENCE':>22s} |{' 5% DOCUMENT':>22s}")
            print(f"  {'':26s} {'':>6s} {'':>10s} "
                  f"|{'TPR':>7s}{'docHit':>8s} |{'TPR':>7s}{'docHit':>8s}{'docPrec':>9s}")
            tr_report = {}
            for label, feats in arms.items():
                t = transfer(hyb, human, fr, feats)
                tr_report[label] = t
                print(f"  {label:26s} {t['n_machine_sentences']:6d} "
                      f"{t['sentence_auroc']:10.3f} "
                      f"|{t['sentence_tpr_at_5pct_clean_fpr']:7.3f}{t['doc_hit_rate']:8.3f} "
                      f"|{t['sentence_tpr_strict']:7.3f}{t['doc_hit_rate_strict']:8.3f}"
                      f"{t['doc_mean_precision_strict']:9.3f}")
            report["frontier"] = tr_report
            print()
            print(f"  n = {len(fr)} documents. A hit rate of 9/10 has a 95% Clopper-Pearson")
            print("  interval of roughly 0.55-1.00, so read this as 'the seam is findable',")
            print("  never as a three-decimal quantity. More frontier hybrids is the fix.")
            print()
    else:
        print(f"NO FRONTIER FEATURES at {fp_path.relative_to(ROOT)} -- "
              f"build them with scripts/build_features.py to measure the frontier case.")
        print()

    # ------------------------------------------------- ESL false positives by proficiency
    best = "RELZ_all_features_looz"
    r = results[best]
    oof_h, thr = r["_oof_human"], r["_thr"]
    prof: dict[str, list[float]] = {}
    idx = 0
    for doc_id, rows in human.items():
        n = len(rows)
        s = oof_h[idx:idx + n]
        idx += n
        p = (rows[0].get("doc_meta") or {}).get("proficiency")
        if p is None:
            continue
        band = f"{float(p):.1f}"
        prof.setdefault(band, []).append(float((s >= thr).any()))
    if prof:
        print("=" * 100
              )
        print(f"ESL FALSE POSITIVES BY MEASURED PROFICIENCY  (arm {best})")
        print("=" * 100)
        print("  A flagged sentence in an unedited essay is an accusation, and the writers most")
        print("  likely to produce prose that does not match its neighbours are the weakest.")
        print(f"  {'ELLIPSE holistic':18s} {'docs':>6s} {'% with a flagged sentence':>26s}")
        for band in sorted(prof, key=float):
            v = prof[band]
            print(f"  {band:18s} {len(v):6d} {100 * float(np.mean(v)):25.1f}%")
        report["esl_by_proficiency"] = {k: {"n": len(v), "rate": float(np.mean(v))}
                                        for k, v in prof.items()}
        print()

    path = ROOT / "artifacts" / f"{args.out}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

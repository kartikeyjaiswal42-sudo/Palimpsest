#!/usr/bin/env python
"""Fit and validate the genre gate.

    python scripts/fit_genre_gate.py --suffix _remote

The gate refuses documents that are not admissions personal statements, because the detector
fails confidently rather than gracefully outside that genre -- see detect/genre.py for the
two measurements that motivate it.

Three things are validated, and the second is the one that matters:

  1. IN-DOMAIN PASS RATE. Wrongly refusing a real admissions essay makes the product useless,
     so the threshold is calibrated to pass a stated fraction of known in-domain essays and
     that fraction is reported on documents the fit did not see.

  2. AUTHORSHIP BLINDNESS. A gate fitted only against human in-domain essays would learn
     authorship and become a silent second detector, refusing precisely the documents the
     real detector was about to flag -- laundering a low recall figure into a high abstention
     figure. Both classes are therefore in-domain during fitting, and the pass rates for
     human and machine admissions essays are compared here. If they diverge, the gate is
     reading authorship and must not ship.

  3. OUT-OF-DOMAIN REJECTION. What it actually catches, per genre.

The gate cannot weaken the detector: it never touches the sentence model, the document model
or their thresholds. It only decides whether they are consulted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.genre import (  # noqa: E402
    GENRE_FEATURES,
    GenreGate,
    document_genre_features,
)

#: Admissions personal statements. BOTH classes, deliberately -- see docstring point 2.
IN_DOMAIN = {
    "train": "mixed",            # liang_college_human + jhu + liang_college_gpt3 + modern_train
    "modern_holdout": "machine",
    "modern_claude": "machine",
}
#: Everything else the corpus contains: argumentative coursework, TOEFL responses, and
#: Hewlett-era school essays. None of these is what the detector was calibrated for.
OUT_OF_DOMAIN = ("esl", "domain_shift")


def load_docs(path: Path) -> dict[str, tuple[list[dict], int]]:
    """{doc_id: (sentence feature dicts, is_machine)}"""
    out: dict[str, tuple[list[dict], int]] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        feats, lab = out.setdefault(r["doc_id"], ([], 0))
        feats.append(r["features"])
        if r["label"]:
            out[r["doc_id"]] = (feats, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_remote")
    ap.add_argument("--false-refusal-budget", type=float, default=0.02,
                    help="max rate at which REAL admissions essays are refused, as a 95%% "
                         "upper bound")
    ap.add_argument("--drop-features", default="",
                    help="comma-separated gate features to exclude, for ablation")
    ap.add_argument("--add-features", default="",
                    help="comma-separated extra document features, for ablation only -- "
                         "used to rebuild retired variants (length, sentence rhythm) so "
                         "their cost can be re-measured rather than remembered")
    ap.add_argument("--out-suffix", default=None,
                    help="write to genre_gate<OUT_SUFFIX>.json instead of overwriting")
    args = ap.parse_args()

    drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
    add = tuple(f.strip() for f in args.add_features.split(",") if f.strip())
    features = tuple(f for f in GENRE_FEATURES if f not in drop) + add
    if drop or add:
        print(f"ablation: -{sorted(drop)} +{list(add)} -> {len(features)} features\n")

    feats_dir = ROOT / "data" / "features"
    rows: list[dict] = []
    labels: list[int] = []
    authorship: list[int] = []
    provenance: list[str] = []

    for name in IN_DOMAIN:
        for _doc, (sf, is_m) in load_docs(feats_dir / f"{name}{args.suffix}.jsonl").items():
            rows.append(document_genre_features(sf))
            labels.append(1)
            authorship.append(is_m)
            provenance.append(name)
    n_in = len(rows)
    for name in OUT_OF_DOMAIN:
        for _doc, (sf, is_m) in load_docs(feats_dir / f"{name}{args.suffix}.jsonl").items():
            rows.append(document_genre_features(sf))
            labels.append(0)
            authorship.append(is_m)
            provenance.append(name)

    y = np.array(labels)
    print(f"in-domain {n_in} documents ({sum(authorship[:n_in])} machine / "
          f"{n_in - sum(authorship[:n_in])} human)")
    print(f"out-of-domain {len(rows) - n_in} documents\n")
    if n_in == 0 or len(rows) == n_in:
        print("! need both classes; build features first")
        return 1

    # Half/half split so the pass rate is reported on documents the fit did not see.
    rng = np.random.default_rng(20260811)
    idx = rng.permutation(len(rows))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]

    gate = GenreGate(feature_names=features).fit([rows[i] for i in tr], y[tr])
    p_te = np.array([gate.probability(rows[i]) for i in te])
    y_te = y[te]

    # Threshold: bound the FALSE-REFUSAL rate, do not merely hit a pass quantile.
    #
    # Refusing to score somebody's essay is a claim ("this is not the kind of writing I was
    # built for") and it needs evidence like any other. The first version simply took the 5%
    # quantile of in-domain scores, which accepts a 5% false-refusal rate by construction and
    # never states it. A real browser found the consequence immediately: the application's own
    # showcase example -- a 260-word admissions essay -- was refused, scoring 0.3293 against a
    # 0.3440 threshold. Losing by 0.015 is not a genre judgement, it is noise.
    #
    # Clopper-Pearson, matching scripts/train.py and scripts/fit_bands.py: pick the highest
    # threshold whose 95% upper bound on refusing a genuine admissions essay stays inside
    # budget. It costs out-of-domain rejection, which is the correct direction to err --
    # a missed out-of-genre document falls through to the bands and is very likely told
    # "insufficient evidence" anyway, whereas a wrongly refused essay is a dead end for the
    # user with no recourse.
    from scipy.stats import beta

    def upper(k: int, n: int, alpha: float = 0.05) -> float:
        return 1.0 if n == 0 or k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))

    in_p = np.sort(p_te[y_te == 1])
    n_dom = len(in_p)
    gate.threshold = 0.0
    for cut in np.unique(np.round(np.concatenate([in_p, [0.0]]), 4))[::-1]:
        if upper(int((in_p < cut).sum()), n_dom) <= args.false_refusal_budget:
            gate.threshold = float(cut)
            break
    k_ref = int((in_p < gate.threshold).sum())
    print(f"false-refusal bound: {k_ref}/{n_dom} = {k_ref / max(n_dom,1):.3f} observed, "
          f"95% upper {upper(k_ref, n_dom):.3f} <= budget {args.false_refusal_budget}")
    gate.metadata = {"falseRefusalBudget": args.false_refusal_budget,
                     "observedFalseRefusal": k_ref / max(n_dom, 1),
                     "nInDomain": int(n_in), "nOutOfDomain": int(len(rows) - n_in),
                     "observer": "remote" if args.suffix == "_remote" else "gpt2"}

    passed = p_te >= gate.threshold
    print(f"threshold P(in-domain) >= {gate.threshold:.4f}\n")
    print(f"  in-domain admissions essays SCORED : {passed[y_te == 1].mean():.1%}  "
          f"(false-refusal budget {args.false_refusal_budget:.0%})")
    print(f"  out-of-domain essays REFUSED       : {(~passed[y_te == 0]).mean():.1%}\n")

    # -- the validation that decides whether this may ship -------------------------
    auth_te = np.array(authorship)[te]
    hum = passed[(y_te == 1) & (auth_te == 0)]
    mac = passed[(y_te == 1) & (auth_te == 1)]
    gap = abs(float(hum.mean()) - float(mac.mean())) if len(hum) and len(mac) else float("nan")
    print("AUTHORSHIP BLINDNESS (in-domain only) -- the gate must not become a 2nd detector")
    print(f"  human admissions essays passed  : {hum.mean():.1%}  (n={len(hum)})")
    print(f"  machine admissions essays passed: {mac.mean():.1%}  (n={len(mac)})")
    print(f"  gap {gap:.1%}  -> {'OK' if gap <= 0.10 else 'FAIL: gate is reading authorship'}")

    prov_te = np.array(provenance)[te]
    print("\nBY SOURCE (refusal rate):")
    for name in sorted(set(provenance)):
        m = prov_te == name
        if m.sum():
            print(f"  {name:20s} n={int(m.sum()):4d}  refused {(~passed[m]).mean():6.1%}")

    if gap > 0.10:
        print("\n! not saving: an authorship-correlated gate would launder low recall into "
              "high abstention")
        return 1

    out = ROOT / "artifacts" / f"genre_gate{args.out_suffix or args.suffix}.json"
    gate.save(out)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Would the genre gate refuse a genuine English-learner's ADMISSIONS essay?

    python scripts/esl_gate_probe.py --suffix _remote

THE GAP THIS ADDRESSES. The corpus has native-authored admissions essays and it has
English-learner writing, but the cell where those two meet is empty: no ESL-authored personal
statements. PROJECT.md called that "the most important untested question in the project",
which was true, but "we have no data" was too quick a surrender. The effect of ESL authorship
on the gate's five features can be MEASURED in the data we do hold, and then applied to the
admissions essays we do hold. That does not fill the empty cell -- it bounds what would
happen if it were filled.

Three measurements, in increasing order of how much they assume:

  1. MATCHED-PROMPT ELL EFFECT (assumes least). PERSUADE is a matched control: the same
     prompts written by ELL-flagged and non-ELL students. Differencing WITHIN a prompt holds
     genre, topic and task constant, so what remains is close to the authorship effect alone.
     This yields a per-feature shift vector and a shift in the gate's log-odds.

  2. PROFICIENCY GRADIENT (assumes little). ELLIPSE carries a holistic proficiency score,
     1.0-5.0, on 260 English-learner documents. If the gate were reading language proficiency
     rather than genre, P(in-domain) would fall as proficiency falls. The correlation is
     reported with a permutation p-value.

  3. TRANSPLANT (assumes most, and it is stated). Add the shift vector from (1) to every real
     admissions essay's features and re-run the gate. This asks: if an admissions essay
     carried the feature signature of an English-learner author, would it still be scored?

     Two assumptions ride on this and neither is verifiable here: that the effect is additive
     in raw feature space, and that an effect measured in argumentative coursework transfers
     to personal narrative. It is a sensitivity analysis. It is not a measurement of the
     missing cell, and it must not be reported as one.

A note on what a NULL result means here. If the gate turns out to be insensitive to the ESL
shift, that is evidence it reads genre rather than authorship -- the same property
fit_genre_gate.py checks for machine-vs-human, checked again along the axis where a false
refusal would do the most harm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.detect.genre import GenreGate, document_genre_features  # noqa: E402

#: Admissions personal statements -- the genre the detector and the gate were fitted for.
IN_DOMAIN = ("train", "modern_holdout", "modern_claude")
RNG = np.random.default_rng(20260812)


def load_docs(path: Path) -> dict[str, tuple[list[dict], dict]]:
    """{doc_id: (sentence feature dicts, doc_meta)}"""
    out: dict[str, tuple[list[dict], dict]] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        feats, _ = out.setdefault(r["doc_id"], ([], r.get("doc_meta") or {}))
        feats.append(r["features"])
    return out


def logodds(gate: GenreGate, f: dict[str, float]) -> float:
    p = min(max(gate.probability(f), 1e-9), 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def boot_ci(v: np.ndarray, n: int = 5000) -> tuple[float, float]:
    if len(v) == 0:
        return (float("nan"), float("nan"))
    m = np.array([RNG.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_remote")
    ap.add_argument("--gate-suffix", default=None,
                    help="score with a different gate artifact, for ablation comparisons")
    args = ap.parse_args()

    feats_dir = ROOT / "data" / "features"
    gate_path = ROOT / "artifacts" / f"genre_gate{args.gate_suffix or args.suffix}.json"
    if not gate_path.exists():
        print(f"! no gate at {gate_path.relative_to(ROOT)}; run fit_genre_gate.py first")
        return 1
    gate = GenreGate.load(gate_path)
    names = list(gate.feature_names)
    print(f"gate: threshold {gate.threshold:.4f}, {len(names)} features\n")

    # -- documents -----------------------------------------------------------------
    esl = load_docs(feats_dir / f"esl{args.suffix}.jsonl")
    rows = {d: (document_genre_features(sf), meta) for d, (sf, meta) in esl.items()}

    admissions: list[dict[str, float]] = []
    for name in IN_DOMAIN:
        for _d, (sf, _m) in load_docs(feats_dir / f"{name}{args.suffix}.jsonl").items():
            admissions.append(document_genre_features(sf))
    if not admissions or not rows:
        print("! need admissions + esl features; build them first")
        return 1

    # ================================================================= 1. matched prompts
    print("=" * 78)
    print("1. MATCHED-PROMPT ELL EFFECT  (PERSUADE: same prompts, ELL flag is the difference)")
    print("=" * 78)

    by_prompt: dict[str, dict[bool, list[dict]]] = defaultdict(lambda: {True: [], False: []})
    for doc, (f, meta) in rows.items():
        if doc.startswith("persuade") and meta.get("ell") is not None:
            by_prompt[meta.get("prompt_name") or "?"][bool(meta["ell"])].append(f)

    shared = {p: v for p, v in by_prompt.items() if v[True] and v[False]}
    n_ell = sum(len(v[True]) for v in shared.values())
    n_nat = sum(len(v[False]) for v in shared.values())
    print(f"{len(shared)} prompts written by both groups: {n_ell} ELL, {n_nat} non-ELL")
    if n_ell < 5:
        print("! too few matched ELL documents to difference; skipping")
        shift = dict.fromkeys(names, 0.0)
    else:
        # Within-prompt difference of means, then average over prompts weighted by ELL
        # count. Differencing inside the prompt is the point: it removes topic, task and
        # rubric, which a pooled comparison would leave in.
        per_prompt_d, per_prompt_w, lo_d = [], [], []
        for _p, v in shared.items():
            d = [np.nanmean([x[n] for x in v[True]]) - np.nanmean([x[n] for x in v[False]])
                 for n in names]
            per_prompt_d.append(d)
            per_prompt_w.append(len(v[True]))
            lo_d.append(np.mean([logodds(gate, x) for x in v[True]])
                        - np.mean([logodds(gate, x) for x in v[False]]))
        w = np.array(per_prompt_w, dtype=float)
        shift_vec = np.nansum(np.array(per_prompt_d) * w[:, None], axis=0) / w.sum()
        shift = dict(zip(names, shift_vec.tolist()))

        lo_arr = np.array(lo_d, dtype=float)
        lo_mean = float(np.nansum(lo_arr * w) / w.sum())
        lo_lo, lo_hi = boot_ci(lo_arr[np.isfinite(lo_arr)])

        print("\n  per-feature shift (ELL minus non-ELL, same prompt), in raw units and in")
        print("  standard deviations of the gate's own scaling:")
        for i, n in enumerate(names):
            sd = shift_vec[i] / float(gate.scale[i]) if gate.scale is not None else float("nan")
            print(f"    {n:24s} {shift_vec[i]:+9.4f}   {sd:+5.2f} sd")
        print(f"\n  gate log-odds shift: {lo_mean:+.3f}  (95% CI over prompts "
              f"{lo_lo:+.3f} to {lo_hi:+.3f})")
        print("  negative = ELL authorship pushes a document TOWARD refusal")

    # ================================================================ 2. proficiency
    print("\n" + "=" * 78)
    print("2. PROFICIENCY GRADIENT  (ELLIPSE, 1.0 = weakest English, 5.0 = strongest)")
    print("=" * 78)
    prof, pin = [], []
    for doc, (f, meta) in rows.items():
        if doc.startswith("ellipse") and meta.get("proficiency") is not None:
            prof.append(float(meta["proficiency"]))
            pin.append(gate.probability(f))
    if len(prof) >= 20:
        prof_a, pin_a = np.array(prof), np.array(pin)

        def spearman(a: np.ndarray, b: np.ndarray) -> float:
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])

        rho = spearman(prof_a, pin_a)
        null = np.array([spearman(prof_a, RNG.permutation(pin_a)) for _ in range(5000)])
        p = float((np.abs(null) >= abs(rho)).mean())
        print(f"  n={len(prof)}   Spearman rho(proficiency, P(in-domain)) = {rho:+.3f}"
              f"   permutation p = {p:.3f}")
        print("  a strong POSITIVE rho would mean the gate refuses weaker English;")
        print(f"  {'-> no significant gradient' if p >= 0.05 else '-> GRADIENT PRESENT'}")
        print("\n  refusal rate by proficiency band:")
        for lo_b, hi_b in ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.5)):
            m = (prof_a >= lo_b) & (prof_a < hi_b)
            if m.sum():
                ref = float((pin_a[m] < gate.threshold).mean())
                print(f"    proficiency {lo_b:.1f}-{hi_b:.1f}  n={int(m.sum()):4d}  "
                      f"refused {ref:6.1%}   mean P(in-domain) {pin_a[m].mean():.3f}")
    else:
        print("  ! too few scored documents")

    # ================================================================ 3. transplant
    print("\n" + "=" * 78)
    print("3. TRANSPLANT  (real admissions essays, shifted by the measured ELL signature)")
    print("=" * 78)
    print("  ASSUMES additivity in raw feature space, and that an effect measured in")
    print("  argumentative coursework carries to personal narrative. Neither is verifiable")
    print("  with the data we hold. This bounds the risk; it does not measure it.\n")

    base_p = np.array([gate.probability(f) for f in admissions])
    base_ref = float((base_p < gate.threshold).mean())
    print(f"  {len(admissions)} real admissions essays, as they are: "
          f"refused {base_ref:.2%}")

    results = {}
    for mult, label in ((1.0, "1x ELL shift"), (2.0, "2x  (a stronger effect than measured)"),
                        (3.0, "3x  (deliberately pessimistic)")):
        shifted = [{**f, **{n: f.get(n, float("nan")) + mult * shift[n] for n in names}}
                   for f in admissions]
        p_s = np.array([gate.probability(f) for f in shifted])
        ref = float((p_s < gate.threshold).mean())
        lo_c, hi_c = boot_ci((p_s < gate.threshold).astype(float))
        results[label] = ref
        print(f"  + {label:38s} refused {ref:6.2%}   (95% CI {lo_c:.2%}-{hi_c:.2%})")

    # ================================================================ 4. proficiency slope
    #
    # Tests 1 and 3 disagree with test 2, and the disagreement is not a wash -- it is a
    # difference in statistical power. Test 1 differences a BINARY flag over 24 ELL
    # documents; test 2 reads a GRADED score over 260. If proficiency matters continuously,
    # a binary split of mostly-stronger writers would miss it, which is exactly the pattern
    # observed. So the shift vector is re-derived from the graded score, where the effect
    # was actually detected, and transplanted the same way.
    #
    # This is the sharper instrument, and it is measured entirely WITHIN one genre
    # (ELLIPSE is all argumentative), so genre is held constant by construction rather
    # than by matching.
    print("\n" + "=" * 78)
    print("4. PROFICIENCY-SLOPE TRANSPLANT  (the test that actually answers the question)")
    print("=" * 78)
    slope_ok = len(prof) >= 20
    if slope_ok:
        E = [(document_genre_features(sf), meta) for d, (sf, meta) in esl.items()
             if d.startswith("ellipse") and (meta or {}).get("proficiency") is not None]
        pv = np.array([m["proficiency"] for _f, m in E], dtype=float)
        print("  which feature carries the proficiency signal (within one genre):")
        slope = {}
        for n in names:
            fv = np.array([f.get(n, np.nan) for f, _m in E], dtype=float)
            ok = np.isfinite(fv)
            if ok.sum() > 10:
                b = float(np.polyfit(pv[ok], fv[ok], 1)[0])   # feature per +1 proficiency
                r = float(np.corrcoef(pv[ok], fv[ok])[0, 1])
                sd = b / float(gate.scale[names.index(n)])
                slope[n] = b
                print(f"    {n:24s} r={r:+.3f}   {b:+9.4f} per proficiency point "
                      f"({sd:+.2f} sd)")
            else:
                slope[n] = 0.0

        # A weak-English applicant, relative to the admissions essays we hold. ELLIPSE's
        # mean is ~3.1; dropping 1.5 points reaches the 1.5-2.0 band, where refusal was
        # 100%. If the gate is reading genre and not proficiency, this shift should barely
        # move an admissions essay.
        print(f"\n  transplanting a proficiency DROP onto {len(admissions)} real admissions "
              "essays:")
        prof_res = {}
        for drop in (0.5, 1.0, 1.5, 2.0):
            shifted = [{**f, **{n: f.get(n, float("nan")) - drop * slope[n] for n in names}}
                       for f in admissions]
            p_s = np.array([gate.probability(f) for f in shifted])
            ref = float((p_s < gate.threshold).mean())
            lo_c, hi_c = boot_ci((p_s < gate.threshold).astype(float))
            prof_res[f"-{drop}"] = ref
            print(f"    -{drop:.1f} proficiency points   refused {ref:6.2%}   "
                  f"(95% CI {lo_c:.2%}-{hi_c:.2%})   baseline {base_ref:.2%}")
    else:
        prof_res, slope = {}, {}

    print("\n" + "=" * 78)
    print("READING THIS HONESTLY")
    print("=" * 78)
    print("  The empty cell is still empty: we hold no ESL-authored admissions essays, and")
    print("  no arithmetic here creates one. What these measurements do is convert")
    print("  'unknown' into a bounded estimate, and say which way the evidence points.")

    out = ROOT / "artifacts" / f"esl_gate_probe{args.gate_suffix or args.suffix}.json"
    out.write_text(json.dumps({
        "threshold": gate.threshold,
        "matchedPrompts": len(shared),
        "nEll": n_ell, "nNonEll": n_nat,
        "shift": shift,
        "baselineRefusal": base_ref,
        "transplantRefusal": results,
        "proficiencySlope": slope,
        "proficiencyTransplantRefusal": prof_res,
        "nAdmissions": len(admissions),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

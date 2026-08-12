#!/usr/bin/env python
"""The honest supervised number: hold out an entire GENERATION PIPELINE, not a generator.

    python scripts/cross_pipeline_probe.py

docs/09-frontier-ceiling.md records a supervised classifier that reached AUROC 1.000 on
held-out Claude Opus and detected nothing -- it flagged 0% of real GPT-3.5 essays from
another collection and 17% of real students. It had learned which pipeline produced a file.
Every internal split preserved that cue because every machine essay in the project came from
one pipeline: ours.

DAIGT fixes the corpus rather than the model. Its machine half was contributed by many
different people using many different models and prompts, so `meta.pipeline` identifies an
independent producer and holding one out removes the shortcut by construction.

Two experiments, answering different questions:

  A. LEAVE-ONE-PIPELINE-OUT, within DAIGT. Domain, prompts and human class are all held
     fixed; only the producer changes. This is the cleanest available measurement of whether
     a supervised detector generalises to a generator it has never seen.

  B. TRAIN ON DAIGT, TEST ON OUR CORPUS. Trained exclusively on other people's machine text
     and other people's students, then asked about our Claude/Gemini essays and our human
     sets. Nothing in training shares a pipeline with anything in test, so the provenance
     shortcut cannot operate at all.

     B additionally changes DOMAIN -- DAIGT is argumentative coursework on PERSUADE prompts,
     our corpus is admissions personal statements. A weak result in B is therefore ambiguous
     between "does not transfer across pipelines" and "does not transfer across domains",
     and must not be read as the former alone. A STRONG result in B is unambiguous, because
     it would have survived both shifts at once.

Headline is TPR at 5% FPR, not AUROC: the product question is what the detector catches at a
false-accusation rate a customer could defend.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("sp", ROOT / "scripts" / "stylometry_probe.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

SEED = 20260811
MIN_PIPELINE = 40  # a pipeline with fewer documents than this cannot support a fold


def load_daigt() -> tuple[list[str], dict[str, list[str]]]:
    """Return (human essays, {pipeline: machine essays}), typography- and length-normalised.

    The same normalisation as stylometry_probe: curly punctuation folded to ASCII and every
    document truncated to a common word count. Both were label leaks in the earlier corpus
    and there is no reason to assume DAIGT is free of them.
    """
    human: list[str] = []
    machine: dict[str, list[str]] = defaultdict(list)
    path = ROOT / "data" / "raw" / "daigt.jsonl"
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        words = sp.normalise_typography(" ".join(d["text"].split())).split()
        if len(words) < sp.TRUNCATE_WORDS:
            continue
        text = " ".join(words[: sp.TRUNCATE_WORDS])
        if d["authorship"] == "human":
            human.append(text)
        else:
            machine[d["meta"]["pipeline"]].append(text)
    return human, dict(machine)


def evaluate(clf, prep, pos_texts: list[str], neg_texts: list[str]) -> tuple[float, float]:
    p = clf.predict_proba([prep(t) for t in pos_texts])[:, 1]
    n = clf.predict_proba([prep(t) for t in neg_texts])[:, 1]
    return sp.auroc(p, n), sp.tpr_at(p, n, 0.05)


def main() -> int:
    if not (ROOT / "data" / "raw" / "daigt.jsonl").exists():
        print("run scripts/fetch_external.py --dataset daigt first")
        return 1

    human, machine = load_daigt()
    machine = {k: v for k, v in machine.items() if len(v) >= MIN_PIPELINE}
    print(f"DAIGT: {len(human)} human, {sum(map(len, machine.values()))} machine "
          f"across {len(machine)} pipelines")
    for k, v in sorted(machine.items(), key=lambda x: -len(x[1])):
        print(f"    {k:36s} {len(v):4d}")
    print()

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(human))
    h_tr = [human[i] for i in idx[: len(human) // 2]]
    h_te = [human[i] for i in idx[len(human) // 2:]]
    results = {}

    # -- A: leave one pipeline out, inside DAIGT --------------------------------------
    for kind, label in (("char", "char 2-4gram"), ("func", "function words only")):
        print("=" * 74)
        print(f"A. LEAVE-ONE-PIPELINE-OUT (within DAIGT)   features: {label}")
        print("=" * 74)
        print(f"{'held-out pipeline':36s} {'n':>4s} {'AUROC':>7s} {'TPR@5%FPR':>10s}")
        print("-" * 62)
        aurocs, tprs = [], []
        for target in machine:
            train_m = [t for k, v in machine.items() if k != target for t in v]
            prep = sp.function_words_only if kind == "func" else (lambda s: s)
            clf = sp.build(kind)
            clf.fit([prep(t) for t in h_tr + train_m],
                    np.r_[np.zeros(len(h_tr)), np.ones(len(train_m))])
            a, t = evaluate(clf, prep, machine[target], h_te)
            aurocs.append(a)
            tprs.append(t)
            results[f"A|{kind}|{target}"] = {"auroc": a, "tpr_at_5fpr": t,
                                             "n": len(machine[target])}
            print(f"{target:36s} {len(machine[target]):4d} {a:7.3f} {t:10.3f}")
        print(f"{'MEDIAN':36s} {'':4s} {np.median(aurocs):7.3f} {np.median(tprs):10.3f}\n")

    # -- B: train on DAIGT only, test on our corpus -----------------------------------
    ours = {n: sp.load(p, m, 999) for n, p, m in sp.MACHINE_SETS}
    our_humans = {
        "liang_college_human": sp.load("data/raw/liang_college_human.jsonl", None, 999),
        "liang_hewlett_human": sp.load("data/raw/liang_hewlett_human.jsonl", None, 999),
        "ellipse_esl": sp.load("data/raw/ellipse.jsonl", None, 400),
    }
    all_m = [t for v in machine.values() for t in v]

    for kind, label in (("char", "char 2-4gram"), ("func", "function words only")):
        prep = sp.function_words_only if kind == "func" else (lambda s: s)
        clf = sp.build(kind)
        clf.fit([prep(t) for t in h_tr + all_m],
                np.r_[np.zeros(len(h_tr)), np.ones(len(all_m))])
        # Threshold from DAIGT's own held-out humans: the only humans this model has any
        # right to calibrate on, since every other human set is also a test set here.
        thr = float(np.quantile(clf.predict_proba([prep(t) for t in h_te])[:, 1], 0.95))

        print("=" * 74)
        print(f"B. TRAINED ON DAIGT ONLY -> OUR CORPUS   features: {label}")
        print("   (threshold = 5% FPR on DAIGT held-out humans; domain also shifts)")
        print("=" * 74)
        print(f"{'probe set':34s} {'n':>5s} {'flagged machine':>16s}   expected")
        for name, rows in ours.items():
            if not rows:
                continue
            p = clf.predict_proba([prep(t) for t in rows])[:, 1]
            r = float((p >= thr).mean())
            results[f"B|{kind}|{name}"] = {"flag_rate": r, "n": len(rows)}
            print(f"  MACHINE {name:26s} {len(rows):5d} {r:15.1%}   high")
        for name, rows in our_humans.items():
            if not rows:
                continue
            p = clf.predict_proba([prep(t) for t in rows])[:, 1]
            r = float((p >= thr).mean())
            results[f"B|{kind}|{name}"] = {"flag_rate": r, "n": len(rows)}
            print(f"  HUMAN   {name:26s} {len(rows):5d} {r:15.1%}   <=5%")
        print()

    (ROOT / "artifacts" / "cross_pipeline_probe.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print("wrote artifacts/cross_pipeline_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

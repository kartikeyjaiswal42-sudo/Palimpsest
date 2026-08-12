#!/usr/bin/env python
"""Can a supervised classifier reading RAW TEXT detect frontier models when statistics cannot?

    python scripts/stylometry_probe.py

docs/09-frontier-ceiling.md measured seven perplexity-family statistics fused by logistic
regression and found 0% recall at a 5% false-positive budget on Claude Opus, Claude Sonnet
and mid-tier Gemini. That is a real result about *those statistics*. It is not yet a result
about detection, because a hand-made statistic is a very narrow model class: it can express
"this text is unsurprising" but not "this text prefers the em-dash and the tricolon and the
word 'nuanced'".

Commercial detectors are supervised transformers over raw tokens, and character n-gram
stylometry is the classical baseline that has beaten far fancier things at authorship work
for thirty years. Both read the surface directly. If frontier machine prose carries a
lexical fingerprint at all, this is the cheapest instrument that would see it.

Design, deliberately hostile to the result we would like:

  * LEAVE-ONE-GENERATOR-OUT. The generator being scored is absent from training. Fitting on
    Opus and scoring Opus measures memorisation of one corpus, which is exactly the mistake
    docs/08 caught the project making with vendor.
  * The human class is pooled native + ESL, so the false-positive budget is charged on the
    population that actually gets falsely accused.
  * TPR at 5% FPR is the headline. AUROC is printed but it is not the product.
  * A TOPIC CONTROL. The machine essays were generated from a fixed subject list; the human
    essays were not. A classifier can score well by recognising the topics rather than the
    authorship, which would be a corpus artifact wearing a detector's clothes. Word-level
    features see topic directly. Character n-grams within words see it too. So the run is
    repeated with FUNCTION WORDS ONLY -- a closed class carrying almost no topical content,
    the standard stylometric control -- and if the number survives that, it is style.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEED = 20260811
PER_SET = 120  # take as much as each file holds, up to this

# Four human corpora from THREE independent collections. `persuade` and `ellipse` are not
# Liang files at all, which is what makes them a real test: holding out a second Liang file
# still leaves the collection's own conventions in the training data.
HUMAN_SETS = [
    ("human_native", "data/raw/liang_college_human.jsonl", None),
    ("human_hewlett", "data/raw/liang_hewlett_human.jsonl", None),
    ("human_ellipse", "data/raw/ellipse.jsonl", None),
    ("human_persuade", "data/raw/persuade.jsonl", None),
]

#: Held out entirely under the `unseen-human` protocol: a different collection, different
#: student population, different preprocessing. If the classifier survives this, the signal
#: is not a provenance cue.
UNSEEN_HUMAN = "human_persuade"
MACHINE_SETS = [
    ("gemini_flash_lite", "data/generated/modern_holdout.jsonl", None),
    ("gemini_flash", "data/generated/modern_unseen_family.jsonl", None),
    ("claude_haiku", "data/generated/claude_modern_heldout.jsonl", "haiku"),
    ("claude_sonnet", "data/generated/claude_modern_heldout.jsonl", "sonnet"),
    ("claude_opus", "data/generated/claude_modern_heldout.jsonl", "opus"),
    ("claude_fable", "data/generated/claude_modern_heldout.jsonl", "fable"),
]

# A closed class: articles, prepositions, pronouns, auxiliaries, conjunctions. These carry
# grammar rather than subject matter, which is what makes them the standard control for
# "did the classifier learn style, or did it learn what the essays were about".
FUNCTION_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself me more most my myself no nor not of off on once only or other ought our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why with would you your yours yourself yourselves
""".split())


#: Words kept from each document. Two confounds are removed by truncating to a common
#: length: essay length itself (TOEFL runs ~104 words, our generations ~640, and the
#: function-word representation encodes token count directly), and the fact that a longer
#: document simply supports a more confident estimate.
TRUNCATE_WORDS = 300

_TYPO = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "—": "-", "–": "-", "…": "...", " ": " ", "′": "'",
})


def normalise_typography(text: str) -> str:
    """Fold smart punctuation to ASCII.

    Measured on this corpus: 90% of Liang's college essays contain a curly apostrophe and
    80% contain curly quotes, while **0%** of every generated set does -- our generation
    pipeline normalised them away. A character n-gram model finds that instantly and scores
    a perfect AUROC without learning anything about authorship.

    This is the same class of leak the project already documented when paragraph structure
    turned out to be a source marker (see `flatten` in scripts/build_features.py). Both
    classes are folded identically here so the cue cannot exist.

    NOTE this discards real signal: an em-dash habit is genuinely stylistic. It is discarded
    anyway, because in THIS corpus the em-dash is perfectly confounded with provenance, and
    a feature that is 100% predictive for the wrong reason is worse than no feature.
    """
    return text.translate(_TYPO)


def load(path: str, model: str | None, n: int) -> list[str]:
    out = []
    p = ROOT / path
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        d = json.loads(line)
        if model and (d.get("meta") or {}).get("model") != model:
            continue
        words = normalise_typography(" ".join((d.get("text") or "").split())).split()
        if len(words) >= TRUNCATE_WORDS:
            out.append(" ".join(words[:TRUNCATE_WORDS]))
    random.Random(SEED).shuffle(out)
    return out[:n]


def function_words_only(text: str) -> str:
    """Strip everything but the closed-class words, preserving order.

    Content words become a placeholder rather than vanishing, so sentence rhythm and the
    *positions* of function words survive -- that structure is a large part of the signal.
    """
    toks = re.findall(r"[a-z']+|[.,;:!?]", text.lower())
    return " ".join(t if (t in FUNCTION_WORDS or not t.isalpha()) else "#" for t in toks)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    return float((pos >= float(np.quantile(neg, 1.0 - fpr))).mean())


def build(kind: str):
    if kind == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                              max_features=60000, sublinear_tf=True)
    else:  # function-word n-grams: style with the topic taken out
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3,
                              max_features=40000, sublinear_tf=True)
    return make_pipeline(vec, StandardScaler(with_mean=False),
                         LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))


def main() -> int:
    human, machine = {}, {}
    for name, path, m in HUMAN_SETS:
        rows = load(path, m, PER_SET)
        if rows:
            human[name] = rows
        print(f"  {name:20s} {len(rows):4d}")
    for name, path, m in MACHINE_SETS:
        rows = load(path, m, PER_SET)
        if rows:
            machine[name] = rows
        print(f"  {name:20s} {len(rows):4d}")
    print()

    results = {}

    # Two protocols. The difference between them is the whole point of this script.
    #
    # "shared-human" holds out only the GENERATOR: the human test essays come from the same
    #   corpora as the human training essays. Any cue that separates "Liang's files" from
    #   "essays we generated" -- unicode quote style, genre, preprocessing, essay length --
    #   is a free win, and the classifier will take it.
    # "unseen-human" additionally holds out an entire HUMAN CORPUS. A corpus artifact cannot
    #   help here, because the human documents being scored were never shown to the fit.
    #
    # A detector that reads authorship performs similarly under both. A detector that reads
    # corpus collapses under the second. The first protocol reported AUROC 1.000 on held-out
    # Claude Opus, which is not a number real detection produces, and is why this control
    # exists at all.
    for protocol in ("shared-human", "unseen-human"):
        for kind, label in (("char", "char 2-4gram (style + topic)"),
                            ("func", "function words only (topic removed)")):
            print("=" * 82)
            print(f"PROTOCOL: {protocol}   FEATURES: {label}")
            if protocol == "unseen-human":
                print("  (human test corpus never seen in training -- corpus shortcut removed)")
            print("=" * 82)
            print(f"{'held-out generator':22s} {'human test set':18s} {'n+':>4s} {'n-':>4s} "
                  f"{'AUROC':>7s} {'TPR@5%FPR':>10s}")
            print("-" * 82)

            for target in machine:
                train_m = [t for k, v in machine.items() if k != target for t in v]

                if protocol == "shared-human":
                    # Random half/half over the pooled humans: every corpus is in both sides.
                    rng = np.random.default_rng(SEED)
                    pooled = [t for v in human.values() for t in v]
                    idx = rng.permutation(len(pooled))
                    h_tr = [pooled[i] for i in idx[: len(pooled) // 2]]
                    h_te = [pooled[i] for i in idx[len(pooled) // 2:]]
                    h_name = "pooled (seen)"
                else:
                    # Hold out an entire independent COLLECTION, not just another file
                    # from the same one.
                    h_name = f"{UNSEEN_HUMAN} (unseen)"
                    h_te = human[UNSEEN_HUMAN]
                    h_tr = [t for k, v in human.items() if k != UNSEEN_HUMAN for t in v]

                prep = function_words_only if kind == "func" else (lambda s: s)
                clf = build(kind)
                clf.fit([prep(t) for t in h_tr + train_m],
                        np.r_[np.zeros(len(h_tr)), np.ones(len(train_m))])
                p_pos = clf.predict_proba([prep(t) for t in machine[target]])[:, 1]
                p_neg = clf.predict_proba([prep(t) for t in h_te])[:, 1]

                a, t = auroc(p_pos, p_neg), tpr_at(p_pos, p_neg, 0.05)
                results[f"{protocol}|{kind}|{target}"] = {
                    "auroc": a, "tpr_at_5fpr": t, "n_target": len(machine[target]),
                    "n_human_test": len(h_te), "human_test": h_name}
                print(f"{target:22s} {h_name:18s} {len(machine[target]):4d} {len(h_te):4d} "
                      f"{a:7.3f} {t:10.3f}")
            print()

    (ROOT / "artifacts" / "stylometry_probe.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print("wrote artifacts/stylometry_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

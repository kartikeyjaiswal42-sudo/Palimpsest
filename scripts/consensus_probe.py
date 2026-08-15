#!/usr/bin/env python
"""Do INDEPENDENT observers agree about machine prose and disagree about human prose?

    python scripts/consensus_probe.py --pilot 20 --budget 2000    # cheap look first
    python scripts/consensus_probe.py --budget 6000               # the full protocol

docs/09-frontier-ceiling.md closes a door and, in closing it, says exactly where the wall is.
Every statistic tried so far reads ONE observer's ABSOLUTE surprisal: how unlikely was this
token under qwen3-30B. That family is confounded by how hard the *content* is, and the
document that made the confound undeniable is in that doc -- a Gemini-written essay carrying
real proper nouns (``TheNectar``, a solar induction calculation) whose ``mean_logprob`` was
**2.26 SD further from the machine distribution than human essays are**. The observer found
it harder to predict than typical human prose, so every absolute statistic voted "human".

Rare tokens are hard for *every* model. That is the opening. If content difficulty is
common-mode across independent observers, then a statistic built from the DISAGREEMENT between
observers cancels it, and what survives is not "was this text surprising" but "was this text
surprising in a way that is idiosyncratic to one model's view of English".

The hypothesis in one sentence: **frontier machine prose sits where independent strong models
agree, and human prose contains choices no model would have made.** A word that qwen, mistral
and llama all rank first is a word three separately-trained models converged on. Human writing
is full of words that only one of them expects.

This is a different axis from "buy a bigger observer", which docs/09 measured and correctly
rejected: 124 M -> 30 B moved Opus from 0.458 to 0.695 AUROC and 0% recall to 0% recall. A
fourth observer of the same kind would not help. A *ratio between* observers is not more of
the same instrument; it is a different measurement, and it is the one Binoculars is built on.

WHAT THIS IS AND IS NOT

  * It is not Binoculars. Binoculars divides perplexity by *cross-perplexity*, which needs the
    full next-token distribution from both models at every position. Workers AI gives the
    realised token's log-probability and rank, plus at most a top-20 head. So the denominator
    here is the observed spread of realised-token log-probabilities across models, which is a
    proxy, and ``x_binoc`` is named for the family rather than the method.
  * It is not a second detector wearing a hat. The observers are never prompted and never
    asked for a verdict; three forward passes are read exactly as one was.
  * It costs real neurons. qwen3-30B is a 3 B-active MoE and prices at 3.1 neurons for a
    571-word essay; mistral-24B costs 21.3 and llama-70B 18.0 for the same text. The free
    allowance is 10,000/day for the whole account, so ``--budget`` is enforced rather than
    hoped for: the run refuses to start if the pre-flight estimate exceeds it and stops mid-run
    if the meter says so. Cached documents are free and are not counted.

PROTOCOL -- deliberately identical to scripts/fusion_probe.py, which is imported rather than
copied so the estimator, the null and the metric cannot drift apart. Leave-one-generator-out:
the generator being scored is held out entirely, because in production the next model is always
one you did not fit on. Three arms are reported side by side:

    A  baseline    the 7 single-observer statistics fusion_probe already published
    B  consensus   the cross-observer statistics alone
    C  both

Arm A exists as a control on this script: it must reproduce fusion_probe's published numbers
(haiku 0.800, opus 0.000 TPR@5%FPR). If it does not, the corpus selection drifted and B and C
mean nothing. Read TPR@5%FPR, not AUROC -- a score can rank machine above human on average
while catching nothing at a false-accusation rate anyone would accept.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_observer as bench  # noqa: E402  -- reuse the exact corpus selection
import fusion_probe as fp  # noqa: E402  -- reuse the exact estimator, null and metrics
from stylometry_probe import normalise_typography  # noqa: E402  -- the same fold, not a copy

from palimpsest.scorer.remote_lm import RemoteLMScorer  # noqa: E402

# Every Workers AI model that returns ``prompt_logprobs``, with the measured cost of one
# 571-word essay. llama-3.1-8b is in the worker's allow-list but returns the legacy
# ``{response}`` shape with no logprobs at all, so it cannot be an observer here.
OBSERVERS = {
    "qwen": ("@cf/qwen/qwen3-30b-a3b-fp8", 3.1),
    "llama": ("@cf/meta/llama-3.3-70b-instruct-fp8-fast", 18.0),
    "mistral": ("@cf/mistralai/mistral-small-3.1-24b-instruct", 21.3),
}

#: An INDEPENDENT in-genre human reference. docs/09 Result 3 is explicit that holding out
#: another file from the same collection proves nothing, because the collection's conventions
#: stay in training -- only a foreign corpus broke the stylometry probe. JHU's *Essays That
#: Worked* are real admissions essays gathered by somebody else, which makes them the control
#: that can distinguish "reads machine-ness" from "reads Liang's file".
JHU_HUMAN = ("human_jhu", "data/raw/jhu.jsonl", None)

#: Words at the very start of a document are unconditioned under every tokenizer, so their
#: log-probabilities describe the prior rather than the text. Dropped for all observers
#: identically, which keeps the alignment honest.
SKIP_WORDS = 3

WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------- alignment

def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def per_word(scores, spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a model's tokens onto word spans.

    Three tokenizers cut the same essay into 666, 667 and 665 tokens, so nothing can be
    compared token-by-token. Character offsets are the common frame: every observer reports
    ``char_start``/``char_end`` against the same string, so a word is the unit all three can
    be asked about.

    Per word: the MEAN log-probability of the tokens covering it (a word split into three
    pieces is one decision by the writer, not three), and the BEST rank among them -- "did
    this model expect this word at all" is the question, and a model that ranked the first
    piece 1st has expected the word.

    Returns ``(logprob, log_rank)`` with NaN where the model emitted no token for a word.
    """
    lp = np.full(len(spans), np.nan)
    lr = np.full(len(spans), np.nan)
    if len(scores) == 0:
        return lp, lr
    centre = (scores.char_start + scores.char_end) / 2.0
    order = np.argsort(centre, kind="mergesort")
    centre = centre[order]
    logprob = np.asarray(scores.logprob, dtype=np.float64)[order]
    rank = np.asarray(scores.rank, dtype=np.float64)[order]
    starts = np.array([s for s, _ in spans], dtype=np.float64)
    ends = np.array([e for _, e in spans], dtype=np.float64)
    lo = np.searchsorted(centre, starts, side="left")
    hi = np.searchsorted(centre, ends, side="left")
    for i in range(len(spans)):
        a, b = lo[i], hi[i]
        if b > a:
            lp[i] = float(np.mean(logprob[a:b]))
            lr[i] = float(np.log(max(np.min(rank[a:b]), 1.0)))
    return lp, lr


# --------------------------------------------------------------------------- features

def consensus_features(L: np.ndarray, R: np.ndarray) -> dict[str, float]:
    """Cross-observer statistics for one document.

    ``L`` and ``R`` are (n_words x n_observers) log-probability and log-rank, already aligned.
    Rows where any observer is missing are dropped -- an average over a different subset per
    observer is not a comparison.

    Every feature here is a statement about DISAGREEMENT, not about surprisal. That is the
    whole point: absolute surprisal is what docs/09 measured to a wall.
    """
    ok = np.all(np.isfinite(L), axis=1) & np.all(np.isfinite(R), axis=1)
    L, R = L[ok], R[ok]
    if len(L) < 30 or L.shape[1] < 2:
        return {}

    sd = L.std(axis=1)
    rng = L.max(axis=1) - L.min(axis=1)
    rank_sd = R.std(axis=1)
    top1 = R <= 1e-9  # log(1) == 0, so rank 1 is the only row that lands on zero

    # Mean pairwise correlation of the observers' per-word views of the same text. High means
    # the observers rise and fall together across the document.
    cors = []
    for i in range(L.shape[1]):
        for j in range(i + 1, L.shape[1]):
            a, b = L[:, i], L[:, j]
            if a.std() > 1e-9 and b.std() > 1e-9:
                cors.append(float(np.corrcoef(a, b)[0, 1]))

    feats = {
        # -- agreement about which words were predictable -------------------------------
        "x_consensus_top1": float(np.mean(np.all(top1, axis=1))),
        "x_any_top1": float(np.mean(np.any(top1, axis=1))),
        "x_no_top1": float(np.mean(~np.any(top1, axis=1))),
        # -- dispersion of the observers' views ----------------------------------------
        "x_spread_lp": float(np.mean(sd)),
        "x_spread_lp_sd": float(np.std(sd)),
        "x_range_lp": float(np.mean(rng)),
        "x_spread_rank": float(np.mean(rank_sd)),
        # A word one observer ranked 1st and another ranked outside the top 100. This is the
        # sharpest form of "only one model saw this coming".
        "x_hard_disagree": float(np.mean(rng > np.log(100.0))),
        "x_corr_lp": float(np.mean(cors)) if cors else float("nan"),
        # -- the Binoculars-family ratio ------------------------------------------------
        # Surprisal NORMALISED by cross-observer disagreement. The numerator is what docs/09
        # measured to a wall; the denominator is the term that carries content difficulty,
        # because a rare proper noun is rare to every observer. Dividing is the cancellation
        # the whole script exists to test.
        "x_binoc": float(-L.mean() / max(float(np.mean(sd)), 1e-6)),
        "x_binoc_rank": float(R.mean() / max(float(np.mean(rank_sd)), 1e-6)),
        # -- the envelope: best and worst observer per word -----------------------------
        # If machine text is text every model finds easy, the MINIMUM across observers rises
        # faster than the mean does.
        "x_min_lp": float(np.mean(L.min(axis=1))),
        "x_max_lp": float(np.mean(L.max(axis=1))),
        "x_best_gap": float(np.mean(L.max(axis=1) - L.mean(axis=1))),
        "x_n_words": float(len(L)),
    }
    return feats


CONSENSUS_FEATS = (
    "x_consensus_top1", "x_any_top1", "x_no_top1",
    "x_spread_lp", "x_spread_lp_sd", "x_range_lp", "x_spread_rank",
    "x_hard_disagree", "x_corr_lp", "x_binoc", "x_binoc_rank",
    "x_min_lp", "x_max_lp", "x_best_gap",
)


# --------------------------------------------------------------------------- budget

class Budget:
    """A hard ceiling on neurons, because the allowance is 10,000/day for the whole account.

    Enforced in two places: a pre-flight estimate that refuses to start a run it cannot
    afford, and a check after every document that stops the run rather than overrunning.
    """

    def __init__(self, cap: float) -> None:
        self.cap = float(cap)
        self.spent = 0.0
        self.stopped = False

    def would_exceed(self, extra: float) -> bool:
        return self.spent + extra > self.cap

    def charge(self, amount: float) -> None:
        self.spent += float(amount)
        if self.spent >= self.cap:
            self.stopped = True


# --------------------------------------------------------------------------- scoring

def score_corpus(corpus: dict[str, list[dict]], observers: list[str], budget: Budget,
                 workers: int, normalise: bool = False
                 ) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Score every document under every observer and build its cross-observer features.

    A document is kept only if EVERY observer scored it. A row where one observer is missing
    would silently become a different measurement, and the whole point is the comparison.
    """
    scorers = {name: RemoteLMScorer(model=OBSERVERS[name][0]) for name in observers}

    # THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS REAL.
    #
    # Measured on the exact sets this script scores: 83% of the human_native documents carry a
    # curly apostrophe and 70% carry curly quotes, against 0% of EVERY machine set -- our
    # generation pipeline folded them to ASCII. docs/09 Result 3 records the same leak sinking
    # a stylometry probe that scored AUROC 1.000 and detected nothing.
    #
    # Cross-observer features are *more* exposed to it than absolute perplexity is, and the
    # reason is mechanical rather than statistical: three tokenizers disagree about how to cut
    # a curly apostrophe, so a document containing one has genuinely higher cross-model
    # disagreement -- which is the quantity every feature here is built from. A detector
    # trained on that reads punctuation provenance and calls it authorship.
    #
    # So the fold is applied to BOTH classes identically, and the run is reported both ways.
    if normalise:
        corpus = {k: [{**r, "text": normalise_typography(r["text"])} for r in rows]
                  for k, rows in corpus.items()}

    # -- pre-flight: what is not already on disk is what this run costs ---------------
    uncached = 0.0
    n_uncached = 0
    for rows in corpus.values():
        for r in rows:
            for name in observers:
                if not scorers[name]._cache_path(r["text"]).exists():
                    uncached += OBSERVERS[name][1]
                    n_uncached += 1
    print(f"  pre-flight: {n_uncached} uncached (model, document) pairs, "
          f"~{uncached:.0f} neurons estimated, budget {budget.cap:.0f}")
    if uncached > budget.cap:
        print(f"  REFUSING TO START: estimate {uncached:.0f} > budget {budget.cap:.0f}.\n"
              f"  Raise --budget deliberately, or lower --pilot / --per-set.")
        return {}, {}
    print()

    out: dict[str, list[dict]] = {}
    # Why a document was dropped is part of the result. A silent drop is how a corpus quietly
    # becomes a different corpus: the observer occasionally returns a token stream it cannot
    # align to the text (it warns, and `per_word` then yields all-NaN for that observer), and
    # a run that reported only the surviving count would hide it.
    dropped: dict[str, int] = {"short": 0, "empty_score": 0, "unalignable": 0, "no_base": 0}
    for set_name, rows in corpus.items():
        kept: list[dict] = []
        for r in rows:
            if budget.stopped:
                break
            text = r["text"]
            spans = word_spans(text)[SKIP_WORDS:]
            if len(spans) < 60:
                dropped["short"] += 1
                continue

            # Fetch the observers for one document in parallel; they are independent calls.
            def one(name: str, _text: str = text):
                return name, scorers[name].score(_text)

            with ThreadPoolExecutor(max_workers=min(workers, len(observers))) as ex:
                got = dict(ex.map(one, observers))

            if any(len(got[n]) == 0 for n in observers):
                dropped["empty_score"] += 1
                continue
            cols_lp, cols_lr = [], []
            for name in observers:
                lp, lr = per_word(got[name], spans)
                cols_lp.append(lp)
                cols_lr.append(lr)
            feats = consensus_features(np.column_stack(cols_lp), np.column_stack(cols_lr))
            if not feats:
                # Fewer than 30 words where EVERY observer resolved a token: either the
                # document is tiny or one observer's stream would not align. Either way the
                # cross-observer comparison does not exist for this document.
                dropped["unalignable"] += 1
                continue

            # The single-observer arm must be the SAME statistics fusion_probe published, so
            # they are recomputed here from the default observer rather than re-derived.
            base = bench.stats_from(got[observers[0]].logprob, got[observers[0]].rank)
            if not base:
                dropped["no_base"] += 1
                continue
            kept.append({**base, **feats, "id": r["id"]})

            spent_now = sum(s.spent_neurons for s in scorers.values())
            budget.spent = spent_now
            if spent_now >= budget.cap:
                budget.stopped = True

        out[set_name] = kept
        spent = sum(s.spent_neurons for s in scorers.values())
        calls = sum(s.n_calls for s in scorers.values())
        cached = sum(s.n_cached for s in scorers.values())
        print(f"  {set_name:20s} {len(kept):3d}/{len(rows):3d} docs   "
              f"calls {calls:4d} cached {cached:5d} neurons {spent:8.1f}")
        if budget.stopped:
            print(f"  STOPPED: budget {budget.cap:.0f} neurons reached.")
            break
    if any(dropped.values()):
        print("  dropped: " + "  ".join(f"{k}={v}" for k, v in dropped.items() if v))
    print()
    return out, dropped


# --------------------------------------------------------------------------- protocol

def arm(data: dict[str, list[dict]], feats: tuple[str, ...], rng,
        human_sets: tuple[str, ...] = fp.HUMAN) -> dict[str, dict]:
    """One leave-one-generator-out sweep over a named feature set. fusion_probe's protocol."""
    human = [r for s in human_sets for r in data.get(s, [])]
    if not human:
        return {}
    H = np.asarray([[r.get(f, np.nan) for f in feats] for r in human], dtype=np.float64)
    machine = {m: np.asarray([[r.get(f, np.nan) for f in feats] for r in data.get(m, [])],
                             dtype=np.float64)
               for m in fp.MACHINE if data.get(m)}

    results = {}
    for target, Xt in machine.items():
        others = [v for k, v in machine.items() if k != target]
        if not others:
            continue
        Xm = np.vstack(others)
        idx = rng.permutation(len(H))
        h_tr, h_te = H[idx[: len(H) // 2]], H[idx[len(H) // 2:]]
        X = np.vstack([h_tr, Xm])
        y = np.concatenate([np.zeros(len(h_tr)), np.ones(len(Xm))])
        Xs, h_te_s, tgt_s = fp.standardise(X, h_te, Xt)
        w = fp.fit_logreg(Xs, y)
        p_pos, p_neg = fp.predict(w, tgt_s), fp.predict(w, h_te_s)
        a = fp.auroc(p_pos, p_neg)
        t = fp.tpr_at_fpr(p_pos, p_neg, 0.05)

        scores = np.concatenate([p_pos, p_neg])
        n_pos = len(p_pos)
        nulls = np.empty(2000)
        for i in range(2000):
            sh = rng.permutation(len(scores))
            nulls[i] = fp.auroc(scores[sh[:n_pos]], scores[sh[n_pos:]])
        results[target] = {
            "auroc": a, "tpr_at_5fpr": t, "n": len(Xt),
            "p_value": float((nulls >= a).mean()),
            "null_p95": float(np.nanquantile(nulls, 0.95)),
        }
    return results


def single_feature_table(data: dict[str, list[dict]],
                         human_sets: tuple[str, ...] = fp.HUMAN) -> dict[str, dict]:
    """Every cross-observer feature alone, unfitted, against the pooled human reference.

    Published for the same reason docs/01 publishes single-feature AUROCs: so a reader can see
    which signals carry weight and which are decoration, and so a fitted gain cannot be
    mistaken for a discovered one.
    """
    human = [r for s in human_sets for r in data.get(s, [])]
    out = {}
    for feat in CONSENSUS_FEATS:
        neg = np.asarray([r[feat] for r in human if feat in r], dtype=np.float64)
        for m in fp.MACHINE:
            rows = data.get(m) or []
            pos = np.asarray([r[feat] for r in rows if feat in r], dtype=np.float64)
            if not len(pos) or not len(neg):
                continue
            a = fp.auroc(pos, neg)
            # Direction is not assumed. A feature that separates the wrong way is reported as
            # 1-AUROC with a flag, never quietly flipped into looking predictive.
            out[f"{feat}|{m}"] = {"auroc": a, "inverted": bool(a < 0.5),
                                  "auroc_oriented": max(a, 1.0 - a)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observers", default="qwen,llama",
                    help="comma-separated names from " + ",".join(OBSERVERS))
    ap.add_argument("--per-set", type=int, default=30)
    ap.add_argument("--pilot", type=int, default=0,
                    help="run only claude_opus + humans, this many docs per set")
    ap.add_argument("--budget", type=float, default=6000.0, help="hard neuron ceiling")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="consensus_probe")
    ap.add_argument("--normalise", action="store_true",
                    help="fold curly punctuation to ASCII in BOTH classes before scoring "
                         "(the typography control -- see score_corpus)")
    ap.add_argument("--drop", default="",
                    help="comma-separated consensus features to exclude, for ablations")
    args = ap.parse_args()

    observers = [o.strip() for o in args.observers.split(",") if o.strip()]
    for o in observers:
        if o not in OBSERVERS:
            print(f"unknown observer {o!r}; choose from {list(OBSERVERS)}")
            return 1
    if len(observers) < 2:
        print("a consensus needs at least two observers")
        return 1

    per_set = args.pilot or args.per_set
    sets = bench.HUMAN_SETS + bench.MACHINE_SETS
    if args.pilot:
        # The pilot spends as little as possible on the only question that matters: is there
        # anything at the frontier? If Opus does not move, nothing else needs paying for.
        sets = bench.HUMAN_SETS + [s for s in bench.MACHINE_SETS if s[0] == "claude_opus"]

    print(f"observers: {', '.join(OBSERVERS[o][0] for o in observers)}")
    print(f"documents: {per_set} per set" + ("  [PILOT: opus only]" if args.pilot else ""))
    print()

    corpus: dict[str, list[dict]] = {}
    for name, path, model in sets:
        rows = bench.load(path, model, per_set)
        if rows:
            corpus[name] = rows
    for k, v in corpus.items():
        print(f"  {k:20s} {len(v):3d} docs")
    print()

    if args.normalise:
        print("  TYPOGRAPHY CONTROL ON: curly punctuation folded to ASCII in both classes\n")
    budget = Budget(args.budget)
    data, dropped = score_corpus(corpus, observers, budget, args.workers, args.normalise)
    if not data:
        return 1

    consensus = tuple(f for f in CONSENSUS_FEATS
                      if f not in {d.strip() for d in args.drop.split(",") if d.strip()})
    if len(consensus) != len(CONSENSUS_FEATS):
        print(f"  ABLATION: dropped {args.drop} -> {len(consensus)} consensus features\n")

    rng = np.random.default_rng(fp.SEED)

    print("=" * 92)
    print("SINGLE CROSS-OBSERVER FEATURES, UNFITTED  (AUROC vs pooled human reference)")
    print("=" * 92)
    singles = single_feature_table(data)
    machine_sets = [m for m in fp.MACHINE if data.get(m)]
    print(f"  {'feature':20s}" + "".join(f"{m[:14]:>16s}" for m in machine_sets))
    for feat in CONSENSUS_FEATS:
        cells = []
        for m in machine_sets:
            e = singles.get(f"{feat}|{m}")
            cells.append("" if not e else
                         f"{e['auroc']:.3f}{'*' if e['inverted'] else ' '}")
        print(f"  {feat:20s}" + "".join(f"{c:>16s}" for c in cells))
    print("  (* = separates in the INVERTED direction: machine scored lower, not higher)")
    print()

    arms = {
        "A_baseline_single_observer": fp.FEATS,
        "B_consensus_only": consensus,
        "C_both": tuple(fp.FEATS) + consensus,
    }
    report: dict[str, dict] = {}
    for label, feats in arms.items():
        print("=" * 92)
        print(f"ARM {label}   ({len(feats)} features)")
        print("=" * 92)
        print(f"  {'held-out generator':22s} {'n':>4s} {'AUROC':>8s} {'TPR@5%':>8s} "
              f"{'null p95':>10s} {'p':>8s}")
        res = arm(data, feats, np.random.default_rng(fp.SEED))
        for g, r in res.items():
            print(f"  {g:22s} {r['n']:4d} {r['auroc']:8.3f} {r['tpr_at_5fpr']:8.3f} "
                  f"{r['null_p95']:10.3f} {r['p_value']:8.4f}")
        report[label] = res
        print()

    payload = {
        "observers": [OBSERVERS[o][0] for o in observers],
        "per_set": per_set,
        "pilot": bool(args.pilot),
        "normalised_typography": bool(args.normalise),
        "dropped_features": args.drop,
        # Per-document rows, so every ablation and re-fit downstream costs zero neurons. Not
        # dumping these is what made the first run's ablations unaffordable.
        "rows": {k: v for k, v in data.items()},
        "neurons_spent": budget.spent,
        "budget": budget.cap,
        "n_docs": {k: len(v) for k, v in data.items()},
        "dropped": dropped,
        "single_features": singles,
        "arms": report,
    }
    path = ROOT / "artifacts" / f"{args.out}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"neurons spent this run: {budget.spent:.1f} of {budget.cap:.0f} budgeted")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

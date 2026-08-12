#!/usr/bin/env python
"""Split the modern-generator corpus into a training pool and two held-out sets.

    python scripts/split_modern.py

Training on model output changes what the detector can honestly claim, so the split is a
design decision rather than a convenience, and it is made here in one place instead of
inside the feature builder.

Four destinations, answering four progressively harder questions:

``modern_train``          essays the detector is fitted on. Answers nothing by itself.
``modern_holdout``        unseen essays from the SAME checkpoint. "Does it detect this
                          generator" -- the weakest question, because the fit has seen that
                          checkpoint's habits.
``modern_unseen``         every essay from a checkpoint withheld **entirely**. "Does it
                          generalise to a generator we did not train on."
``modern_unseen_family``  essays from a different model family, also withheld entirely, and
                          the closest thing here to next year's model.

Splitting only at random would leave the last two questions unasked, and they are the ones
this project already got wrong: a detector fitted on GPT-3.5 scored 0.96 in-domain and then
missed a Gemini essay completely -- 18 sentences, none flagged. A random split would have
hidden that, because every generator in the pool would have been in training.

Each set is reported separately and they must not be averaged. The three recall numbers will
differ, and the gap between the first and the last IS the finding: it measures how much of
the detector's skill is generator-specific memorisation rather than a grasp of machine prose.

The assignment is written to ``artifacts/modern_split.json`` so a rebuild reproduces it
exactly and so a reader can check that no essay appears on both sides.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl, write_jsonl  # noqa: E402

SRC = ROOT / "data" / "generated" / "modern_essays.jsonl"
GEN = ROOT / "data" / "generated"
ARTIFACT = ROOT / "artifacts" / "modern_split.json"

#: Held out whole: a checkpoint the fit never sees, kept back to answer "does this
#: generalise to a generator we did not train on".
#:
#: This was going to be gemini-3.6-flash, the newest model, which is the right choice on
#: principle. It is not the choice here, and the reason is worth stating rather than hiding:
#: the thinking-heavy checkpoints produce roughly one essay every four minutes against the
#: lite checkpoints' four a minute, and two of them exhaust their daily free quota outright.
#: After 45 minutes gemini-3.6-flash had produced 8 essays. A recall figure on 8 documents
#: has a 95% interval about 35 points wide -- wide enough to be consistent with almost any
#: claim, which makes it useless as evidence and dangerous as a headline.
#:
#: So the whole-model holdout is the largest checkpoint that is NOT in training, and the
#: thinking-heavy models are kept as a second, smaller held-out set of their own. Both are
#: reported. See UNSEEN_FAMILY.
UNSEEN_MODEL = "gemini-3.5-flash-lite"

#: Also held out, and the harder test: a different model family entirely. Small -- these are
#: the quota-limited checkpoints -- so the number carries a wide interval and is reported
#: with its n rather than as a rate on its own.
#:
#: Holding these out costs the training pool its only non-lite examples. That is deliberate.
#: A detector fitted on every family in the corpus can only ever be asked the easy question,
#: and the easy question is the one this project already answered wrongly: 0.96 in-domain,
#: then a complete miss on the first modern essay anyone actually pasted in.
UNSEEN_FAMILY = ["gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.6-flash"]

#: How many modern essays are allowed into training. The rest are held out.
#:
#: This is the most consequential number in the file and it is not a tuning knob, it is a
#: safety limit. Measured on the existing pool: 3,544 human sentences against 469 machine,
#: so the detector was fitted at a prior of 11.7% machine. A modern essay segments to 19.5
#: sentences (measured, not assumed), so pouring in every essay that is not the whole-model
#: holdout looks like this:
#:
#:     +80 essays   35% machine          +200 essays   54% machine
#:     +135 essays  47% machine          +400 essays   69% machine
#:
#: At +400 the training pool believes three documents in four are machine-written. A
#: classifier fitted on that prior does not become a better detector, it becomes a more
#: willing accuser, and the cost lands on exactly the group this project spends a whole
#: document apologising to: real students, disproportionately those writing in a second
#: language. The document operating point is chosen against a false-positive budget and
#: would absorb some of it, but the sentence threshold is chosen on precision and the
#: highlighting would light up on human prose.
#:
#: 135 puts the pool at roughly even. Everything above that is worth more as held-out
#: evidence than as training signal -- a 300-essay test set measures recall to about +/-3
#: points, which is tighter than any claim being made from it.
TRAIN_CAP = 135

#: Of the capped training candidates, none are wasted: whatever the cap excludes joins the
#: held-out set. This fraction only sets the floor when the corpus is small.
HOLDOUT_FRACTION = 0.35

SEED = 20260809


def collect() -> list:
    """Every generated essay, from the single file or from the per-model parts.

    Generation runs one process per model, because each model carries its own rate limit and
    a single round-robin process spends most of its time waiting on the slowest one -- 4
    essays a minute against 34. The parts are merged here rather than by the generator so
    that a partial run is still splittable and re-running either half is safe.
    """
    docs = list(read_jsonl(SRC)) if SRC.exists() else []
    for part in sorted((GEN / "parts").glob("*.jsonl")):
        docs.extend(read_jsonl(part))
    # The parts are written independently, so a hash collision across them is possible in
    # principle even though each process de-duplicates its own output.
    seen, unique = set(), []
    for d in docs:
        if d.sha256 in seen:
            continue
        seen.add(d.sha256)
        unique.append(d)
    if len(unique) != len(docs):
        print(f"dropped {len(docs) - len(unique)} duplicate essays across parts")
    return unique


def main() -> int:
    docs = collect()
    if not docs:
        raise SystemExit(
            "no essays to split -- run scripts/generate_modern.py first "
            f"(looked in {SRC.relative_to(ROOT)} and {(GEN / 'parts').relative_to(ROOT)}/)"
        )

    by_model: dict[str, list] = defaultdict(list)
    for d in docs:
        by_model[d.meta.get("model", d.source_id)].append(d)

    if UNSEEN_MODEL not in by_model:
        raise SystemExit(
            f"{UNSEEN_MODEL} is not in the corpus; the whole-model holdout would be empty. "
            f"present: {sorted(by_model)}"
        )

    rng = random.Random(SEED)
    train, holdout = [], []
    unseen = sorted(by_model[UNSEEN_MODEL], key=lambda d: d.id)
    family = sorted((d for m in UNSEEN_FAMILY for d in by_model.get(m, [])),
                    key=lambda d: d.id)

    # Take the training pool as evenly across the contributing models as their sizes allow,
    # rather than filling it from whichever model generated fastest. The lite checkpoints
    # produce an essay roughly eight times quicker than the thinking-heavy ones and are the
    # only ones with spare daily quota, so a first-come cap would fit the detector almost
    # entirely on two cheap models and then claim to cover the generation.
    #
    # The pools are very uneven for that reason, so the allocation is a smallest-first fill:
    # a model that cannot supply its equal share gives up the remainder to the models that
    # can, instead of the training pool silently coming in under TRAIN_CAP.
    withheld = {UNSEEN_MODEL, *UNSEEN_FAMILY}
    contributing = [m for m in sorted(by_model) if m not in withheld]
    if not contributing:
        raise SystemExit("every model is withheld; nothing left to train on")
    pools = {}
    for model in contributing:
        pool = sorted(by_model[model], key=lambda d: d.id)
        rng.shuffle(pool)
        pools[model] = pool

    budget, remaining = TRAIN_CAP, list(contributing)
    allocation: dict[str, int] = {}
    for model in sorted(remaining, key=lambda m: len(pools[m])):
        share = budget // max(len(remaining), 1)
        # Never take a model's whole pool: each contributing model must also be represented
        # in the held-out set, or "unseen essays from a seen generator" is unmeasurable
        # for it.
        allocation[model] = min(share, int(len(pools[model]) * (1 - HOLDOUT_FRACTION)))
        budget -= allocation[model]
        remaining.remove(model)

    for model in contributing:
        keep = allocation[model]
        train.extend(pools[model][:keep])
        holdout.extend(pools[model][keep:])

    # A duplicate across the boundary would inflate every number that follows. The generator
    # already discards repeated text by hash, so this should be impossible -- which is
    # exactly why it is asserted rather than assumed.
    parts = {"train": train, "holdout": holdout, "unseen": unseen, "family": family}
    for a in parts:
        for b in parts:
            if a >= b:
                continue
            assert not ({d.id for d in parts[a]} & {d.id for d in parts[b]}), \
                f"an essay is in both {a} and {b}"
            assert not ({d.sha256 for d in parts[a]} & {d.sha256 for d in parts[b]}), \
                f"identical text in both {a} and {b}"

    for name, part in (("modern_train", train), ("modern_holdout", holdout),
                       ("modern_unseen", unseen), ("modern_unseen_family", family)):
        write_jsonl(sorted(part, key=lambda d: d.id), GEN / f"{name}.jsonl")
        counts = Counter(d.meta.get("model") for d in part)
        print(f"{name:16s} {len(part):4d} essays  {dict(counts)}")

    # The prior the detector will be fitted at, stated rather than discovered later. About
    # 19.5 sentences per modern essay, measured; the pool is 3,544 human / 469 machine.
    machine = 469 + 19.5 * len(train)
    share = machine / (3544 + machine)
    print(f"\ntraining pool prior: ~{100 * share:.0f}% machine "
          f"(was 11.7%; TRAIN_CAP holds this near even on purpose)")
    if share > 0.65:
        print("  ! above 0.65 the fit starts accusing human prose -- lower TRAIN_CAP")

    ARTIFACT.write_text(json.dumps({
        "seed": SEED,
        "unseenModel": UNSEEN_MODEL,
        "unseenFamily": UNSEEN_FAMILY,
        "trainCap": TRAIN_CAP,
        "holdoutFraction": HOLDOUT_FRACTION,
        "approxMachineShareAfter": round(share, 3),
        "counts": {"train": len(train), "holdout": len(holdout), "unseen": len(unseen),
                   "family": len(family)},
        "trainIds": sorted(d.id for d in train),
        "holdoutIds": sorted(d.id for d in holdout),
        "unseenIds": sorted(d.id for d in unseen),
        "familyIds": sorted(d.id for d in family),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {ARTIFACT.relative_to(ROOT)}")
    print("next: python scripts/build_features.py --sets train modern_holdout modern_unseen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

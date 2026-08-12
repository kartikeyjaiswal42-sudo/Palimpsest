#!/usr/bin/env python
"""Generate admissions essays from modern models, for training and for measurement.

    python scripts/generate_modern.py                     # run until the corpus is full
    python scripts/generate_modern.py --target 100
    python scripts/generate_modern.py --models gemini-3.6-flash

Every real machine essay this detector was originally fitted on is GPT-3.5 output from 2022,
and it showed: a Gemini-written essay pasted into the interface scored 18 sentences, none
flagged, a median sentence probability of 0.019 and a machine share of 0.0%. Not a near
miss and not a threshold that wanted nudging -- a generator the features had never seen.
This script builds the corpus that fixes it.

Three things make the corpus worth fitting on rather than just a pile of model output:

**The prompts are the GPT-3.5 set's own prompts,** read out of
``data/raw/liang_college_gpt3.jsonl`` rather than written here, so a comparison between the
old machine set and the new one varies the generator and nothing else.

**Subject is steered, style never is.** See ``SUBJECTS``. The first machine corpus in this
project's history was composed to *sound* machine-generated and the detector learned the
instructions instead of the authorship (docs/04-failures.md #1). Nothing here describes prose.

**Length is drawn to match the human class, not the old machine class.** See ``WHY_LENGTH``.

The essays are NOT all held out. ``scripts/split_modern.py`` divides them three ways -- a
training pool, a held-out set from the same models, and one model withheld entirely -- and
that script is where the honesty of every number downstream is decided. Run it next.

The key is read from the environment or from a local ``.env``; it is never written into the
corpus. What lands on disk is the essay, the model that wrote it, the prompt and subject it
answered and a SHA-256, the same provenance record every other source in ``data/`` carries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import Document, read_jsonl, write_jsonl  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: Three checkpoints spanning the Gemini 3 series, oldest to newest.
#:
#: The intended design was one model per year -- a 2024, a 2025 and a 2026 checkpoint -- so
#: the result would be a trend against generator age rather than a single number. Every
#: Gemini 2.x model and every `pro` model on the key available here answers 429: the free
#: tier's daily quota for those families is spent. Re-run with `--models` on a key that has
#: them to recover the full series; the sets below are what was actually reachable, and the
#: headline comparison does not depend on them, because the trained-on generator (GPT-3.5,
#: 2022) is already four years older than anything here.
DEFAULT_MODELS = [
    ("gemini-3-flash-preview", "3.0"),
    ("gemini-3.1-flash-lite", "3.1"),
    ("gemini-3.5-flash", "3.5"),
    ("gemini-3.5-flash-lite", "3.5-lite"),
    ("gemini-3.6-flash", "3.6"),
]

#: Concrete subjects, drawn one per essay alongside the prompt.
#:
#: This is the one place the script deliberately steers the model, and it steers SUBJECT,
#: never style. Four prompts repeated five hundred times do not give five hundred essays;
#: they give five hundred variations on robotics club, the soup kitchen and the community
#: centre, which is what the first eighteen generations produced. A detector fitted on that
#: learns the topic distribution of one afternoon of sampling and calls it authorship --
#: precisely the mistake in docs/04-failures.md #1, arriving by a different road.
#:
#: What must NOT appear here is any instruction about voice: no "write naturally", no "vary
#: sentence length", no "sound like a teenager". The moment the prompt describes prose, the
#: detector starts measuring the prompt.
SUBJECTS = [
    "a summer job gutting fish at a wholesale market",
    "learning Carnatic violin from a strict grandmother",
    "rebuilding the brakes on a scooter with no manual",
    "running the scoreboard at a small-town baseball field",
    "translating for a parent at hospital appointments",
    "keeping bees on an apartment rooftop",
    "a failed attempt to start a samosa stall outside school",
    "competitive crossword construction",
    "volunteering in a municipal seed library",
    "repairing sewing machines for a tailoring co-operative",
    "a year of insomnia spent reading train timetables",
    "coaching a younger sibling through a stammer",
    "restoring a flooded darkroom",
    "moderating an online forum for amateur astronomers",
    "working the night shift at a 24-hour pharmacy",
    "growing chillies on a balcony in poor soil",
    "learning to weld from a neighbour",
    "cataloguing a grandfather's cassette collection",
    "a school debate lost on a technicality",
    "building a rain gauge network for a farming village",
    "teaching swimming to adults who feared water",
    "an apprenticeship with a sign painter",
    "mapping potholes on a daily cycle route",
    "cooking for a hundred at a temple kitchen",
    "learning chess from an uncle who never let anyone win",
    "salvaging usable parts from discarded printers",
    "a long correspondence with a pen pal in another country",
    "keeping the accounts for a family kirana shop",
    "recording birdsong before a wetland was drained",
    "learning to read Braille alongside a blind classmate",
    "reviving a defunct school literary magazine",
    "a monsoon that flooded the ground floor three years running",
    "assisting at a veterinary clinic during calving season",
    "building a low-cost prosthetic hand for a school project",
    "learning tabla rhythms by counting on a bus commute",
    "sorting donations at a disaster relief warehouse",
    "an obsession with the mathematics of origami",
    "running a repair cafe for broken headphones",
    "documenting a dying dialect spoken by four relatives",
    "a first attempt at brewing kombucha that went badly wrong",
]

#: Asking for an essay produces an essay with a title, section headings and a bulleted
#: "Reflection". None of that survives contact with a real application form, and all of it
#: would be detected as formatting rather than as prose -- which is the mistake truepen made
#: and had to unpick. So the instruction constrains the SHAPE and says nothing about style:
#: no "write naturally", no "vary your sentence length", no "sound human". Steering the style
#: is how the first machine corpus in this project ended up measuring our own instructions
#: instead of the model (docs/04-failures.md #1), and it is not repeated here.
SHAPE = (
    "Write it as continuous prose in {n} paragraphs, about {words} words in total. "
    "Output only the essay itself: no title, no headings, no markdown, no closing note."
)

#: Why these essays are 400-500 words and not the 261 the GPT-3.5 set averages.
#:
#: Measured across the corpus: the human essays run to a median of 632 words
#: (liang_college_human) and 715 (jhu), while every machine essay already in the pool runs
#: to 261. That is a 2.5x gap that lines up exactly with the label, and it is the reason
#: `log_sentences` had to be deleted from the document model -- it had taken the largest
#: weight in the whole fit at -3.09, "shorter means machine", learned entirely from how the
#: two halves of the corpus were collected.
#:
#: Generating a new machine set at the old machine length would have rebuilt that artifact
#: at five times the scale and handed it to the sentence model as well. Drawing the length
#: from [400, 500] puts the machine class inside the range a real applicant writes in --
#: the Common App caps essays at 650 words -- so length stops separating the classes and
#: the fit has to find something else. Per-essay rather than fixed, because a corpus where
#: every machine document is the same length is its own giveaway.
WHY_LENGTH = "human median 632-715 words vs machine 261; drawn per essay to close the gap"


def load_prompts() -> list[str]:
    """The prompts the held-out GPT-3.5 essays answered, in corpus order."""
    path = ROOT / "data" / "raw" / "liang_college_gpt3.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} -- run scripts/fetch_corpus.py first. "
            "The prompts are taken from that file on purpose, so the comparison holds "
            "the task fixed and varies only the generator."
        )
    seen, out = set(), []
    for doc in read_jsonl(path):
        p = (doc.meta or {}).get("prompt", "").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    if not out:
        raise SystemExit("no prompts found in liang_college_gpt3.jsonl")
    return out


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    for env in (ROOT / ".env", ROOT.parent / "job-search-automation" / ".env"):
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit(
        "no GEMINI_API_KEY in the environment or in a .env file. This script is the only "
        "part of the project that needs a network or a key; everything else runs offline."
    )


def generate(client: httpx.Client, model: str, key: str, prompt: str, words: int) -> str | None:
    """One essay, or None if the model declined or the call failed.

    A refusal and a transport error are both returned as None deliberately: neither is an
    essay, and silently substituting a retry of a different prompt would break the pairing
    with the GPT-3.5 set that makes this a controlled comparison.
    """
    body = {
        "contents": [{"parts": [{"text": prompt + "\n\n" + SHAPE.format(n=5, words=words)}]}],
        # 16384, not the 4096 this started with. Gemini 3 models spend a large and invisible
        # share of the output budget on thinking tokens -- measured here at 1,055 to 1,949 on
        # a single 450-word essay -- and the essay is truncated to pay for them. The symptom
        # is not an error: the call returns 200 with finishReason STOP and a 143-word essay,
        # so a corpus generated at the old ceiling is quietly a third of the requested length
        # and skewed toward whichever models think least.
        #
        # `thinkingConfig: {thinkingBudget: 0}` would be the tidier fix and produces slightly
        # longer essays on the models that accept it, but gemini-3.5-flash-lite and
        # gemini-3.6-flash reject it with a 400, so it cannot be used across the set.
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 16384},
    }
    # A 429 here is usually the per-MINUTE rate limit on the free tier, not the daily one,
    # and it clears by waiting. Treating the two as the same thing silently halved a run:
    # 14 requested, 6 written, no error visible except a smaller corpus at the end.
    for attempt in range(4):
        try:
            r = client.post(API.format(model=model), params={"key": key}, json=body,
                            timeout=120)
        except httpx.HTTPError as err:
            print(f"    ! transport error: {type(err).__name__}")
            return None
        # 429 is the per-minute rate limit far more often than the daily one, and 503 is
        # "this model is busy". Both clear by waiting; neither means stop. Treating them as
        # fatal silently halved a run once -- 14 requested, 6 written, no visible error.
        if r.status_code in (429, 503) and attempt < 3:
            wait = 20 * (attempt + 1)
            print(f"    . HTTP {r.status_code}, waiting {wait}s")
            time.sleep(wait)
            continue
        break
    if r.status_code != 200:
        print(f"    ! HTTP {r.status_code}: {r.text[:160]}")
        return None
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        print(f"    ! no candidate returned ({str(data.get('promptFeedback'))[:120]})")
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="total essays wanted on disk")
    ap.add_argument("--models", nargs="*", help="model ids (default: the reachable series)")
    ap.add_argument("--words-min", type=int, default=400)
    ap.add_argument("--words-max", type=int, default=500,
                    help="length is drawn per essay from [min, max]; see WHY_LENGTH")
    ap.add_argument("--out", default="data/generated/modern_essays.jsonl")
    ap.add_argument("--no-subjects", action="store_true",
                    help="answer the bare prompt, with no subject steering -- builds the "
                         "topic-clean control set (see the note below)")
    args = ap.parse_args()

    # Why a control set exists at all.
    #
    # Steering subject buys diversity, and it buys a confound with it: every modern essay in
    # the corpus is about one of 40 subjects chosen here, while the human essays are about
    # whatever their authors chose. "Machine" and "these 40 topics" are then perfectly
    # correlated, and a detector could score well on the modern sets by recognising
    # beekeeping rather than by recognising machine prose.
    #
    # The direction of that bias is probably conservative -- unusual subjects raise
    # corpus_surprisal_mean, whose weight points toward HUMAN, so odd topics should make
    # these essays harder to catch. Probably is not measured. `--no-subjects` generates
    # essays from the bare original prompts, and recall on that set is the number that says
    # whether the gain is about authorship or about topic.

    models = ([(m, "") for m in args.models] if args.models else DEFAULT_MODELS)
    prompts = load_prompts()
    key = api_key()

    # Resume rather than restart. Free-tier quota is spent per call and not refunded, and a
    # 500-essay run cannot complete inside one rate-limit window, so this script is expected
    # to be run repeatedly until the corpus is full.
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = list(read_jsonl(out)) if out.exists() else []
    seen = {d.sha256 for d in docs}
    start = len(docs)
    print(f"{len(prompts)} prompts x {len(SUBJECTS)} subjects | {len(models)} models")
    print(f"length {args.words_min}-{args.words_max} words -- {WHY_LENGTH}")
    print(f"on disk: {start} | target: {args.target}")

    rng = random.Random(20260809 + start)
    stalled: set[str] = set()
    cooled = False
    with httpx.Client() as client:
        while len(docs) < args.target:
            live = [m for m in models if m[0] not in stalled]
            if not live:
                # Every model has failed its retries. That is usually one bad minute rather
                # than an exhausted key, so give them one long cooling-off period and try the
                # whole set again before concluding the run is over.
                if cooled:
                    print("\nevery model is still failing after a cool-off. Stopping; "
                          "re-run this script later to continue from where it left off.")
                    break
                print("\nevery model is failing. Cooling off for 120s, then trying again.")
                time.sleep(120)
                stalled.clear()
                cooled = True
                continue
            for model, era in live:
                if len(docs) >= args.target:
                    break
                prompt = rng.choice(prompts)
                subject = None if args.no_subjects else rng.choice(SUBJECTS)
                words = rng.randint(args.words_min, args.words_max)
                ask = prompt if subject is None else f"{prompt}\n\nWrite about {subject}."
                text = generate(client, model, key, ask, words)
                if text is None:
                    stalled.add(model)
                    print(f"    ! {model} stalled, dropping it from this run")
                    continue
                # The length band is a property of the corpus, not a hint to the model, so it
                # is enforced here rather than hoped for. A truncated generation is still a
                # fluent essay and would be invisible in the corpus except as a short one --
                # and short machine documents against long human ones is the exact artifact
                # this length choice exists to remove.
                n_words = len(text.split())
                if not (args.words_min * 0.8 <= n_words <= args.words_max * 1.3):
                    print(f"    . {model} returned {n_words} words, outside the band -- discarded")
                    continue
                doc = Document(
                    id=f"modern_{model}:{len(docs):04d}",
                    source_id=f"modern_{model}",
                    text=text,
                    authorship="machine",
                    role="unseen-generator",
                    meta={"prompt": prompt, "subject": subject, "model": model, "era": era,
                          "generatedAt": time.strftime("%Y-%m-%d")},
                )
                # Identical text from two samples is one essay's worth of signal counted
                # twice, and at temperature 1.0 with a repeated (prompt, subject) pair it
                # does happen. Duplicates would land on both sides of a train/test split.
                if doc.sha256 in seen:
                    print("    . duplicate, discarded")
                    continue
                seen.add(doc.sha256)
                cooled = False
                docs.append(doc)
                if len(docs) % 10 == 0 or len(docs) - start < 5:
                    print(f"  [{len(docs):4d}] {model:24s} {doc.n_words:4d}w  {text[:52]!r}")
                write_jsonl(docs, out)
                time.sleep(0.3)

    if not docs:
        raise SystemExit("no essays generated -- nothing written")
    print(f"\nadded {len(docs) - start} this run")
    words = sorted(d.n_words for d in docs)
    print(f"\nwrote {len(docs)} essays -> {out.relative_to(ROOT)}")
    print(f"  median {words[len(words) // 2]} words (GPT-3.5 held-out set: 261)")
    print("\nnext: python scripts/build_features.py --sets modern")
    print("      python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

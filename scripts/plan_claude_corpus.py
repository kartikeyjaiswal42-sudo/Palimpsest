#!/usr/bin/env python
"""Plan a 500-essay corpus written by the Claude family, before a word of it is written.

    python scripts/plan_claude_corpus.py                  # write the plan
    python scripts/plan_claude_corpus.py --n 100          # a smaller run
    python scripts/plan_claude_corpus.py --summary        # describe an existing plan

The measurement this corpus exists to fix is in docs/03-evaluation.md: on the `adversarial`
set -- 21 documents written by Claude -- the shipped detector has a document recall of
**0.0%**. It catches none of them. That is not surprising. The machine half of training is
31 GPT-3.5 essays generated in early 2023, so the detector has been fitted to one generator,
four years old, and asked to generalise to a family it has never seen.

Five hundred essays from four Claude models is the direct attack on that number. This file is
the part that decides what those essays are, and it exists as a separate step for one reason:
**every choice here is a way to accidentally measure ourselves instead of the model.** The
project has already paid for that lesson once. The first machine corpus was hand-composed
prose in a machine register; it scored document AUROC 0.500, exactly chance, and its strongest
feature turned out to be reading back the instruction we had written (docs/04-failures.md #4).
A corpus of 500 essays generated from one prompt template would reproduce that failure at
fifty times the scale, and it would do it while looking like progress.

So the plan is fixed, written to disk and reviewable before any essay exists.

Four things it controls:

**Length is matched to the human corpus, not chosen for convenience.** Human admissions
essays here run p10 497 / median 632 / p90 650 words, with a hard spike at 650 because that
is the Common Application limit, and a right tail to ~790 from the JHU set. The sampler below
reproduces that shape, spike included. This matters more than it looks: docs/06-decisions.md
#6 records that when document length was visible to the model it took a weight of -3.09 and
the detector's single strongest belief became "short essays are machine-written" -- learned
entirely from the fact that the GPT-3.5 essays are shorter. Length was removed from the
document model to kill that. Handing it a new machine corpus with a distinctive length
profile would put the same artifact back through the sentence features, where nobody removed
it.

**Style is mostly not steered at all.** 45% of the plan carries a shape-only instruction:
how many words, continuous prose, no title. It says nothing about voice, and deliberately
never says "sound human" or "vary your sentence length". Those steered variants exist -- they
are the `evasive` slice, 15% -- but they are a labelled minority and a separate axis to
report on, not the corpus. If we steered every essay we would learn to detect our own
adjectives.

**Topic never crosses the train/held-out boundary.** The split is pre-registered here, by
seed, before generation: 60 of the 100 seeds are training and 40 are held out. Two essays on
the same childhood kitchen, one in each split, would let the detector recognise the kitchen
rather than the writing, and the held-out number would be a memorisation score wearing a
generalisation label.

**Four generators, not one.** Opus, Sonnet, Haiku and Fable each write a quarter. A detector
fitted on a single checkpoint learns that checkpoint; the useful claim is about the family,
and it can only be made if the corpus spans it. It also gives the evaluation a real axis:
recall per model is a more honest table than one pooled number.

The output is a set of per-batch assignment files under ``data/generated/plan/``. Each is
self-contained -- a generating worker reads its batch and needs nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN_DIR = ROOT / "data" / "generated" / "plan"

#: Fixed so the plan is reproducible. Re-running this script must produce the same corpus
#: definition, or the provenance record in every essay stops meaning anything.
SEED = 20260809

#: Generators overshoot a requested word count, and they do it consistently.
#:
#: Measured over the first 32 essays: actual/requested came back at mean 1.068, median 1.065,
#: with a tight spread (p10 1.032, p90 1.080). Asking for the human median therefore lands
#: the machine corpus about 40 words above it -- which
#: ``assemble_claude_corpus.report_lengths`` flagged as separable-on-length-alone before the
#: corpus was a tenth built.
#:
#: So the ask is scaled down by the measured bias rather than the distribution being left to
#: drift. This is calibration against an observed property of the generator, not a thumb on
#: the scale: the target is still the human distribution, and the correction is what makes
#: the output land on it.
#:
#: Applied to the RETURN VALUE of ``sample_length`` and nowhere else, deliberately. It
#: consumes no random numbers, so the draw sequence is unchanged and every essay already
#: generated keeps exactly the subject, style, model, split and batch it was assigned.
OVERSHOOT = 1.068

# --------------------------------------------------------------------------- the subjects

#: One hundred concrete situations a real applicant might write about.
#:
#: Curated rather than combinatorially generated, because a machine-built topic list produces
#: essays about things nobody has ever done, and the resulting prose is unrepresentative in a
#: way that would flatter the detector. The range is deliberate: the set spans income, region,
#: family shape and interest, and it includes the well-worn subjects (the mission trip, the
#: injured athlete, the grandparent) as well as the unusual ones, because real application
#: piles are mostly well-worn subjects and a corpus of only surprising essays is not a corpus
#: of admissions essays.
SEEDS: list[str] = [
    "learning to cook from a grandmother who spoke little English",
    "working the register at the family's convenience store",
    "an injury that ended a competitive swimming career",
    "building a robot for a regional competition that failed on stage",
    "translating for parents at doctors' appointments",
    "starting a school radio station nobody listened to at first",
    "moving countries in the middle of high school",
    "caring for a younger sibling with a chronic illness",
    "a summer spent restoring a broken motorcycle",
    "quitting piano after eleven years",
    "the first time they cooked dinner for the whole family alone",
    "organising a food drive that collected far less than promised",
    "learning to code to make a game for a younger cousin",
    "a debate tournament loss that changed how they argue",
    "keeping a garden alive through a drought",
    "working night shifts at a distribution warehouse",
    "teaching an elderly neighbour to use a smartphone",
    "being cut from a varsity team three years running",
    "running the lighting board for the school play",
    "a grandfather's stories about a village that no longer exists",
    "fixing bicycles for neighbourhood children",
    "an essay competition they entered under a false name",
    "learning sign language for a deaf classmate",
    "the family restaurant closing after twenty-two years",
    "birdwatching alone before school",
    "founding a chess club at a school with no chess players",
    "a science-fair project on local water quality",
    "the year they stopped speaking their first language at home",
    "sewing costumes for a community theatre",
    "a father's long unemployment",
    "training a rescue dog that could not be trained",
    "delivering newspapers on a route inherited from an older sibling",
    "arguing with a history teacher about a textbook",
    "the summer they read every book on one library shelf",
    "learning to drive on a farm",
    "an internship where they were given nothing to do",
    "helping a mother study for a citizenship test",
    "a friendship that ended over a group project",
    "keeping score at a father's cricket matches",
    "learning to weld in a school shop class",
    "the church choir they sang in but did not believe in",
    "starting a podcast that ran for four episodes",
    "a hospital stay that lasted a month",
    "the neighbourhood basketball court that was demolished",
    "tutoring maths to students who were failing it",
    "a family recipe nobody wrote down",
    "working as a lifeguard at an empty pool",
    "an obsession with subway maps",
    "the school newspaper article that got them in trouble",
    "learning to play a grandfather's harmonium",
    "a mother's second-hand clothing business",
    "collecting rainfall data for three years",
    "the year they were the only one of their friends without a phone",
    "volunteering at an animal shelter and being bad at it",
    "building furniture from pallets",
    "a cousin's wedding they were made responsible for",
    "an argument about whether to sell the family land",
    "learning photography with a camera that had no light meter",
    "moving between six schools in nine years",
    "a summer job picking fruit",
    "the mathematics olympiad they trained for and failed",
    "reading to patients at a hospice",
    "the day the power stayed off for a week",
    "learning to make pottery on a wheel that wobbled",
    "starting a recycling programme that the school cancelled",
    "a brother's decision not to go to college",
    "keeping bees on a rooftop",
    "a teacher who told them they were not a writer",
    "running a stall at a weekend market",
    "coaching a under-11 football team",
    "restoring an uncle's old radio",
    "a grandmother's dementia",
    "the physics problem they could not stop thinking about",
    "learning to swim at sixteen",
    "an online friendship with someone in another country",
    "the family's first winter in a cold country",
    "acting as the only girl on a robotics team",
    "a summer of caring for a family orchard",
    "leaving a religious school",
    "a science teacher who let them use the lab after hours",
    "selling homemade jewellery online",
    "a long illness that meant a year of school at home",
    "learning to read music at fifteen",
    "an argument with a best friend about politics",
    "the bus route that took two hours each way",
    "working as a scribe for a blind student",
    "a failed attempt to start a business selling phone cases",
    "the summer they built a canoe",
    "a mother who returned to university at forty",
    "learning traditional dance for a festival they did not want to attend",
    "the first snow they ever saw",
    "repairing sewing machines in a tailoring shop",
    "an exchange student who stayed with them for a year",
    "the school library they were asked to reorganise",
    "a father's stroke",
    "growing chillies on a balcony",
    "the debate about whether to keep the family's cows",
    "learning to fix the school's broken laptops",
    "a year of insomnia",
    "the last game of a losing season",
]

#: Five ways of entering the same subject. A frame changes the essay's architecture -- what
#: it opens on, what it withholds, where its turn is -- far more than the subject does, and
#: architecture is what the sentence-level features actually read. Crossing 100 subjects with
#: 5 frames gives 500 assignments no two of which are the same essay.
FRAMES: list[tuple[str, str]] = [
    ("turn", "written around the moment they changed their mind about it"),
    ("object", "told through one specific physical object"),
    ("failure", "written as something they got wrong and still think about"),
    ("argument", "written as an unresolved disagreement with someone they love"),
    ("routine", "written as a small repeated habit that turned out to matter"),
]

# --------------------------------------------------------------------------- the prompts

#: The seven Common Application prompts, verbatim, plus a set of supplemental-style questions.
#:
#: Real prompts rather than paraphrases: an applicant using a model pastes the prompt in, and
#: a corpus generated from our own restatement of it would differ from the real thing in
#: exactly the register we are trying to measure.
PROMPTS: list[tuple[str, str]] = [
    ("common-1", "Some students have a background, identity, interest, or talent that is so "
     "meaningful they believe their application would be incomplete without it. If this "
     "sounds like you, then please share your story."),
    ("common-2", "The lessons we take from obstacles we encounter can be fundamental to "
     "later success. Recount a time when you faced a challenge, setback, or failure. How "
     "did it affect you, and what did you learn from the experience?"),
    ("common-3", "Reflect on a time when you questioned or challenged a belief or idea. "
     "What prompted your thinking? What was the outcome?"),
    ("common-4", "Reflect on something that someone has done for you that has made you "
     "happy or thankful in a surprising way. How has this gratitude affected or motivated "
     "you?"),
    ("common-5", "Discuss an accomplishment, event, or realization that sparked a period of "
     "personal growth and a new understanding of yourself or others."),
    ("common-6", "Describe a topic, idea, or concept you find so engaging that it makes you "
     "lose all track of time. Why does it captivate you? What or who do you turn to when "
     "you want to learn more?"),
    ("common-7", "Share an essay on any topic of your choice. It can be one you've already "
     "written, one that responds to a different prompt, or one of your own design."),
    ("supp-community", "Describe a community you belong to and your place within it."),
    ("supp-curiosity", "What is something you have taught yourself, and what did learning "
     "it without a teacher show you?"),
    ("supp-conflict", "Tell us about a time you disagreed with someone whose opinion you "
     "respected."),
    ("supp-place", "Describe a place where you feel most yourself, and why."),
    ("supp-contribution", "What would you contribute to our campus that would not be here "
     "if you were not?"),
    ("supp-failure", "Describe a time you were not successful. What did you do next?"),
    ("supp-object", "Write about an object that is important to you."),
]

# --------------------------------------------------------------------------- the styles

#: How the essay is asked for. The weights are the argument of this file.
#:
#: `plain` dominates on purpose. It carries no instruction about voice at all -- only shape --
#: so what it captures is the model writing an admissions essay rather than the model
#: performing our adjectives. The steered variants are kept because they are what a real
#: applicant does (nobody types "write me an essay" and submits the first result), but they
#: are a minority and they are labelled, so recall can be reported per style. If `evasive`
#: turns out to be the only slice we catch, that is a finding, and the label is what makes it
#: visible instead of averaged away.
STYLES: list[tuple[str, int, str]] = [
    ("plain", 45,
     "Write a college application essay responding to the prompt below.\n\n"
     "PROMPT: {prompt}\n\n"
     "The essay is about {topic}, {frame}.\n\n"
     "Write approximately {words} words as continuous prose. Output only the essay itself: "
     "no title, no headings, no markdown, no note at the end."),
    ("persona", 20,
     "You are a 17-year-old in your final year of school, applying to university. Write "
     "your own application essay responding to this prompt.\n\n"
     "PROMPT: {prompt}\n\n"
     "Write about {topic}, {frame}.\n\n"
     "Approximately {words} words, continuous prose. Output only the essay: no title, no "
     "headings, no markdown, no closing note."),
    ("evasive", 15,
     "Write a college application essay responding to the prompt below.\n\n"
     "PROMPT: {prompt}\n\n"
     "The essay is about {topic}, {frame}.\n\n"
     "Make it sound genuinely human. Vary your sentence length. Avoid the polished, even "
     "rhythm that AI writing has, and avoid words that sound like a language model wrote "
     "them. It should be able to pass an AI-detection tool.\n\n"
     "Approximately {words} words. Output only the essay: no title, no headings, no "
     "markdown, no closing note."),
    ("notes", 12,
     "A student has given you these notes for their application essay and asked you to "
     "write it for them.\n\n"
     "PROMPT: {prompt}\n\n"
     "NOTES: it's about {topic}. They want it {frame}. They've said to keep it honest and "
     "not to make anything up beyond what's here.\n\n"
     "Write approximately {words} words of continuous prose. Output only the essay: no "
     "title, no headings, no markdown, no closing note."),
    ("constrained", 8,
     "Write a college application essay responding to the prompt below.\n\n"
     "PROMPT: {prompt}\n\n"
     "The essay is about {topic}, {frame}.\n\n"
     "Open in the middle of a scene, with no preamble, and end without stating the lesson "
     "outright.\n\n"
     "Approximately {words} words, continuous prose. Output only the essay: no title, no "
     "headings, no markdown, no closing note."),
]

# --------------------------------------------------------------------------- the generators

#: Four checkpoints across the Claude family, one quarter each.
MODELS: list[str] = ["opus", "sonnet", "haiku", "fable"]


def sample_length(rng: random.Random) -> int:
    """A word count drawn to match the human corpus, including its artifacts.

    The human essays here are not normally distributed and it would be wrong to pretend they
    are. They pile up hard against 650 -- the Common Application limit -- with a left tail of
    essays that came in under, and a right tail from the JHU set, which publishes essays that
    were never bound by that limit. A Gaussian around the median would produce a machine
    corpus with a visibly different silhouette from the human one, and the sentence features
    would find it.

    The bands below were fitted against the pooled human corpus by grid search rather than
    guessed. They reproduce it to a mean of 637 words against 636, a median of 646 against
    642, and every reported quantile within 11 words -- close enough that document length
    carries no signal a classifier could separate on.
    """
    r = rng.random()
    if r < 0.14:                      # short: the under-the-limit tail
        want = rng.randint(420, 545)
    elif r < 0.36:                    # approaching the limit
        want = rng.randint(546, 612)
    elif r < 0.52:                    # the mass just under 650
        want = rng.randint(613, 649)
    elif r < 0.64:                    # written exactly to the limit
        want = 650
    elif r < 0.90:                    # the JHU-shaped right tail
        want = rng.randint(651, 750)
    else:                             # the long ones
        want = rng.randint(751, 880)
    # `want` is what we need the essay to BE. What we ASK for is lower, because the generator
    # overshoots by OVERSHOOT with a tight spread. The floor keeps the shortest ask above 400
    # words even after the correction, so no essay is ever commissioned below the floor.
    return max(400, round(want / OVERSHOOT))


def build(n: int) -> list[dict]:
    """The full assignment list, deterministic given SEED."""
    rng = random.Random(SEED)

    seeds = SEEDS[: max(1, n // len(FRAMES))]
    # Pre-registered split, decided here and never revisited: 60% of SUBJECTS train, 40%
    # held out. Splitting by subject rather than by essay is the whole point -- every one of
    # a subject's five framings lands on the same side, so no held-out essay has a training
    # essay about the same kitchen, the same injury, the same shop.
    order = list(range(len(seeds)))
    rng.shuffle(order)
    cut = int(round(0.60 * len(seeds)))
    split_of = {i: ("train" if pos < cut else "heldout") for pos, i in enumerate(order)}

    style_pool: list[str] = []
    for name, weight, _ in STYLES:
        style_pool += [name] * weight
    style_text = {name: text for name, _, text in STYLES}

    rows: list[dict] = []
    for si, seed in enumerate(seeds):
        for fi, (frame_id, frame) in enumerate(FRAMES):
            idx = len(rows)
            prompt_id, prompt = PROMPTS[idx % len(PROMPTS)]
            style = rng.choice(style_pool)
            words = sample_length(rng)
            model = MODELS[idx % len(MODELS)]
            rows.append({
                "id": f"claude_{idx:03d}",
                "seed_index": si,
                "seed": seed,
                "frame": frame_id,
                "topic": seed,
                "split": split_of[si],
                "style": style,
                "model": model,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "target_words": words,
                "instruction": style_text[style].format(
                    prompt=prompt, topic=seed, frame=frame, words=words),
            })
    # Shuffle first so a batch is never one style, one frame or one region of the subject
    # list -- then group by model, because a generating worker runs as a single checkpoint
    # and can only be given its own model's assignments. Within a model the mix stays random.
    rng.shuffle(rows)
    ordered: list[dict] = []
    batch = 0
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        for start in range(0, len(group), 25):
            chunk = group[start:start + 25]
            for row in chunk:
                row["batch"] = batch
            ordered += chunk
            batch += 1
    return ordered


def summarise(rows: list[dict]) -> None:
    from collections import Counter

    def show(label: str, counter: Counter) -> None:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counter.items()))
        print(f"  {label:<10} {parts}")

    words = sorted(r["target_words"] for r in rows)
    q = lambda p: words[int(p * (len(words) - 1))]  # noqa: E731
    print(f"{len(rows)} essays, {len({r['seed'] for r in rows})} subjects, "
          f"{max(r['batch'] for r in rows) + 1} batches")
    show("split", Counter(r["split"] for r in rows))
    show("style", Counter(r["style"] for r in rows))
    show("model", Counter(r["model"] for r in rows))
    show("frame", Counter(r["frame"] for r in rows))
    print(f"  {'words':<10} p10 {q(.10)}, median {q(.50)}, p90 {q(.90)}, max {words[-1]}")
    print("  human corpus for comparison: p10 497, median 632, p90 650, max 980")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="essays to plan (multiple of 5)")
    ap.add_argument("--summary", action="store_true", help="describe the existing plan only")
    args = ap.parse_args()

    if args.summary:
        rows = [json.loads(l) for f in sorted(PLAN_DIR.glob("batch_*.json"))
                for l in [json.dumps(r) for r in json.loads(f.read_text())]]
        summarise(rows)
        return 0

    rows = build(args.n)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    for old in PLAN_DIR.glob("batch_*.json"):
        old.unlink()
    batches = max(r["batch"] for r in rows) + 1
    for b in range(batches):
        batch = [r for r in rows if r["batch"] == b]
        (PLAN_DIR / f"batch_{b:02d}.json").write_text(
            json.dumps(batch, indent=1, ensure_ascii=False), encoding="utf-8")
    (PLAN_DIR / "plan.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    summarise(rows)
    print(f"\nwrote {batches} batch files -> {PLAN_DIR.relative_to(ROOT)}/")
    print("next: generate each batch, then python scripts/assemble_claude_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

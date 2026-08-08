"""Word lists used by the surface features.

Provenance and honesty note
---------------------------
These lists are hand-built. They were seeded from the assistant-register vocabulary that
shows up repeatedly in instruction-tuned model output, then pruned against the human half
of our corpus: any term that was *more* frequent in human essays than in machine ones was
removed, because it was measuring register rather than authorship.

Three limitations you should know before trusting anything built on them:

1. They are English-only and American-leaning. "Whilst" and "amongst" are common in Indian
   and British student writing and are NOT in the machine list, deliberately.
2. They are gameable. Anyone who reads this file can avoid these words. They are a minority
   of the evidence for exactly that reason -- see ``docs/01-approach.md``.
3. Formal registers share vocabulary with machine prose. A careful human writer using
   "moreover" is not suspicious; the features below are therefore *densities* combined with
   other evidence, never a trigger on their own.
"""

from __future__ import annotations

__all__ = [
    "MACHINE_LEANING_PHRASES",
    "MACHINE_LEANING_WORDS",
    "HEDGES",
    "BOOSTERS",
    "DISCOURSE_MARKERS",
    "FUNCTION_WORDS",
    "CONCRETE_MARKERS",
]

# Multi-word constructions. Matched case-insensitively on word boundaries.
MACHINE_LEANING_PHRASES: tuple[str, ...] = (
    "it is important to note",
    "it is worth noting",
    "plays a crucial role",
    "plays a vital role",
    "plays a significant role",
    "in today's world",
    "in today's fast-paced",
    "in an ever-changing",
    "ever-evolving",
    "a testament to",
    "serves as a reminder",
    "navigate the complexities",
    "navigating the complexities",
    "delve into",
    "delving into",
    "shed light on",
    "a beacon of",
    "the tapestry of",
    "rich tapestry",
    "at its core",
    "more than just",
    "not merely",
    "not just a",
    "underscores the importance",
    "highlights the importance",
    "fostering a sense of",
    "instilled in me",
    "taught me valuable lessons",
    "valuable lesson",
    "a profound impact",
    "profound understanding",
    "deeply resonated",
    "resonated with me",
    "pivotal moment",
    "defining moment",
    "little did i know",
    "as i reflect",
    "reflecting on this",
    "this experience taught me",
    "i came to realize",
    "i realized that",
    "the power of",
    "embrace the",
    "embark on",
    "embarked on a journey",
    "journey of self-discovery",
    "paved the way",
    "opened my eyes",
    "eye-opening experience",
    "stepping out of my comfort zone",
    "outside my comfort zone",
    "in conclusion",
    "ultimately taught me",
    "continues to shape",
    "shaped who i am",
    "who i am today",
)

# Single words that skew machine in our corpus. Densities, not triggers.
MACHINE_LEANING_WORDS: frozenset[str] = frozenset(
    """
    moreover furthermore additionally consequently therefore thus hence
    crucial pivotal profound invaluable multifaceted nuanced holistic
    intricate intricacies myriad plethora realm landscape framework
    endeavor endeavour aspiration resilience perseverance determination
    unwavering steadfast relentless meticulous diligent
    foster fostering cultivate cultivating harness harnessing leverage
    transformative empowering enriching fulfilling rewarding
    showcase showcasing underscore underscores exemplify exemplifies
    testament culmination trajectory
    """.split()
)

# Epistemic hedging. Humans hedge unevenly; instruction-tuned models hedge steadily.
HEDGES: frozenset[str] = frozenset(
    """
    perhaps maybe possibly probably arguably seemingly apparently
    somewhat rather quite fairly relatively generally typically usually
    often sometimes occasionally suggest suggests indicate indicates
    tend tends appear appears seem seems might could may
    """.split()
)

# Intensity. Over-use of boosters alongside heavy hedging is itself an odd combination.
BOOSTERS: frozenset[str] = frozenset(
    """
    very extremely incredibly truly deeply utterly absolutely completely
    entirely thoroughly immensely tremendously significantly substantially
    remarkably profoundly undoubtedly certainly definitely clearly obviously
    """.split()
)

# Explicit connectives. Machine prose signposts its structure more than people do.
DISCOURSE_MARKERS: frozenset[str] = frozenset(
    """
    however therefore moreover furthermore additionally consequently nevertheless
    nonetheless meanwhile similarly likewise conversely instead besides
    accordingly subsequently ultimately overall finally firstly secondly thirdly
    """.split()
)

# Closed-class words. Their *rate* is a classic authorship signal because it is largely
# unconscious and survives topic change.
FUNCTION_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of to in on at by for with about against between
    into through during before after above below from up down out off over under
    again further then once here there all any both each few more most other some
    such no nor not only own same so than too very can will just should now
    i me my myself we our ours ourselves you your yours he him his she her hers
    it its they them their what which who whom this that these those am is are was
    were be been being have has had having do does did doing would could shall
    """.split()
)

# Signals of specific, checkable detail. Their ABSENCE is what is informative: a personal
# essay with no names, numbers, or dates in it is doing something unusual.
CONCRETE_MARKERS: frozenset[str] = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november december
    morning afternoon evening midnight noon
    """.split()
)

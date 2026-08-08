"""The feature catalogue: one entry per feature, carrying everything the UI needs to explain it.

``expected_direction`` records what we *predicted* the sign would be before fitting. It is
not used by the model -- the classifier learns its own signs from data. It exists so that
``palimpsest report`` can print the disagreements between what we expected and what the
corpus actually said. Those disagreements are the most interesting thing in the evaluation:
a feature whose learned sign contradicts the literature is either a bug, a quirk of our
dataset, or a real finding, and we would rather be forced to decide which than never notice.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Feature", "FEATURES", "FEATURE_NAMES", "FEATURES_BY_NAME", "GROUPS"]


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    group: str
    label: str
    description: str
    #: +1 = we expected a higher value to indicate machine authorship, -1 the reverse.
    expected_direction: int
    #: Unit shown in the evidence panel. Empty string for dimensionless quantities.
    unit: str = ""


GROUPS: dict[str, str] = {
    "likelihood": "How predictable the words were",
    "rank": "Where each word sat in the model's ranking",
    "composite": "Published detection statistics",
    "corpus": "Compared against real applicant essays",
    "rhythm": "Sentence shape and punctuation",
    "register": "Vocabulary and construction",
    "context": "Compared against the rest of this essay",
}


def _f(name, group, label, desc, direction, unit=""):
    return Feature(name, group, label, desc, direction, unit)


FEATURES: tuple[Feature, ...] = (
    # ---- likelihood -------------------------------------------------------------------
    _f("mean_logprob", "likelihood", "Average predictability",
       "Mean log-probability the observer model assigned to the words actually used. "
       "Higher means the passage was easier to predict word by word.", +1),
    _f("logprob_sd", "likelihood", "Predictability spread",
       "Standard deviation of that surprisal. People alternate between obvious words and "
       "surprising ones; polished machine prose stays in a narrower band.", -1),
    _f("logprob_iqr", "likelihood", "Predictability range",
       "Interquartile range of surprisal, which ignores the single most surprising word "
       "and so is harder to move with one unusual name.", -1),
    _f("mean_entropy", "likelihood", "Observer uncertainty",
       "How undecided the observer model was at each position. Distinguishes 'confident and "
       "correct' from 'no idea, and the word happened to be common'.", -1),
    _f("entropy_sd", "likelihood", "Uncertainty spread",
       "Variation in that uncertainty across the passage.", -1),
    _f("surprisal_autocorr", "likelihood", "Rhythm of surprise",
       "Lag-1 autocorrelation of surprisal. Human prose alternates predictable connective "
       "tissue with surprising content words, giving a negative value; flat prose gives ~0.", +1),
    # ---- rank -------------------------------------------------------------------------
    _f("frac_rank_top1", "rank", "Words that were the top guess",
       "Share of words that were the observer's single most likely next word.", +1, "%"),
    _f("frac_rank_top10", "rank", "Words in the top 10",
       "Share of words inside the observer's ten most likely continuations.", +1, "%"),
    _f("frac_rank_top100", "rank", "Words in the top 100",
       "Share of words inside the observer's hundred most likely continuations.", +1, "%"),
    _f("frac_rank_tail", "rank", "Genuinely unexpected words",
       "Share of words the observer ranked beyond 1000th. Real writing produces these "
       "constantly -- names, odd collocations, mistakes.", -1, "%"),
    _f("mean_log_rank", "rank", "Average log rank",
       "Mean of log(rank). Less sensitive than probability to how well the observer's "
       "probabilities happen to be calibrated.", -1),
    _f("log_rank_sd", "rank", "Log-rank spread", "Variation in log rank across the passage.", -1),
    # ---- composite --------------------------------------------------------------------
    _f("lrr", "composite", "Likelihood/rank ratio",
       "DetectLLM's LRR. Machine text is both more probable and more highly ranked, but "
       "rank collapses faster, pushing the ratio up.", +1),
    _f("curvature", "composite", "Probability curvature",
       "Fast-DetectGPT's statistic, length-normalised: how far the observed words sit above "
       "what the observer expected to see, in units of its own standard deviation. Machine "
       "text sits near a local peak of the model's distribution; human text does not.", +1),
    # ---- corpus -----------------------------------------------------------------------
    _f("corpus_surprisal_mean", "corpus", "Distance from applicant prose",
       "Surprisal under a trigram model fitted on real admissions essays. Measures 'is this "
       "how applicants write', which is a different question from 'is this fluent English'.", -1),
    _f("corpus_surprisal_sd", "corpus", "Distance spread",
       "Variation in that distance across the passage.", -1),
    _f("novel_trigram_rate", "corpus", "Three-word phrases nobody else wrote",
       "Share of word-triples absent from the human reference corpus. Individual voice "
       "produces these; prose assembled from common constructions does not.", -1, "%"),
    _f("fluency_typicality_gap", "corpus", "Fluent but atypical",
       "Corpus surprisal minus observer surprisal. Positive means the passage is smoother "
       "than general English yet less typical of the genre -- the polished-text signature. "
       "Non-native writing usually shows the opposite sign.", +1),
    # ---- rhythm -----------------------------------------------------------------------
    _f("n_words", "rhythm", "Sentence length", "Words in the sentence.", 0, "words"),
    _f("mean_word_len", "rhythm", "Average word length", "Mean characters per word.", +1, "chars"),
    _f("long_word_rate", "rhythm", "Long words", "Share of words longer than six letters.", +1, "%"),
    _f("root_ttr", "rhythm", "Vocabulary richness",
       "Distinct words divided by the square root of word count, which removes most of the "
       "length dependence of a plain type-token ratio.", -1),
    _f("comma_rate", "rhythm", "Commas", "Commas per hundred words.", +1, "/100w"),
    _f("punct_variety", "rhythm", "Punctuation variety",
       "Number of distinct punctuation marks used.", -1),
    _f("em_dash_rate", "rhythm", "Dashes", "Em/en dashes per hundred words.", +1, "/100w"),
    _f("subordination_rate", "rhythm", "Subordinate clauses",
       "Clause-introducing words per hundred words -- a proxy for syntactic depth.", +1, "/100w"),
    _f("function_word_rate", "rhythm", "Function words",
       "Closed-class words per hundred. Largely unconscious, so a classic authorship signal.",
       +1, "/100w"),
    # ---- register ---------------------------------------------------------------------
    _f("machine_phrase_rate", "register", "Stock phrases",
       "Hits from a curated list of constructions that instruction-tuned models overproduce.",
       +1, "/100w"),
    _f("machine_word_rate", "register", "Stock vocabulary",
       "Single words that skew machine in our corpus.", +1, "/100w"),
    _f("hedge_rate", "register", "Hedging", "Hedging words per hundred.", +1, "/100w"),
    _f("booster_rate", "register", "Intensifiers", "Intensifying words per hundred.", +1, "/100w"),
    _f("discourse_marker_rate", "register", "Signposting",
       "Explicit connectives per hundred. Machine prose signposts its structure more than "
       "people do.", +1, "/100w"),
    _f("first_person_rate", "register", "First person", "First-person pronouns per hundred.",
       -1, "/100w"),
    _f("contraction_rate", "register", "Contractions", "Contractions per hundred words.",
       -1, "/100w"),
    _f("specificity_rate", "register", "Checkable detail",
       "Names, numbers and dates per hundred words. Their absence is the informative case: "
       "a personal essay with nothing checkable in it is doing something unusual.", -1, "/100w"),
    _f("tricolon", "register", "Three-item list",
       "Whether the sentence contains an 'A, B, and C' construction.", +1),
    _f("antithesis", "register", "'Not just X but Y'",
       "Whether the sentence uses the not-merely-but antithesis.", +1),
    # ---- context ----------------------------------------------------------------------
    _f("logprob_z_in_doc", "context", "Smoother than the author's baseline",
       "How far this sentence's predictability sits above the median of the OTHER sentences "
       "in the same essay, in robust standard deviations. The core signal for a paragraph "
       "that was polished inside otherwise human writing.", +1, "sd"),
    _f("curvature_z_in_doc", "context", "Curvature vs the author's baseline",
       "The same leave-one-out comparison applied to probability curvature.", +1, "sd"),
    _f("len_z_in_doc", "context", "Length vs the author's baseline",
       "How unusual this sentence's length is for this essay.", 0, "sd"),
    _f("style_gap_from_doc", "context", "Doesn't sound like the rest",
       "Mean absolute deviation of this sentence's style vector from the median of the rest "
       "of the essay, across rhythm, register and likelihood together.", +1, "sd"),
    _f("local_len_burstiness", "context", "Local rhythm",
       "Coefficient of variation of sentence length in a five-sentence window. Even rhythm "
       "over a stretch of text is one of the more reliable passage-level signals.", -1),
    _f("rel_position", "context", "Position in essay",
       "Where the sentence falls, 0 at the start and 1 at the end. Openings and closings "
       "differ systematically, and the model is allowed to know that.", 0),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature name in registry"

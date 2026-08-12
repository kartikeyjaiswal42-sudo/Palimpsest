/**
 * Port of `features/registry.py` — the catalogue the evidence panel renders from.
 *
 * `expectedDirection` is what we predicted the sign would be before fitting. The classifier
 * never reads it; it exists so a disagreement between prediction and fitted sign is visible
 * rather than invisible.
 */

const f = (name, group, label, description, expectedDirection, unit = '') => ({
  name, group, label, description, expectedDirection, unit,
});

export const GROUPS = {
  likelihood: 'How predictable the words were',
  rank: "Where each word sat in the model's ranking",
  composite: 'Published detection statistics',
  corpus: 'Compared against real applicant essays',
  rhythm: 'Sentence shape and punctuation',
  register: 'Vocabulary and construction',
  context: 'Compared against the rest of this essay',
};

export const FEATURES = [
  f('mean_logprob', 'likelihood', 'Average predictability',
    'Mean log-probability the observer model assigned to the words actually used. Higher means the passage was easier to predict word by word.', +1),
  f('logprob_sd', 'likelihood', 'Predictability spread',
    'Standard deviation of that surprisal. People alternate between obvious words and surprising ones; polished machine prose stays in a narrower band.', -1),
  f('logprob_iqr', 'likelihood', 'Predictability range',
    'Interquartile range of surprisal, which ignores the single most surprising word and so is harder to move with one unusual name.', -1),
  f('mean_entropy', 'likelihood', 'Observer uncertainty',
    "How undecided the observer model was at each position. Distinguishes 'confident and correct' from 'no idea, and the word happened to be common'.", -1),
  f('entropy_sd', 'likelihood', 'Uncertainty spread', 'Variation in that uncertainty across the passage.', -1),
  f('surprisal_autocorr', 'likelihood', 'Rhythm of surprise',
    'Lag-1 autocorrelation of surprisal. Human prose alternates predictable connective tissue with surprising content words, giving a negative value; flat prose gives ~0.', +1),

  f('frac_rank_top1', 'rank', 'Words that were the top guess',
    "Share of words that were the observer's single most likely next word.", +1, '%'),
  f('frac_rank_top10', 'rank', 'Words in the top 10',
    "Share of words inside the observer's ten most likely continuations.", +1, '%'),
  f('frac_rank_top100', 'rank', 'Words in the top 100',
    "Share of words inside the observer's hundred most likely continuations.", +1, '%'),
  f('frac_rank_tail', 'rank', 'Genuinely unexpected words',
    'Share of words the observer ranked beyond 1000th. Real writing produces these constantly -- names, odd collocations, mistakes.', -1, '%'),
  f('mean_log_rank', 'rank', 'Average log rank',
    "Mean of log(rank). Less sensitive than probability to how well the observer's probabilities happen to be calibrated.", -1),
  f('log_rank_sd', 'rank', 'Log-rank spread', 'Variation in log rank across the passage.', -1),

  f('lrr', 'composite', 'Likelihood/rank ratio',
    "DetectLLM's LRR. Machine text is both more probable and more highly ranked, but rank collapses faster, pushing the ratio up.", +1),
  f('curvature', 'composite', 'Probability curvature',
    "Fast-DetectGPT's statistic, length-normalised: how far the observed words sit above what the observer expected to see, in units of its own standard deviation. Machine text sits near a local peak of the model's distribution; human text does not.", +1),

  f('corpus_surprisal_mean', 'corpus', 'Distance from applicant prose',
    "Surprisal under a trigram model fitted on real admissions essays. Measures 'is this how applicants write', which is a different question from 'is this fluent English'.", -1),
  f('corpus_surprisal_sd', 'corpus', 'Distance spread', 'Variation in that distance across the passage.', -1),
  f('novel_trigram_rate', 'corpus', 'Three-word phrases nobody else wrote',
    'Share of word-triples absent from the human reference corpus. Individual voice produces these; prose assembled from common constructions does not.', -1, '%'),
  f('fluency_typicality_gap', 'corpus', 'Fluent but atypical',
    'Corpus surprisal minus observer surprisal. Positive means the passage is smoother than general English yet less typical of the genre -- the polished-text signature. Non-native writing usually shows the opposite sign.', +1),

  f('n_words', 'rhythm', 'Sentence length', 'Words in the sentence.', 0, 'words'),
  f('mean_word_len', 'rhythm', 'Average word length', 'Mean characters per word.', +1, 'chars'),
  f('long_word_rate', 'rhythm', 'Long words', 'Share of words longer than six letters.', +1, '%'),
  f('root_ttr', 'rhythm', 'Vocabulary richness',
    'Distinct words divided by the square root of word count, which removes most of the length dependence of a plain type-token ratio.', -1),
  f('comma_rate', 'rhythm', 'Commas', 'Commas per hundred words.', +1, '/100w'),
  f('punct_variety', 'rhythm', 'Punctuation variety', 'Number of distinct punctuation marks used.', -1),
  f('em_dash_rate', 'rhythm', 'Dashes', 'Em/en dashes per hundred words.', +1, '/100w'),
  f('subordination_rate', 'rhythm', 'Subordinate clauses',
    'Clause-introducing words per hundred words -- a proxy for syntactic depth.', +1, '/100w'),
  f('function_word_rate', 'rhythm', 'Function words',
    'Closed-class words per hundred. Largely unconscious, so a classic authorship signal.', +1, '/100w'),

  f('machine_phrase_rate', 'register', 'Stock phrases',
    'Hits from a curated list of constructions that instruction-tuned models overproduce.', +1, '/100w'),
  f('machine_word_rate', 'register', 'Stock vocabulary', 'Single words that skew machine in our corpus.', +1, '/100w'),
  f('hedge_rate', 'register', 'Hedging', 'Hedging words per hundred.', +1, '/100w'),
  f('booster_rate', 'register', 'Intensifiers', 'Intensifying words per hundred.', +1, '/100w'),
  f('discourse_marker_rate', 'register', 'Signposting',
    'Explicit connectives per hundred. Machine prose signposts its structure more than people do.', +1, '/100w'),
  f('first_person_rate', 'register', 'First person', 'First-person pronouns per hundred.', -1, '/100w'),
  f('contraction_rate', 'register', 'Contractions', 'Contractions per hundred words.', -1, '/100w'),
  f('specificity_rate', 'register', 'Checkable detail',
    'Names, numbers and dates per hundred words. Their absence is the informative case: a personal essay with nothing checkable in it is doing something unusual.', -1, '/100w'),
  f('tricolon', 'register', 'Three-item list', "Whether the sentence contains an 'A, B, and C' construction.", +1),
  f('antithesis', 'register', "'Not just X but Y'", 'Whether the sentence uses the not-merely-but antithesis.', +1),

  f('logprob_z_in_doc', 'context', "Smoother than the author's baseline",
    "How far this sentence's predictability sits above the median of the OTHER sentences in the same essay, in robust standard deviations. The core signal for a paragraph that was polished inside otherwise human writing.", +1, 'sd'),
  f('curvature_z_in_doc', 'context', "Curvature vs the author's baseline",
    'The same leave-one-out comparison applied to probability curvature.', +1, 'sd'),
  f('len_z_in_doc', 'context', "Length vs the author's baseline",
    "How unusual this sentence's length is for this essay.", 0, 'sd'),
  f('style_gap_from_doc', 'context', "Doesn't sound like the rest",
    "Mean absolute deviation of this sentence's style vector from the median of the rest of the essay, across rhythm, register and likelihood together.", +1, 'sd'),
  f('local_len_burstiness', 'context', 'Local rhythm',
    'Coefficient of variation of sentence length in a five-sentence window. Even rhythm over a stretch of text is one of the more reliable passage-level signals.', -1),
  f('rel_position', 'context', 'Position in essay',
    'Where the sentence falls, 0 at the start and 1 at the end. Openings and closings differ systematically, and the model is allowed to know that.', 0),
];

export const FEATURES_BY_NAME = Object.fromEntries(FEATURES.map((x) => [x.name, x]));

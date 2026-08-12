/**
 * Ports of `features/model_based.py`, `features/surface.py`, `features/context.py` and the
 * lexicons they read. The comments that explain *why* a feature is shaped the way it is live
 * in the Python modules; what follows notes only where JavaScript forced a change.
 *
 * The recurring one is regular expressions. Python's `\w` and `\b` are Unicode-aware for
 * `str` patterns; JavaScript's are ASCII-only even under `/u`. Left alone, `café's` counts
 * as a contraction in the Python build and not here — a silent difference in a feature the
 * classifier weights. Every affected pattern is therefore written with explicit
 * `[\p{L}\p{N}_]` classes and hand-rolled boundary lookarounds.
 */

import {
  clip, isFinite_, mean, median, nanMean, nanMedian, percentile, PY_SPACE_CLASS, std1,
} from './pyshim.js';
import { tokenizeWords } from './segment.js';

// -- lexicons (features/lexicons.py) --------------------------------------------------

const words = (s) => new Set(s.split(/\s+/).filter(Boolean));

export const MACHINE_LEANING_PHRASES = [
  "it is important to note", "it is worth noting", "plays a crucial role", "plays a vital role",
  "plays a significant role", "in today's world", "in today's fast-paced", "in an ever-changing",
  "ever-evolving", "a testament to", "serves as a reminder", "navigate the complexities",
  "navigating the complexities", "delve into", "delving into", "shed light on", "a beacon of",
  "the tapestry of", "rich tapestry", "at its core", "more than just", "not merely",
  "not just a", "underscores the importance", "highlights the importance",
  "fostering a sense of", "instilled in me", "taught me valuable lessons", "valuable lesson",
  "a profound impact", "profound understanding", "deeply resonated", "resonated with me",
  "pivotal moment", "defining moment", "little did i know", "as i reflect",
  "reflecting on this", "this experience taught me", "i came to realize", "i realized that",
  "the power of", "embrace the", "embark on", "embarked on a journey",
  "journey of self-discovery", "paved the way", "opened my eyes", "eye-opening experience",
  "stepping out of my comfort zone", "outside my comfort zone", "in conclusion",
  "ultimately taught me", "continues to shape", "shaped who i am", "who i am today",
];

const MACHINE_LEANING_WORDS = words(`
  moreover furthermore additionally consequently therefore thus hence
  crucial pivotal profound invaluable multifaceted nuanced holistic
  intricate intricacies myriad plethora realm landscape framework
  endeavor endeavour aspiration resilience perseverance determination
  unwavering steadfast relentless meticulous diligent
  foster fostering cultivate cultivating harness harnessing leverage
  transformative empowering enriching fulfilling rewarding
  showcase showcasing underscore underscores exemplify exemplifies
  testament culmination trajectory`);

const HEDGES = words(`
  perhaps maybe possibly probably arguably seemingly apparently
  somewhat rather quite fairly relatively generally typically usually
  often sometimes occasionally suggest suggests indicate indicates
  tend tends appear appears seem seems might could may`);

const BOOSTERS = words(`
  very extremely incredibly truly deeply utterly absolutely completely
  entirely thoroughly immensely tremendously significantly substantially
  remarkably profoundly undoubtedly certainly definitely clearly obviously`);

const DISCOURSE_MARKERS = words(`
  however therefore moreover furthermore additionally consequently nevertheless
  nonetheless meanwhile similarly likewise conversely instead besides
  accordingly subsequently ultimately overall finally firstly secondly thirdly`);

const FUNCTION_WORDS = words(`
  a an the and or but if while of to in on at by for with about against between
  into through during before after above below from up down out off over under
  again further then once here there all any both each few more most other some
  such no nor not only own same so than too very can will just should now
  i me my myself we our ours ourselves you your yours he him his she her hers
  it its they them their what which who whom this that these those am is are was
  were be been being have has had having do does did doing would could shall`);

const CONCRETE_MARKERS = words(`
  monday tuesday wednesday thursday friday saturday sunday
  january february march april may june july august september october november december
  morning afternoon evening midnight noon`);

const CLAUSE_MARKERS = words(`
  that which who whom whose because although though while whereas since unless
  until whether if when where after before as`);

const FIRST_PERSON = words('i me my mine myself we us our ours');

// -- model-based features (features/model_based.py) -----------------------------------

export const MIN_TOKENS = 5;

export const MODEL_FEATURE_NAMES = [
  'mean_logprob', 'logprob_sd', 'logprob_iqr', 'frac_rank_top1', 'frac_rank_top10',
  'frac_rank_top100', 'frac_rank_tail', 'mean_log_rank', 'log_rank_sd', 'mean_entropy',
  'entropy_sd', 'lrr', 'curvature', 'surprisal_autocorr',
];

const allNaN = (names) => Object.fromEntries(names.map((n) => [n, NaN]));

function autocorr(x, lag = 1) {
  if (x.length <= lag + 2) return NaN;
  const a = x.slice(0, x.length - lag);
  const b = x.slice(lag);
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    num += da * db;
    sa += da * da;
    sb += db * db;
  }
  const denom = Math.sqrt(sa * sb);
  if (denom < 1e-12) return NaN;
  return num / denom;
}

/**
 * `scores` carries parallel arrays for one span: logprob, rank, entropy, mu, sigma2.
 * The remote observer cannot supply the last three — the API returns only the realised
 * token, never the full predictive distribution — so they arrive as NaN and the three
 * features that need them come out NaN, which the classifier treats as "not measured".
 */
export function extractModelFeatures(scores) {
  const n = scores.logprob.length;
  if (n < MIN_TOKENS) return allNaN(MODEL_FEATURE_NAMES);

  const lp = scores.logprob;
  const rank = scores.rank;
  const logRank = rank.map((r) => Math.log(Math.max(r, 1.0)));

  const q75 = percentile(lp, 75);
  const q25 = percentile(lp, 25);

  const meanLp = mean(lp);
  const meanNll = -meanLp;
  const meanLogRank = mean(logRank);
  // Undefined, not enormous, when the observer ranked every token in the span first --
  // mirrors the same change in features/model_based.py, where the old `max(.., 1e-6)` guard
  // let 11 of 6,941 training sentences score up to 7.4e5 and thereby set the column's
  // standard deviation to 11,625, silencing the feature for every other sentence. Parity
  // with Python is what makes the accuracy figures describe this deployment, so the two
  // implementations have to make the same choice here.
  const lrr = meanLogRank > 0 ? meanNll / meanLogRank : NaN;

  const varMean = mean(scores.sigma2);
  const denom = Math.sqrt(Math.max(varMean, 1e-9));
  const curvature = (meanLp - mean(scores.mu)) / denom;

  const rateAtMost = (limit) => rank.filter((r) => r <= limit).length / n;

  return {
    mean_logprob: meanLp,
    logprob_sd: n > 1 ? std1(lp) : NaN,
    logprob_iqr: q75 - q25,
    frac_rank_top1: rateAtMost(1),
    frac_rank_top10: rateAtMost(10),
    frac_rank_top100: rateAtMost(100),
    frac_rank_tail: rank.filter((r) => r > 1000).length / n,
    mean_log_rank: meanLogRank,
    log_rank_sd: n > 1 ? std1(logRank) : NaN,
    mean_entropy: mean(scores.entropy),
    entropy_sd: n > 1 ? std1(scores.entropy) : NaN,
    lrr,
    curvature,
    surprisal_autocorr: autocorr(lp),
  };
}

// -- surface features (features/surface.py) -------------------------------------------

export const MIN_WORDS = 4;

export const SURFACE_FEATURE_NAMES = [
  'n_words', 'mean_word_len', 'long_word_rate', 'root_ttr', 'comma_rate', 'punct_variety',
  'em_dash_rate', 'subordination_rate', 'function_word_rate', 'machine_phrase_rate',
  'machine_word_rate', 'hedge_rate', 'booster_rate', 'discourse_marker_rate',
  'first_person_rate', 'contraction_rate', 'specificity_rate', 'tricolon', 'antithesis',
];

// Python's `\w` for str patterns is Unicode alphanumerics plus underscore, and its `\b` is
// defined against that class. JavaScript's are ASCII even under /u, so both are written out.
const W = String.raw`[\p{L}\p{N}_]`;
const NW = String.raw`[^\p{L}\p{N}_]`;
const WB = `(?:(?<=${NW}|^)(?=${W})|(?<=${W})(?=${NW}|$))`;
const re = (source, flags) => new RegExp(source, flags);

const PUNCT = /[,;:—–\-()"'!?.]/g;
const EM_DASH = /[—–]|(?<= )-{1,2}(?= )/g;
// Python: `\b\w+['’](?:s|t|re|ve|ll|d|m)\b`, IGNORECASE.
const CONTRACTION = re(`${WB}${W}+['’](?:s|t|re|ve|ll|d|m)${WB}`, 'giu');
// Python `\d` on a str pattern is Unicode decimal digits, not ASCII.
const DIGIT = /\p{Nd}/gu;
const MID_CAPS = /(?<=[a-z,;] )([A-Z][a-z]{2,})/g;
// Python: `\b[\w'’]+(?:\s+[\w'’]+){0,3},\s+[\w'’]+(?:\s+[\w'’]+){0,3},\s+and\s+`
const TOK = String.raw`[\p{L}\p{N}_'’]`;
const S = `${PY_SPACE_CLASS}`; // Python's `\s`, which is not JavaScript's
const TRICOLON = re(
  `${WB}${TOK}+(?:${S}+${TOK}+){0,3},${S}+${TOK}+(?:${S}+${TOK}+){0,3},${S}+and${S}+`,
  'u',
);
const ANTITHESIS = re(
  `${WB}not (?:just|only|merely|simply)${WB}.{0,80}?${WB}but${WB}` +
    `|${WB}was ?n[o'’]t${WB}.{0,60}?${WB}it was${WB}` +
    `|${WB}more than (?:just )?a${WB}`,
  'isu',
);

const countMatches = (re, text) => {
  re.lastIndex = 0;
  let n = 0;
  while (re.exec(text) !== null) n += 1;
  return n;
};

const countIn = (list, set) => list.reduce((n, w) => n + (set.has(w) ? 1 : 0), 0);

export function extractSurfaceFeatures(text) {
  const ws = tokenizeWords(text);
  const n = ws.length;
  if (n < MIN_WORDS) return allNaN(SURFACE_FEATURE_NAMES);

  const lower = text.toLowerCase();
  const per100 = 100.0 / n;
  const lengths = ws.map((w) => w.length);

  let phraseHits = 0;
  for (const p of MACHINE_LEANING_PHRASES) if (lower.includes(p)) phraseHits += 1;

  const specific =
    countMatches(MID_CAPS, text) + countMatches(DIGIT, text) + countIn(ws, CONCRETE_MARKERS);

  PUNCT.lastIndex = 0;
  const punctSeen = new Set();
  let pm;
  while ((pm = PUNCT.exec(text)) !== null) punctSeen.add(pm[0]);

  return {
    n_words: n,
    mean_word_len: mean(lengths),
    long_word_rate: lengths.filter((l) => l > 6).length / n,
    root_ttr: new Set(ws).size / Math.sqrt(n),
    comma_rate: (text.split(',').length - 1) * per100,
    punct_variety: punctSeen.size,
    em_dash_rate: countMatches(EM_DASH, text) * per100,
    subordination_rate: countIn(ws, CLAUSE_MARKERS) * per100,
    function_word_rate: countIn(ws, FUNCTION_WORDS) * per100,
    machine_phrase_rate: phraseHits * per100,
    machine_word_rate: countIn(ws, MACHINE_LEANING_WORDS) * per100,
    hedge_rate: countIn(ws, HEDGES) * per100,
    booster_rate: countIn(ws, BOOSTERS) * per100,
    discourse_marker_rate: countIn(ws, DISCOURSE_MARKERS) * per100,
    first_person_rate: countIn(ws, FIRST_PERSON) * per100,
    contraction_rate: countMatches(CONTRACTION, text) * per100,
    specificity_rate: specific * per100,
    tricolon: TRICOLON.test(text) ? 1 : 0,
    antithesis: ANTITHESIS.test(text) ? 1 : 0,
  };
}

// -- context features (features/context.py) -------------------------------------------

export const MIN_SENTENCES = 5;
const DEGENERATE_Z = 6.0;

export const CONTEXT_FEATURE_NAMES = [
  'logprob_z_in_doc', 'curvature_z_in_doc', 'len_z_in_doc', 'style_gap_from_doc',
  'local_len_burstiness', 'rel_position',
];

const STYLE_KEYS = [
  'mean_logprob', 'logprob_sd', 'mean_log_rank', 'curvature', 'mean_word_len', 'comma_rate',
  'subordination_rate', 'function_word_rate', 'discourse_marker_rate', 'root_ttr', 'n_words',
];

function robustScaleColumns(matrix, nCols) {
  const out = matrix.map(() => new Array(nCols).fill(NaN));
  for (let j = 0; j < nCols; j += 1) {
    const col = matrix.map((row) => row[j]);
    const finite = col.filter(isFinite_);
    if (finite.length < 3) continue;
    const centre = median(finite);
    let mad = median(finite.map((v) => Math.abs(v - centre))) * 1.4826;
    if (mad < 1e-9) {
      const sd = std1(finite);
      mad = sd > 1e-9 ? sd : 1.0;
    }
    for (let i = 0; i < matrix.length; i += 1) out[i][j] = (col[i] - centre) / mad;
  }
  return out;
}

function looZ(base, i, key) {
  const values = base.map((row) => (key in row ? row[key] : NaN));
  const here = values[i];
  if (!isFinite_(here)) return NaN;
  const others = values.filter((_, k) => k !== i).filter(isFinite_);
  if (others.length < 3) return NaN;
  const centre = median(others);
  let mad = median(others.map((v) => Math.abs(v - centre))) * 1.4826;
  if (mad < 1e-9) {
    const spread = std1(others);
    if (spread < 1e-9) {
      const delta = here - centre;
      if (Math.abs(delta) < 1e-9) return 0.0;
      return Math.sign(delta) * DEGENERATE_Z;
    }
    mad = spread;
  }
  return clip((here - centre) / mad, -DEGENERATE_Z, DEGENERATE_Z);
}

function localBurstiness(lengths, i, halfWindow = 2) {
  const lo = Math.max(0, i - halfWindow);
  const hi = Math.min(lengths.length, i + halfWindow + 1);
  const win = lengths.slice(lo, hi).filter(isFinite_);
  if (win.length < 3 || mean(win) < 1e-9) return NaN;
  return std1(win) / mean(win);
}

export function extractContextFeatures(base, nSentences) {
  if (nSentences < MIN_SENTENCES) {
    return base.map(() => allNaN(CONTEXT_FEATURE_NAMES));
  }

  const matrix = base.map((row) => STYLE_KEYS.map((k) => (k in row ? row[k] : NaN)));
  const scaled = robustScaleColumns(matrix, STYLE_KEYS.length);
  const lengths = base.map((row) => ('n_words' in row ? row.n_words : NaN));

  const out = [];
  for (let i = 0; i < nSentences; i += 1) {
    const others = scaled.filter((_, k) => k !== i);
    const centre = STYLE_KEYS.map((_, j) => nanMedian(others.map((r) => r[j])));
    const here = scaled[i];
    const gapTerms = here.map((v, j) => Math.abs(v - centre[j]));
    const styleGap = gapTerms.some(isFinite_) ? nanMean(gapTerms) : NaN;

    out.push({
      logprob_z_in_doc: looZ(base, i, 'mean_logprob'),
      curvature_z_in_doc: looZ(base, i, 'curvature'),
      len_z_in_doc: looZ(base, i, 'n_words'),
      style_gap_from_doc: styleGap,
      local_len_burstiness: localBurstiness(lengths, i),
      rel_position: i / Math.max(nSentences - 1, 1),
    });
  }
  return out;
}

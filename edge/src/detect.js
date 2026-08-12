/**
 * Ports of `detect/classifier.py`, `detect/document.py` and `detect/genre.py`.
 *
 * Nothing is re-fitted here. Every weight, mean, scale, calibration knot and threshold is
 * read from the artifacts the Python training scripts wrote, so this file is arithmetic
 * over numbers produced elsewhere. That is the only way the accuracy figures in `docs/`
 * describe the thing that is actually deployed.
 */

import { average, clip, interp, isFinite_, mean, percentile, sigmoid } from './pyshim.js';
import { PCG64 } from './numpy_random.js';

/** Standardised values are clipped here. See the long note in classifier.py: it exists
 *  because three run-on ESL essays segmented to one 300+ word "sentence" and `n_words`
 *  alone then carried a +7.5 logit term. */
export const Z_CLIP = 5.0;

export class SentenceDetector {
  constructor(d) {
    this.featureNames = d.feature_names;
    this.mean = d.mean;
    this.scale = d.scale;
    this.coef = d.coef;
    this.intercept = d.intercept;
    this.calibrationX = d.calibration_x;
    this.calibrationY = d.calibration_y;
    this.flagThreshold = d.flag_threshold ?? 0.5;
    this.metadata = d.metadata ?? {};
  }

  /** Feature dict -> standardised vector. NaN is imputed with the training mean, which
   *  standardises to exactly zero, so an unmeasured feature contributes nothing. */
  standardise(features) {
    const z = new Array(this.featureNames.length);
    for (let i = 0; i < this.featureNames.length; i += 1) {
      const raw = features[this.featureNames[i]];
      const filled = isFinite_(raw) ? raw : this.mean[i];
      z[i] = clip((filled - this.mean[i]) / this.scale[i], -Z_CLIP, Z_CLIP);
    }
    return z;
  }

  calibrate(p) {
    if (!this.calibrationX || !this.calibrationY) return p;
    return interp(p, this.calibrationX, this.calibrationY);
  }

  predict(features) {
    const z = this.standardise(features);
    const terms = z.map((v, i) => this.coef[i] * v);
    // Sum the terms first, then add the intercept, matching NumPy's `terms.sum() + b`.
    let acc = 0;
    for (const t of terms) acc += t;
    const logit = acc + this.intercept;

    const contributions = this.featureNames.map((name, i) => {
      const value = features[name];
      const measured = isFinite_(value);
      return {
        name,
        value: measured ? value : NaN,
        z: z[i],
        weight: this.coef[i],
        contribution: terms[i],
        measured,
      };
    });

    return {
      probability: this.calibrate(sigmoid(logit)),
      logit,
      intercept: this.intercept,
      contributions,
    };
  }
}

// -- document layer (detect/document.py) ----------------------------------------------

export const FLAG_THRESHOLD = 0.65;
export const MAX_SENTENCE_WORDS = 90;
export const DOC_FEATURES = ['mean_p', 'max_p', 'q90_p', 'share'];

/** Length-weighted moving average over a +/-1 sentence window; the sentence counts double. */
export function smoothProbabilities(probs, weights) {
  const n = probs.length;
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const lo = Math.max(0, i - 1);
    const hi = Math.min(n, i + 2);
    const w = weights.slice(lo, hi);
    w[i - lo] *= 2.0;
    let total = 0;
    for (const v of w) total += v;
    if (total > 0) {
      let acc = 0;
      for (let k = lo; k < hi; k += 1) acc += probs[k] * w[k - lo];
      out[i] = acc / total;
    } else {
      out[i] = probs[i];
    }
  }
  return out;
}

export function findPassages(sentences, threshold = FLAG_THRESHOLD) {
  const passages = [];
  let run = [];
  const flush = () => {
    if (!run.length) return;
    passages.push({
      start: run[0].start,
      end: run[run.length - 1].end,
      sentenceIndices: run.map((s) => s.index),
      nWords: run.reduce((a, s) => a + s.nWords, 0),
      meanProbability: average(run.map((s) => s.probability), run.map((s) => s.nWords)),
      peakProbability: Math.max(...run.map((s) => s.probability)),
    });
  };
  for (const s of sentences) {
    if (s.smoothed >= threshold) {
      run.push(s);
    } else {
      flush();
      run = [];
    }
  }
  flush();
  return passages;
}

export function documentStatistics(probs, words, threshold) {
  if (!probs.length) {
    return { mean_p: 0, max_p: 0, q90_p: 0, share: 0, log_sentences: 0 };
  }
  let total = 0;
  for (const w of words) total += w;
  total = Math.max(total, 1.0);
  let flaggedWords = 0;
  for (let i = 0; i < probs.length; i += 1) if (probs[i] >= threshold) flaggedWords += words[i];
  return {
    mean_p: average(probs, words.map((w) => Math.max(w, 1.0))),
    max_p: Math.max(...probs),
    q90_p: percentile(probs, 90),
    share: flaggedWords / total,
    log_sentences: Math.log1p(probs.length),
  };
}

export class DocumentDetector {
  constructor(d) {
    this.coef = d?.coef ?? null;
    this.intercept = d?.intercept ?? 0;
    this.mean = d?.mean ?? null;
    this.scale = d?.scale ?? null;
    this.threshold = d?.threshold ?? 0.5;
    this.metadata = d?.metadata ?? {};
  }

  predict(stats) {
    // Unfitted fallback: the strongest single sentence. Monotone, and it never collapses
    // to zero the way a thresholded count does.
    if (!this.coef) return stats.max_p ?? 0;
    let acc = 0;
    for (let i = 0; i < DOC_FEATURES.length; i += 1) {
      const z = (stats[DOC_FEATURES[i]] - this.mean[i]) / this.scale[i];
      acc += z * this.coef[i];
    }
    return sigmoid(acc + this.intercept);
  }
}

/** Percentile bootstrap for the machine share, seeded so one essay always gives one answer. */
function bootstrapShare(probs, words, seed, threshold, nBoot = 400) {
  const n = probs.length;
  if (n < 3) return [0.0, 1.0];
  const rng = new PCG64(seed);
  const shares = new Array(nBoot);
  for (let b = 0; b < nBoot; b += 1) {
    const idx = rng.integers(n, n);
    let total = 0;
    let flagged = 0;
    for (let k = 0; k < n; k += 1) {
      const w = words[idx[k]];
      total += w;
      if (probs[idx[k]] >= threshold) flagged += w;
    }
    shares[b] = total > 0 ? flagged / total : 0.0;
  }
  return [percentile(shares, 5), percentile(shares, 95)];
}

export function aggregate(sentences, threshold = FLAG_THRESHOLD, docModel = null, rngSeed = 0) {
  const empty = {
    machineShare: 0, machineShareLow: 0, machineShareHigh: 0, anyMachineProbability: 0,
    nSentences: 0, nWords: 0, nReliableSentences: 0,
  };
  if (!sentences.length) return empty;

  // A span the tool has said it cannot measure must not decide the answer: unreliable
  // sentences leave both the numerator and the denominator.
  const scored = sentences.filter((s) => s.reliable);
  if (!scored.length) {
    return {
      ...empty,
      nSentences: sentences.length,
      nWords: Math.trunc(sentences.reduce((a, s) => a + s.nWords, 0)),
    };
  }

  const probs = scored.map((s) => s.probability);
  const words = scored.map((s) => s.nWords);
  const totalWords = words.reduce((a, w) => a + w, 0);

  let flaggedWords = 0;
  for (let i = 0; i < probs.length; i += 1) if (probs[i] >= threshold) flaggedWords += words[i];
  const share = totalWords > 0 ? flaggedWords / totalWords : 0.0;
  const [low, high] = bootstrapShare(probs, words, rngSeed, threshold);
  const stats = documentStatistics(probs, words, threshold);

  return {
    machineShare: share,
    machineShareLow: low,
    machineShareHigh: high,
    anyMachineProbability: (docModel ?? new DocumentDetector(null)).predict(stats),
    nSentences: sentences.length,
    nWords: Math.trunc(totalWords),
    nReliableSentences: sentences.filter((s) => s.reliable).length,
  };
}

// -- genre gate (detect/genre.py) ------------------------------------------------------

export const GENRE_FEATURES = [
  'corpus_surprisal_mean', 'novel_trigram_rate', 'first_person_rate', 'specificity_rate',
];

/** Word-weighted document means. NaN is skipped, never imputed: a feature the pipeline
 *  could not measure must not be replaced by an average and then used to refuse an essay. */
export function documentGenreFeatures(sentenceFeatures) {
  if (!sentenceFeatures.length) {
    return Object.fromEntries(GENRE_FEATURES.map((n) => [n, NaN]));
  }
  const w = sentenceFeatures.map((f) => {
    const v = Number(f.n_words || 0);
    return Number.isFinite(v) && v > 0 ? v : 0;
  });
  const total = w.reduce((a, v) => a + v, 0);

  // `float(f.get(name, nan) or nan)` in genre.py: Python's `or` treats a genuine 0.0 as
  // falsy, so a sentence with (say) no first-person pronouns is DROPPED from the weighted
  // mean rather than contributing a zero. Arguably a wart, but the gate's mean, scale and
  // threshold were all fitted through it, so reproducing it is not optional.
  const wmean = (name) => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < sentenceFeatures.length; i += 1) {
      const raw = sentenceFeatures[i][name];
      const v = raw === undefined || raw === null || raw === 0 || raw === false ? NaN : Number(raw);
      if (isFinite_(v) && w[i] > 0) {
        num += v * w[i];
        den += w[i];
      }
    }
    return den > 0 ? num / den : NaN;
  };

  return {
    corpus_surprisal_mean: wmean('corpus_surprisal_mean'),
    novel_trigram_rate: wmean('novel_trigram_rate'),
    first_person_rate: wmean('first_person_rate'),
    specificity_rate: wmean('specificity_rate'),
    mean_sentence_words: total / Math.max(sentenceFeatures.length, 1),
  };
}

export class GenreGate {
  constructor(d) {
    this.featureNames = d?.feature_names ?? GENRE_FEATURES;
    this.mean = d?.mean ?? null;
    this.scale = d?.scale ?? null;
    this.coef = d?.coef ?? null;
    this.intercept = d?.intercept ?? 0;
    this.threshold = d?.threshold ?? 0;
    this.metadata = d?.metadata ?? {};
  }

  probability(docFeatures) {
    if (!this.coef) return 1.0; // an unfitted gate must never refuse anything
    let acc = 0;
    for (let i = 0; i < this.featureNames.length; i += 1) {
      const raw = docFeatures[this.featureNames[i]];
      const x = raw === undefined || raw === null ? NaN : raw;
      let z = (x - this.mean[i]) / this.scale[i];
      if (!Number.isFinite(z)) z = 0; // np.nan_to_num
      acc += clip(z, -5.0, 5.0) * this.coef[i];
    }
    return sigmoid(acc + this.intercept);
  }

  inDomain(docFeatures) {
    return !this.coef || this.probability(docFeatures) >= this.threshold;
  }
}

export { mean, percentile };

/**
 * Port of `palimpsest/analyze.py` and the verdict wording in `api/app.py`.
 *
 *     segment -> observe -> 43 features -> sentence model -> document model -> gate -> band
 *
 * The observer is passed in. It is an instrument: one forward pass whose logits are read,
 * never a prompt and never a question. Nothing in this file, or anything it calls, asks a
 * model for an opinion.
 */

import { MAX_OBSERVER_CHARS } from './observer.js';
import { splitSentences, tokenizeWords } from './segment.js';
import {
  extractContextFeatures, extractModelFeatures, extractSurfaceFeatures, MIN_TOKENS,
} from './features.js';
import { extractCorpusFeatures } from './ngram.js';
import {
  aggregate, documentGenreFeatures, findPassages, FLAG_THRESHOLD, MAX_SENTENCE_WORDS,
  smoothProbabilities,
} from './detect.js';
import { FEATURES_BY_NAME } from './registry.js';

const RANK_BUCKETS = [[10, 'top10'], [100, 'top100'], [1000, 'top1000']];

/**
 * Python's `round()`, exactly: correct rounding of the true binary value, ties to EVEN.
 *
 * The comment that used to sit here said the harness compares raw values rather than their
 * rendering. That was wrong -- the verdict block is exported already rounded -- and the two
 * languages then disagreed by 1e-4 on a real document (jhu:and-the-secret-ingredient-is,
 * share = 0.10625). Both JS primitives are wrong for this in different ways:
 *
 *   Math.round(v * 10**k)  multiplies FIRST, and the multiplication is itself rounded. It
 *                          turned 0.10625 into exactly 1062.5 and then rounded half UP.
 *   Number(v.toFixed(k))   rounds the true value correctly, but breaks ties AWAY from zero.
 *
 * Ties are reachable here rather than theoretical: `share` is a ratio of word counts, so any
 * k/32 (0.03125, 0.09375, ...) lands exactly on one. So: take toFixed's correctly-rounded
 * digits, and detect a genuine tie by reading far enough into the exact decimal expansion
 * that a non-tie double -- which differs from the tie by at least one ulp -- cannot hide.
 */
const round = (v, k) => {
  if (!Number.isFinite(v)) return null;
  const exact = Math.abs(v).toFixed(Math.min(100, k + 25));
  const tail = exact.slice(exact.indexOf('.') + 1 + k);
  if (!/^50*$/.test(tail)) return Number(v.toFixed(k)); // not a tie: toFixed is correct
  const down = Number(exact.slice(0, exact.length - tail.length)); // truncated, magnitude
  const step = 10 ** -k;
  const even = Math.round(down * 10 ** k) % 2 === 0 ? down : down + step;
  return Number((v < 0 ? -even : even).toFixed(k));
};

/** Tokens whose CENTRE lies in [start, end), matching TokenScores.select. */
function selectTokens(tokens, start, end) {
  const out = { logprob: [], rank: [], entropy: [], mu: [], sigma2: [] };
  let n = 0;
  for (const t of tokens) {
    const centre = (t.start + t.end) / 2.0;
    if (centre >= start && centre < end) {
      out.logprob.push(t.logprob);
      out.rank.push(t.rank);
      out.entropy.push(NaN);
      out.mu.push(NaN);
      out.sigma2.push(NaN);
      n += 1;
    }
  }
  out.n = n;
  return out;
}

export function featuresFor(text, tokens, reference) {
  const spans = splitSentences(text);
  if (!spans.length) return { spans: [], features: [] };

  const base = spans.map((span) => {
    const spanTokens = selectTokens(tokens, span.start, span.end);
    const modelFeats = extractModelFeatures(spanTokens);
    const surfaceFeats = extractSurfaceFeatures(span.text);
    const corpusFeats = extractCorpusFeatures(span.text, reference, modelFeats.mean_logprob);
    return { ...modelFeats, ...surfaceFeats, ...corpusFeats };
  });

  const context = extractContextFeatures(base, spans.length);
  return { spans, features: base.map((b, i) => ({ ...b, ...context[i] })) };
}

function tokenViews(tokens) {
  return tokens.map((t) => {
    let bucket = 'tail';
    for (const [limit, name] of RANK_BUCKETS) {
      if (t.rank <= limit) {
        bucket = name;
        break;
      }
    }
    return {
      text: t.token,
      start: t.start,
      end: t.end,
      logprob: round(t.logprob, 3),
      rank: t.rank,
      bucket,
    };
  });
}

function band(score, bands) {
  if (!bands || bands.unfitted) {
    return {
      band: 'insufficient_evidence',
      bandLabel: 'Not calibrated',
      bandDetail: 'No operating point has been fitted, so no verdict is offered.',
      canExonerate: false,
    };
  }
  if (score >= bands.tMachine) {
    return {
      band: 'likely_machine',
      bandLabel: 'Likely machine-written',
      bandDetail:
        `Above the threshold calibrated so that at most ${(bands.fprBudget * 100).toFixed(0)}% ` +
        `of at-risk human essays are flagged (observed ${(bands.observedFpr * 100).toFixed(1)}% ` +
        `on ${bands.nHuman} held-out documents).`,
      canExonerate: false,
    };
  }
  if (score <= bands.tHuman) {
    return {
      band: 'no_evidence',
      bandLabel: 'No evidence of machine writing',
      bandDetail:
        'This is not a finding that a person wrote it. It means the signals this tool ' +
        'measures are absent, which is also true of capable models -- ' +
        `${(bands.observedMiss * 100).toFixed(0)}% of known machine essays land here.`,
      canExonerate: false,
    };
  }
  return {
    band: 'insufficient_evidence',
    bandLabel: 'Insufficient evidence',
    bandDetail:
      'This document falls between the two calibrated thresholds. The tool declines to ' +
      'answer rather than guess; it abstains on ' +
      `${(bands.abstainHuman * 100).toFixed(0)}% of human and ` +
      `${(bands.abstainMachine * 100).toFixed(0)}% of machine essays by design.`,
    canExonerate: false,
  };
}

/**
 * `models` carries the fitted artifacts: { detector, documentModel, gate, bands, reference,
 * limitations }. `observation` is the observer's reply for this exact text.
 */
export function analyze(text, observation, models, { includeTokens = true, topK = 6, startedAt } = {}) {
  // Timed from the caller's entry point, not from here. A Worker's `Date.now()` is frozen
  // for the duration of synchronous execution — a deliberate defence against timing attacks
  // — so a timer that opens and closes around pure computation always reports 0 ms, which
  // the interface then displayed as the analysis time.
  const started = startedAt ?? Date.now();
  const { detector, documentModel, gate, bands, reference } = models;
  const tokens = observation.tokens ?? [];

  const { spans, features } = featuresFor(text, tokens, reference);

  if (!spans.length) {
    return {
      verdict: { ...aggregate([]), ...band(0, bands), inDomainProbability: null },
      sentences: [],
      passages: [],
      tokens: [],
      meta: meta(observation, models, 0, 0),
      flagThreshold: round(detector.flagThreshold, 4),
      limitations: models.limitations,
    };
  }

  const predictions = features.map((fv) => detector.predict(fv));
  const probs = predictions.map((p) => p.probability);
  const wordCounts = spans.map((s) => tokenizeWords(s.text).length);

  const tokenCounts = spans.map((s) => selectTokens(tokens, s.start, s.end).n);
  const reliableFlags = spans.map((s, i) =>
    tokenCounts[i] >= MIN_TOKENS && wordCounts[i] <= MAX_SENTENCE_WORDS);
  // Smoothing is told which spans are measurable. It is weighted by word count and the
  // unmeasurable spans are the long ones, so without this a 466-word span the verdict
  // refuses to score still decided its 20-word neighbour's smoothed value, and the smoothed
  // value is what draws passages. See smooth_probabilities in detect/document.py.
  const smoothed = smoothProbabilities(probs, wordCounts.slice(), reliableFlags);

  // The last character the observer actually read, used to tell "never looked at" apart from
  // the other reasons a span is unmeasurable.
  const observedChars = tokens.length ? tokens[tokens.length - 1].end : 0;
  const whyUnmeasurable = (i, span) => {
    if (observation.clipped && tokenCounts[i] === 0 && span.start >= observedChars) {
      return 'beyond_observer_window';
    }
    if (wordCounts[i] > MAX_SENTENCE_WORDS) return 'too_long';
    if (tokenCounts[i] < MIN_TOKENS) return 'too_short';
    return null;
  };

  const verdicts = spans.map((span, i) => ({
    index: i,
    start: span.start,
    end: span.end,
    text: span.text,
    probability: probs[i],
    nWords: wordCounts[i],
    smoothed: smoothed[i],
    reliable: reliableFlags[i],
    unreliableReason: whyUnmeasurable(i, span),
  }));

  // The document verdict uses the FITTED sentence threshold; passages use the display
  // threshold. Both match the Python serving path, which does the same thing.
  const verdict = aggregate(verdicts, detector.flagThreshold, documentModel);
  const passages = findPassages(verdicts, FLAG_THRESHOLD);

  // The gate runs after scoring and before the verdict is worded. Scoring anyway keeps the
  // per-sentence evidence visible to a curious reader; an out-of-domain document must still
  // never be handed a band, because the thresholds were calibrated on admissions essays.
  const gfeat = documentGenreFeatures(features);
  const pIn = gate.probability(gfeat);
  const wording = gate.inDomain(gfeat)
    ? band(verdict.anyMachineProbability, bands)
    : {
        band: 'out_of_scope',
        bandLabel: "Outside this tool's scope",
        bandDetail:
          'This does not read as a college admissions personal statement, which is the ' +
          'only kind of writing this detector was fitted and calibrated on. It refuses ' +
          'rather than guessing: pointed at other genres it fails confidently, not ' +
          "gracefully -- it once called a school student's argumentative essay " +
          'machine-written at 97% confidence. Sentence scores are still shown as evidence, ' +
          'but no verdict is offered.',
        canExonerate: false,
      };

  return {
    verdict: {
      machineShare: round(verdict.machineShare, 4),
      machineShareLow: round(verdict.machineShareLow, 4),
      machineShareHigh: round(verdict.machineShareHigh, 4),
      anyMachineProbability: round(verdict.anyMachineProbability, 4),
      nSentences: verdict.nSentences,
      nWords: verdict.nWords,
      nReliableSentences: verdict.nReliableSentences,
      ...wording,
      inDomainProbability: round(pIn, 4),
    },
    sentences: spans.map((span, i) => sentenceReport(i, span, predictions[i], probs[i],
      smoothed[i], wordCounts[i], verdicts[i], topK)),
    passages: passages.map((p) => ({
      start: p.start,
      end: p.end,
      sentenceIndices: p.sentenceIndices,
      nWords: p.nWords,
      meanProbability: round(p.meanProbability, 4),
      peakProbability: round(p.peakProbability, 4),
    })),
    ...(includeTokens ? { tokens: tokenViews(tokens) } : {}),
    meta: meta(observation, models, Date.now() - started, tokens.length),
    flagThreshold: round(detector.flagThreshold, 4),
    // The document threshold the band was decided against. Published so the interface can
    // print "flags at 93%" beside a 16% score, which is what makes an abstention legible as
    // a distance from the line rather than as a low mark out of 100.
    documentThreshold: bands && !bands.unfitted ? round(bands.tMachine, 4) : null,
    limitations: models.limitations,
  };
}

function sentenceReport(i, span, prediction, probability, smoothed, nWords, verdict, topK) {
  const sorted = prediction.contributions
    .map((c, idx) => ({ c, idx }))
    .sort((a, b) => Math.abs(b.c.contribution) - Math.abs(a.c.contribution) || a.idx - b.idx)
    .slice(0, topK);
  const shown = new Set(sorted.map((s) => s.idx));
  let remainder = 0;
  prediction.contributions.forEach((c, idx) => {
    if (!shown.has(idx)) remainder += c.contribution;
  });

  return {
    index: i,
    start: span.start,
    end: span.end,
    text: span.text,
    probability: round(probability, 4),
    smoothed: round(smoothed, 4),
    nWords,
    reliable: verdict.reliable,
    unreliableReason: verdict.unreliableReason,
    logit: round(prediction.logit, 4),
    evidence: sorted.map(({ c }) => {
      const m = FEATURES_BY_NAME[c.name] ?? {};
      return {
        name: c.name,
        label: m.label ?? c.name,
        group: m.group ?? '',
        description: m.description ?? '',
        value: c.measured ? round(c.value, 4) : null,
        z: round(c.z, 3),
        weight: round(c.weight, 4),
        contribution: round(c.contribution, 4),
        toward: c.contribution > 0 ? 'machine' : 'human',
        measured: c.measured,
      };
    }),
    // The bars are the k largest terms, which is NOT the whole logit. Publishing the
    // intercept and the omitted remainder lets a reader check that the arithmetic closes.
    intercept: round(prediction.intercept, 4),
    evidenceRemainder: round(remainder, 4),
    nFeaturesShown: Math.min(topK, prediction.contributions.length),
    nFeaturesTotal: prediction.contributions.length,
  };
}

function meta(observation, models, elapsedMs, nTokens) {
  return {
    observer: observation.model ?? models.observerName,
    device: 'workers-ai',
    hasCorpusReference: Boolean(models.reference),
    nObserverTokens: nTokens,
    elapsedMs,
    clipped: Boolean(observation.clipped),
    // Published so the interface can name the window without hard-coding it. A constant
    // repeated in the page is a constant that drifts from the one the observer used.
    observerCharLimit: MAX_OBSERVER_CHARS,
    trainedOn: models.detector.metadata,
  };
}

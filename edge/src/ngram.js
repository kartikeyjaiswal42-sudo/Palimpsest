/**
 * Port of `palimpsest/scorer/ngram.py`, reading the binary produced by
 * `edge/scripts/build_ngram_bin.py`.
 *
 * The reference is the second observer: an interpolated trigram model fitted on the *human*
 * half of the corpus, so the question it answers is "is this how applicants write" rather
 * than "is this fluent English". The Python docstring explains why it is deliberately not
 * fitted as a human-vs-machine likelihood ratio.
 *
 * All this file changes is storage. Counts are exact — the compiler verifies all 793,205
 * entries round-trip before writing — because `novel_trigram_rate` is a membership test and
 * an approximate structure would quietly shift a feature the classifier carries a weight for.
 */

import { std1, mean } from './pyshim.js';
import { tokenizeWords } from './segment.js';

const LAMBDA3 = 0.5;
const LAMBDA2 = 0.3;
const LAMBDA1 = 0.2;
const BOUNDARY_ID = 0;
const MAGIC = 'PALNGRM1';

export class NgramReference {
  constructor(fields) {
    Object.assign(this, fields);
  }

  static parse(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    if (String.fromCharCode(...bytes.subarray(0, 8)) !== MAGIC) {
      throw new Error('ngram.bin: bad magic; rebuild with edge/scripts/build_ngram_bin.py');
    }
    let off = 8;
    const u32 = () => {
      const v = view.getUint32(off, true);
      off += 4;
      return v;
    };
    const nVocab = u32();
    u32(); // nUnigrams, informational: absent tokens are stored as a zero count
    const nBi = u32();
    const nTri = u32();
    const totalTokens = u32();
    const vocabSize = u32();
    const nDocuments = u32();

    const blobLen = u32();
    const vocab = new TextDecoder().decode(bytes.subarray(off, off + blobLen)).split('\n');
    off += blobLen + ((-blobLen % 4) + 4) % 4;

    const slice = (n) => {
      // Copy rather than subarray: the asset buffer's byte offset is not guaranteed to be
      // 4-byte aligned on every runtime, and an unaligned Uint32Array view throws.
      const out = new Uint32Array(n);
      for (let i = 0; i < n; i += 1) out[i] = view.getUint32(off + 4 * i, true);
      off += 4 * n;
      return out;
    };

    const uniCount = slice(nVocab);
    const biKey = slice(nBi);
    const biCount = slice(nBi);
    const triHi = slice(nTri);
    const triLo = slice(nTri);
    const triCount = slice(nTri);

    const index = new Map();
    for (let i = 0; i < vocab.length; i += 1) index.set(vocab[i], i);

    return new NgramReference({
      index, uniCount, biKey, biCount, triHi, triLo, triCount,
      totalTokens, vocabSize, nDocuments,
    });
  }

  // -- exact count lookups ------------------------------------------------------------

  bigramCount(a, b) {
    const key = a * 65536 + b;
    const keys = this.biKey;
    let lo = 0;
    let hi = keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const v = keys[mid];
      if (v === key) return this.biCount[mid];
      if (v < key) lo = mid + 1;
      else hi = mid - 1;
    }
    return 0;
  }

  trigramIndex(a, b, c) {
    const lokey = b * 65536 + c;
    let lo = 0;
    let hi = this.triHi.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const h = this.triHi[mid];
      const l = this.triLo[mid];
      if (h === a && l === lokey) return mid;
      if (h < a || (h === a && l < lokey)) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  trigramCount(a, b, c) {
    const i = this.trigramIndex(a, b, c);
    return i < 0 ? 0 : this.triCount[i];
  }

  // -- interpolated model -------------------------------------------------------------

  pUnigram(w) {
    return (this.uniCount[w] + 1.0) / (this.totalTokens + this.vocabSize);
  }

  pBigram(prev, w) {
    const denom = prev !== BOUNDARY_ID ? this.uniCount[prev] : this.nDocuments;
    if (denom === 0) return this.pUnigram(w);
    return this.bigramCount(prev, w) / denom;
  }

  pTrigram(a, b, w) {
    const denom = this.bigramCount(a, b);
    if (denom === 0) return this.pBigram(b, w);
    return this.trigramCount(a, b, w) / denom;
  }

  logprob(a, b, w) {
    const p =
      LAMBDA3 * this.pTrigram(a, b, w) + LAMBDA2 * this.pBigram(b, w) + LAMBDA1 * this.pUnigram(w);
    return Math.log(Math.max(p, 1e-12));
  }

  /** Word ids for `text`, with anything outside the closed vocabulary mapped to `<unk>`. */
  ids(text) {
    const unk = this.index.get('<unk>');
    const out = [];
    for (const w of tokenizeWords(text)) {
      const id = this.index.get(w);
      // `<s>` (id 0) is never in the unigram table, so a token that somehow decoded to it
      // is out-of-vocabulary exactly as Python's `w in self.unigrams` would decide.
      out.push(id === undefined || id === BOUNDARY_ID ? unk : id);
    }
    return out;
  }

  surprisals(text) {
    const words = this.ids(text);
    const padded = [BOUNDARY_ID, BOUNDARY_ID, ...words];
    const out = [];
    for (let i = 2; i < padded.length; i += 1) {
      out.push(-this.logprob(padded[i - 2], padded[i - 1], padded[i]));
    }
    return out;
  }

  novelTrigramRate(text) {
    const words = this.ids(text);
    if (words.length < 3) return NaN;
    let unseen = 0;
    let total = 0;
    for (let i = 2; i < words.length; i += 1) {
      total += 1;
      if (this.trigramIndex(words[i - 2], words[i - 1], words[i]) < 0) unseen += 1;
    }
    return unseen / total;
  }
}

export const CORPUS_FEATURE_NAMES = [
  'corpus_surprisal_mean',
  'corpus_surprisal_sd',
  'novel_trigram_rate',
  'fluency_typicality_gap',
];

/** Port of `features/corpus.py`. */
export function extractCorpusFeatures(text, reference, meanLogprob) {
  const nan = Object.fromEntries(CORPUS_FEATURE_NAMES.map((n) => [n, NaN]));
  if (!reference) return nan;

  const surprisals = reference.surprisals(text);
  if (surprisals.length < 4) return nan;

  const corpusMean = mean(surprisals);
  const gap = Number.isFinite(meanLogprob) ? corpusMean - -meanLogprob : NaN;

  return {
    corpus_surprisal_mean: corpusMean,
    corpus_surprisal_sd: surprisals.length > 1 ? std1(surprisals) : NaN,
    novel_trigram_rate: reference.novelTrigramRate(text),
    fluency_typicality_gap: gap,
  };
}

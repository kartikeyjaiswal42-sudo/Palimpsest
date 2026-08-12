/**
 * Python and NumPy semantics that JavaScript does not share.
 *
 * This file exists because the port must reproduce the Python pipeline's numbers, not
 * merely resemble them. The differences that bite are small and silent:
 *
 *   - `str.strip()` strips U+001C-U+001F and U+0085, which `String.trim()` leaves alone;
 *     `trim()` strips U+FEFF, which Python leaves alone. A stray one of either shifts every
 *     character offset in the document and every span the UI highlights.
 *   - Python's `\w` and `\b` are Unicode-aware for `str` patterns. JavaScript's are ASCII
 *     even under `/u`, so `café's` is a contraction in Python and not one here unless the
 *     boundary is written out by hand.
 *   - `np.percentile` interpolates linearly between order statistics. The textbook
 *     "sort and take the middle" answer disagrees with it on most inputs.
 *   - `np.std()` defaults to ddof=0; every call in this project passes ddof=1.
 *
 * Each of these is small enough to look like a rounding difference in review and large
 * enough to move a sentence across a threshold. test/parity.test.mjs checks the whole
 * pipeline against Python output rather than trusting this comment.
 */

// -- Python character classes ---------------------------------------------------------

/** Exactly the characters for which Python's `str.isspace()` is true. */
const PY_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
  0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
  0x2029, 0x202f, 0x205f, 0x3000,
]);

export function isSpace(ch) {
  return ch !== undefined && PY_SPACE.has(ch.codePointAt(0));
}

/**
 * The same set as a regex character class, for patterns that use Python's `\s`.
 *
 * JavaScript's `\s` is not a synonym: it omits U+001C-U+001F and U+0085, and it adds
 * U+FEFF. `\n\s*\n` is the paragraph splitter, so either difference silently changes where
 * paragraphs — and therefore sentences, and therefore every highlight — begin.
 */
export const PY_SPACE_CLASS =
  '[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';

const RE_ALPHA = /^\p{L}$/u;
const RE_UPPER = /^\p{Lu}$/u;
const RE_LOWER = /^\p{Ll}$/u;
const RE_DIGIT = /^\p{Nd}$/u;

export const isAlpha = (ch) => ch !== undefined && RE_ALPHA.test(ch);
export const isUpper = (ch) => ch !== undefined && RE_UPPER.test(ch);
export const isLower = (ch) => ch !== undefined && RE_LOWER.test(ch);
export const isDigit = (ch) => ch !== undefined && RE_DIGIT.test(ch);

/** `str.lstrip()`, returned as the count of characters that would be removed.
 *  Callers strip from the right with `isSpace` directly, because they need the offsets. */
export function leadingSpace(s) {
  let i = 0;
  while (i < s.length && isSpace(s[i])) i += 1;
  return i;
}

/** `str.rstrip('.')`. */
export function rstripDots(s) {
  let j = s.length;
  while (j > 0 && s[j - 1] === '.') j -= 1;
  return s.slice(0, j);
}

// -- NumPy reductions -----------------------------------------------------------------

export const isFinite_ = (v) => typeof v === 'number' && Number.isFinite(v);

export function mean(xs) {
  if (!xs.length) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** `np.std(x, ddof=1)`. Returns NaN for n < 2, matching the guards at every call site. */
export function std1(xs) {
  const n = xs.length;
  if (n < 2) return NaN;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / (n - 1));
}

/**
 * `np.percentile(xs, q)` with the default linear interpolation.
 *
 * NumPy's "linear" method places the qth percentile at index q/100 * (n-1) of the sorted
 * data and interpolates between the two neighbouring order statistics. Rounding to a
 * single index instead — the obvious implementation — disagrees on most inputs.
 */
export function percentile(xs, q) {
  const n = xs.length;
  if (!n) return NaN;
  const sorted = Float64Array.from(xs).sort();
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export const median = (xs) => percentile(xs, 50);

/** `np.nanmedian` over a list: NaN in, ignored; all-NaN in, NaN out. */
export function nanMedian(xs) {
  const ok = xs.filter(isFinite_);
  return ok.length ? median(ok) : NaN;
}

/** `np.nanmean`. */
export function nanMean(xs) {
  const ok = xs.filter(isFinite_);
  return ok.length ? mean(ok) : NaN;
}

/** `np.average(values, weights=w)`. */
export function average(values, weights) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < values.length; i += 1) {
    num += values[i] * weights[i];
    den += weights[i];
  }
  return num / den;
}

export const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * `np.interp(x, xp, fp)` — piecewise-linear, clamped at both ends.
 * `xp` is assumed ascending, as it is for the calibration knots.
 */
export function interp(x, xp, fp) {
  const n = xp.length;
  if (!n) return NaN;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xp[hi] - xp[lo];
  if (span === 0) return fp[lo];
  return fp[lo] + ((fp[hi] - fp[lo]) * (x - xp[lo])) / span;
}

export const sigmoid = (x) => 1 / (1 + Math.exp(-clip(x, -30, 30)));

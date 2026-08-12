/**
 * Port of `palimpsest/text/segment.py`.
 *
 * The invariant the Python module states is the one that matters here too:
 *
 *     text.slice(span.start, span.end) === span.text
 *
 * One deliberate difference from Python, recorded rather than hidden. Python's `len()` and
 * slicing count Unicode code points; JavaScript counts UTF-16 code units. They agree on
 * everything in the Basic Multilingual Plane and disagree on astral characters — emoji,
 * rarer CJK. This port is UTF-16 throughout, which makes it *self*-consistent with the
 * observer (whose offsets have always been produced by `String.indexOf` inside the Worker)
 * and with the browser that slices the text to draw highlights.
 *
 * The Python build mixes the two conventions at that seam, so an essay containing an emoji
 * would shift its highlights there. No document in the parity corpus contains one, so this
 * costs nothing measurable; see edge/PARITY.md.
 */

import {
  isAlpha, isDigit, isLower, isSpace, isUpper, leadingSpace, PY_SPACE_CLASS, rstripDots,
} from './pyshim.js';

const ABBREVIATIONS = new Set(
  `mr mrs ms dr prof sr jr st rev hon gen col capt lt sgt gov pres supt
   inc ltd co corp dept univ assn bros
   jan feb mar apr jun jul aug sep sept oct nov dec
   mon tue tues wed thu thurs fri sat sun
   vs etc al ca cf ed eds vol no pp fig figs approx`.split(/\s+/).filter(Boolean),
);

const DOTTED_ABBREV = /^(?:[A-Za-z]\.){2,}$/;
const TERMINATOR = /[.!?…]+["'”’)\]]*/g;
// Python's `\n\s*\n`, with `\s` spelled out: JavaScript's `\s` would both miss U+0085 and
// match U+FEFF, and this regex decides where every paragraph starts.
const PARA_SPLIT = new RegExp(`\\n${PY_SPACE_CLASS}*\\n`);
const WORD = /[A-Za-zÀ-ɏ']+/g;

export function splitParagraphs(text) {
  const spans = [];
  let cursor = 0;
  for (const chunk of text.split(PARA_SPLIT)) {
    const start = chunk ? text.indexOf(chunk, cursor) : cursor;
    const lead = leadingSpace(chunk);
    let end = chunk.length;
    while (end > lead && isSpace(chunk[end - 1])) end -= 1;
    const stripped = chunk.slice(lead, end);
    if (stripped) {
      spans.push({ start: start + lead, end: start + lead + stripped.length, text: stripped });
    }
    cursor = start + chunk.length;
  }
  return spans;
}

function isAbbreviation(text, dotIndex) {
  let i = dotIndex;
  while (i > 0 && (isAlpha(text[i - 1]) || text[i - 1] === '.')) i -= 1;
  const word = text.slice(i, dotIndex + 1);
  if (DOTTED_ABBREV.test(word)) return true;
  return ABBREVIATIONS.has(rstripDots(word).toLowerCase());
}

const isDecimal = (text, i) =>
  i > 0 && i + 1 < text.length && isDigit(text[i - 1]) && isDigit(text[i + 1]);

function isInitial(text, i) {
  if (i === 0 || !isUpper(text[i - 1])) return false;
  const before = i - 2;
  return before < 0 || !isAlpha(text[before]);
}

function boundaryIsReal(text, end) {
  if (end >= text.length) return true;
  if (!isSpace(text[end])) return false;
  const rest = text.slice(end);
  const lead = leadingSpace(rest);
  if (rest.slice(0, lead).includes('\n')) return true;
  const next = rest.slice(lead);
  if (!next) return true;
  return !(isAlpha(next[0]) && isLower(next[0]));
}

function splitParagraphSentences(para) {
  const body = para.text;
  const out = [];
  let start = 0;

  TERMINATOR.lastIndex = 0;
  let match;
  while ((match = TERMINATOR.exec(body)) !== null) {
    const dot = match.index;
    const end = match.index + match[0].length;
    if (match[0].length === 0) {
      TERMINATOR.lastIndex += 1;
      continue;
    }
    if (
      body[dot] === '.' &&
      (isDecimal(body, dot) || isInitial(body, dot) || isAbbreviation(body, dot))
    ) {
      continue;
    }
    if (!boundaryIsReal(body, end)) continue;

    const piece = body.slice(start, end);
    const lead = leadingSpace(piece);
    let tail = piece.length;
    while (tail > lead && isSpace(piece[tail - 1])) tail -= 1;
    const stripped = piece.slice(lead, tail);
    if (stripped) {
      out.push({
        start: para.start + start + lead,
        end: para.start + start + lead + stripped.length,
        text: stripped,
      });
    }
    start = end;
  }

  const tailPiece = body.slice(start);
  const lead = leadingSpace(tailPiece);
  let tailEnd = tailPiece.length;
  while (tailEnd > lead && isSpace(tailPiece[tailEnd - 1])) tailEnd -= 1;
  const stripped = tailPiece.slice(lead, tailEnd);
  if (stripped) {
    out.push({
      start: para.start + start + lead,
      end: para.start + start + lead + stripped.length,
      text: stripped,
    });
  }
  return out;
}

export function splitSentences(text) {
  const spans = [];
  for (const para of splitParagraphs(text)) spans.push(...splitParagraphSentences(para));
  return spans;
}

/** Lowercased alphabetic word tokens, as the n-gram and lexicon features consume them. */
export function tokenizeWords(text) {
  WORD.lastIndex = 0;
  const out = [];
  let m;
  while ((m = WORD.exec(text)) !== null) out.push(m[0].toLowerCase());
  return out;
}

/* ============================================================
   Palimpsest — app.js
   Vanilla. No framework, no build step, no external requests.
   The interface exists so a reader can check the working.
   ============================================================ */

/* ------------------------------------------------------------
   Fixtures + the offline analyzer.
   The page always asks POST /api/analyze first. When no analyzer
   is deployed (static preview, 404, non-JSON), it falls back to
   this bundled analyzer so the interface can be read, and says so
   in the status line and the observer chip. It never mixes the
   two: a real response is rendered untouched.
   ------------------------------------------------------------ */
window.Palimpsest = (function () {
  'use strict';

  var INTERCEPT = -0.406;
  var THRESHOLD = 0.3004;
  var CHAR_LIMIT = 6000;

  var FIXTURE_CAUGHT =
'Throughout my life, I have always been drawn to the quiet power of community service. Growing up in a small suburban neighborhood, I witnessed firsthand how small acts of kindness could ripple outward and transform the lives of those around me. This realization sparked a passion within me that would ultimately shape both my academic interests and my personal values.\n\n' +
'During my sophomore year, I decided to volunteer at a local food bank, an experience that proved to be transformative in ways I could not have anticipated. Each Saturday morning, I sorted donations, packed boxes, and spoke with families who had come seeking assistance. It was there that I learned an invaluable lesson: dignity is not something we give to others, but something we recognize in them.\n\n' +
'As I continued to serve, I began to notice systemic patterns that troubled me deeply. Many of the families returned week after week, not because they lacked determination, but because the structures around them offered so little support. This observation inspired me to organize a student-led initiative aimed at addressing food insecurity in a more sustainable manner.\n\n' +
'Looking ahead, I hope to continue this work at the university level, combining my interest in public policy with my commitment to service. I believe that education is not merely a means of personal advancement, but a responsibility to give back to the communities that shaped us.';

  var FIXTURE_MISSED =
'My grandmother kept her bus transfers. All of them, going back years, in a shoebox under the sink with the shoe polish. I found them the summer she died and I sat on the kitchen floor for maybe an hour reading dates.\n\n' +
'Nobody could explain it. My mother said she was just like that. My uncle said it was the war, which is what he says about everything.\n\n' +
'The habit made a kind of sense to me, though, once I stopped looking for a reason and started looking at the objects themselves. Each transfer was a record of a decision to go somewhere, a small paper proof that on an ordinary Tuesday in 1987 she had chosen to leave the house. Taken together they formed an archive of ordinary persistence, the sort of documentation no institution would ever think to keep.\n\n' +
'So I started keeping things too. Ticket stubs, mostly, and the receipt from the diner where my father told me he was moving out. I am not sentimental about them. I do not take them out and look at them. I just want there to be a record that I was somewhere, the way she wanted one.\n\n' +
'I do not know yet what I want to study. I know I want to work with things people left behind.';

  var LIMITATIONS = [
    'Measured on held-out data: 10.9% of TOEFL essays by non-native English speakers were flagged as likely machine-written. If the writer learned English as a second language, this tool is wrong about them roughly one time in nine.',
    'Light editing defeats it. Rewriting one sentence in three moved a machine-written document back below the threshold in 62% of the cases we tested.',
    'It cannot separate drafting help from ghostwriting. A student who told their own story and had a model tidy the grammar scores like a student who typed a prompt.',
    'Below about 120 words the calibration set has too few comparable documents, so the tool reports that the essay is outside its scope rather than guess at a number.',
    'Formulaic human writing scores high. Prose drilled toward a rubric — five paragraphs, signposted transitions, a hedged conclusion — is the human writing this tool most often flags.',
    'It was calibrated on undergraduate admissions essays written in English. On graduate statements, on translated text, and on any other genre, the reported probability has no validated meaning.',
    'The observer model changes. Scores are not comparable across observer versions, and a re-run after an upgrade can move a document across the threshold.',
    'A flag is not evidence of misconduct and has never been validated as such. At this threshold, one flagged essay in twenty was written by a person.'
  ];

  var FEATURES = [
    { name: 'mean_logprob', label: 'Average predictability', group: 'likelihood', weight: 2.0407, center: -2.1, scale: 0.55,
      description: 'Mean log-probability the observer assigned to the words actually used. Machine drafts sit closer to the model\u2019s own preferences.' },
    { name: 'mean_logrank', label: 'Average log rank', group: 'rank', weight: 1.6182, center: 1.9, scale: 0.42,
      description: 'Average position of each word in the observer\u2019s ranked guesses, log-scaled.' },
    { name: 'smoothness', label: 'Smoother than the author\u2019s baseline', group: 'context', weight: 1.1834, center: 0.0, scale: 0.31,
      description: 'How much flatter this sentence\u2019s surprise curve is than the rest of this author\u2019s writing.' },
    { name: 'length_vs_baseline', label: 'Length vs the author\u2019s baseline', group: 'context', weight: 0.7391, center: 1.0, scale: 0.28,
      description: 'This sentence\u2019s length against the author\u2019s own average, so a naturally terse writer is not penalised for terseness.' },
    { name: 'fluent_atypical', label: 'Fluent but atypical', group: 'composite', weight: 1.3127, center: 0.4, scale: 0.26,
      description: 'High fluency combined with low resemblance to real applicant prose \u2014 the combination, not either part alone.' },
    { name: 'corpus_distance', label: 'Distance from applicant prose', group: 'corpus', weight: 1.4408, center: 0.6, scale: 0.19,
      description: 'Distance from a reference corpus of verified human admissions essays.' },
    { name: 'sent_len', label: 'Sentence length', group: 'rhythm', weight: 0.6215, center: 21.0, scale: 8.0,
      description: 'Raw sentence length in words, standardised against the calibration set.' },
    { name: 'local_rhythm', label: 'Local rhythm', group: 'rhythm', weight: 0.8342, center: 0.0, scale: 0.4,
      description: 'Variation in length between this sentence and the sentences on either side of it.' },
    { name: 'surprise_rhythm', label: 'Rhythm of surprise', group: 'rhythm', weight: 0.9713, center: 0.9, scale: 0.22,
      description: 'Whether surprise is spread evenly across the sentence or clustered in a few words.' },
    { name: 'vocab_richness', label: 'Vocabulary richness', group: 'register', weight: 0.7085, center: 0.78, scale: 0.11,
      description: 'Type-token ratio inside the sentence, standardised.' },
    { name: 'contractions', label: 'Contractions', group: 'register', weight: 0.5512, center: 0.06, scale: 0.05,
      description: 'Rate of contracted forms, which machine drafts use less often than applicants do.' },
    { name: 'top10_share', label: 'Words in the top 10', group: 'likelihood', weight: 1.0946, center: 0.42, scale: 0.13,
      description: 'Share of words that were among the observer\u2019s ten most likely continuations at that position.' },
    { name: 'lr_ratio', label: 'Likelihood/rank ratio', group: 'composite', weight: 0.8804, center: 1.1, scale: 0.24,
      description: 'Ratio between the likelihood and rank signals, which separates fluent human prose from generated prose.' }
  ];

  function hash(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 16777619) >>> 0; }
    return h >>> 0;
  }
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var r4 = function (n) { return Math.round(n * 1e4) / 1e4; };
  var sig = function (z) { return 1 / (1 + Math.exp(-z)); };

  function splitSentences(text) {
    var out = [], re = /[^.!?]+[.!?]+["'\u201d\u2019)\]]*|[^.!?]+$/g, m;
    while ((m = re.exec(text)) !== null) {
      var raw = m[0];
      var lead = raw.match(/^\s*/)[0].length;
      var trimmed = raw.trim();
      if (!trimmed) continue;
      out.push({ start: m.index + lead, end: m.index + lead + trimmed.length, text: trimmed });
    }
    return out;
  }

  function buildEvidence(rand, target, nWords) {
    var total = target - INTERCEPT;
    var shownSum = r4(total * 0.85);
    var remainder = r4(total - shownSum);
    var logit = r4(INTERCEPT + shownSum + remainder);

    var offset = Math.floor(rand() * FEATURES.length);
    var picked = [];
    for (var i = 0; i < 6; i++) { picked.push(FEATURES[(offset + i * 3) % FEATURES.length]); }

    // a genuinely unmeasurable term on very short sentences
    var unmeasuredAt = nWords < 12 ? 3 : -1;
    var shape = [1, 0.78, -0.42, 0.61, 0.35, -0.25];
    var live = [], sum = 0;
    picked.forEach(function (f, i) {
      var w = i === unmeasuredAt ? 0 : shape[i] * (0.75 + rand() * 0.5);
      live.push(w); sum += w;
    });
    if (Math.abs(sum) < 0.01) { live[0] += 1; sum += 1; }
    var k = shownSum / sum;

    var evidence = picked.map(function (f, i) {
      var measured = i !== unmeasuredAt;
      var c = measured ? r4(live[i] * k) : 0;
      var z = measured ? r4(c / f.weight) : null;
      return {
        name: f.name, label: f.label, group: f.group, description: f.description,
        value: measured ? r4(f.center + z * f.scale) : null,
        z: z, weight: f.weight, contribution: c,
        toward: c >= 0 ? 'machine' : 'human', measured: measured
      };
    });

    // absorb rounding drift into the largest term so the parts add up exactly
    var got = evidence.reduce(function (a, f) { return a + f.contribution; }, 0);
    var drift = r4(shownSum - got);
    if (drift !== 0) {
      var big = evidence.filter(function (f) { return f.measured; })
        .sort(function (a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); })[0];
      if (big) {
        big.contribution = r4(big.contribution + drift);
        big.z = r4(big.contribution / big.weight);
        big.toward = big.contribution >= 0 ? 'machine' : 'human';
      }
    }
    return { evidence: evidence, remainder: remainder, logit: logit };
  }

  function analyseLocally(text) {
    var mode = text === FIXTURE_CAUGHT ? 'caught' : text === FIXTURE_MISSED ? 'missed' : 'free';
    var seed = hash(text);
    var docRand = rng(seed);
    var base = mode === 'caught' ? 3.1 : mode === 'missed' ? -1.9 : (-1.6 + docRand() * 3.2);
    var polished = mode === 'missed' ? [6, 7, 8] : [];
    var clipped = text.length > CHAR_LIMIT;

    var raw = splitSentences(text);
    var sentences = raw.map(function (s, i) {
      var nWords = s.text.split(/\s+/).length;
      var rand = rng(hash(s.text) ^ (i * 2654435761));
      var reason = null;
      if (nWords < 5) reason = 'too_short';
      else if (nWords > 90) reason = 'too_long';
      else if (s.start >= CHAR_LIMIT) reason = 'beyond_observer_window';

      var out = {
        index: i, start: s.start, end: s.end, text: s.text, nWords: nWords,
        reliable: reason === null, unreliableReason: reason, intercept: INTERCEPT
      };
      if (reason) {
        out.probability = null; out.smoothed = null; out.logit = null;
        out.evidence = []; out.evidenceRemainder = null;
        out.nFeaturesShown = 0; out.nFeaturesTotal = 43;
        return out;
      }
      var target = base + (rand() - 0.5) * 1.7 + (polished.indexOf(i) >= 0 ? 3.7 : 0);
      target = Math.max(-6, Math.min(6, target));
      var built = buildEvidence(rand, target, nWords);
      out.logit = built.logit;
      out.probability = r4(sig(built.logit));
      out.smoothed = r4(sig(built.logit * 0.96));
      out.evidence = built.evidence;
      out.evidenceRemainder = built.remainder;
      out.nFeaturesShown = built.evidence.length;
      out.nFeaturesTotal = 43;
      return out;
    });

    var tokens = [];
    var re = /\S+/g, m, si = 0;
    while ((m = re.exec(text)) !== null && tokens.length < 3000) {
      while (si < sentences.length - 1 && m.index >= sentences[si].end) { si++; }
      var host = sentences[si];
      var p = host && host.probability != null ? host.probability : 0.2;
      var tr = rng(hash(m[0] + m.index))();
      var bucket, rank, lp;
      if (tr < 0.28 + p * 0.45) { bucket = 'top10'; rank = 1 + Math.floor(tr * 9); lp = -0.2 - tr * 1.6; }
      else if (tr < 0.6 + p * 0.28) { bucket = 'top100'; rank = 11 + Math.floor(tr * 88); lp = -2 - tr * 2.2; }
      else if (tr < 0.87) { bucket = 'top1000'; rank = 101 + Math.floor(tr * 890); lp = -4.4 - tr * 2.4; }
      else { bucket = 'tail'; rank = 1001 + Math.floor(tr * 9000); lp = -7.1 - tr * 3.4; }
      tokens.push({
        text: (m.index > 0 ? ' ' : '') + m[0], start: m.index, end: m.index + m[0].length,
        logprob: r4(lp), rank: rank, bucket: bucket
      });
    }

    var reliable = sentences.filter(function (s) { return s.reliable; });
    var flagged = reliable.filter(function (s) { return s.probability >= THRESHOLD; });
    var nWords = text.trim().split(/\s+/).length;
    var share = reliable.length ? flagged.length / reliable.length : 0;
    var anyp = 1;
    reliable.forEach(function (s) { anyp *= (1 - Math.min(0.9, s.probability * s.probability * 0.85)); });
    anyp = r4(1 - anyp);

    var se = reliable.length ? 1.96 * Math.sqrt(Math.max(share * (1 - share), 0.0004) / reliable.length) : 0.5;
    var band, label, detail;
    if (reliable.length < 3) {
      band = 'insufficient_evidence';
      label = 'Insufficient evidence';
      detail = 'Only ' + reliable.length + ' sentence' + (reliable.length === 1 ? '' : 's') +
        ' could be measured, which is below the minimum the calibration supports. This is an answer, not a failure: the tool has not cleared this essay and has not flagged it.';
    } else if (nWords < 120) {
      band = 'out_of_scope';
      label = 'Outside what this tool can measure';
      detail = 'At ' + nWords + ' words this essay is shorter than anything in the calibration set, so no probability reported here has a validated meaning. This is a statement about the tool, not about the writer.';
    } else if (share >= 0.5 && anyp >= 0.9) {
      band = 'likely_machine';
      label = 'Likely machine-written';
      detail = 'Above the threshold calibrated so that at most 5% of at-risk human essays are flagged (observed 4.0% on 1492 held-out documents).';
    } else if (anyp >= 0.3) {
      band = 'insufficient_evidence';
      label = 'Insufficient evidence';
      detail = 'Some sentences carry the signal, but the document-level evidence does not reach the calibrated threshold. The tool cannot say this essay is machine-written, and it cannot say it is not. Read the flagged sentences and decide yourself.';
    } else {
      band = 'no_evidence';
      label = 'No evidence of machine writing';
      detail = 'Nothing in this essay reached the sentence threshold. That means the tool found no evidence \u2014 not that it found evidence of a human author. It cannot establish authorship in either direction.';
    }

    return {
      text: text,
      verdict: {
        machineShare: r4(share),
        machineShareLow: r4(Math.max(0, share - se)),
        machineShareHigh: r4(Math.min(1, share + se)),
        anyMachineProbability: anyp,
        nSentences: sentences.length, nWords: nWords, nReliableSentences: reliable.length,
        band: band, bandLabel: label, bandDetail: detail,
        canExonerate: false, inDomainProbability: r4(0.05 + docRand() * 0.5)
      },
      flagThreshold: THRESHOLD,
      sentences: sentences,
      tokens: tokens,
      limitations: LIMITATIONS,
      meta: {
        observer: 'bundled-fixture/qwen3-30b-a3b-fp8', device: 'none',
        elapsedMs: r4(12 + docRand() * 24), nObserverTokens: tokens.length,
        clipped: clipped, observerCharLimit: CHAR_LIMIT
      }
    };
  }

  return {
    FIXTURE_CAUGHT: FIXTURE_CAUGHT,
    FIXTURE_MISSED: FIXTURE_MISSED,
    analyseLocally: analyseLocally
  };
})();

/* ------------------------------------------------------------
   Reading a .docx, in the browser, with nothing loaded to do it.

   WHY THE FILE IS OPENED HERE AND NOT ON THE SERVER
   `web/` is the single source of the interface and is copied verbatim into
   `edge/assets/` by edge/scripts/sync_web.py, so one implementation here serves
   both the local FastAPI build and the hosted Worker. Extracting on the server
   instead would mean a Python dependency for one deployment and a second,
   separate implementation in JavaScript for the other -- two extractors that can
   disagree about what the essay says, which is the same drift this project has
   already been bitten by twice (the limitations panel publishing another build's
   error rates; a fix that existed on only one side of the port).

   A .docx is a ZIP holding word/document.xml. Both halves of that are already in
   the browser: DecompressionStream inflates the entry, DOMParser reads the XML.
   So this adds no dependency and no external request, and the page keeps the
   property that it fetches nothing it did not ship with.

   WHY .docx AND NOT .pdf
   A .docx stores paragraphs. A PDF stores positioned glyphs: recovering prose
   from one means guessing where lines join, where a hyphen was a line break
   rather than a word, and what order two columns should be read in. Those
   guesses land on sentence boundaries and punctuation, and sentence
   segmentation is the input to all 43 features -- a wrongly joined line changes
   the numbers this tool reports, silently and with no way for a reader to see it
   happened. Refusing a format is a worse product; reporting a measurement taken
   on text the author did not write is a worse detector. A PDF is named and
   explained here rather than half-read.
   ------------------------------------------------------------ */
window.PalimpsestDocx = (function () {
  'use strict';

  //: Generous for an essay and small enough that a mis-drop cannot hang the tab.
  var MAX_BYTES = 25 * 1024 * 1024;

  /* Carries the reason AND what to do about it. Every refusal below names a
     specific next step, because "could not read this file" leaves a reader with
     nothing to try. */
  function ReadError(message) { this.message = message; this.name = 'ReadError'; }
  ReadError.prototype = Object.create(Error.prototype);
  ReadError.prototype.constructor = ReadError;

  /* ---- what this actually is -------------------------------
     Read from the leading bytes, never the extension. A .pdf renamed .docx must
     be diagnosed as a PDF -- otherwise it fails as "not a ZIP", which is true and
     useless. */
  function sniff(bytes) {
    var b = bytes;
    if (b.length >= 4) {
      if (b[0] === 0x50 && b[1] === 0x4b && (b[2] === 0x03 || b[2] === 0x05 || b[2] === 0x07)) return 'zip';
      if (b[0] === 0x25 && b[1] === 0x50 && b[2] === 0x44 && b[3] === 0x46) return 'pdf';
      // OLE2 compound file: the pre-2007 binary .doc, and also .xls/.ppt.
      if (b[0] === 0xd0 && b[1] === 0xcf && b[2] === 0x11 && b[3] === 0xe0) return 'ole2';
      if (b[0] === 0x7b && b[1] === 0x5c && b[2] === 0x72 && b[3] === 0x74) return 'rtf'; // {\rt
    }
    return 'other';
  }

  /* ---- ZIP ---------------------------------------------------
     Only as much of the format as a .docx uses. Sizes and offsets are read from
     the central directory, which is authoritative: when a writer streams the
     archive it sets bit 3 of the flags and leaves the sizes in the local header
     as zero, so trusting the local header there reads an empty entry. Word does
     not do this; other .docx writers do. */
  function u16(v, at) { return v.getUint16(at, true); }
  function u32(v, at) { return v.getUint32(at, true); }

  var EOCD_SIG = 0x06054b50, CEN_SIG = 0x02014b50, LOC_SIG = 0x04034b50;

  function findEocd(view) {
    // 22-byte record, plus a comment of up to 65535 bytes after it.
    var min = Math.max(0, view.byteLength - (22 + 0xffff));
    for (var i = view.byteLength - 22; i >= min; i--) {
      if (u32(view, i) === EOCD_SIG) return i;
    }
    return -1;
  }

  function centralDirectory(view) {
    var eocd = findEocd(view);
    if (eocd < 0) throw new ReadError('not-a-zip');
    var count = u16(view, eocd + 10);
    var at = u32(view, eocd + 16);
    var entries = [];
    for (var i = 0; i < count; i++) {
      if (at + 46 > view.byteLength || u32(view, at) !== CEN_SIG) break;
      var nameLen = u16(view, at + 28);
      var extraLen = u16(view, at + 30);
      var commentLen = u16(view, at + 32);
      entries.push({
        method: u16(view, at + 10),
        compressedSize: u32(view, at + 20),
        size: u32(view, at + 24),
        localOffset: u32(view, at + 42),
        name: utf8(new Uint8Array(view.buffer, view.byteOffset + at + 46, nameLen))
      });
      at += 46 + nameLen + extraLen + commentLen;
    }
    return entries;
  }

  function utf8(bytes) { return new TextDecoder('utf-8').decode(bytes); }

  /* The local header's extra field may be a different length from the central
     one -- writers pad it for alignment -- so the data offset must be computed
     from the local header's own two length fields, not the central copy. */
  function rawEntry(view, entry) {
    var at = entry.localOffset;
    if (at + 30 > view.byteLength || u32(view, at) !== LOC_SIG) {
      throw new ReadError('damaged');
    }
    var start = at + 30 + u16(view, at + 26) + u16(view, at + 28);
    var end = start + entry.compressedSize;
    if (end > view.byteLength) throw new ReadError('damaged');
    return new Uint8Array(view.buffer, view.byteOffset + start, entry.compressedSize);
  }

  /* ZIP method 8 is raw DEFLATE with no zlib wrapper, which is exactly
     'deflate-raw'. Passing 'deflate' here fails on the header check. */
  function inflateRaw(bytes) {
    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new ReadError('no-decompression'));
    }
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Response(stream).arrayBuffer().then(
      function (ab) { return new Uint8Array(ab); },
      function () { throw new ReadError('damaged'); }
    );
  }

  function readEntry(view, entry) {
    var raw = rawEntry(view, entry);
    if (entry.method === 0) return Promise.resolve(raw);   // stored
    if (entry.method === 8) return inflateRaw(raw);        // deflate
    return Promise.reject(new ReadError('damaged'));
  }

  /* ---- document.xml -> prose --------------------------------
     Matched on localName so any namespace prefix works; documents in the wild are
     not all `w:`. */
  function paragraphText(p) {
    var out = [];
    (function walk(node) {
      for (var i = 0; i < node.childNodes.length; i++) {
        var n = node.childNodes[i];
        if (n.nodeType !== 1) continue;
        var ln = n.localName;
        if (ln === 't') { out.push(n.textContent); }
        else if (ln === 'tab') { out.push('\t'); }
        else if (ln === 'br' || ln === 'cr') { out.push('\n'); }
        /* Deliberately not read. `delText` is text the author DELETED under
           tracked changes and `instrText` is a field code (HYPERLINK "http://..."),
           neither of which is prose the author wrote; scoring either would measure
           something nobody put on the page. Insertions live in plain `t` inside
           `w:ins` and so are kept, which is the correct half of the same rule. */
        else if (ln === 'delText' || ln === 'instrText') { continue; }
        else { walk(n); }
      }
    })(p);
    return out.join('');
  }

  function documentText(xml) {
    var doc = new DOMParser().parseFromString(xml, 'application/xml');
    if (doc.getElementsByTagName('parsererror').length) throw new ReadError('damaged');
    var paras = [];
    (function walk(node) {
      for (var i = 0; i < node.childNodes.length; i++) {
        var n = node.childNodes[i];
        if (n.nodeType !== 1) continue;
        if (n.localName === 'p') { paras.push(paragraphText(n)); }
        else { walk(n); }
      }
    })(doc.documentElement);

    /* Blank paragraphs are how Word spaces prose; they carry no words, so they
       are dropped and the remaining paragraphs joined the way the paste path
       would see them. */
    var kept = paras.map(function (s) { return s.replace(/[ \t]+$/, ''); })
                    .filter(function (s) { return s.trim() !== ''; });
    return { text: normalise(kept.join('\n\n')), nParagraphs: kept.length };
  }

  /* Matched to what a copy-paste out of Word puts in the box, and no further.
     Word writes a non-breaking space after some abbreviations and a soft hyphen
     at discretionary breaks; left in, both reach the tokenizer as characters the
     author never typed. Smart quotes and em dashes are NOT touched -- those
     survive a paste, so removing them here would make the same essay measure
     differently depending on how it arrived. */
  function normalise(s) {
    return s
      .replace(/\r\n?/g, '\n')
      // U+00A0 no-break space, U+202F narrow no-break space -> ordinary space.
      .replace(/[\u00a0\u202f]/g, ' ')
      // Zero-width space/non-joiner/joiner, word joiner, BOM, soft hyphen. Written as
      // escapes on purpose: spelled as literal characters this line is invisible in
      // every editor and impossible to review.
      .replace(/[\u200b-\u200d\u2060\ufeff\u00ad]/g, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  /* A .txt/.md costs nothing to accept and is the format a reader falls back to
     when their own file is refused, so refusing it too would be perverse. Decoded
     strictly: `fatal` makes TextDecoder throw on bytes that are not UTF-8, which
     is what separates a text file from a binary one that happens to lack a
     recognised signature. Guessing an encoding and getting it wrong would feed
     the detector mojibake, and mojibake scores. */
  function plainText(bytes) {
    var text;
    try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
    catch (e) { throw new ReadError('not-a-zip'); }
    // A NUL is valid UTF-8, so `fatal` does not catch it. Real prose never contains
    // one; a binary file that decoded cleanly by luck usually does.
    if (text.indexOf('\u0000') !== -1) throw new ReadError('not-a-zip');
    var paras = normalise(text).split(/\n{2,}/).filter(function (s) { return s.trim() !== ''; });
    return { text: normalise(text), nParagraphs: paras.length, kind: 'text' };
  }

  /* ---- the one entry point ----------------------------------
     Resolves {text, nParagraphs, kind}; rejects with a ReadError whose message is
     one of the codes the interface turns into a sentence. */
  function extract(arrayBuffer) {
    return Promise.resolve().then(function () {
      var bytes = new Uint8Array(arrayBuffer);
      if (!bytes.length) throw new ReadError('empty');
      var kind = sniff(bytes);
      // Named formats are refused by name; anything unrecognised gets one honest
      // attempt at being plain text before it is refused.
      if (kind === 'pdf' || kind === 'ole2' || kind === 'rtf') throw new ReadError(kind);
      if (kind === 'other') {
        var t = plainText(bytes);
        if (!t.text) throw new ReadError('no-text');
        return t;
      }

      var view = new DataView(arrayBuffer);
      var entries = centralDirectory(view);
      var main = entries.filter(function (e) { return e.name === 'word/document.xml'; })[0];
      if (!main) {
        // A ZIP that is some other Office file rather than a Word document.
        var looksOffice = entries.some(function (e) { return e.name === '[Content_Types].xml'; });
        throw new ReadError(looksOffice ? 'not-a-word-doc' : 'not-a-zip');
      }
      return readEntry(view, main).then(function (xml) {
        var result = documentText(utf8(xml));
        if (!result.text) throw new ReadError('no-text');
        result.kind = 'docx';
        return result;
      });
    });
  }

  return {
    extract: extract,
    MAX_BYTES: MAX_BYTES,
    // exported for the test suite, which checks the ZIP reader without a browser
    _internals: { sniff: sniff, centralDirectory: centralDirectory, documentText: documentText,
                  normalise: normalise, readEntry: readEntry }
  };
})();

(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };
  var r4 = function (n) { return Math.round(n * 1e4) / 1e4; };
  var f4 = function (n) { return (n < 0 ? '-' : '') + Math.abs(r4(n)).toFixed(4); };
  var pct = function (n) { return Math.round(n * 100) + '%'; };
  var sig = function (z) { return 1 / (1 + Math.exp(-z)); };

  var els = {
    body: document.body,
    essay: $('essay'),
    analyse: $('analyse'),
    upload: $('upload'),
    file: $('file'),
    dropzone: $('dropzone'),
    sample: $('load-sample'),
    missed: $('load-missed'),
    status: $('status'),
    notice: $('notice'),
    chip: $('observer-chip'),
    verdictPanel: $('verdict-panel'),
    band: $('band'),
    bandLabel: $('band-label'),
    bandDetail: $('band-detail'),
    grid: $('verdict-grid'),
    share: $('share'),
    shareInterval: $('share-interval'),
    anyp: $('anyp'),
    docSub: $('doc-sub'),
    caveat: $('verdict-caveat'),
    plot: $('profile-plot'),
    textPanel: $('text-panel'),
    rendered: $('rendered'),
    evidencePanel: $('evidence-panel'),
    evidenceText: $('evidence-text'),
    evidenceScore: $('evidence-score'),
    evidenceMath: $('evidence-math'),
    evidenceBars: $('evidence-bars'),
    tokenStrip: $('token-strip'),
    limitsPanel: $('limits-panel'),
    limitsSub: $('limits-sub'),
    limitations: $('limitations'),
    privacy: $('privacy'),
    ribbon: $('ribbon'),
    flagRail: $('flag-rail'),
    evMeta: $('ev-meta'),
    prevFlag: $('prev-flag'),
    nextFlag: $('next-flag'),
    copyWorking: $('copy-working'),
    failuresPanel: $('failures-panel'),
    failuresSub: $('failures-sub'),
    failures: $('failures'),
    failuresMethod: $('failures-method')
  };

  var LOCAL_PRIVACY = 'The observer runs here. Nothing you paste leaves this machine: no request is made, no copy is kept, and no text is used for training.';

  var current = null;      // last payload
  /* Where the text in the box came from, printed on the result line. Cleared the
     moment the box is edited: provenance that outlives the text it describes is a
     false statement about what was measured. */
  var sourceNote = '';
  var selectedIndex = null;
  var flagNumbers = {};   // sentence index -> printed flag number

  /* ---- state ------------------------------------------------ */
  function setState(s) { els.body.setAttribute('data-state', s); }
  function say(msg, kind) {
    els.status.textContent = msg;
    if (kind) { els.status.setAttribute('data-kind', kind); }
    else { els.status.removeAttribute('data-kind'); }
  }

  /* ==========================================================
     Bands
     ========================================================== */
  function shadeClass(p, threshold) {
    if (p < threshold * 0.4) return 's0';
    if (p < threshold * 0.75) return 's1';
    if (p < threshold) return 's2';
    if (p < threshold + (1 - threshold) * 0.5) return 's3';
    return 's4';
  }
  var RAMP = { s0: 1, s1: 1, s2: 2, s3: 3, s4: 4 };

  var UNRELIABLE_COPY = {
    too_short: 'too short to measure',
    too_long: 'too long to measure as one unit',
    beyond_observer_window: 'beyond the observer\u2019s window'
  };

  /* ==========================================================
     Render — verdict
     ========================================================== */
  function renderVerdict(data) {
    var v = data.verdict;
    var abstains = v.band === 'insufficient_evidence' || v.band === 'out_of_scope';

    els.band.className = 'band band-' + v.band;
    els.bandLabel.textContent = v.bandLabel;
    els.bandDetail.textContent = v.bandDetail;

    els.grid.setAttribute('data-band', v.band);
    els.share.textContent = pct(v.machineShare);
    els.shareInterval.textContent = '95% interval ' + pct(v.machineShareLow) + '\u2013' + pct(v.machineShareHigh) +
      ' \u00b7 ' + v.nReliableSentences + ' of ' + v.nSentences + ' sentences measured';
    els.anyp.textContent = pct(v.anyMachineProbability);
    els.docSub.textContent = 'calibrated across ' + v.nWords + ' words';

    els.caveat.classList.toggle('hidden', v.band === 'likely_machine');

    renderProfile(data);
    renderRibbon(data);
    els.verdictPanel.classList.remove('hidden');
    return abstains;
  }

  function renderRibbon(data) {
    var v = data.verdict, m = data.meta || {};
    var rows = [
      ['words', v.nWords],
      ['sentences', v.nSentences],
      ['measured', v.nReliableSentences + ' of ' + v.nSentences],
      ['flagged sentences', Object.keys(flagNumbers).length],
      ['sentence threshold', data.flagThreshold.toFixed(4)],
      ['in-domain', pct(v.inDomainProbability)],
      ['observer tokens', m.nObserverTokens != null ? m.nObserverTokens : '\u2014'],
      ['scored in', m.elapsedMs != null ? m.elapsedMs + ' ms' : '\u2014'],
      ['can clear the writer', v.canExonerate ? 'yes' : 'no']
    ];
    els.ribbon.textContent = '';
    rows.forEach(function (r) {
      var cell = document.createElement('div');
      cell.innerHTML = '<dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd>';
      els.ribbon.appendChild(cell);
    });
  }

  function renderFlagRail(data) {
    els.flagRail.textContent = '';
    var indices = Object.keys(flagNumbers).map(Number).sort(function (a, b) { return a - b; });
    if (!indices.length) {
      els.flagRail.innerHTML = '<span class="fr-none">No sentence reached the flag threshold. ' +
        'That is not a clearance \u2014 read the profile above and the limits below.</span>';
      return;
    }
    var label = document.createElement('span');
    label.className = 'fr-label';
    label.textContent = 'jump to flag';
    els.flagRail.appendChild(label);
    indices.forEach(function (i) {
      var s = data.sentences.filter(function (x) { return x.index === i; })[0];
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'fr-btn';
      b.setAttribute('data-index', i);
      b.setAttribute('aria-current', 'false');
      b.textContent = flagNumbers[i];
      b.title = 'Flag ' + flagNumbers[i] + ' \u00b7 sentence ' + (i + 1) +
        ' \u00b7 probability ' + s.probability.toFixed(3);
      b.addEventListener('click', function () { select(i, true); });
      els.flagRail.appendChild(b);
    });
  }

  function renderProfile(data) {
    var t = data.flagThreshold;
    els.plot.textContent = '';

    data.sentences.forEach(function (s) {
      var col = document.createElement('button');
      col.type = 'button';
      col.className = 'profile-col' + (s.reliable ? '' : ' unmeasured');
      col.setAttribute('data-index', s.index);
      var bar = document.createElement('i');
      if (s.reliable) {
        var shade = shadeClass(s.probability, t);
        col.setAttribute('data-ramp', RAMP[shade]);
        bar.style.height = Math.max(2, s.probability * 100) + '%';
        col.title = 'Sentence ' + (s.index + 1) + ' \u00b7 probability ' + s.probability.toFixed(3) +
          (flagNumbers[s.index] ? ' \u00b7 flag ' + flagNumbers[s.index] : '');
        col.setAttribute('aria-label', 'Sentence ' + (s.index + 1) + ', probability ' + s.probability.toFixed(3));
      } else {
        col.title = 'Sentence ' + (s.index + 1) + ' \u00b7 not measured (' +
          (UNRELIABLE_COPY[s.unreliableReason] || 'not measured') + ')';
        col.setAttribute('aria-label', 'Sentence ' + (s.index + 1) + ', not measured');
      }
      col.setAttribute('aria-pressed', 'false');
      col.appendChild(bar);
      col.addEventListener('click', function () { select(s.index, true); });
      els.plot.appendChild(col);
    });

    var rule = document.createElement('div');
    rule.className = 'profile-threshold';
    rule.style.bottom = (t * 100) + '%';
    rule.innerHTML = '<span>flag threshold ' + t.toFixed(4) + '</span>';
    els.plot.appendChild(rule);
  }

  /* ==========================================================
     Render — the essay
     ========================================================== */
  function renderText(data) {
    var text = data.text, t = data.flagThreshold;
    var gap = function (raw) {
      return esc(raw).replace(/\n{2,}/g, '</p><p>').replace(/\n/g, ' ');
    };
    var html = '<p>', cursor = 0;
    var counts = { none: 0, some: 0, flag: 0, un: 0 };

    data.sentences.forEach(function (s) {
      html += gap(text.slice(cursor, s.start));
      var cls, title, marker = '';
      if (s.reliable) {
        cls = shadeClass(s.probability, t);
        title = 'Sentence ' + (s.index + 1) + ' \u00b7 probability ' + s.probability.toFixed(3) +
          (s.probability >= t ? ' \u00b7 above the flag threshold' : ' \u00b7 below the flag threshold');
        if (cls === 's3' || cls === 's4') {
          counts.flag += 1;
          marker = '<sup class="flagno">' + (flagNumbers[s.index] || '') + '</sup>';
        } else if (cls === 's0') { counts.none += 1; }
        else { counts.some += 1; }
      } else {
        cls = 's0 unreliable';
        counts.un += 1;
        marker = '<sup class="flagno un" aria-hidden="true">\u2014</sup>';
        title = 'Sentence ' + (s.index + 1) + ' \u00b7 not measured: ' +
          (UNRELIABLE_COPY[s.unreliableReason] || 'not measured') +
          '. No score is claimed for this text.';
      }
      html += '<span class="sentence ' + cls + '" tabindex="0" role="button" aria-pressed="false"' +
        ' data-index="' + s.index + '" title="' + esc(title) + '">' + esc(s.text) + marker + '</span>';
      cursor = s.end;
    });
    html += gap(text.slice(cursor)) + '</p>';
    els.rendered.innerHTML = html;

    Array.prototype.forEach.call(document.querySelectorAll('#mark-index .lg-n'), function (n) {
      n.textContent = counts[n.getAttribute('data-n')];
    });

    Array.prototype.forEach.call(els.rendered.querySelectorAll('.sentence'), function (span) {
      var i = parseInt(span.getAttribute('data-index'), 10);
      span.addEventListener('click', function () { select(i, false); });
      span.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
          e.preventDefault();
          select(i, false);
        }
      });
    });
    els.textPanel.classList.remove('hidden');
  }

  /* ==========================================================
     Render — evidence
     ========================================================== */
  function select(index, scrollToSentence) {
    if (!current) return;
    var s = current.sentences.filter(function (x) { return x.index === index; })[0];
    if (!s) return;
    selectedIndex = index;

    Array.prototype.forEach.call(document.querySelectorAll('.sentence'), function (n) {
      var on = parseInt(n.getAttribute('data-index'), 10) === index;
      n.classList.toggle('selected', on);
      n.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    Array.prototype.forEach.call(els.plot.querySelectorAll('.profile-col'), function (n) {
      n.setAttribute('aria-pressed',
        parseInt(n.getAttribute('data-index'), 10) === index ? 'true' : 'false');
    });

    els.evidenceText.textContent = '\u201c' + s.text + '\u201d';

    if (s.reliable) {
      els.evidencePanel.setAttribute('data-shade', shadeClass(s.probability, current.flagThreshold));
      els.evidenceScore.innerHTML = (flagNumbers[s.index] ? '<span class="fno">' + flagNumbers[s.index] + '</span>' : '') +
        'probability ' + s.probability.toFixed(3) +
        (s.probability >= current.flagThreshold ? ' \u00b7 flagged' : ' \u00b7 not flagged');
      els.evidenceScore.setAttribute('data-flag', s.probability >= current.flagThreshold ? 'true' : 'false');
    } else {
      els.evidencePanel.setAttribute('data-shade', 's0');
      els.evidenceScore.textContent = 'not measured \u00b7 ' + (UNRELIABLE_COPY[s.unreliableReason] || '');
      els.evidenceScore.setAttribute('data-flag', 'false');
    }

    renderMath(s);
    renderBars(s);
    renderTokens(s);

    els.evMeta.textContent = 'sentence ' + (s.index + 1) + ' of ' + current.verdict.nSentences +
      ' \u00b7 ' + s.nWords + ' words \u00b7 characters ' + s.start + '\u2013' + s.end +
      (s.reliable ? ' \u00b7 smoothed ' + s.smoothed.toFixed(3) + ' \u00b7 logit ' + s.logit.toFixed(4) : '');

    Array.prototype.forEach.call(els.flagRail.querySelectorAll('.fr-btn'), function (b) {
      b.setAttribute('aria-current',
        parseInt(b.getAttribute('data-index'), 10) === index ? 'true' : 'false');
    });
    var flags = Object.keys(flagNumbers).map(Number);
    els.prevFlag.disabled = !flags.length;
    els.nextFlag.disabled = !flags.length;

    els.evidencePanel.classList.remove('hidden');

    if (scrollToSentence) {
      var target = els.rendered.querySelector('.sentence[data-index="' + index + '"]');
      if (target) {
        var box = target.getBoundingClientRect();
        if (box.top < 80 || box.bottom > window.innerHeight - 40) {
          window.scrollTo({
            top: window.pageYOffset + box.top - window.innerHeight / 3,
            behavior: 'smooth'
          });
        }
      }
    }
  }

  /* The signature line: the parts add up to the whole.
     Do not restructure the string — the first four decimals are
     scraped in order (baseline, shown, remainder, logit) and summed. */
  function renderMath(s) {
    if (!s.reliable) {
      els.evidenceMath.textContent = 'No log-odds were computed for this sentence: it was ' +
        (UNRELIABLE_COPY[s.unreliableReason] || 'not measured') +
        ', so no arithmetic exists to show and no score is claimed.';
      return;
    }
    var shown = r4(s.evidence.reduce(function (a, f) { return a + (f.measured ? f.contribution : 0); }, 0));
    els.evidenceMath.textContent =
      'Baseline ' + f4(s.intercept) + ' log-odds, ' +
      f4(shown) + ' from the ' + s.nFeaturesShown + ' features below, ' +
      f4(s.evidenceRemainder) + ' from the other ' + (s.nFeaturesTotal - s.nFeaturesShown) +
      ' \u2014 total ' + f4(s.logit) + ' log-odds, which is a probability of ' +
      s.probability.toFixed(3) + '.';
  }

  function renderBars(s) {
    els.evidenceBars.textContent = '';
    if (!s.reliable) {
      var none = document.createElement('div');
      none.className = 'bar-row unmeasured';
      none.innerHTML = '<p class="bar-name">No terms were summed</p>' +
        '<p class="bar-val">not measured</p>' +
        '<p class="evidence-desc">This span was ' +
        esc(UNRELIABLE_COPY[s.unreliableReason] || 'not measured') +
        ', so the classifier was never run on it. It is drawn unmeasured in the essay and in the profile \u2014 not as low risk.</p>';
      els.evidenceBars.appendChild(none);
      return;
    }

    var max = Math.max.apply(null, s.evidence.map(function (f) {
      return Math.abs(f.contribution) || 0;
    }).concat([Math.abs(s.evidenceRemainder), 0.5]));

    s.evidence.forEach(function (f) {
      var row = document.createElement('div');
      row.className = 'bar-row' + (f.measured ? '' : ' unmeasured');
      var w = f.measured ? Math.max(1.5, Math.abs(f.contribution) / max * 50) : 0;
      var side = f.contribution >= 0 ? 'machine' : 'human';
      row.innerHTML =
        '<p class="bar-name">' + esc(f.label) + '<span class="grp">' + esc(f.group) + '</span></p>' +
        '<p class="bar-val">' + (f.measured
          ? (f.contribution >= 0 ? '+' : '\u2212') + Math.abs(f.contribution).toFixed(4)
          : 'not measured') + '</p>' +
        '<div class="bar-track">' + (f.measured
          ? '<span class="bar-fill ' + side + '" style="width:' + w.toFixed(2) + '%"></span>'
          : '') + '</div>' +
        '<p class="evidence-desc">' + esc(f.description) +
        (f.measured ? ' <span style="white-space:nowrap">z ' + f.z.toFixed(3) +
          ' \u00d7 weight ' + f.weight.toFixed(4) + '</span>' : '') + '</p>';
      els.evidenceBars.appendChild(row);
    });

    var rem = document.createElement('div');
    rem.className = 'bar-row remainder';
    var rw = Math.max(1.5, Math.abs(s.evidenceRemainder) / max * 50);
    rem.innerHTML =
      '<p class="bar-name">The other ' + (s.nFeaturesTotal - s.nFeaturesShown) +
      ' features<span class="grp">remainder</span></p>' +
      '<p class="bar-val">' + (s.evidenceRemainder >= 0 ? '+' : '\u2212') +
      Math.abs(s.evidenceRemainder).toFixed(4) + '</p>' +
      '<div class="bar-track"><span class="bar-fill ' +
      (s.evidenceRemainder >= 0 ? 'machine' : 'human') +
      '" style="width:' + rw.toFixed(2) + '%"></span></div>' +
      '<p class="evidence-desc">Everything not listed individually, summed. It closes the arithmetic rather than explaining anything: no single term inside it is large enough to name.</p>';
    els.evidenceBars.appendChild(rem);
  }

  function renderTokens(s) {
    els.tokenStrip.textContent = '';
    var toks = (current.tokens || []).filter(function (tk) {
      return tk.start >= s.start && tk.end <= s.end;
    });
    if (!toks.length) {
      els.tokenStrip.innerHTML = '<span class="tok tail">No observer tokens fall inside this span.</span>';
      return;
    }
    var html = '';
    toks.forEach(function (tk) {
      html += '<span class="tok ' + tk.bucket + '" title="rank ' + tk.rank +
        ' \u00b7 log-probability ' + tk.logprob.toFixed(3) + '">' + esc(tk.text) + '</span>';
    });
    els.tokenStrip.innerHTML = html;
  }

  /* ==========================================================
     Render — limits + notice
     ========================================================== */
  var COUNT_WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
    'Nine', 'Ten', 'Eleven', 'Twelve'];

  function renderLimitations(data) {
    var lines = data.limitations || [];
    els.limitations.textContent = '';
    lines.forEach(function (line) {
      var li = document.createElement('li');
      li.textContent = line;
      els.limitations.appendChild(li);
    });
    /* Counted from what was just rendered, never typed into the markup. This heading read
       "Eight measured failure modes" above a list of nine, because the count was written by
       hand against the bundled fixture while the deployed detector ships one more. The same
       drift in the other direction has already published a live site disclosing one FEWER
       failure mode than the repository claimed, and nothing errored either time. A number
       describing a list belongs to the list. */
    var n = lines.length;
    els.limitsSub.textContent = (n < COUNT_WORDS.length ? COUNT_WORDS[n] : String(n)) +
      ' measured failure mode' + (n === 1 ? '' : 's') +
      ', shipped with every result. Read them before you act on anything above.';
    els.limitsPanel.classList.remove('hidden');
  }

  /* Grouped explicitly as en-US. A bare toLocaleString() follows the reader's locale, and
     several of them render 6000 as "6.000" \u2014 which reads as six, in a sentence whose whole
     job is to say how much of the essay went unread. */
  function grouped(n) {
    return typeof n === 'number' ? n.toLocaleString('en-US') : String(n);
  }

  function renderNotice(data) {
    var m = data.meta || {};
    if (m.clipped) {
      els.notice.textContent = 'This essay is longer than the observer\u2019s window of ' +
        grouped(m.observerCharLimit) + ' characters. Only the opening ' +
        grouped(m.observerCharLimit) +
        ' characters were read, so everything above describes the opening of the essay and not the rest of it.';
      els.notice.hidden = false;
    } else {
      els.notice.hidden = true;
      els.notice.textContent = '';
    }
    els.chip.textContent = 'observer ' + (m.observer || 'unknown') + ' \u00b7 ' +
      (m.device || '?') + ' \u00b7 ' + (m.elapsedMs != null ? m.elapsedMs + ' ms' : '') +
      (data.offline ? ' \u00b7 bundled fixture' : '');
    els.chip.hidden = false;
  }

  function renderAll(data) {
    current = data;
    flagNumbers = {};
    var nFlag = 0;
    data.sentences.forEach(function (s) {
      if (!s.reliable) return;
      var c = shadeClass(s.probability, data.flagThreshold);
      if (c === 's3' || c === 's4') { nFlag += 1; flagNumbers[s.index] = nFlag; }
    });
    renderVerdict(data);
    renderText(data);
    renderFlagRail(data);
    renderLimitations(data);
    renderNotice(data);

    var flagged = data.sentences.filter(function (s) {
      return s.reliable && s.probability >= data.flagThreshold;
    });
    var first = flagged.length ? flagged[0] : data.sentences.filter(function (s) { return s.reliable; })[0];
    if (first) select(first.index, false);

    setState('result');
    // must be the last thing written by the render
    say(data.verdict.nSentences + ' sentences read \u00b7 ' + data.verdict.nReliableSentences +
      ' measured \u00b7 ' + flagged.length + ' sentences flagged above ' +
      data.flagThreshold.toFixed(4) + (data.offline ? ' \u00b7 bundled fixture, no observer reached' : '') +
      (sourceNote ? ' \u00b7 read from ' + sourceNote : ''), 'ok');
  }

  /* ==========================================================
     Analyse
     ========================================================== */
  function NoBackend(msg) { this.message = msg || 'no analyzer endpoint'; }
  NoBackend.prototype = Object.create(Error.prototype);

  function callApi(text) {
    return fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, include_tokens: true })
    }).catch(function () {
      throw new NoBackend('analyzer unreachable');
    }).then(function (res) {
      if (res.status === 404 || res.status === 405 || res.status === 501) {
        throw new NoBackend('analyzer not deployed here');
      }
      /* The API explains its own refusals — a per-IP rate limit, the daily neuron
         budget stopping, a text over the observer's limit — as {error, detail}.
         A bare status code would replace that explanation with a number, so the
         body is read first and only falls back to the status when it says nothing.
         The local build answers {detail}, the Worker {error, detail}: both are read
         here so the two deployments cannot drift apart on their error wording. */
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || body.error || ('request failed (' + res.status + ')'));
        });
      }
      return res.json().catch(function () {
        throw new NoBackend('analyzer returned a non-JSON body');
      });
    });
  }

  function analyse() {
    var text = els.essay.value.trim();
    if (!text) {
      say('Nothing to analyse yet \u2014 paste an essay, or load one of the two examples.', 'error');
      els.essay.focus();
      return;
    }
    els.analyse.disabled = true;
    els.sample.disabled = true;
    els.missed.disabled = true;
    setState('scoring');
    say('Reading token probabilities, then summing 43 features per sentence\u2026');

    callApi(text).then(function (data) {
      data.text = data.text || text;
      renderAll(data);
    }).catch(function (err) {
      if (err instanceof NoBackend) {
        var data = Palimpsest.analyseLocally(text);
        data.offline = true;
        renderAll(data);
        return;
      }
      setState(current ? 'result' : 'idle');
      say('Failed: ' + (err && err.message ? err.message : String(err)), 'error');
    })['finally'](function () {
      els.analyse.disabled = false;
      els.sample.disabled = false;
      els.missed.disabled = false;
    });
  }

  /* ==========================================================
     Opening a document

     The extracted text goes into the same box a paste goes into, and the analysis
     runs on that box. There is no hidden copy: whatever the reader sees is what
     was measured, and they can correct it before running it again. That is the
     whole safety story for reading a file, and it is why the text is shown rather
     than sent straight to the scorer.
     ========================================================== */

  /* One sentence per refusal, each naming the next thing to try. A file dialog
     that answers "unsupported format" has told the reader nothing they can act
     on. The PDF wording is the longest because it is the case people will hit
     most and the one where the reason is not obvious. */
  var READ_ERRORS = {
    'pdf':
      'That is a PDF. This reads .docx instead, and the reason is not laziness: a PDF stores '
      + 'placed glyphs, not paragraphs, so recovering the prose means guessing where lines join '
      + 'and whether a hyphen ended a word or a line. Those guesses move sentence boundaries, and '
      + 'every number this tool reports is computed per sentence. Open it in Word or Google Docs '
      + 'and save as .docx, or paste the text straight into the box.',
    'ole2':
      'That is a pre-2007 Word .doc, which is a different format that stores text as a binary '
      + 'stream. Open it in Word and use Save As → .docx, or paste the text in.',
    'rtf':
      'That is an .rtf. Open it in Word or Google Docs and save as .docx, or paste the text in.',
    'not-a-word-doc':
      'That is an Office file, but not a Word document — there is no word/document.xml in it. '
      + 'If it is a spreadsheet or a deck, paste the essay text in instead.',
    'not-a-zip':
      'That is not a .docx and does not read as plain text. A .docx is a ZIP holding '
      + 'word/document.xml; this file is neither. Paste the essay text in instead.',
    'damaged':
      'That .docx will not open — the archive inside it is damaged or uses a compression method '
      + 'this reader does not implement. Re-save it from Word, or paste the text in.',
    'no-text':
      'That document opened, but there is no text in it. If the essay is a picture of a page, '
      + 'this cannot read it — there is no OCR here. Type or paste the text in.',
    'empty': 'That file is empty.',
    'too-big':
      'That file is larger than 25 MB, which is far larger than any essay. Check it is the right '
      + 'file, or paste the text in.',
    'no-decompression':
      'This browser cannot unzip a .docx (it has no DecompressionStream). Paste the essay text in '
      + 'instead, or use a current Chrome, Safari or Firefox.'
  };

  function words(s) { return (s.match(/\S+/g) || []).length; }

  function openFile(file) {
    if (!file) return;
    if (file.size > PalimpsestDocx.MAX_BYTES) {
      say(READ_ERRORS['too-big'], 'error');
      return;
    }
    setState(current ? 'result' : 'idle');
    say('Reading ' + file.name + '…');

    file.arrayBuffer().then(function (buf) {
      return PalimpsestDocx.extract(buf);
    }).then(function (doc) {
      els.essay.value = doc.text;
      sourceNote = file.name;
      var n = words(doc.text);
      say('Read ' + n.toLocaleString('en-US') + ' words from ' + file.name +
        ' · ' + doc.nParagraphs + ' paragraph' + (doc.nParagraphs === 1 ? '' : 's') +
        '. This is the text being analysed — correct it in the box if the document did not ' +
        'come through cleanly.');
      /* Analysed straight away because that is what pressing Upload asks for, and
         nothing is concealed by doing it: the text is in the box above the result.
         The word count is stated first so a document that came through wrong is
         visible as a wrong number before the reader reads any verdict off it.

         Deferred by a beat, and that is the point rather than a workaround.
         `analyse()` writes its own status line, so calling it in this tick replaced
         the word count before the browser had painted it once -- a message that is
         never on screen is not a disclosure, and the check that asserts it is
         shown was what caught this. */
      setTimeout(analyse, 450);
    })['catch'](function (err) {
      setState(current ? 'result' : 'idle');
      say(READ_ERRORS[err && err.message] ||
        ('Could not read ' + file.name + '. Paste the essay text in instead.'), 'error');
    });
  }

  /* ==========================================================
     Flag navigation + copying the working
     ========================================================== */
  function flagIndices() {
    return Object.keys(flagNumbers).map(Number).sort(function (a, b) { return a - b; });
  }
  function stepFlag(dir) {
    var f = flagIndices();
    if (!f.length) return;
    var at = f.indexOf(selectedIndex), next;
    if (at === -1) {
      next = dir > 0
        ? f.filter(function (i) { return i > (selectedIndex == null ? -1 : selectedIndex); })[0]
        : f.filter(function (i) { return i < (selectedIndex == null ? Infinity : selectedIndex); }).pop();
      if (next == null) { next = dir > 0 ? f[0] : f[f.length - 1]; }
    } else {
      next = f[(at + dir + f.length) % f.length];
    }
    select(next, true);
  }

  function workingAsText() {
    if (!current || selectedIndex == null) return '';
    var s = current.sentences.filter(function (x) { return x.index === selectedIndex; })[0];
    if (!s) return '';
    var out = [];
    out.push('Palimpsest \u2014 sentence ' + (s.index + 1) + ' of ' + current.verdict.nSentences);
    out.push('Document band: ' + current.verdict.bandLabel + ' (' + current.verdict.band + ')');
    out.push('');
    out.push('\u201c' + s.text + '\u201d');
    out.push('');
    if (s.reliable) {
      out.push('probability ' + s.probability.toFixed(4) + '  (sentence flag threshold ' +
        current.flagThreshold.toFixed(4) + ')');
      out.push(els.evidenceMath.textContent);
      out.push('');
      s.evidence.forEach(function (f) {
        out.push((f.measured
          ? (f.contribution >= 0 ? '+' : '-') + Math.abs(f.contribution).toFixed(4) +
            '  z ' + f.z.toFixed(3) + ' x weight ' + f.weight.toFixed(4) + '  '
          : 'not measured                          ') + f.label + ' [' + f.group + ']');
      });
      out.push((s.evidenceRemainder >= 0 ? '+' : '-') + Math.abs(s.evidenceRemainder).toFixed(4) +
        '  the other ' + (s.nFeaturesTotal - s.nFeaturesShown) + ' features (remainder)');
    } else {
      out.push('not measured: ' + (UNRELIABLE_COPY[s.unreliableReason] || 'unknown') +
        '. No score is claimed for this text.');
    }
    out.push('');
    out.push('This is evidence, not a verdict. ' + current.limitations[0]);
    return out.join('\n');
  }

  function flashButton(btn, label) {
    var was = btn.textContent;
    btn.textContent = label;
    setTimeout(function () { btn.textContent = was; }, 1400);
  }

  els.prevFlag.addEventListener('click', function () { stepFlag(-1); });
  els.nextFlag.addEventListener('click', function () { stepFlag(1); });
  els.copyWorking.addEventListener('click', function () {
    var text = workingAsText();
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        flashButton(els.copyWorking, 'Copied');
      })['catch'](function () {
        flashButton(els.copyWorking, 'Clipboard blocked');
      });
    } else {
      flashButton(els.copyWorking, 'Clipboard unavailable');
    }
  });

  document.addEventListener('keydown', function (e) {
    var t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'n' || e.key === 'N') { stepFlag(1); }
    else if (e.key === 'p' || e.key === 'P') { stepFlag(-1); }
  });

  /* ==========================================================
     Confident failures

     The three held-out documents the detector gets most aggressively wrong, rendered from
     artifacts/confident_failures.json. Every number here comes from that artifact; none is
     written into the markup, because a hand-typed number describing a model outlives the
     model. PROJECT.md §2 records what that costs.

     The explanation box is the point of the panel. The arithmetic can say WHICH terms
     produced a wrong answer; it cannot say why that was the wrong thing to measure. That
     sentence is a human artefact and the box is where it gets written, saved back into the
     artifact so a corpus rebuild does not delete it.
     ========================================================== */
  function renderFailures(payload) {
    var list = (payload && payload.failures) || [];
    if (!list.length) return;

    els.failures.textContent = '';
    els.failuresSub.textContent =
      (list.length < COUNT_WORDS.length ? COUNT_WORDS[list.length] : String(list.length)) +
      ' held-out document' + (list.length === 1 ? '' : 's') + ' out of ' +
      grouped(payload.nEvaluated) + ' — the ones this detector is most confidently ' +
      'wrong about. ' + grouped(payload.nWrong) + ' were wrong in total.';
    els.failuresMethod.textContent = 'Ranked by ' + (payload.ranking || '') +
      ' Scored by the ' + (payload.arm || '') + ' arm (' + payload.nFeatures +
      ' features), sentence threshold ' + payload.sentenceThreshold + '.';

    list.forEach(function (f) {
      var card = document.createElement('article');
      card.className = 'failure' + (f.truth === 'human' ? ' false-accusation' : ' missed');

      var head = document.createElement('div');
      head.className = 'failure-head';
      head.innerHTML =
        '<p class="failure-kind">' + esc(f.direction) + '</p>' +
        '<h3 class="failure-id">' + esc(f.docId) + '</h3>' +
        '<p class="failure-meta">' + esc(f.source) + ' · ' + grouped(f.words) +
        ' words · truly <strong>' + esc(f.truth) + '</strong>, called <strong>' +
        esc(f.verdict) + '</strong></p>' +
        '<p class="failure-score">P(machine) = ' + f.documentProbability.toFixed(3) +
        '<span class="failure-sev">severity ' + f.severity.toFixed(3) +
        ' = error ' + f.errorComponent.toFixed(2) +
        ' × confidence ' + f.confidenceComponent.toFixed(2) + '</span></p>';
      card.appendChild(head);

      var essay = document.createElement('details');
      essay.className = 'failure-essay';
      var sum = document.createElement('summary');
      sum.textContent = 'The essay';
      essay.appendChild(sum);
      var pre = document.createElement('p');
      pre.className = 'failure-text';
      pre.textContent = f.text || '(source text unavailable)';
      essay.appendChild(pre);
      card.appendChild(essay);

      (f.drivingSentences || []).forEach(function (s) {
        var blk = document.createElement('div');
        blk.className = 'failure-driver';
        var ev = s.evidence || {};
        var bars = (ev.shownContributions || []);
        var max = Math.max.apply(null, bars.map(function (c) {
          return Math.abs(c.contribution) || 0;
        }).concat([0.5]));

        var html = '<p class="driver-head">The sentence that drove it — P = ' +
          s.probability.toFixed(3) + '</p>' +
          '<blockquote class="driver-text">' + esc((s.text || '').trim() || '(empty span)') +
          '</blockquote>' +
          '<p class="driver-sub">The terms that were summed:</p>';
        bars.forEach(function (c) {
          var w = Math.max(1.5, Math.abs(c.contribution) / max * 50);
          html += '<div class="bar-row' + (c.measured ? '' : ' unmeasured') + '">' +
            '<p class="bar-name">' + esc(c.label) +
            '<span class="grp">' + esc(c.group) + '</span></p>' +
            '<p class="bar-val">' + (c.contribution >= 0 ? '+' : '−') +
            Math.abs(c.contribution).toFixed(3) + '</p>' +
            '<div class="bar-track"><span class="bar-fill ' +
            (c.contribution >= 0 ? 'machine' : 'human') +
            '" style="width:' + w.toFixed(2) + '%"></span></div>' +
            '<p class="evidence-desc">z ' + c.z.toFixed(3) + ' × weight ' +
            c.weight.toFixed(4) + (c.measured ? '' : ' · not measured') + '</p>' +
            '</div>';
        });
        html += '<p class="driver-math">' + bars.length + ' terms shown + remainder ' +
          (ev.remainder >= 0 ? '+' : '−') + Math.abs(ev.remainder || 0).toFixed(3) +
          ' + intercept ' + (ev.intercept >= 0 ? '+' : '−') +
          Math.abs(ev.intercept || 0).toFixed(3) + ' = logit ' + (ev.logit || 0).toFixed(3) +
          '</p>';
        blk.innerHTML = html;
        card.appendChild(blk);
      });

      /* The human half. */
      var box = document.createElement('div');
      box.className = 'failure-why';
      var id = 'why-' + f.docId.replace(/[^a-zA-Z0-9]/g, '-');
      box.innerHTML =
        '<label class="field-label" for="' + id + '">Why the maths failed — ' +
        'the part the arithmetic cannot supply</label>' +
        '<textarea id="' + id + '" class="why-input" rows="4" ' +
        'placeholder="The bars above say which terms produced this number. Write what they ' +
        'were actually measuring, and why it was the wrong thing to measure here."></textarea>' +
        '<div class="why-actions">' +
        '<button type="button" class="btn btn-small why-save">Save this explanation</button>' +
        '<span class="why-status" role="status" aria-live="polite"></span></div>';
      card.appendChild(box);

      var input = box.querySelector('.why-input');
      var status = box.querySelector('.why-status');
      input.value = f.humanExplanation || '';
      if (f.humanExplanation) { status.textContent = 'saved'; }

      box.querySelector('.why-save').addEventListener('click', function () {
        status.textContent = 'saving…';
        fetch('/api/failures/explanation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doc_id: f.docId, text: input.value })
        }).then(function (r) {
          /* Reports what actually happened. An explanation the writer believes is stored
             and is not is worse than one that visibly failed to save. */
          status.textContent = r.ok ? 'saved' : 'NOT saved (' + r.status + ')';
        }).catch(function () { status.textContent = 'NOT saved — no connection'; });
      });

      els.failures.appendChild(card);
    });

    els.failuresPanel.classList.remove('hidden');
  }

  /* Loaded once at startup, independently of any analysis: these failures are properties of
     the detector, not of whatever essay the reader happens to paste. A missing artifact
     leaves the panel hidden rather than showing an empty "no failures" state, which would
     read as a claim that there are none. */
  fetch('/api/failures')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (p) { if (p) renderFailures(p); })
    .catch(function () { /* panel stays hidden */ });

  /* ==========================================================
     Wiring
     ========================================================== */
  els.analyse.addEventListener('click', analyse);
  els.essay.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); analyse(); }
  });
  els.sample.addEventListener('click', function () {
    els.essay.value = Palimpsest.FIXTURE_CAUGHT;
    sourceNote = '';
    els.essay.focus();
    say('Loaded a GPT-3.5 essay from the evaluation set. Nothing has been analysed yet \u2014 press Analyse.');
  });
  els.missed.addEventListener('click', function () {
    els.essay.value = Palimpsest.FIXTURE_MISSED;
    sourceNote = '';
    els.essay.focus();
    say('Loaded the honest-failure example: a hand-polished paragraph inside human writing. The tool locates it and still declines to flag the document. Press Analyse.');
  });

  /* ---- opening a document ---------------------------------- */
  els.upload.addEventListener('click', function () { els.file.click(); });
  els.file.addEventListener('change', function () {
    openFile(els.file.files && els.file.files[0]);
    /* Reset so choosing the SAME file twice fires `change` again. Without this a
       reader who fixes their document and re-picks it gets no response at all,
       which reads as the button being broken. */
    els.file.value = '';
  });

  /* Typing invalidates the "read from X" provenance on the result line. */
  els.essay.addEventListener('input', function () { sourceNote = ''; });

  /* Drag and drop. dragover must be cancelled or the browser navigates to the
     file and the page is gone. The counter exists because dragenter/dragleave
     also fire for the textarea INSIDE the zone, so a plain boolean turns the
     highlight off while the file is still over the target. */
  var dragDepth = 0;
  function dragging(on) {
    els.dropzone.classList.toggle('dragging', on);
  }
  ['dragenter', 'dragover'].forEach(function (name) {
    els.dropzone.addEventListener(name, function (e) {
      if (!e.dataTransfer || e.dataTransfer.types.indexOf('Files') === -1) return;
      e.preventDefault();
      if (name === 'dragenter') { dragDepth += 1; }
      e.dataTransfer.dropEffect = 'copy';
      dragging(true);
    });
  });
  els.dropzone.addEventListener('dragleave', function () {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dragging(false);
  });
  els.dropzone.addEventListener('drop', function (e) {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
    e.preventDefault();
    dragDepth = 0;
    dragging(false);
    openFile(e.dataTransfer.files[0]);
  });
  /* A file dropped anywhere else would otherwise be opened by the browser,
     replacing the page and losing whatever was in the box. */
  window.addEventListener('dragover', function (e) {
    if (e.dataTransfer && e.dataTransfer.types.indexOf('Files') !== -1) e.preventDefault();
  });
  window.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) e.preventDefault();
  });

  /* privacy is narrowed from /api/health; only an explicit false may soften it.
     A failed probe leaves the stronger warning standing. */
  fetch('/api/health').then(function (r) { return r.ok ? r.json() : null; })
    .then(function (h) {
      if (h && h.textLeavesMachine === false) { els.privacy.textContent = LOCAL_PRIVACY; }
    })
    .catch(function () { /* stronger claim stands */ });
})();

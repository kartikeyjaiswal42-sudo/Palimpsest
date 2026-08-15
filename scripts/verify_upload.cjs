// End-to-end check of opening a document, in a real browser.
//
//   node scripts/verify_upload.cjs [baseUrl]
//
// With no argument this serves web/ itself on a spare port and drives that, so the reader
// path can be checked without uvicorn running. The extraction is client-side, so what is
// tested here is the same code the hosted Worker serves -- edge/assets/app.js is a copy of
// web/app.js made by edge/scripts/sync_web.py, and a check pointed at the deployed URL
// exercises the copy.
//
// Fixtures are BUILT HERE rather than committed. A .docx checked into the repository is an
// opaque blob: nobody can see from a diff whether it still contains what a test claims it
// does, and the one case worth testing hardest -- tracked changes -- is invisible in every
// viewer that renders it correctly. Written out as XML in this file, each case can be read.

const { chromium } = require('playwright');
const zlib = require('node:zlib');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', 'web');

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok, detail });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

// Same fallback as verify_ui.cjs: a machine without `npx playwright install` should get a
// working run off system Chrome rather than a stack trace.
async function launch() {
  try {
    return await chromium.launch();
  } catch {
    console.log('  (bundled chromium unavailable, trying system Chrome)');
    return await chromium.launch({ channel: 'chrome' });
  }
}

/* ============================================================
   Building the fixtures
   ============================================================ */

// Enough of ZIP to write one. `store` skips deflate so the method-0 branch of the reader is
// exercised by a real file rather than assumed to work.
function zip(entries) {
  const chunks = [];
  const central = [];
  let offset = 0;

  for (const e of entries) {
    const name = Buffer.from(e.name, 'utf8');
    const raw = Buffer.from(e.data, 'utf8');
    const store = !!e.store;
    const body = store ? raw : zlib.deflateRawSync(raw);
    const crc = zlib.crc32(raw);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);                 // version needed
    local.writeUInt16LE(0, 6);                  // flags
    local.writeUInt16LE(store ? 0 : 8, 8);      // method
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);                 // extra length

    const cen = Buffer.alloc(46);
    cen.writeUInt32LE(0x02014b50, 0);
    cen.writeUInt16LE(20, 4);
    cen.writeUInt16LE(20, 6);
    cen.writeUInt16LE(0, 8);
    cen.writeUInt16LE(store ? 0 : 8, 10);
    cen.writeUInt32LE(crc, 16);
    cen.writeUInt32LE(body.length, 20);
    cen.writeUInt32LE(raw.length, 24);
    cen.writeUInt16LE(name.length, 28);
    cen.writeUInt32LE(offset, 42);

    chunks.push(local, name, body);
    central.push(Buffer.concat([cen, name]));
    offset += 30 + name.length + body.length;
  }

  const dir = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(dir.length, 12);
  eocd.writeUInt32LE(offset, 16);

  return Buffer.concat([...chunks, dir, eocd]);
}

const CONTENT_TYPES =
  '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
  '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
  '<Default Extension="xml" ContentType="application/xml"/></Types>';

function wordDoc(bodyXml) {
  return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">' +
    `<w:body>${bodyXml}</w:body></w:document>`;
}
const p = (...runs) => `<w:p>${runs.join('')}</w:p>`;
const t = (s) => `<w:r><w:t xml:space="preserve">${s}</w:t></w:r>`;

function docx(bodyXml, opts = {}) {
  return zip([
    { name: '[Content_Types].xml', data: CONTENT_TYPES },
    { name: 'word/document.xml', data: wordDoc(bodyXml), store: !!opts.store },
  ]);
}

const ESSAY_A = 'My grandmother kept her bus transfers.';
const ESSAY_B = 'Nobody could explain it. My mother said she was just like that.';

const FIXTURES = {
  // The ordinary case: two paragraphs, deflated, as Word writes them.
  plain: docx(p(t(ESSAY_A)) + p(t(ESSAY_B))),

  // Same content stored uncompressed, so method 0 is covered by a real file.
  stored: docx(p(t(ESSAY_A)) + p(t(ESSAY_B)), { store: true }),

  // Blank paragraphs are how Word spaces prose and must not become blank lines in the box.
  blanks: docx(p(t(ESSAY_A)) + '<w:p/>' + '<w:p/>' + p(t(ESSAY_B))),

  // Tracked changes. The inserted clause is the author's current text and must be read; the
  // deleted one is text they took out and must not be. Getting this backwards would score
  // words nobody chose to leave on the page.
  tracked: docx(p(
    t('The habit made a kind of sense to me, '),
    '<w:ins><w:r><w:t xml:space="preserve">though, once I stopped looking for a reason.</w:t></w:r></w:ins>',
    '<w:del><w:r><w:delText xml:space="preserve">DELETED-SENTENCE-SHOULD-NOT-APPEAR</w:delText></w:r></w:del>'
  )),

  // A hyperlink field. The instruction text is machinery, not prose.
  field: docx(p(
    t('See '),
    '<w:r><w:instrText xml:space="preserve"> HYPERLINK "http://example.com/NOT-PROSE" </w:instrText></w:r>',
    t('the source.')
  )),

  // Tab, explicit line break, non-breaking space, soft hyphen.
  whitespace: docx(p(
    t('One'), '<w:r><w:tab/></w:r>', t('two'),
    '<w:r><w:br/></w:r>', t('three\u00a0four\u00adfive')
  )),

  // A ZIP that is an Office file but not a Word document.
  notword: zip([{ name: '[Content_Types].xml', data: CONTENT_TYPES },
                { name: 'xl/workbook.xml', data: '<workbook/>' }]),

  // A .docx holding no text at all: a scan, in practice.
  notext: docx('<w:p/>'),

  pdf: Buffer.from('%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n'),
  doc: Buffer.concat([Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]),
                      Buffer.alloc(600)]),
  rtf: Buffer.from('{\\rtf1\\ansi Hello.}'),
  txt: Buffer.from(`${ESSAY_A}\n\n${ESSAY_B}\n`, 'utf8'),
  empty: Buffer.alloc(0),
  garbage: Buffer.from([0x00, 0x01, 0x02, 0xff, 0xfe, 0x00, 0x99, 0x42, 0x00, 0x17]),
};

const DOCX_MIME =
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
const file = (name, buffer, mimeType = DOCX_MIME) => ({ name, mimeType, buffer });

/* ============================================================
   A static server, so no uvicorn is needed
   ============================================================ */
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml' };

function serveWeb() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const rel = req.url.split('?')[0] === '/' ? '/index.html' : req.url.split('?')[0];
      const full = path.join(WEB, path.normalize(rel).replace(/^(\.\.[/\\])+/, ''));
      if (!full.startsWith(WEB) || !fs.existsSync(full) || !fs.statSync(full).isFile()) {
        // /api/* lands here. 404 is what the page treats as "no analyzer deployed", which
        // makes it fall back to the bundled analyzer -- fine, the reader path is the subject.
        res.writeHead(404).end('not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': TYPES[path.extname(full)] || 'application/octet-stream' });
      res.end(fs.readFileSync(full));
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

/* ============================================================ */

(async () => {
  const passedBase = process.argv[2];
  let server = null, base = passedBase;
  if (!base) {
    const s = await serveWeb();
    server = s.server;
    base = `http://127.0.0.1:${s.port}`;
  }

  console.log(`\n${'='.repeat(72)}\nVERIFYING UPLOAD: ${base}`);
  console.log(passedBase ? 'target: the URL you passed.'
    : 'target: web/ served from disk by this script (no uvicorn needed).');
  console.log('='.repeat(72) + '\n');

  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } });
  const errors = [];
  /* When this script serves web/ itself there is no /api/*, on purpose: the page is supposed
     to fall back to its bundled analyzer, and it is checked above that it does. The browser
     still logs those 404s, so they are dropped HERE and only here -- narrowly, by URL, and
     only when we are the ones serving. Against a URL the caller passed, an /api/ 404 is a
     real defect and is counted. A blanket "ignore console errors" would have hidden it. */
  const ownApi404 = (m) =>
    !passedBase && m.text().includes('404') && /\/api\//.test(m.location()?.url || '');
  page.on('console', (m) => {
    if (m.type() === 'error' && !ownApi404(m)) errors.push(`${m.text()} @ ${m.location()?.url || ''}`);
  });
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(base, { waitUntil: 'domcontentloaded' });

  const essay = () => page.inputValue('#essay');
  const status = () => page.textContent('#status');

  // Waits for a real signal rather than sleeping: `say()` sets data-kind only on the two
  // terminal outcomes ('ok' after a render, 'error' after a refusal), so that attribute is
  // an unambiguous "finished" flag and a fixed pause -- the usual source of flake in a UI
  // suite -- is not needed.
  //
  // Every status message is recorded, not just the last one. Reading a document writes
  // "Read N words from X", then the analysis it kicks off immediately overwrites that with
  // its own line; sampling once at the end would therefore never see the message this suite
  // most needs to assert, and the check would look broken rather than the product.
  async function open(f) {
    await page.fill('#essay', '');
    await page.evaluate(() => {
      const el = document.getElementById('status');
      el.textContent = '';
      el.removeAttribute('data-kind');
      window.__seen = [];
      if (window.__obs) window.__obs.disconnect();
      window.__obs = new MutationObserver(() => window.__seen.push(el.textContent));
      window.__obs.observe(el, { childList: true, characterData: true, subtree: true });
    });
    await page.setInputFiles('#file', f);
    await page.waitForFunction(
      () => {
        const k = document.getElementById('status').getAttribute('data-kind');
        return k === 'ok' || k === 'error';
      },
      undefined,
      { timeout: 20000 }
    );
    return {
      text: await essay(),
      status: await status(),
      seen: (await page.evaluate(() => window.__seen || [])).join(' || '),
    };
  }

  /* ---- the controls exist ---- */
  check('an Upload button is offered', await page.isVisible('#upload'));
  check('the file input accepts .docx',
    (await page.getAttribute('#file', 'accept') || '').includes('.docx'));

  /* ---- the ordinary case ---- */
  {
    const r = await open(file('essay.docx', FIXTURES.plain));
    check('a .docx lands in the essay box', r.text.includes(ESSAY_A) && r.text.includes(ESSAY_B),
      JSON.stringify(r.text.slice(0, 60)));
    check('paragraphs are separated by a blank line', r.text === `${ESSAY_A}\n\n${ESSAY_B}`,
      JSON.stringify(r.text));
    check('the status names the file and the word count',
      /\b18 words from essay\.docx/.test(r.seen), r.seen.slice(0, 120));
    check('opening a document starts the analysis without being asked again',
      /Reading token probabilities|analyz|analys/i.test(r.seen), r.seen.slice(0, 160));

    // WHAT THIS USED TO CHECK, AND WHY IT CHANGED. This block asserted that the finished
    // result line said "read from essay.docx". This suite runs with NO SERVER, so the only
    // reason a result line ever existed here was that the page manufactured one locally when
    // the analyzer could not be reached -- a seeded PRNG producing a verdict, a machine share
    // and evidence bars, which has since been removed for the obvious reason. The check was
    // therefore reading a fabricated render, not the product. What can honestly be verified
    // without a backend is that the reader records the source and that the failure is
    // reported as a failure; the source note ON A REAL RESULT is covered by verify_ui.cjs,
    // which runs against a live analyzer.
    const ended = await status();
    check('with no analyzer reachable it reports that, rather than inventing a result',
      /nothing was measured/i.test(ended), ended.slice(0, 110));
    check('and it does not paint a verdict it did not compute',
      !(await page.isVisible('#verdict-panel')));
  }

  /* ---- uncompressed entries ---- */
  {
    const r = await open(file('stored.docx', FIXTURES.stored));
    check('a stored (uncompressed) .docx reads the same',
      r.text === `${ESSAY_A}\n\n${ESSAY_B}`, JSON.stringify(r.text.slice(0, 60)));
  }

  /* ---- Word's blank spacer paragraphs ---- */
  {
    const r = await open(file('blanks.docx', FIXTURES.blanks));
    check('empty spacer paragraphs do not become blank lines',
      r.text === `${ESSAY_A}\n\n${ESSAY_B}`, JSON.stringify(r.text));
  }

  /* ---- tracked changes: the case that matters most ---- */
  {
    const r = await open(file('tracked.docx', FIXTURES.tracked));
    check('tracked-change INSERTIONS are read', r.text.includes('once I stopped looking'),
      JSON.stringify(r.text.slice(0, 90)));
    check('tracked-change DELETIONS are not read',
      !r.text.includes('DELETED-SENTENCE-SHOULD-NOT-APPEAR'), JSON.stringify(r.text));
  }

  /* ---- field codes ---- */
  {
    const r = await open(file('field.docx', FIXTURES.field));
    check('hyperlink field codes are not read as prose', !r.text.includes('NOT-PROSE'),
      JSON.stringify(r.text));
    check('the words around a field are kept', r.text === 'See the source.',
      JSON.stringify(r.text));
  }

  /* ---- whitespace ---- */
  {
    const r = await open(file('ws.docx', FIXTURES.whitespace));
    check('a tab is preserved', r.text.includes('One\ttwo'), JSON.stringify(r.text));
    check('an explicit break becomes a newline', r.text.includes('two\nthree'),
      JSON.stringify(r.text));
    check('a no-break space becomes an ordinary space',
      r.text.includes('three four') && !/\u00a0/.test(r.text), JSON.stringify(r.text));
    check('a soft hyphen is removed',
      r.text.includes('fourfive') && !/\u00ad/.test(r.text), JSON.stringify(r.text));
  }

  /* ---- plain text ---- */
  {
    const r = await open(file('essay.txt', FIXTURES.txt, 'text/plain'));
    check('a .txt is read too', r.text === `${ESSAY_A}\n\n${ESSAY_B}`,
      JSON.stringify(r.text.slice(0, 60)));
  }

  /* ---- every refusal ----
     Each asserts two things: the reason is named, and the box is left alone. A failed read
     that wipes what the reader had already pasted is worse than no upload button. */
  const REFUSALS = [
    ['a PDF is refused as a PDF, with the reason', file('essay.pdf', FIXTURES.pdf, 'application/pdf'), /PDF/],
    ['a legacy .doc is named as a .doc', file('old.doc', FIXTURES.doc, 'application/msword'), /pre-2007|Save As/],
    ['an .rtf is named as an .rtf', file('x.rtf', FIXTURES.rtf, 'application/rtf'), /\.rtf/],
    ['a non-Word Office file says so', file('book.xlsx', FIXTURES.notword), /not a Word document/],
    ['a .docx with no text says so, and mentions there is no OCR', file('scan.docx', FIXTURES.notext), /no text in it|OCR/],
    ['an empty file is named as empty', file('nothing.docx', FIXTURES.empty), /empty/],
    ['binary junk is refused', file('junk.docx', FIXTURES.garbage), /not a \.docx|does not read as plain text/],
  ];
  for (const [name, f, re] of REFUSALS) {
    await page.fill('#essay', 'PASTED TEXT THE READER ALREADY HAD');
    await page.setInputFiles('#file', f);
    await page.waitForFunction(
      () => (document.getElementById('status').getAttribute('data-kind') === 'error'),
      undefined, { timeout: 10000 }
    ).catch(() => {});
    const s = await status();
    check(name, re.test(s), s.slice(0, 110));
    check(`  …and the box is left as it was (${f.name})`,
      (await essay()) === 'PASTED TEXT THE READER ALREADY HAD');
  }

  /* ---- the same file twice ----
     The input's value is cleared after each read for this reason. Without it the second
     pick of a corrected document fires no event and the button looks dead. */
  {
    await open(file('essay.docx', FIXTURES.plain));
    await page.fill('#essay', 'CLEARED');
    await page.setInputFiles('#file', file('essay.docx', FIXTURES.plain));
    const again = await page.waitForFunction(
      () => (document.getElementById('essay').value || '').startsWith('My grandmother'),
      undefined, { timeout: 15000 }
    ).then(() => true).catch(() => false);
    check('picking the same file a second time reads it again', again);
  }

  /* ---- drag and drop ---- */
  {
    const has = await page.evaluate(() => {
      const zone = document.getElementById('dropzone');
      const dt = new DataTransfer();
      dt.items.add(new File(['x'], 'a.docx'));
      zone.dispatchEvent(new DragEvent('dragenter', { dataTransfer: dt, bubbles: true }));
      const on = zone.classList.contains('dragging');
      zone.dispatchEvent(new DragEvent('dragleave', { dataTransfer: dt, bubbles: true }));
      return { on, off: !zone.classList.contains('dragging') };
    });
    check('dragging a file over the box highlights it', has.on);
    check('dragging back out clears the highlight', has.off);
  }

  /* ---- a real Word file, if one was built ----
     The fixtures above are written by this script, so they can only prove the reader handles
     what this script emits. This one is produced by python-docx and is the check that it
     handles what a word processor emits. */
  const REAL = path.join(require('node:os').tmpdir(), 'palimpsest-real.docx');
  if (fs.existsSync(REAL)) {
    const r = await open(file('real.docx', fs.readFileSync(REAL)));
    check('a document written by a real word processor reads correctly',
      r.text.includes('bus transfers') && r.text.includes('shoebox'),
      JSON.stringify(r.text.slice(0, 80)));
  } else {
    console.log('  SKIP  real-word-processor document (run scripts/make_test_docx.py first)');
  }

  check('no console errors during any of it', errors.length === 0, errors.slice(0, 3).join(' | '));

  await browser.close();
  if (server) server.close();

  const failed = checks.filter((c) => !c.ok);
  console.log(`\n${'='.repeat(72)}`);
  console.log(`${checks.length - failed.length}/${checks.length} passed`);
  if (failed.length) {
    console.log('\nFAILED:');
    failed.forEach((c) => console.log(`  - ${c.name}${c.detail ? `  — ${c.detail}` : ''}`));
  }
  console.log('='.repeat(72) + '\n');
  process.exit(failed.length ? 1 : 0);
})();

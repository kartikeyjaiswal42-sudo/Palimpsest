// Does the interface actually stop a low number from reading as an acquittal?
//
// The 23-check suite drives an essay that lands in "likely machine", so it never renders the
// abstention state at all. This drives the exact essay that prompted the work: Gemini-written,
// scored 13% machine share, verdict "insufficient evidence".
const { chromium } = require('playwright');
const fs = require('fs');

const BASE = process.argv[2];
const ESSAY = fs.readFileSync(process.argv[3], 'utf8');
const checks = [];
const check = (n, ok, d = '') => { checks.push(ok); console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${n}${d ? `  — ${d}` : ''}`); };

(async () => {
  let browser;
  try { browser = await chromium.launch(); }
  catch { browser = await chromium.launch({ channel: 'chrome' }); }
  const page = await browser.newPage({ viewport: { width: 390, height: 900 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(String(e)));

  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.fill('textarea', ESSAY);
  await page.click('button[type="submit"], #analyze, button:has-text("Analy")');
  await page.waitForFunction(
    () => { const e = document.getElementById('verdict-grid'); return e && e.getAttribute('data-band'); },
    { timeout: 120000 });

  const band = await page.getAttribute('#verdict-grid', 'data-band');
  check('verdict lands in the abstention band', band === 'insufficient_evidence', `band=${band}`);

  const label = (await page.textContent('#band-label').catch(() => '')) || '';
  check('the band is named in words, above the numbers', /insufficient/i.test(label),
    label.trim().slice(0, 60));

  // Ordering is the actual fix: a reader who stops at the first big number must hit the
  // band, not "13%". Assert the DOM order rather than trusting the stylesheet.
  const bandComesFirst = await page.evaluate(() => {
    const band = document.getElementById('band');
    const grid = document.getElementById('verdict-grid');
    if (!band || !grid) return false;
    return !!(band.compareDocumentPosition(grid) & Node.DOCUMENT_POSITION_FOLLOWING);
  });
  check('the band is rendered before the percentages', bandComesFirst);

  const caveatVisible = await page.isVisible('#verdict-caveat');
  const caveatText = (await page.textContent('#verdict-caveat')) || '';
  check('the caveat is actually visible, not just present', caveatVisible);
  check('the caveat says a low number is not a clearance',
    /not|cannot|does not/i.test(caveatText) && caveatText.length > 40,
    caveatText.trim().slice(0, 110));

  const opacity = await page.evaluate(() => {
    const v = document.querySelector('#verdict-grid .metric-value');
    return v ? getComputedStyle(v).opacity : null;
  });
  check('the big percentages are visually demoted', opacity !== null && parseFloat(opacity) < 0.95,
    `opacity=${opacity}`);

  const share = (await page.textContent('#share, #machine-share').catch(() => '')) || '';
  check('the share is still shown as evidence, not hidden', share.trim().length > 0, share.trim());

  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('no horizontal overflow at 390px', overflow <= 0, `overflow ${overflow}px`);
  check('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));

  await page.screenshot({ path: '/tmp/palimpsest-ui/abstention-390.png', fullPage: true });
  await browser.close();
  const passed = checks.filter(Boolean).length;
  console.log(`\n${passed}/${checks.length} checks passed`);
  process.exit(passed === checks.length ? 0 : 1);
})();

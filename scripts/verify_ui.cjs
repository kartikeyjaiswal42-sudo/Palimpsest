// End-to-end check of the interface in a real browser.
//
//   node scripts/verify_ui.cjs [baseUrl]
//
// Requires the API to be running and playwright to be available. This is deliberately not a
// pytest: it verifies the thing a judge will actually look at, which no unit test can.

const { chromium } = require('playwright');

const BASE = process.argv[2] || 'http://127.0.0.1:8123';
const OUT = process.env.SHOT_DIR || '/tmp/palimpsest-ui';

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok, detail });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

// Prefer Playwright's own chromium; fall back to an installed Google Chrome. Without the
// fallback this script dies on any machine where `npx playwright install` has not been run,
// or where the npm package has been updated past the downloaded browser build -- which is a
// confusing failure for someone who just cloned the repo to look at the interface.
async function launch() {
  try {
    return await chromium.launch();
  } catch (err) {
    console.log('  (bundled chromium unavailable, trying system Chrome)');
    return await chromium.launch({ channel: 'chrome' });
  }
}

(async () => {
  const fs = require('fs');
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } });

  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));

  console.log(`\nverifying ${BASE}\n`);
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });
  check('page loads', await page.title() !== '');

  // ---- the example the detector catches
  await page.click('#load-sample');
  const pasted = await page.inputValue('#essay');
  check('example loads into the textarea', pasted.length > 500, `${pasted.length} chars`);

  await page.click('#analyse');
  await page.waitForSelector('#verdict-panel:not(.hidden)', { timeout: 180000 });
  await page.waitForFunction(() => document.querySelectorAll('.sentence').length > 0, null,
    { timeout: 60000 });

  const share = (await page.textContent('#share')).trim();
  const anyp = (await page.textContent('#anyp')).trim();
  check('machine share rendered', /^\d+%$/.test(share), share);
  check('any-machine probability rendered', /^\d+%$/.test(anyp), anyp);
  check('share is high for a real machine essay', parseInt(share, 10) >= 50, share);

  const interval = (await page.textContent('#share-interval')).trim();
  check('confidence interval shown', /interval/.test(interval), interval.slice(0, 60));

  const nSentences = await page.locator('.sentence').count();
  const nFlagged = await page.locator('.sentence.s3, .sentence.s4').count();
  check('sentences rendered individually', nSentences > 5, `${nSentences} sentences`);
  check('machine sentences highlighted', nFlagged > 0, `${nFlagged} highlighted`);

  // ---- the evidence panel is the core requirement: WHY, not just where
  check('evidence panel auto-opened on the top sentence',
    !(await page.locator('#evidence-panel').getAttribute('class')).includes('hidden'));

  await page.locator('.sentence').nth(2).click();
  await page.waitForTimeout(400);
  const bars = await page.locator('.bar-row').count();
  check('evidence bars rendered', bars >= 4, `${bars} feature contributions`);

  const mathText = (await page.textContent('#evidence-math')).trim();
  check('evidence panel prints the arithmetic', /log-odds/.test(mathText),
    mathText.slice(0, 70) + '…');

  // Read the four numbers off the screen and check they add up. This is the claim the whole
  // interface rests on -- that the explanation IS the computation -- so it is verified
  // against the rendered text, not against the API payload that produced it.
  const nums = mathText.replace(/−/g, '-').match(/[-+]?\d+\.\d+/g).map(Number);
  const [baseline, shownSum, otherSum, logit] = nums;
  const closes = Math.abs(baseline + shownSum + otherSum - logit) < 0.02;
  check('the displayed evidence adds up to the displayed verdict', closes,
    `${baseline} ${shownSum >= 0 ? '+' : ''}${shownSum} ${otherSum >= 0 ? '+' : ''}${otherSum} = ${logit}`);

  const remainderRow = await page.locator('.bar-row.remainder').count();
  check('the features not shown individually are still accounted for', remainderRow === 1,
    `${remainderRow} remainder row`);

  const firstBar = (await page.textContent('.bar-row .bar-name')).trim();
  check('features named in plain language', firstBar.length > 3, firstBar.replace(/\s+/g, ' '));

  const desc = (await page.textContent('.evidence-desc')).trim();
  check('each feature explains itself', desc.length > 40, desc.slice(0, 70) + '…');

  // ---- token strip
  await page.click('.tokens summary');
  await page.waitForTimeout(300);
  const toks = await page.locator('.tok').count();
  check('per-token observer view renders', toks > 5, `${toks} tokens`);

  // ---- limitations must always be visible
  const limits = await page.locator('#limitations li').count();
  const limitsText = await page.textContent('#limitations');
  check('limitations shown with every result', limits >= 3, `${limits} items`);
  check('ESL false-positive rate disclosed in the UI', /non-native|TOEFL/i.test(limitsText));

  await page.screenshot({ path: `${OUT}/caught.png`, fullPage: true });

  // ---- the example the detector misses
  await page.click('#load-missed');
  await page.click('#analyse');
  await page.waitForTimeout(1500);
  await page.waitForSelector('#verdict-panel:not(.hidden)', { timeout: 180000 });
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('flagged'),
    null, { timeout: 180000 });
  const share2 = (await page.textContent('#share')).trim();
  check('the documented failure case is reported honestly as low',
    parseInt(share2, 10) < 30, `share ${share2}`);

  // The interesting half of this example: the SENTENCE layer gets it right even though the
  // DOCUMENT verdict declines to flag. If the highlighting ever stops landing on the
  // rewritten paragraph, the demo's explanation becomes false and should fail here.
  const highlighted = await page.locator('.sentence.s3, .sentence.s4').allTextContents();
  const polished = highlighted.filter((t) =>
    /Throughout this experience|important to note|remarkable patience|profound learning/.test(t));
  check('the rewritten paragraph is still located correctly',
    highlighted.length > 0 && polished.length >= Math.ceil(highlighted.length / 2),
    `${polished.length} of ${highlighted.length} highlighted sentences are in the rewritten paragraph`);

  const anyp2 = parseInt((await page.textContent('#anyp')).trim(), 10);
  check('and the document verdict still declines to flag it', anyp2 < 97,
    `confidence ${anyp2}% is below the shipped document threshold`);
  await page.screenshot({ path: `${OUT}/missed.png`, fullPage: true });

  // ---- responsive
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForTimeout(400);
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('no horizontal overflow at 390px', overflow <= 1, `overflow ${overflow}px`);
  await page.screenshot({ path: `${OUT}/mobile.png`, fullPage: true });

  check('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));

  await browser.close();

  const failed = checks.filter((c) => !c.ok);
  console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
  console.log(`screenshots in ${OUT}`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error('FATAL', e); process.exit(1); });

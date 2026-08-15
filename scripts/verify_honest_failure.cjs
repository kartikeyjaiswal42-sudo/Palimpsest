// Two things the interface must do when it has nothing to report.
//
//   node scripts/verify_honest_failure.cjs [baseUrl]
//
// 1. AN UNREACHABLE ANALYZER MUST NOT PRODUCE A RESULT. The page used to answer a failed
//    request with a locally manufactured one -- a verdict band, a machine share, evidence
//    bars and token ranks, all from a seeded PRNG, with "bundled fixture" in a metadata chip
//    as the only disclosure. Every request is aborted here, which is what a dropped
//    connection looks like to fetch(), and the page must say nothing was measured and paint
//    no verdict.
//
// 2. A DOCUMENT WITH NOTHING MEASURABLE IN IT MUST ABSTAIN. Zero reliable sentences makes
//    the aggregate report anyMachineProbability = 0, which is the ABSENCE of a measurement;
//    it used to fall below the human threshold and come back "No evidence of machine
//    writing" for text the tool never scored a word of.

const { chromium } = require('playwright');

const BASE = process.argv[2] || 'http://127.0.0.1:8789';

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

async function launch() {
  try { return await chromium.launch(); }
  catch { return await chromium.launch({ channel: 'chrome' }); }
}

(async () => {
  const browser = await launch();
  console.log(`\n${'='.repeat(72)}\nVERIFYING HONEST FAILURE: ${BASE}\n${'='.repeat(72)}\n`);

  // ---- 1. the analyzer cannot be reached -------------------------------------------
  {
    const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });

    // Abort only the analysis call, so the page itself still loads normally.
    await page.route('**/api/analyze', (route) => route.abort('failed'));

    await page.click('#load-sample');
    await page.click('#analyse');
    await page.waitForTimeout(3500);

    const state = await page.evaluate(() => {
      const panel = document.querySelector('#verdict-panel');
      const status = document.querySelector('#status');
      return {
        verdictVisible: !!panel && !panel.classList.contains('hidden')
          && panel.getBoundingClientRect().height > 0,
        status: status ? status.textContent.trim() : '',
        anyPercent: !!document.querySelector('#verdict-panel .share, #verdict-panel .big'),
        bodyState: document.body.getAttribute('data-state'),
      };
    });

    check('no verdict is painted when the analyzer cannot be reached', !state.verdictVisible,
      `verdict panel visible=${state.verdictVisible} · body state=${state.bodyState}`);
    check('the page says nothing was measured',
      /nothing was measured/i.test(state.status), JSON.stringify(state.status.slice(0, 110)));
    check('it does not call the manufactured output a "bundled fixture"',
      !/bundled fixture/i.test(state.status));
    check('no uncaught error on the failure path', errors.length === 0, errors[0] || '');
    await page.close();
  }

  // ---- 2. nothing in the document could be measured ---------------------------------
  {
    const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } });
    await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });

    // Spans that are each under the minimum token count: nothing here is measurable.
    const unmeasurable = 'No. Yes. Ok. Fine. Sure. Maybe. Right. True. Good. Stop.';
    await page.fill('#essay', unmeasurable);
    await page.click('#analyse');
    await page.waitForSelector('#verdict-panel:not(.hidden)', { timeout: 180000 });
    await page.waitForTimeout(500);

    const payload = await page.evaluate(async (text) => {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, include_tokens: false }),
      });
      return res.json();
    }, unmeasurable);

    const v = payload.verdict || {};
    check('the API measured nothing in this text', v.nReliableSentences === 0,
      `${v.nReliableSentences} of ${v.nSentences} spans measurable`);
    check('an unmeasurable document is NOT cleared', v.band !== 'no_evidence',
      `band=${v.band} · "${v.bandLabel}"`);
    check('it abstains and says why',
      v.band === 'insufficient_evidence' || v.band === 'out_of_scope',
      `"${String(v.bandDetail || '').slice(0, 100)}…"`);
    check('the verdict does not claim to exonerate', v.canExonerate === false);
    await page.close();
  }

  await browser.close();
  const failed = checks.filter((c) => !c.ok);
  console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch((err) => { console.error('FATAL', err); process.exit(1); });

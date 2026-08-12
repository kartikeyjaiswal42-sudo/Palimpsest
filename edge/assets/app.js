// Palimpsest interface.
//
// The one rule this file follows: never show a number the backend did not compute. Every
// bar in the evidence panel is a term the classifier actually summed, and the panel prints
// the arithmetic so a reader can check that the parts add up to the whole.

const $ = (id) => document.getElementById(id);

const els = {
  essay: $("essay"),
  analyse: $("analyse"),
  sample: $("load-sample"),
  sampleMissed: $("load-missed"),
  status: $("status"),
  verdict: $("verdict-panel"),
  band: $("band"),
  bandLabel: $("band-label"),
  bandDetail: $("band-detail"),
  share: $("share"),
  shareInterval: $("share-interval"),
  anyp: $("anyp"),
  docSub: $("doc-sub"),
  verdictGrid: $("verdict-grid"),
  verdictCaveat: $("verdict-caveat"),
  textPanel: $("text-panel"),
  rendered: $("rendered"),
  evidence: $("evidence-panel"),
  evidenceText: $("evidence-text"),
  evidenceScore: $("evidence-score"),
  evidenceMath: $("evidence-math"),
  bars: $("evidence-bars"),
  tokenStrip: $("token-strip"),
  limits: $("limits-panel"),
  limitations: $("limitations"),
  privacy: $("privacy"),
};

// The footer claims something about the user's text, so it is answered by the server that
// handles the text rather than by whatever the page was built with. `textLeavesMachine`
// tracks the observer, and only a definite `false` narrows the claim -- a failed probe
// leaves the stronger warning standing, because the cost of the two errors is not
// symmetric: overstating what leaves the machine misleads nobody into pasting an essay.
async function statePrivacyHonestly() {
  if (!els.privacy) return;
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const health = await res.json();
    if (health.textLeavesMachine === false) {
      els.privacy.textContent =
        "The language model is read for token probabilities only — it is never asked for " +
        `a verdict. The observer (${health.observer}) runs in this process, so nothing you ` +
        "paste leaves this machine.";
    }
  } catch {
    /* keep the stronger claim */
  }
}
statePrivacyHonestly();

let current = null;

// Two examples, deliberately: one the detector catches and one it misses. A demo that only
// shows the success case is a sales pitch, not an instrument.
//
// CAUGHT: a real GPT-3.5 admissions essay from Liang et al. (2023), "GPT detectors are
// biased against non-native English writers", Patterns 4(7):100779. Machine-generated text,
// reproduced as a single demo fixture with attribution.
const SAMPLE_MACHINE = `The unexpected kindness of others has often left me feeling grateful, but one instance in particular stands out in my memory. Last summer, I was going through a tough time and feeling pretty down. Out of the blue, one of my friends surprised me with a care package that included all of my favorite snacks, a new book by my favorite author, and a heartfelt note of encouragement.

This small gesture of kindness had a big impact on me. It made me realize how lucky I was to have such caring and thoughtful friends, and it reminded me that even in difficult times, there are always reasons to be grateful. It also inspired me to pay it forward and look for opportunities to do something kind for others.

Since then, I've made an effort to be more intentional about expressing gratitude and kindness in my daily life. Whether it's thanking a teacher for their hard work or offering a listening ear to a friend in need, I've found that small acts of kindness can have a ripple effect and make a big difference.

In terms of how this experience has affected my motivation, it has helped me to refocus on what really matters in life. Instead of getting caught up in my own problems and stresses, I try to take a step back and appreciate the people and experiences that bring joy and meaning to my life. Ultimately, I hope to carry this gratitude and compassion forward as I continue to grow and pursue my goals in college and beyond.`;

// LETS THROUGH: an essay written by hand whose middle paragraph was rewritten by hand in a
// polished register. This is the more interesting kind of failure, and it is not the one we
// first described. The sentence layer gets it RIGHT -- the highlighting lands on the
// rewritten paragraph and almost nothing else. What fails is the document verdict: about
// 74% confidence against a shipped threshold of 97.4%, so the essay is not flagged.
//
// That is the operating point doing exactly what docs/03-evaluation.md says it does. We
// calibrated it so the tool almost never accuses a real student, and this is what that
// costs, on screen, in the demo. Left in deliberately.
const SAMPLE_MISSED = `I burned the rice. Again. My grandmother watched from the doorway of our kitchen in Lucknow and said nothing, which was worse than if she had said something. She had been making biryani for forty-one years. I had been making it for three weekends. The difference, she told me later, was not the recipe. It was that I kept checking. Lifting the lid, stirring, worrying. Steam escapes, she said. You have to trust the pot.

Throughout this experience, I came to realize that patience is a virtue that extends far beyond the culinary realm. It is important to note that the kitchen has always been more than just a room; it has been a place of profound learning, growth, and self-discovery. My grandmother, a woman of remarkable patience and unwavering dedication, taught me invaluable lessons that continue to shape who I am today.

I am not good at trusting the pot. In chemistry lab I check my burette readings four times. My lab partner Aditi calls me the recount. But that October my grandmother's hands had started shaking, and she wanted someone to know how to do it. So I learned. Badly at first, then less badly. The rice still catches sometimes. I have stopped apologising for it.`;

// ---------------------------------------------------------------- rendering

/** Five bands, not a gradient. See the note in style.css. */
function band(p, threshold) {
  if (p >= threshold) return p >= threshold + 0.35 ? "s4" : "s3";
  if (p >= threshold * 0.66) return "s2";
  if (p >= threshold * 0.33) return "s1";
  return "s0";
}

function renderText(data) {
  const { text } = current;
  const frag = document.createDocumentFragment();
  let cursor = 0;

  data.sentences.forEach((s) => {
    if (s.start > cursor) {
      frag.appendChild(document.createTextNode(text.slice(cursor, s.start)));
    }
    const span = document.createElement("span");
    span.className = `sentence ${band(s.probability, data.flagThreshold)}`;
    if (!s.reliable) span.classList.add("unreliable");
    span.textContent = text.slice(s.start, s.end);
    span.dataset.index = String(s.index);
    span.title = `${(s.probability * 100).toFixed(0)}% machine-like` +
      (s.reliable ? "" : " — too short to measure reliably");
    span.addEventListener("click", () => selectSentence(s.index));
    frag.appendChild(span);
    cursor = s.end;
  });

  if (cursor < text.length) frag.appendChild(document.createTextNode(text.slice(cursor)));
  els.rendered.replaceChildren(frag);
}

function renderVerdict(data) {
  const v = data.verdict;

  // The calibrated band, before any percentage. `band` is one of likely_machine,
  // insufficient_evidence, no_evidence; the class drives the colour so an abstention can
  // never be styled like a finding.
  els.band.className = `band band-${v.band || "insufficient_evidence"}`;
  els.bandLabel.textContent = v.bandLabel || "Insufficient evidence";
  els.bandDetail.textContent = v.bandDetail || "";
  els.share.textContent = `${(v.machineShare * 100).toFixed(0)}%`;
  els.shareInterval.textContent =
    `90% interval ${(v.machineShareLow * 100).toFixed(0)}–${(v.machineShareHigh * 100).toFixed(0)}%` +
    ` · ${v.nWords} words, ${v.nSentences} sentences`;
  els.anyp.textContent = `${(v.anyMachineProbability * 100).toFixed(0)}%`;

  const unreliable = v.nSentences - v.nReliableSentences;
  const flagAt = typeof data.documentThreshold === "number"
    ? ` · flags at ${(data.documentThreshold * 100).toFixed(0)}%` : "";
  els.docSub.textContent = (unreliable > 0
    ? `${unreliable} sentence${unreliable === 1 ? "" : "s"} too short to measure`
    : `all sentences long enough to measure`) + flagAt;

  // A percentage is a quantity; a verdict is a decision. They come apart precisely when the
  // tool abstains, and the failure is asymmetric: nobody reads "91% machine, insufficient
  // evidence" as a conviction, but everybody reads "19% machine" as an acquittal. Both live
  // failures that motivated the three-band design were of that second kind -- a Gemini essay
  // at 35% and an Opus statement of purpose at 0%, both read as clearing the writer.
  const band = v.band || "insufficient_evidence";
  const decided = band === "likely_machine" || band === "no_evidence";
  if (els.verdictGrid) els.verdictGrid.dataset.band = band;
  if (els.verdictCaveat) {
    els.verdictCaveat.hidden = decided;
    els.verdictCaveat.textContent = decided ? "" : band === "out_of_scope"
      ? "These numbers are evidence, not a verdict. This does not read as an admissions " +
        "essay, which is the only writing this tool was calibrated on, so no verdict is " +
        "offered at any percentage — high or low."
      : "These numbers are evidence, not a verdict. This document sits between the two " +
        "calibrated thresholds, so the tool declines to answer. A low percentage here is " +
        "NOT a finding that a person wrote it — capable models routinely land in this " +
        "range, and so does honest human writing.";
  }
}

function renderTokens(sentence) {
  if (!current.data.tokens) return;
  const toks = current.data.tokens.filter(
    (t) => t.start >= sentence.start && t.end <= sentence.end
  );
  els.tokenStrip.replaceChildren(
    ...toks.map((t) => {
      const el = document.createElement("span");
      el.className = `tok ${t.bucket}`;
      el.textContent = t.text;
      el.title = `rank ${t.rank} · log-probability ${t.logprob}`;
      return el;
    })
  );
}

function renderEvidence(sentence) {
  els.evidenceText.textContent = sentence.text;
  els.evidenceScore.textContent = `${(sentence.probability * 100).toFixed(1)}% machine-like`;

  // Print a sum that closes. Showing the biggest six bars and waving at "the rest" is not
  // an explanation you can check: the omitted terms outweigh every bar displayed on 30.9%
  // of sentences, and the baseline term is bigger than both. intercept + shown + remainder
  // is exactly the logit, so a reader can verify the explanation produces the verdict
  // rather than trusting that it does.
  const shownTotal = sentence.evidence.reduce((a, c) => a + c.contribution, 0);
  const sign = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
  els.evidenceMath.textContent =
    `Every bar is a term the classifier added up, and these add up to the verdict: ` +
    `baseline ${sign(sentence.intercept)} ` +
    `${sign(shownTotal)} from the ${sentence.nFeaturesShown} features shown ` +
    `${sign(sentence.evidenceRemainder)} from the other ` +
    `${sentence.nFeaturesTotal - sentence.nFeaturesShown} ` +
    `= ${sentence.logit.toFixed(2)} log-odds, which is ` +
    `${(sentence.probability * 100).toFixed(1)}% after calibration.`;

  const widest = Math.max(...sentence.evidence.map((c) => Math.abs(c.contribution)), 0.01);
  const frag = document.createDocumentFragment();

  sentence.evidence.forEach((c) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (c.measured ? "" : " unmeasured");

    const name = document.createElement("div");
    name.className = "bar-name";
    name.innerHTML = `${escapeHtml(c.label)}<span class="grp">${escapeHtml(c.group)}</span>`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${c.contribution >= 0 ? "machine" : "human"}`;
    fill.style.width = `${(Math.abs(c.contribution) / widest) * 48}%`;
    track.appendChild(fill);

    const val = document.createElement("div");
    val.className = "bar-val";
    val.textContent = c.measured
      ? `${c.contribution >= 0 ? "+" : ""}${c.contribution.toFixed(2)}`
      : "not measured";

    row.append(name, track, val);
    frag.appendChild(row);

    const desc = document.createElement("div");
    desc.className = "evidence-desc";
    desc.textContent = c.measured
      ? `${c.description} Value here: ${formatValue(c.value)} (${c.z >= 0 ? "+" : ""}${c.z.toFixed(1)} sd vs the corpus).`
      : `${c.description} This sentence was too short to measure it, so it contributed nothing.`;
    frag.appendChild(desc);
  });

  // The omitted features as one row, on the same scale as the rest. Without it the panel
  // silently drops most of the model, and a reader comparing the bars to the verdict would
  // find they disagree.
  const others = sentence.nFeaturesTotal - sentence.nFeaturesShown;
  if (others > 0) {
    const row = document.createElement("div");
    row.className = "bar-row remainder";

    const name = document.createElement("div");
    name.className = "bar-name";
    name.innerHTML = `${others} smaller terms<span class="grp">everything else</span>`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${sentence.evidenceRemainder >= 0 ? "machine" : "human"}`;
    fill.style.width = `${Math.min(Math.abs(sentence.evidenceRemainder) / widest, 1) * 48}%`;
    track.appendChild(fill);

    const val = document.createElement("div");
    val.className = "bar-val";
    val.textContent = `${sentence.evidenceRemainder >= 0 ? "+" : ""}` +
      `${sentence.evidenceRemainder.toFixed(2)}`;

    row.append(name, track, val);
    frag.appendChild(row);

    const desc = document.createElement("div");
    desc.className = "evidence-desc";
    desc.textContent =
      `The rest of the ${sentence.nFeaturesTotal} features, summed. Each is individually ` +
      `smaller than the bars above, but there are many of them, so together they can ` +
      `outweigh everything shown. Shown so the arithmetic above adds up.`;
    frag.appendChild(desc);
  }

  els.bars.replaceChildren(frag);
  renderTokens(sentence);
}

function selectSentence(index) {
  const sentence = current.data.sentences[index];
  if (!sentence) return;
  els.rendered.querySelectorAll(".sentence").forEach((el) =>
    el.classList.toggle("selected", el.dataset.index === String(index))
  );
  els.evidence.classList.remove("hidden");
  renderEvidence(sentence);
  els.evidence.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------- plumbing

function formatValue(v) {
  if (v === null || v === undefined) return "—";
  return Math.abs(v) >= 100 || Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function analyse() {
  const text = els.essay.value.trim();
  if (!text) {
    els.status.textContent = "Paste an essay first.";
    return;
  }
  els.analyse.disabled = true;
  els.status.textContent = "Scoring…";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, include_tokens: true }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || body.error || `request failed (${res.status})`);
    }
    const data = await res.json();
    current = { text, data };

    renderVerdict(data);
    renderText(data);
    els.limitations.replaceChildren(
      ...(data.limitations || []).map((l) => {
        const li = document.createElement("li");
        li.textContent = l;
        return li;
      })
    );
    const notice = document.getElementById("notice");
    if (notice) {
      // The observer scores at most 6,000 characters. Beyond that, sentences carry no
      // observer tokens, are marked unreliable, and are dropped from the document verdict --
      // so the verdict describes a prefix. Presenting that as a verdict on the whole essay
      // would be the same class of quiet overclaim this project exists to avoid.
      if (data.meta && data.meta.clipped) {
        notice.textContent =
          "This essay is longer than the observer scores in one pass (6,000 characters). " +
          "The verdict describes the first 6,000 characters only; the rest is shown " +
          "unhighlighted and was not measured.";
        notice.classList.remove("hidden");
      } else {
        notice.classList.add("hidden");
      }
    }
    [els.verdict, els.textPanel, els.limits].forEach((p) => p.classList.remove("hidden"));
    els.evidence.classList.add("hidden");

    const flagged = data.sentences.filter((s) => s.probability >= data.flagThreshold);
    els.status.textContent =
      `${data.meta.elapsedMs} ms · ${data.meta.observer} on ${data.meta.device} · ` +
      `${flagged.length} of ${data.sentences.length} sentences flagged`;
    if (flagged.length) selectSentence(flagged[0].index);
  } catch (err) {
    els.status.textContent = `Failed: ${err.message}`;
  } finally {
    els.analyse.disabled = false;
  }
}

els.analyse.addEventListener("click", analyse);
els.sample.addEventListener("click", () => {
  els.essay.value = SAMPLE_MACHINE;
  els.status.textContent = "Loaded a real GPT-3.5 essay (Liang et al. 2023). Press Analyse.";
});
els.sampleMissed.addEventListener("click", () => {
  els.essay.value = SAMPLE_MISSED;
  els.status.textContent =
    "Loaded a case we let through. The highlighting finds the rewritten paragraph — " +
    "but the document confidence lands under our threshold, so we do not flag it.";
});
els.essay.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyse();
});

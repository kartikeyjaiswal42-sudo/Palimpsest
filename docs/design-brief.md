# Palimpsest — UI/UX redesign brief

**Paste everything below the line into Claude Design or v0.app.** It is written as a
self-contained prompt: product context, visual direction, a screen-by-screen spec, every
button's behaviour, the real data shape, validated colour tokens, and a hard integration
contract.

Return three files — `index.html`, `style.css`, `app.js` — and paste them back into Claude
Code. They will be diffed against the current `web/` and checked against a 30-assertion
browser suite before deploy.

The **Integration contract** section is not stylistic advice. Every id, class and literal
string listed there is read by `web/app.js`, by `scripts/verify_ui.cjs` (30 checks), by
`scripts/verify_abstention.cjs` (9 checks), or by `edge/scripts/sync_web.py` (which aborts
the public build if a patch anchor moves). Breaking one is a failed build, not a nitpick.

---

# Redesign the interface of Palimpsest, an AI-text detector for college admissions essays

## What the product is

A teacher or admissions reader pastes a student's essay. The tool returns **where** in the
essay reads as machine-written and **why** — sentence by sentence, with the actual
arithmetic behind every score on screen.

It is not a verdict machine. A language model is read for token probabilities only; it is
never asked "is this AI". Every number displayed is arithmetic done on those probabilities,
and the interface exists to let a reader **check the working** rather than trust a score.

The name is the concept: a palimpsest is a manuscript where the earlier writing shows
through the later. That is the product — the human draft showing through the polish.

## The one rule that outranks every design instinct

**This interface must never let a number read as an acquittal.**

Two real failures caused the current design: an AI-written essay reported as "35% machine"
and another as "0%", both read by a human as clearing the student. The tool has three
possible answers — *likely machine-written*, *insufficient evidence*, *no evidence* — and
"insufficient evidence" is a **real answer, not a degraded one.**

Consequences you must preserve, and should make *stronger* rather than prettier:

1. **The calibrated band comes first, above the percentages, always.** A reader who stops
   at the first big thing on the page must hit words, not a number.
2. **When the tool abstains, the two big percentages are visually demoted** (muted, lighter
   weight) while the band stays at full contrast. They remain on screen as evidence; they
   lose their claim to be the answer.
3. **Nothing is ever green.** The tool cannot clear anybody, so no state gets the colour
   that means "all good". Success-green anywhere on the verdict is a design error.
4. **The "what this tool gets wrong" panel ships with every result.** It is not a
   collapsed disclaimer, not a footer, not fine print. It discloses a 10.9% false-positive
   rate on essays by non-native English speakers. Design it as a first-class panel.
5. **Text the tool refused to measure gets no shade and no percentage.** It is drawn as
   explicitly unmeasured, never as low-risk.

If a layout decision fights one of these, the rule wins.

## Visual direction: elevate the manuscript

Keep the paper identity the name implies — a document under examination — but make it
deliberate rather than beige-by-default.

- **The essay is paper.** Warm off-white stock, generous measure, serif body. It should
  read as an object you are holding and marking up.
- **The apparatus is chrome.** Everything that is *about* the essay — verdict, evidence,
  limits — sits on slightly recessed surfaces around the paper, in sans, quieter.
- **Highlighting is annotation, not fill.** The current design paints whole sentences in
  solid orange blocks; a flagged essay becomes an unreadable orange slab. Replace with
  marginal-annotation logic: a light tint plus a **weighted underline** that carries the
  band, so the words stay legible and the marking reads as a reader's pen.
- **Editorial type.** A real modular scale, tight display letterspacing, proper measure
  (~68ch for the essay). Serif for the document and quotes; sans for the apparatus; mono
  for numbers and tokens.
- Restrained, scholarly, high-contrast. No gradients on data, no shadows on data, no
  rounded-blob card soup.

## Hard constraints

- **Vanilla HTML + CSS + JS. No framework, no build step, no bundler, no Tailwind, no
  React.** The output is served as three static files.
- **Zero external requests. No Google Fonts, no CDN, no remote images, no icon packs.**
  This is load-bearing, not a preference: the page makes an explicit claim about whether
  the user's text leaves their machine, and a font request out to a third party
  contradicts it. Use system/local font stacks and inline SVG only.
- **Full dark mode via `prefers-color-scheme`.** The current stylesheet recolours only the
  verdict bands for dark, leaving light-red text on cream — that is an actual bug to fix,
  not a feature to add. Every token needs both modes. No theme toggle needed.
- **No horizontal overflow at 390px.** Asserted at 390×900.
- **Zero console errors.** Asserted.
- **Keyboard accessible.** Sentences are currently click-only `<span>`s with no tabindex —
  fix this. Visible `:focus-visible` everywhere.
- **Do not change any copy.** Every sentence of visible text was written to prevent a
  specific misreading, and several strings are regex-asserted. You may *add* new labels
  (axis captions, legends, section headings for things you introduce). You may not edit
  or reword existing text.

---

## Screens and states

There is one page. It moves through three states — drive them off a
`data-state` attribute on `<body>`: `idle` → `scoring` → `result`.

### 1. Masthead
Wordmark "Palimpsest" + the tagline. Small, confident, not a hero banner. Room for a
quiet observer/status chip on the right at desktop width.

### 2. Composer (always present, all three states)
- Label "Paste an essay", one `<textarea id="essay">`, and the control row.
- **In `idle`:** roomy textarea (~12 rows). Add a short "how this works" strip — the tool's
  method is genuinely interesting and currently nowhere on the page: *two observers, neither
  of which gives a verdict → 43 interpretable features → a calibrated probability whose
  logit is the explanation.* Three steps or a small diagram. This strip disappears in
  `result`.
- **In `scoring`:** controls disabled, a real progress affordance (indeterminate shimmer
  or skeleton), not just a text status. Scoring can take seconds.
- **In `result`:** the composer **compacts** — textarea shrinks to roughly 6–8 rows —
  so the answer sits near the top of the page instead of below a wall of input.
  > **It must stay visible and editable in `result`.** The test suite refills `#essay` and
  > re-runs without touching anything else. `display:none`, `visibility:hidden`,
  > zero height, or a collapsed `<details>` around it will hang the suite.

### 3. Verdict (`#verdict-panel`) — full width, the headline
DOM order is fixed: **`#band` first, then `#verdict-grid`.**

- **`#band`** — the calibrated answer in words. Four colour states, and the middle two are
  as visually prominent as the finding: `likely_machine`, `insufficient_evidence`,
  `no_evidence`, `out_of_scope`. Give this real presence; it is the answer.
- **`#verdict-grid`** — two metrics side by side, carrying `data-band="<band>"`:
  - "How much reads as machine-written" → `#share`, with `#share-interval` beneath.
  - "Confidence any of it is machine-written" → `#anyp`, with `#doc-sub` beneath.
  - When `data-band` is `insufficient_evidence` or `out_of_scope`, `.metric-value` must
    compute to **opacity < 0.95** and lighter weight. This is asserted from computed style.
- **`#verdict-caveat`** — the "these numbers are evidence, not a verdict" paragraph. Shown
  only when abstaining, and it must be **genuinely visible** (asserted via `isVisible`), not
  merely present.
- Then the standing "two different questions, kept apart on purpose" note.

**New element to add here — the sentence profile.** The single biggest missing affordance.
One small ordered column chart, one column per sentence, height = that sentence's
probability, with a horizontal reference rule at the real `flagThreshold` from the payload.
It shows the *shape* of the evidence at a glance — an all-machine essay is a plateau; one
polished paragraph is a spike — which is exactly what the two-number split is trying to
say in words. Make each column click-to-select the matching sentence, with hover tooltips.

Chart rules (these come from a validated visualization standard, don't improvise):
- Single ordinal ramp, one hue, light→dark with magnitude. Ramp values are in the tokens
  below and have been validated in both modes; use them as given.
- Columns ≤24px wide, 4px rounded top, square at the baseline, 2px surface gap between
  neighbours. Threshold rule is a 1px solid recessive line — never dashed rainbow.
- **Unmeasurable sentences get a hatched empty slot at the baseline, never a height and
  never a colour.** Rule 5 above applies to the chart exactly as it applies to the text.
- One series, so no legend box; label the threshold rule directly.
- Text never wears the data colour — labels stay in ink tokens.

### 4. The essay (`#text-panel`) and 5. Evidence (`#evidence-panel`)

**Make these two a side-by-side workspace at ≥1080px: the document on the left, the
evidence rail sticky on the right.** Today they are stacked, so clicking a sentence scrolls
the sentence you are investigating off the screen — you cannot see the words and their
explanation at once. That is the worst interaction problem in the product. Below 1080px,
stack them (essay first, evidence second) as now.

**The essay panel** renders the original text with every sentence wrapped in a clickable,
focusable span, plus a legend (unremarkable / some signal / flagged) and the hint line.
The legend's "unremarkable" swatch is currently invisible — give it a real outline.

**The evidence panel** answers "why did this sentence score what it did":
- The sentence quoted, plus its score as a pill.
- **`#evidence-math`** — the arithmetic line. This is the product's signature claim: the
  explanation *is* the computation, and the parts add up to the whole. Give it real
  typographic treatment — mono numerals, a bit of space — rather than the grey mush it is
  now. **Do not restructure the string** (see contract).
- **The feature bars.** Each is one term the classifier summed. A diverging layout around a
  centre axis: contributions toward *machine* extend right in the machine colour,
  toward *human* extend left in the human colour. Add a single axis caption
  (`toward human ← → toward machine`) once at the top — currently the direction is
  unlabelled and unguessable.
  Current layout wastes the row: a cramped right-aligned 190px name column, then a bar,
  then a value, then a description indented 202px underneath. Rework it — name and group on
  one line, bar and signed value on the axis, description full-width and muted beneath.
  Must degrade to a sane stacked layout on mobile.
- A **remainder row** for the ~33 features not shown individually, separated by a rule
  because it closes the sum rather than explaining anything.
- Rows for features that could not be measured are dimmed and read "not measured".
- **The token strip**, inside a `<details>`: every word tinted by how predictable the
  observer found it, in four buckets. Currently an undifferentiated monospace wall — give it
  breathing room and a small legend naming the four buckets (top-10 / top-100 / top-1000 /
  tail). Keep it collapsed by default.

### 6. Limits (`#limits-panel`) — full width
"What this tool gets wrong", 8 measured failure modes. Treat as a first-class panel with a
caution accent — the project's honesty is the feature. Two columns at desktop is fine.

### 7. Footer
The privacy paragraph, `#privacy`. Restyle freely; **do not touch the words.**

### 8. Notice (`#notice`)
An amber advisory shown only when the essay was longer than the observer's window, so the
verdict describes only the opening. Lives in the composer area. Hidden via the `hidden`
attribute — and because an author `display` rule beats `[hidden]`, the stylesheet must
carry `#notice[hidden]{display:none !important}`.

---

## Every button and control

| Control | id | Behaviour |
|---|---|---|
| **Analyse** | `analyse` | POSTs `{text, include_tokens:true}` to `/api/analyze`. Disables itself, sets state `scoring`. On success renders verdict → text → limitations → notice, reveals the four panels, auto-selects the first flagged sentence, sets state `result`. On failure writes `Failed: <message>` to `#status`. Re-enables in `finally`. **Its label must keep starting with "Analy"** (a test clicks it by text). |
| **Example it catches** | `load-sample` | Fills `#essay` with a real GPT-3.5 essay fixture, writes a status line. Does **not** auto-analyse. |
| **Example it lets through** | `load-missed` | Fills `#essay` with the honest-failure fixture (a hand-polished paragraph inside human writing — the tool locates it and still declines to flag the document). Writes a status line. Does **not** auto-analyse. |
| **⌘/Ctrl + Enter** in the textarea | — | Runs Analyse. |
| **A sentence** | `.sentence` | Click **or Enter/Space** selects it: marks `.selected`, opens `#evidence-panel`, renders its bars and tokens. Must be `tabindex="0"` with an appropriate role and a visible focus ring. |
| **A profile column** (new) | — | Selects the same sentence as clicking the sentence itself. Keyboard reachable, or excluded from tab order if the sentence spans already cover it. |
| **Word-by-word view** | `.tokens summary` | Native `<details>` disclosure. Keep it a real `<summary>`; a test clicks it. |

Loading, empty and error states all need a designed treatment — today they are one grey
string in `#status`.

---

## The data you are rendering

Real response from `POST /api/analyze`, trimmed:

```jsonc
{
  "verdict": {
    "machineShare": 1.0, "machineShareLow": 1.0, "machineShareHigh": 1.0,
    "anyMachineProbability": 0.9716,
    "nSentences": 11, "nWords": 260, "nReliableSentences": 11,
    "band": "likely_machine",              // | insufficient_evidence | no_evidence | out_of_scope
    "bandLabel": "Likely machine-written",
    "bandDetail": "Above the threshold calibrated so that at most 5% of at-risk human essays are flagged (observed 4.0% on 1492 held-out documents).",
    "canExonerate": false, "inDomainProbability": 0.1154
  },
  "flagThreshold": 0.3004,                 // the sentence threshold — the chart's reference rule
  "sentences": [{
    "index": 2, "start": 190, "end": 355, "text": "Out of the blue, …",
    "probability": 0.965, "smoothed": 0.96, "nWords": 34,
    "reliable": true,
    "unreliableReason": null,              // | too_short | too_long | beyond_observer_window
    "logit": 4.6, "intercept": -0.406,
    "evidenceRemainder": -0.5373, "nFeaturesShown": 6, "nFeaturesTotal": 39,
    "evidence": [{
      "name": "mean_logprob",
      "label": "Average predictability",
      "group": "likelihood",               // likelihood | rank | corpus | context | rhythm | register | composite
      "description": "Mean log-probability the observer model assigned to the words actually used…",
      "value": -1.4125, "z": 1.292, "weight": 2.0407,
      "contribution": 2.6369,              // signed: + toward machine, − toward human
      "toward": "machine", "measured": true
    }]
  }],
  "tokens": [{ "text": " others", "start": 26, "end": 33,
               "logprob": -4.725, "rank": 5,
               "bucket": "top10" }],       // top10 | top100 | top1000 | tail
  "limitations": ["Measured on held-out data: 10.9% of TOEFL essays…", "…"],  // 8 strings
  "meta": { "observer": "@cf/qwen/qwen3-30b-a3b-fp8", "device": "remote",
            "elapsedMs": 19.8, "nObserverTokens": 286,
            "clipped": false, "observerCharLimit": 6000 }
}
```

Feature labels you will see, so you can size the name column honestly: *Average
predictability · Average log rank · Smoother than the author's baseline · Length vs the
author's baseline · Fluent but atypical · Distance from applicant prose · Sentence length ·
Local rhythm · Rhythm of surprise · Vocabulary richness · Contractions · Words in the top 10
· Likelihood/rank ratio*.

**Never display a number the response did not contain.** No invented confidence bars, no
derived "risk scores", no document-level threshold marker — the payload has no
`documentThreshold`, so nothing may draw one.

---

## Design tokens

Contrast and ramp values below are already validated (WCAG contrast, and the ordinal ramp
against monotone-lightness / step-gap / light-end-contrast / single-hue checks in both
modes). **Use them as given; if you change one, it needs re-validating.**

```css
/* ---- light ---- */
--paper:      #f6f3ec;   /* page */
--card:       #fffdf8;   /* panel / the essay's stock */
--ink:        #1b1a17;   /* body            17.1:1 */
--ink-soft:   #5c584f;   /* secondary        7.0:1 */
--ink-faint:  #7a7468;   /* metadata         4.6:1  (was #8a8478 = 3.65:1, below AA — fixed) */
--rule:       #ddd6c7;
--accent:     #7a4a2b;   /* 7.3:1 — primary action */
--machine:    #a8402b;   /* 6.0:1 — toward machine */
--human:      #3f6f52;   /* 5.7:1 — toward human   */
--focus:      #2f5d8a;

/* ---- dark (prefers-color-scheme: dark) ---- */
--paper:      #17150f;
--card:       #1e1b15;
--ink:        #ece7db;   /* 13.9:1 */
--ink-soft:   #b3ac9c;   /*  7.6:1 */
--ink-faint:  #8a8375;   /*  4.6:1 */
--rule:       #332f26;
--machine:    #e0866a;   /*  6.4:1 */
--human:      #7fbf98;   /*  8.0:1 */

/* ---- sentence-profile ordinal ramp (validated) ---- */
light, on #f6f3ec:  #d9a071 → #c87a4a → #b0552f → #8f3320   /* low → high */
dark,  on #17150f:  #78351f → #9c4e2c → #c2703f → #e0a06a   /* low → high */
```

**Highlight bands.** Five steps `s0`–`s4`, deliberately *bands and not a continuous
gradient*: a smooth ramp invites a reader to over-read a 3% difference the model cannot
actually resolve. `s0` is unmarked. In the annotation direction, keep the tints light and
let a **weighted underline** carry most of the signal, so the marking survives dark mode and
never buries the words. Current light tints, as a starting point:
`s1 #f6e9dd · s2 #f2d9c4 · s3 #ecc3a5 · s4 #e3a986`. Dark mode needs its own set —
low-alpha washes of the machine hue plus the underline.

**Band colours** (verdict states, all four need both modes; nothing green):
`likely_machine` red-toned · `insufficient_evidence` amber-toned · `no_evidence`
slate-toned · `out_of_scope` informational blue (it is a statement about the tool's scope,
not a finding about the writer).

**Type.** Serif for the document, quotes and wordmark; system sans for the apparatus; mono
for numerals, feature values and the token strip. Local stacks only, e.g.
`"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif` and
`ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace`. Big standalone
figures use proportional numerals; only columns of numbers get `tabular-nums`.

---

## Integration contract — breaking any of this fails the build

### Ids that must exist, with these exact names
`essay` · `analyse` · `load-sample` · `load-missed` · `status` · `notice` ·
`verdict-panel` · `band` · `band-label` · `band-detail` · `verdict-grid` · `share` ·
`share-interval` · `anyp` · `doc-sub` · `verdict-caveat` · `text-panel` · `rendered` ·
`evidence-panel` · `evidence-text` · `evidence-score` · `evidence-math` · `evidence-bars` ·
`token-strip` · `limits-panel` · `limitations` · `privacy`

### Classes that must keep their names and roles
- `.panel` and **`.hidden` must mean `display:none`** — panels are revealed by removing it.
- `.sentence` plus band classes `.s0 .s1 .s2 .s3 .s4`, plus `.unreliable`, `.selected`.
- `.bar-row`, `.bar-row.remainder`, `.bar-row.unmeasured`, `.bar-name`, `.bar-name .grp`,
  `.bar-track`, `.bar-fill.machine`, `.bar-fill.human`, `.bar-val`, `.evidence-desc`.
- `.tokens` (a `<details>`) containing a `<summary>`, `.token-strip`, `.tok` with buckets
  `.top10 .top100 .top1000 .tail`.
- `.band-likely_machine`, `.band-insufficient_evidence`, `.band-no_evidence`,
  `.band-out_of_scope`.
- `.metric-value`, `.metric-label`, `.metric-sub`, `.swatch`, `.pill`.

### Structural rules asserted by tests
1. **`#band` precedes `#verdict-grid`** in document order.
2. `#verdict-grid[data-band="insufficient_evidence"] .metric-value` and the
   `out_of_scope` equivalent compute to **opacity < 0.95**.
3. `#verdict-caveat` is genuinely visible when abstaining.
4. **Exactly one `<textarea>` on the page**, and it stays visible and editable in every
   state including `result`.
5. **Exactly one element matches `button[type="submit"], #analyze, button:has-text("Analy")`,
   and it is the Analyse button.** So: no `type="submit"` anywhere, no `id="analyze"`, and
   no other button whose text contains "Analy".
6. **Only real sentences carry `.sentence`.** The profile chart's columns must use different
   class names — a test counts `.sentence` and requires `< 4` for a single-span run-on.
   Flagged = `.sentence.s3, .sentence.s4`.
7. An unmeasurable span gets `s0` + `.unreliable`, **no shade**, and a `title` that does not
   contain "machine-like".
8. `#status` must contain the literal phrase **`sentences flagged`** after a successful run,
   and must be the **last** thing written by the render (the suite waits on it to know the
   whole render finished — including the notice).
9. `#notice[hidden] { display: none !important; }` must be in the stylesheet.
10. `#limitations` renders `<li>` children, ≥3, and the text must keep matching
    `/non-native|TOEFL/i`.
11. `.tokens summary` must be clickable and reveal `.tok` elements.
12. No horizontal overflow at 390×900. No console errors.

### Literal strings that must survive byte-for-byte
- **`#privacy`'s paragraph.** Asserted against `/sent to|leaves? (this|your) machine/i`
  for the remote observer and `/nothing you paste leaves this machine/i` for the local one
  — and `app.js` rewrites it from `/api/health`. Restyle, never reword.
- **`#evidence-math`'s string.** A test scrapes the first four decimals in order —
  baseline, features shown, the remainder, the logit — and asserts they sum. It also
  requires the substring `log-odds`. Keep the sentence's construction exactly; style the
  element, not the text.
- **The footer's privacy block and the `if (!res.ok)` block in `app.js`** are patch anchors
  for `edge/scripts/sync_web.py`, which builds the public deployment and **aborts** if it
  cannot find them verbatim, indentation and line breaks included. Leave both alone.

### Behaviour that must not regress
- `#privacy` is narrowed from `/api/health` on load: only an explicit
  `textLeavesMachine === false` may soften the claim. A failed probe leaves the *stronger*
  warning standing — the errors are not symmetric.
- Unmeasurable spans report **which** condition failed (`too_short`, `too_long`,
  `beyond_observer_window`), never a generic guess. The old code always said "too short",
  including for a 138-word run-on — precisely backwards, and shown to the second-language
  writers least able to argue with it.
- The over-length notice quotes the character limit **from `meta.observerCharLimit`**, never
  a constant in the stylesheet or script.

---

## Deliverable

Three files — `index.html`, `style.css`, `app.js` — complete and self-contained, no build
step, no external requests, both colour modes, verified at 390px and desktop.

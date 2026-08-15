#!/usr/bin/env python3
"""Build the REDACTED failure artifact the hosted Worker serves.

    python3 edge/scripts/build_failures.py

Reads ``artifacts/confident_failures.json`` and writes ``edge/assets/failures.json`` with
every full essay removed.

Why this is a build step and not a flag on the serving code
-----------------------------------------------------------
``confident_failures.json`` embeds the complete text of the documents the detector got most
wrong, and on this corpus those are **real second-language students' essays** (ELLIPSE) and
a hybrid spliced onto a Liang essay. The repository already treats that class of text as not
ours to redistribute -- ``.gitignore`` keeps ``data/raw/`` out of git for exactly this
reason, and the same logic applies with more force to a public URL.

Redaction could have been a conditional in the Worker. It is a separate file instead,
because a conditional is one edit away from being inverted and nothing would notice: the
page would look correct and quietly serve the essays. Here the assets directory physically
cannot contain the full text, so the failure mode requires someone to copy the wrong file in
rather than to make a mistake.

What survives, and why that is enough
-------------------------------------
Everything the panel exists to show:

* the score, the band, the severity and its two components;
* the per-feature contributions that reconstruct the logit;
* **the one sentence that drove the verdict**, which is what makes the arithmetic legible.

A single quoted sentence carrying the analysis of why it was misread is ordinary academic
practice. Republishing the whole essay is not, and the difference is the whole point of this
file.

What is removed
---------------
* ``text`` -- the full essay, replaced by ``null`` plus a stated reason;
* ``humanExplanation`` is KEPT, because it is ours and it is the analysis.

The local build (``palimpsest.api.app``) reads the unredacted artifact and is unaffected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "confident_failures.json"
TARGET = ROOT / "edge" / "assets" / "failures.json"

REASON = (
    "The full essay is withheld from the hosted build. These are real students' essays, "
    "including second-language writers, and republishing them at a public URL is not "
    "something this project has the right to do. The sentence that drove the verdict is "
    "quoted below with its arithmetic; run the tool locally to see the whole document."
)

#: Longest single quoted sentence. A cap is needed rather than "the sentence as segmented"
#: because the documents this panel surfaces are, disproportionately, the unpunctuated
#: run-on essays second-language students write -- and there ONE segmented sentence can be
#: most of the document. Quoting it whole would defeat the redaction while looking like a
#: quotation.
MAX_QUOTE_CHARS = 320

#: And a whole-document ceiling, because two capped quotes from a short essay can still add
#: up to most of it. Floor so that a very short document still shows something legible.
MAX_QUOTE_FRACTION = 0.35
MIN_QUOTE_BUDGET = 200

_ELLIPSIS = " […]"


def _trim(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - len(_ELLIPSIS), 0)].rstrip() + _ELLIPSIS


def redact(payload: dict) -> dict:
    out = dict(payload)
    out["redacted"] = True
    out["redactionReason"] = REASON
    out["quoteCapChars"] = MAX_QUOTE_CHARS
    out["failures"] = []

    for entry in payload.get("failures", []):
        e = dict(entry)
        essay = entry.get("text") or ""
        e["text"] = None
        e["textWithheld"] = True

        budget = max(MIN_QUOTE_BUDGET, int(len(essay) * MAX_QUOTE_FRACTION))
        sentences = []
        for s in entry.get("drivingSentences") or []:
            s = dict(s)
            quoted = _trim(s.get("text") or "", min(MAX_QUOTE_CHARS, budget))
            budget -= len(quoted)
            s["text"] = quoted
            s["quoteTruncated"] = quoted.endswith(_ELLIPSIS)
            sentences.append(s)
        e["drivingSentences"] = sentences
        out["failures"].append(e)
    return out


def main() -> int:
    if not SOURCE.exists():
        print(f"no {SOURCE.relative_to(ROOT)} -- run scripts/confident_failures.py first",
              file=sys.stderr)
        return 1

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    out = redact(payload)

    # Assert the redaction rather than trusting the loop above: this file is the only thing
    # standing between a student's essay and a public URL.
    #
    # The FIRST version of this guard was wrong and rejected a correct build. It asserted
    # that no 60-character slice of the essay appeared anywhere in the output -- but the
    # driving sentence IS a slice of the essay, deliberately, so the check could never pass
    # while the panel still showed its evidence. "Any text survived" is the wrong question.
    # The right one is HOW MUCH survived, which is what these three checks measure.
    blob = json.dumps(out, indent=2)
    for original, e in zip(payload.get("failures", []), out["failures"], strict=True):
        essay = original.get("text") or ""
        doc_id = original["docId"]

        if e["text"] is not None:
            raise SystemExit(f"REFUSED: {doc_id} still carries a full essay.")

        quoted = sum(len(s.get("text") or "") for s in e["drivingSentences"])
        if quoted > max(MIN_QUOTE_BUDGET, len(essay) * MAX_QUOTE_FRACTION) + 1:
            raise SystemExit(
                f"REFUSED: {doc_id} retains {quoted} of {len(essay)} characters "
                f"({quoted / max(len(essay), 1):.0%}), over the "
                f"{MAX_QUOTE_FRACTION:.0%} ceiling."
            )
        for s in e["drivingSentences"]:
            if len(s.get("text") or "") > MAX_QUOTE_CHARS:
                raise SystemExit(
                    f"REFUSED: a quoted sentence in {doc_id} is "
                    f"{len(s['text'])} chars, over the {MAX_QUOTE_CHARS} cap."
                )

        # The essay must not be reconstructable from the payload as a contiguous run
        # longer than the quotes we intended to publish.
        if len(essay) > 400 and essay[: MAX_QUOTE_CHARS + 40] in blob:
            raise SystemExit(
                f"REFUSED: a contiguous opening run of {doc_id} longer than the quote cap "
                "appears in the output."
            )

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(blob, encoding="utf-8")

    kept = sum(len(f.get("drivingSentences") or []) for f in out["failures"])
    print(f"wrote {TARGET.relative_to(ROOT)}")
    print(f"  {len(out['failures'])} failures, full essays removed, "
          f"{kept} driving sentences kept")
    print(f"  {SOURCE.stat().st_size / 1024:.1f} KB -> {TARGET.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

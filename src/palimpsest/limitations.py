"""The error rates shown under every verdict, rendered from the measurements themselves.

Why this is a module and not four lines inside ``api/app.py``
-------------------------------------------------------------
It used to be those four lines, and they were wrong in a way nothing caught.

``api/app.py`` selects an artifact set with ``PALIMPSEST_SUFFIX`` -- detector, bands and
genre gate all move together, and the default observer is the remote 30 B model, so the
served set is ``_remote``. The limitations panel alone read ``evaluation.json``
unconditionally. The application therefore served one detector and published another
detector's error rates, unlabelled, under every verdict it produced.

That is the precise failure ``_limitations()``'s own docstring was written to prevent --
"an earlier version hard-coded the percentages, the model was retrained, and the interface
went on confidently displaying the previous run's error rates" -- reached by a different
route: not stale prose, but a stale *file*.

The hosted build fixed it in ``edge/scripts/build_artifacts.py`` and the Python application
did not, so the two shipped different honesty panels for the same detector. Both callers now
render from here, which is what makes them agree by construction rather than by review.

What the numbers do when a set was not measured on the served build
-------------------------------------------------------------------
``evaluation_remote.json`` covers fewer sets than the GPT-2 run does -- the adversarial and
evade-detection suites were never re-run against the 30 B observer. Dropping those statements
would quietly shorten the disclosure; printing them as if they described the served build
would be the original bug again. They are carried over from the GPT-2 run and **labelled**,
because a number measured on a different instrument is still worth showing and is not worth
showing silently.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["CARRIED", "GENERIC", "render"]

GENERIC = (
    "Short passages carry little evidence. Anything under about five sentences is reported "
    "as unreliable rather than scored confidently."
)

#: Appended to any statement whose number was measured on the GPT-2 observer rather than on
#: the observer actually serving requests.
CARRIED = " (measured on the GPT-2-observer build; not re-run for this observer)"

_UNMEASURED = "Error rates have not been measured for this build."


def render(artifacts: Path, suffix: str = "", extra: list[str] | None = None) -> list[str]:
    """Statements for the artifact set ``suffix``, in the order the interface shows them.

    ``extra`` is appended before the closing generic statement, for a disclosure that is true
    of one deployment only -- the hosted build's observer repeatability, which the Python
    build never had to answer because it scored each essay once and cached the result.
    """
    served = artifacts / f"evaluation{suffix}.json"
    fallback = artifacts / "evaluation.json"
    if not served.exists() and not fallback.exists():
        return [_UNMEASURED, GENERIC]

    sets = json.loads(served.read_text(encoding="utf-8")).get("sets", {}) if served.exists() else {}
    other = (
        json.loads(fallback.read_text(encoding="utf-8")).get("sets", {})
        if fallback.exists() and fallback != served
        else {}
    )

    def pick(name: str) -> tuple[dict, str]:
        """The set as measured on the served build, else on the GPT-2 build, with a tag."""
        if name in sets:
            return sets[name], ""
        return other.get(name, {}), CARRIED

    out: list[str] = []

    esl, tag = pick("esl")
    toefl = (esl.get("breakdown") or {}).get("liang_toefl", {})
    if toefl.get("documentFPR") is not None:
        out.append(
            f"Measured on held-out data: {toefl['documentFPR']:.1%} of TOEFL essays written "
            f"by non-native speakers were wrongly flagged "
            f"({esl.get('documentFPR', 0):.1%} across all English-language-learner essays)"
            f"{tag}. Do not use this as evidence against a student."
        )

    # The gap between "a generator we trained on" and "a family we did not" is the single
    # most misleading thing a detector can hide, so both numbers go in front of every user.
    unseen, unseen_tag = pick("modern_unseen")
    family, family_tag = pick("modern_unseen_family")
    if unseen.get("documentRecall") is not None and family.get("documentRecall") is not None:
        out.append(
            f"Trained on GPT-3.5 and Gemini 3 only. On a model checkpoint held out of "
            f"training it catches {unseen['documentRecall']:.0%} of essays{unseen_tag}, but "
            f"on a different model family it catches {family['documentRecall']:.0%} "
            f"({family.get('nDocuments', 0)} essays){family_tag}. Expect it to be worse on a "
            "generator nobody has measured, including whatever ships next."
        )

    # The ceiling, in front of the reader rather than in a design document. This is the
    # number that makes "no evidence found" mean what the bottom band says it means.
    claude, tag = pick("modern_claude")
    if claude.get("documentRecall") is not None:
        out.append(
            "On frontier prose (Claude opus/sonnet/haiku) it catches "
            f"{claude['documentRecall']:.1%} of essays{tag}. A low score is not evidence "
            "that a person wrote this."
        )

    prompting, tag = pick("unseen_prompting")
    if prompting.get("documentRecall") is not None:
        out.append(
            "When a generator is prompted to evade detection, only "
            f"{prompting['documentRecall']:.1%} of essays were caught{tag}."
        )

    adversarial, tag = pick("adversarial")
    if adversarial.get("documentRecall") is not None:
        out.append(
            "Prose deliberately composed to imitate a model was caught "
            f"{adversarial['documentRecall']:.0%} of the time{tag}."
        )

    loc, tag = pick("localisation")
    if loc.get("recall") is not None:
        out.append(
            f"Inside a part-machine essay we find {loc['recall']:.0%} of the machine "
            f"sentences (precision {loc.get('precision', 0):.0%}){tag}, so an unhighlighted "
            "sentence is not evidence of anything."
        )

    out.extend(extra or [])
    out.append(GENERIC)
    return out

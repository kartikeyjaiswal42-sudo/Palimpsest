"""The error rates shown under a verdict must be the served detector's own.

This test exists because the application did not do that. ``api/app.py`` selects its
detector, bands and genre gate with ``PALIMPSEST_SUFFIX`` -- default ``_remote``, the 30 B
observer -- and read ``evaluation.json`` for the limitations panel regardless. So the tool
served one detector and published a different one's measured error rates beneath it, with
nothing marking them as another build's numbers.

Nothing caught it: no unit test touched the panel, and ``verify_ui.cjs`` only asserts that
some ESL rate is disclosed, which is just as true of the wrong rate. The hosted build had
already fixed it in its own copy of the renderer, so the two shipped different honesty
panels for the same detector and the discrepancy lived in the gap between them.

Both now render from ``palimpsest.limitations``. These checks fail on the old behaviour:
the first two on the numbers themselves, the last on the two builds disagreeing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from palimpsest.limitations import CARRIED, GENERIC, render

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SERVED_SUFFIX = "_remote"


def _sets(name: str) -> dict:
    path = ARTIFACTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present; run scripts/evaluate.py")
    return json.loads(path.read_text(encoding="utf-8")).get("sets", {})


def _first_percent(statement: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)%", statement)
    assert m, f"no percentage in: {statement}"
    return float(m.group(1))


def test_panel_quotes_the_served_build_not_the_default_file():
    """The headline false-positive rate must be the one the served detector measured."""
    served = _sets(f"evaluation{SERVED_SUFFIX}")
    other = _sets("evaluation")

    served_fpr = (served["esl"]["breakdown"]["liang_toefl"])["documentFPR"]
    other_fpr = (other["esl"]["breakdown"]["liang_toefl"])["documentFPR"]
    if round(served_fpr, 3) == round(other_fpr, 3):
        pytest.skip("the two evaluation runs agree here, so this cannot discriminate")

    esl_line = next(s for s in render(ARTIFACTS, SERVED_SUFFIX) if "TOEFL" in s)
    assert _first_percent(esl_line) == pytest.approx(served_fpr * 100, abs=0.05)
    # The point of the test, stated as its own assertion: not the other build's number.
    assert _first_percent(esl_line) != pytest.approx(other_fpr * 100, abs=0.05)


def test_a_number_from_another_observer_is_never_presented_as_this_one():
    """Carrying a figure over is allowed. Carrying it over silently is the bug."""
    served = _sets(f"evaluation{SERVED_SUFFIX}")
    statements = render(ARTIFACTS, SERVED_SUFFIX)

    # Sets the served evaluation never covered. Their statements may still appear -- dropping
    # them would quietly shorten the disclosure -- but only wearing the label.
    for name, needle in (
        ("unseen_prompting", "prompted to evade detection"),
        ("adversarial", "deliberately composed to imitate"),
    ):
        if name in served:
            continue
        line = next((s for s in statements if needle in s), None)
        if line is None:
            continue
        assert CARRIED.strip() in line, f"unlabelled figure from another observer: {line}"

    # And the converse: a set the served build DID measure must not be labelled as carried.
    esl_line = next(s for s in statements if "TOEFL" in s)
    assert CARRIED.strip() not in esl_line


def test_the_ceiling_is_disclosed_when_it_has_been_measured():
    """A low score is only honest next to the number that says why it is not a clearance."""
    served = _sets(f"evaluation{SERVED_SUFFIX}")
    if served.get("modern_claude", {}).get("documentRecall") is None:
        pytest.skip("frontier recall not measured for the served build")
    statements = render(ARTIFACTS, SERVED_SUFFIX)
    line = next((s for s in statements if "frontier prose" in s), None)
    assert line is not None, "frontier recall measured but not shown to the user"
    assert _first_percent(line) == pytest.approx(
        served["modern_claude"]["documentRecall"] * 100, abs=0.05
    )
    assert "not evidence" in line


def test_the_two_builds_publish_the_same_panel():
    """The hosted bundle and the Python app must not state different rates for one detector.

    ``edge/src/artifacts.js`` is generated, so this compares the shipped bundle against a
    fresh render. A drift here means someone edited one renderer, which is the exact shape
    of the original defect.
    """
    bundle_path = ROOT / "edge" / "src" / "artifacts.js"
    if not bundle_path.exists():
        pytest.skip("hosted bundle not built")
    text = bundle_path.read_text(encoding="utf-8")
    bundle = json.loads(text[text.index("{") : text.rindex("}") + 1])
    shipped = bundle["limitations"]

    local = render(ARTIFACTS, bundle.get("suffix", SERVED_SUFFIX))
    # The hosted build adds one statement the Python build cannot make -- observer
    # repeatability, which only exists where the observer is called live per request.
    extra_only = [s for s in shipped if s not in local]
    assert all("repeatable" in s for s in extra_only), (
        f"hosted build states something the Python build does not: {extra_only}"
    )
    assert [s for s in local if s != GENERIC] == [
        s for s in shipped if s in local and s != GENERIC
    ], "the two builds order or word the shared statements differently"

"""The interface may not manufacture a result it did not measure.

`web/app.js` shipped an offline analyzer. When `POST /api/analyze` could not be reached --
a 404, a non-JSON body, or a dropped connection -- the page fell back to it and rendered a
complete result through the same code that renders a real one: a verdict band, a machine
share, per-sentence probabilities, per-feature evidence bars carrying z-scores and weights,
and per-token ranks. All of it came from a seeded random-number generator. For any text that
was not one of the two bundled example essays the document score was literally

    base = -1.6 + random() * 3.2

so a student pasting their own essay during a network blip could be shown "Likely
machine-written" with a full evidence panel behind it. The only disclosure was the words
"bundled fixture" appended to a metadata chip and to the status line.

That is the precise failure this project argues against everywhere else: `aggregate` refuses
to let an unmeasured span decide the answer, `find_passages` refuses to draw one, the genre
gate refuses to band an out-of-domain essay -- and the interface would invent the whole
thing rather than say the observer was unreachable.

These tests read the shipped interface as text. They are deliberately cheap and structural:
the fabricator is gone, and it must not come back by any name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Both copies. `web/` is the source and `edge/assets/` is the deployed copy that
#: `edge/scripts/sync_web.py` writes; a fabricator restored in either one ships to somebody.
APP_FILES = [ROOT / "web" / "app.js", ROOT / "edge" / "assets" / "app.js"]


def _sources() -> list[tuple[str, str]]:
    return [(p.name if p.parent.name == "web" else f"{p.parent.parent.name}/assets/{p.name}",
             p.read_text(encoding="utf-8"))
            for p in APP_FILES if p.exists()]


def test_the_interface_ships_no_local_analyzer():
    """No function that produces a result without the API."""
    for label, src in _sources():
        for banned in ("analyseLocally", "analyzeLocally", "buildEvidence"):
            assert banned not in src, (
                f"{label} defines or calls {banned!r}. The interface must not be able to "
                f"produce a verdict, a probability or an evidence bar that no observer "
                f"measured."
            )


def test_no_randomness_anywhere_in_the_interface():
    """A detector's interface has no legitimate use for a random number.

    This is the check that would have caught the original defect, because the fabricator's
    tell was not its name -- it was a seeded PRNG feeding numbers the page then presented as
    measurements.
    """
    for label, src in _sources():
        # Strip comments first: the note explaining why the fabricator was removed quotes the
        # expression it used, and a test that trips over its own explanation is noise.
        stripped = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        stripped = re.sub(r"^\s*//.*$", "", stripped, flags=re.M)
        assert "Math.random" not in stripped, (
            f"{label} calls Math.random(). Every number this interface displays must come "
            f"from the analyzer response."
        )
        # The original used a hand-rolled PRNG precisely because Math.random is conspicuous.
        assert not re.search(r"\b0x6D2B79F5\b", stripped, re.I), (
            f"{label} contains the mulberry32 constant, i.e. a hand-rolled PRNG. The offline "
            f"analyzer was built on one; it must not return under another name."
        )


def test_an_unreachable_analyzer_is_reported_rather_than_replaced():
    """The NoBackend path must say nothing was measured, not render something."""
    src = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    catch = src[src.index("function analyse()"):]
    catch = catch[:catch.index("function words(")] if "function words(" in catch else catch

    assert "NoBackend" in catch, "the unreachable-analyzer case is no longer distinguished"
    assert "nothing was measured" in catch, (
        "the unreachable-analyzer branch must tell the reader nothing was measured; it "
        "previously rendered a manufactured result with 'bundled fixture' in a metadata chip"
    )
    # renderAll paints a verdict. It must not be reachable from the failure branch.
    branch = catch[catch.index(".catch("):catch.index("['finally']")]
    assert "renderAll" not in branch, (
        "the failure branch calls renderAll, so a request that produced no result still "
        "paints one"
    )


@pytest.mark.parametrize("phrase", ["bundled fixture", "no observer reached"])
def test_the_old_disclosure_wording_is_gone(phrase):
    """If these strings come back, so has the thing they were failing to disclose."""
    for label, src in _sources():
        stripped = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        assert phrase not in stripped, (
            f"{label} still renders {phrase!r}. That wording only ever existed to label "
            f"fabricated output, and it was not enough of a label."
        )

"""The documentation may not disagree with the artifacts.

This test exists because it caught a real problem. An ablation was measured early, written
into docs/04-failures.md as "removing the length feature halved the harm and cost nothing",
and then the pipeline changed underneath it. The prose went on asserting a number that the
build no longer produced, and nothing complained -- a project whose entire claim is honesty
was shipping a stale figure in its own honesty section.

So every headline number in the README and the docs is checked against the JSON the training
and evaluation scripts write. Retrain, and any claim that no longer holds fails here with the
file and the value that has gone wrong.

The claims are declared, not inferred. Parsing arbitrary prose for "numbers" would be fragile
and would fail for boring reasons; naming each claim explicitly means a failure is always
real. Numbers appearing in prose that this table does not cover are unchecked -- the table is
a floor, not a proof, and adding a headline number without adding it here is a review miss.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))

#: WHICH BUILD THE PROSE IS CHECKED AGAINST, and the whole reason this constant exists.
#:
#: This file used to read `evaluation.json`, `detector.json` and `document_detector.json`
#: unconditionally -- the GPT-2-observer artifacts. But `api/app.py` serves the artifact set
#: named by `SUFFIX`, which defaults to `_remote`, and so does the hosted Worker. So the test
#: written to stop the documentation drifting away from the shipped model was itself pointed
#: at a model nobody runs, and it passed happily while the README's headline table described
#: a different detector from the one behind the button: 0.925 sentence AUROC against a served
#: 0.9576, and a 17.8% TOEFL false-positive rate against a served 10.9% -- the latter printed
#: in the safety warning telling people how often the tool is wrong about a real student.
#:
#: This is the same fault `api/app.py::_limitations` records having fixed in the application
#: ("it read evaluation.json unconditionally while every other artifact on this page was
#: selected by SUFFIX"). The application was fixed and the test was not, which is precisely
#: why nothing caught the drift. Resolved here the same way the server resolves it.
SUFFIX = os.environ.get("PALIMPSEST_SUFFIX") or (
    "_remote" if os.environ.get("PALIMPSEST_OBSERVER", "remote") == "remote" else ""
)


def _artifacts(suffix: str = SUFFIX) -> dict:
    """Every artifact under one namespace so a claim can point anywhere."""
    out = {}
    for name in ("evaluation", "detector", "document_detector", "ablation_length"):
        # ablate_length.py writes one file for the whole experiment, not one per observer.
        path = ARTIFACTS / f"{name}{'' if name == 'ablation_length' else suffix}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _dig(data: dict, path: str):
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


#: The wording ``limitations.py`` already uses to mark a number that was not re-measured on
#: the shipped observer. Any claim declared ``"gpt2"`` below must carry it, so a figure from
#: the research build cannot sit unlabelled in a table describing the deployed one.
GPT2_LABEL = "GPT-2-observer build"

#: (label, regex capturing one number, artifact path, scale, tolerance, build)
#:
#: ``scale`` converts the artifact's units to the units the prose uses -- 100 where the docs
#: write a percentage and the artifact stores a fraction. ``tolerance`` is half a display
#: unit, so a claim rounded to one decimal place passes and a claim that is actually wrong
#: does not.
#:
#: ``build`` is ``"served"`` for a number that must match the artifact set the application
#: actually serves, and ``"gpt2"`` for one that only exists in the GPT-2-observer evaluation
#: -- several held-out sets (the ASAP rewrite comparison, the evade-detection prompt, two of
#: the four modern-generator arms) were never re-run against the remote observer. Those are
#: still worth publishing, but only if the prose says which build produced them, so a
#: ``"gpt2"`` claim is additionally required to carry ``GPT2_LABEL`` on the same line.
CLAIMS: list[tuple[str, str, str, float, float, str]] = [
    (
        "sentence AUROC",
        r"[Ss]entence AUROC,? out-of-fold \| \*\*(\d\.\d+)\*\*",
        "detector.metadata.oofSentenceAuroc", 1.0, 0.0005,
        "served",
    ),
    (
        "document AUROC",
        r"[Dd]ocument AUROC,? out-of-fold \| \*\*(\d\.\d+)\*\*",
        "document_detector.metadata.auroc", 1.0, 0.0005,
        "served",
    ),
    (
        "localisation AUROC",
        r"\*\*AUROC (\d\.\d+)\*\*, seam found within 2 sentences",
        "evaluation.sets.localisation.sentenceAuroc", 1.0, 0.0005,
        "served",
    ),
    (
        "seam within two sentences",
        r"seam found within 2 sentences in \*\*(\d+)%\*\*",
        "evaluation.sets.localisation.seam.withinTwoSentences", 100.0, 0.5,
        "served",
    ),
    (
        "ESL document FPR",
        r"[Ff]alse positives on essays by English-language learners \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.esl.documentFPR", 100.0, 0.05,
        "served",
    ),
    (
        "domain-shift document FPR",
        r"[Ff]alse positives on out-of-domain human essays \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.domain_shift.documentFPR", 100.0, 0.05,
        "served",
    ),
    (
        "TOEFL document FPR",
        r"TOEFL essays by non-native writers, wrongly flagged \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.esl.breakdown.liang_toefl.documentFPR", 100.0, 0.05,
        "served",
    ),
    (
        "prompt-engineered recall",
        r"prompted to evade detection\*\* \| only \*\*(\d+\.\d+)%\*\* of essays caught",
        "evaluation.sets.unseen_prompting.documentRecall", 100.0, 0.05,
        "gpt2",
    ),
    (
        "ASAP originals flagged",
        r"The 88 original student essays \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.ablation.ASAP essays, GPT-simplified.originalFlagged", 100.0, 0.05,
        "gpt2",
    ),
    (
        "ASAP rewrites flagged",
        r"The same 88 essays, rewritten by a model \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.ablation.ASAP essays, GPT-simplified.rewrittenFlagged", 100.0, 0.05,
        "gpt2",
    ),
    (
        # This one is here because it went wrong. The README's safety warning quoted 22%,
        # correct when it was written and stale the moment the operating point was
        # recalibrated -- while the headline table 100 lines above it was updated, because
        # the table was covered here and the warning was not. A tool whose entire argument is
        # honesty about its error rate had the error rate wrong in the sentence telling
        # people not to trust it, and it understated the harm.
        "TOEFL FPR in the safety warning",
        r"An? \*\*(\d+\.\d+)%\*\*\s*\n?false-positive rate on the TOEFL essays",
        "evaluation.sets.esl.breakdown.liang_toefl.documentFPR", 100.0, 0.05,
        "served",
    ),
    # The modern-generator results. These are why the detector was refitted, and the GRADIENT
    # across them is the finding -- so all four are pinned, not only the flattering one. If a
    # later retrain lifts `modern_holdout` while `modern_unseen_family` collapses, that is
    # generator memorisation dressed up as progress, and this is where it surfaces.
    (
        "modern recall, generator in training",
        r"in the training pool, these essays are not \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_holdout.documentRecall", 100.0, 0.05,
        "served",
    ),
    (
        "modern recall, topic control",
        r"no topic steering\*\* \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_control.documentRecall", 100.0, 0.05,
        "gpt2",
    ),
    (
        "modern recall, unseen checkpoint",
        r"a checkpoint withheld from training entirely \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_unseen.documentRecall", 100.0, 0.05,
        "gpt2",
    ),
    (
        "modern recall, unseen family",
        r"a different model family, withheld entirely \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_unseen_family.documentRecall", 100.0, 0.05,
        "served",
    ),
]


@pytest.mark.parametrize("label,pattern,path,scale,tol,build", CLAIMS,
                         ids=[c[0] for c in CLAIMS])
def test_documented_number_matches_artifact(label, pattern, path, scale, tol, build):
    suffix = "" if build == "gpt2" else SUFFIX
    expected = _dig(_artifacts(suffix), path)
    if expected is None:
        if build == "served":
            # Not a skip. A headline claim declared as describing the served build, whose
            # measurement is missing from the served build's evaluation, is the drift this
            # file exists to catch -- skipping would hide it exactly as it was hidden before.
            pytest.fail(
                f"{label!r} is declared as a claim about the served build, but "
                f"artifacts/evaluation{SUFFIX}.json has no {path}. Either re-run "
                f"scripts/evaluate.py --suffix {SUFFIX or '(none)'}, or declare the claim "
                f'"gpt2" and label it in the prose.'
            )
        pytest.skip(f"{path} not present -- run scripts/train.py and scripts/evaluate.py")

    found = [
        (doc.name, m, float(m.group(1)))
        for doc in DOCS
        for m in re.finditer(pattern, doc.read_text(encoding="utf-8"))
    ]
    assert found, f"no documented value for {label!r} matched {pattern!r} in any doc"

    target = expected * scale
    for doc_name, match, claimed in found:
        assert abs(claimed - target) <= tol, (
            f"{doc_name} claims {label} = {claimed}, but artifacts/"
            f"{'evaluation' if path.startswith('evaluation') else path.split('.')[0]}"
            f"{suffix}.json says {target:.4f}. Re-run the evaluation and update the prose, "
            f"or the docs are lying."
        )
        if build == "gpt2":
            # The number is real but was measured on a build that is not deployed. It may be
            # published; it may not be published silently, in a table a reader will take to
            # describe the thing behind the button.
            line = match.string[match.string.rfind("\n", 0, match.start()) + 1:
                                match.string.find("\n", match.end())]
            assert GPT2_LABEL in line, (
                f"{doc_name} publishes {label} = {claimed}, which exists only in the "
                f"GPT-2-observer evaluation, without saying so. The served build "
                f"(artifacts/evaluation{SUFFIX}.json) never measured this set, so the line "
                f"must carry {GPT2_LABEL!r} the way limitations.py already labels it."
            )


def _failure_contribution(sentence_fragment: str, feature_label: str) -> float | None:
    """The contribution a named feature made to a named sentence, from failures.json."""
    path = ARTIFACTS / "failures.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for bucket in data.values():
        for doc in bucket:
            for sent in doc.get("topSentences", []):
                if sentence_fragment not in sent.get("text", ""):
                    continue
                for ev in sent.get("evidence", []):
                    if ev.get("label", "").lower() == feature_label.lower():
                        return float(ev["contribution"])
    return None


#: The per-sentence evidence quoted in the failure write-ups. These went stale once already:
#: find_failures.py had not been re-run after the final retrain, so the docs quoted evidence
#: values from a superseded model while describing the current one.
FAILURE_CLAIMS = [
    ("vocabulary richness on the one-sentence ELLIPSE essay",
     "In respond to everything i do think",
     "Vocabulary richness", r"vocabulary richness \*\*\+(\d+\.\d+)\*\*"),
    ("curvature-vs-baseline on the TOEFL leadership sentence",
     "he should bring a strong feeling of authority",
     "Curvature vs the author's baseline",
     r"\*\*curvature vs the author's baseline \+(\d+\.\d+)\*\*"),
]


@pytest.mark.parametrize("label,fragment,feature,pattern", FAILURE_CLAIMS,
                         ids=[c[0] for c in FAILURE_CLAIMS])
def test_quoted_failure_evidence_matches_the_artifact(label, fragment, feature, pattern):
    expected = _failure_contribution(fragment, feature)
    if expected is None:
        pytest.skip("run scripts/find_failures.py")

    found = [
        (doc.name, float(m.group(1)))
        for doc in DOCS
        for m in re.finditer(pattern, doc.read_text(encoding="utf-8"), re.IGNORECASE)
    ]
    assert found, f"no documented value for {label!r} matched {pattern!r}"
    for doc_name, claimed in found:
        assert abs(claimed - expected) <= 0.005, (
            f"{doc_name} quotes {label} as {claimed:+.2f}, but artifacts/failures.json says "
            f"{expected:+.3f}. Re-run scripts/find_failures.py and update the write-up."
        )


def test_calibration_table_in_the_docs_matches_the_run():
    """The reliability table in docs/03-evaluation.md must be the one training produced.

    It was not. The table was copied out of a run that predated the operating-point
    recalibration and then sat unchanged through it: it listed five bands summing to 3,989
    sentences when the training pool holds 4,895, omitted the 0.3-0.5 bands entirely, and the
    prose beneath it called the 0.5-0.7 band "over-confident on 41 sentences" when that band
    holds 186. Every headline number in the file was right, because every headline number was
    checked here. This one was a hand-copied table, so nothing noticed for three commits.

    train.py now writes the table into the detector artifact and this reads it back.
    """
    det = _artifacts().get("detector")
    if det is None or "calibration" not in det.get("metadata", {}):
        pytest.skip("run scripts/train.py to write metadata.calibration")
    expected = det["metadata"]["calibration"]

    text = (ROOT / "docs" / "03-evaluation.md").read_text(encoding="utf-8")
    # Rows look like: | 0.0–0.1 | 3,691 | 0.013 | 0.016 |   (en dash, thousands separator)
    rows = re.findall(
        r"^\|\s*(\d\.\d)[–-](\d\.\d)\s*\|\s*([\d,]+)\s*\|\s*(\d\.\d+)\s*\|\s*(\d\.\d+)\s*\|$",
        text, re.MULTILINE,
    )
    assert rows, "no calibration table rows found in docs/03-evaluation.md"
    assert len(rows) == len(expected), (
        f"docs/03-evaluation.md lists {len(rows)} calibration bands, the run produced "
        f"{len(expected)}. Bands with fewer than 20 sentences are dropped, so a changed "
        f"count means the table was not regenerated."
    )

    for (lo, hi, n, pred, actual), want in zip(rows, expected, strict=True):
        band = f"{lo}-{hi}"
        assert band == want["band"], f"band {band} out of order; run produced {want['band']}"
        assert int(n.replace(",", "")) == want["n"], (
            f"docs say band {band} holds {n} sentences, the run says {want['n']}"
        )
        assert abs(float(pred) - want["meanPredicted"]) <= 0.0005, (
            f"docs say band {band} predicted {pred}, the run says {want['meanPredicted']}"
        )
        assert abs(float(actual) - want["actual"]) <= 0.0005, (
            f"docs say band {band} was actually {actual}, the run says {want['actual']}"
        )

    total = sum(r["n"] for r in expected)
    assert f"{total:,}" in text, (
        f"docs/03-evaluation.md should state that the bands account for {total:,} sentences"
    )


def test_length_ablation_conclusion_is_not_overstated():
    """The docs must not claim the length ablation helped more than it did.

    The original write-up said removing the document-length feature "halved the harm and cost
    nothing measurable". The re-run says the TOEFL improvement is inside noise and that the
    removal costs in-domain AUROC. If someone restores the flattering phrasing, or the
    experiment starts producing a significant result, this should be noticed deliberately.
    """
    path = ARTIFACTS / "ablation_length.json"
    if not path.exists():
        pytest.skip("run scripts/ablate_length.py")
    arms = json.loads(path.read_text(encoding="utf-8"))
    with_len = arms["with log_sentences"]
    without = arms["shipped (without)"]

    # The claim the docs actually make: removal costs in-domain AUROC and does not reduce
    # aggregate ESL false positives. If either flips, the prose needs rewriting.
    assert without["inDomainDocumentAuroc"] < with_len["inDomainDocumentAuroc"], (
        "Removing the length feature no longer costs in-domain AUROC. "
        "docs/04-failures.md #6 and docs/06-decisions.md #6 say it does."
    )
    # The ESL row is deliberately NOT asserted in either direction. It has changed sign twice
    # during this project -- keeping the feature looked better before the modern corpus went
    # in, worse immediately after, and better again once run-on spans stopped being scored --
    # and the swing each time was 2 to 3 documents out of 395. Pinning a direction here would
    # be pinning noise, and would fail on a rebuild for a reason that is not a defect. What
    # the docs claim about that row is that it is within noise, so the assertion is on the
    # size of the difference rather than its sign.
    assert abs(without["eslDocumentFPR"] - with_len["eslDocumentFPR"]) < 0.03, (
        f"The ESL arms have separated: {without['eslDocumentFPR']:.4f} without vs "
        f"{with_len['eslDocumentFPR']:.4f} with. docs/04-failures.md #6 describes this "
        "difference as noise. If it is now a real effect, that write-up needs rewriting."
    )

    # The TOEFL row is the one the docs make a claim about, and the claim is that the
    # measurements run AGAINST the decision we made.
    assert without["toeflDocumentFPR"] >= with_len["toeflDocumentFPR"], (
        "Removing the length feature now improves TOEFL FPR. docs/04-failures.md #6 says the "
        "measurements disagree with the removal and that we did it on principle anyway. If "
        "the evidence has turned in our favour, say so instead of leaving the apology in."
    )

    failures = ROOT / "docs" / "04-failures.md"
    text = failures.read_text(encoding="utf-8")
    assert "halved the harm and cost nothing" not in text, (
        "The superseded claim is back in docs/04-failures.md; the measurement does not "
        "support it."
    )


def test_the_readme_states_the_real_number_of_tests(request):
    """The README quoted two different test counts, in two places, both wrong.

    "pytest # 194 tests" in the run instructions and "tests/ 127 tests" in the repository
    map -- a file whose argument is that a number describing something belongs to the thing
    it describes. Counted from the collected session rather than typed, for the same reason
    `renderLimitations` counts the list it just rendered.
    """
    collected = request.session.testscollected
    if collected < 150:
        pytest.skip("only part of the suite was collected; nothing to compare a total against")

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = {int(m.group(1)) for m in re.finditer(r"\b(\d{2,4}) tests\b", text)}
    assert claimed, "the README no longer states how many tests there are"
    assert claimed == {collected}, (
        f"README claims {sorted(claimed)} tests; the suite collects {collected}. If more than "
        f"one number appears, the README is disagreeing with itself as well as with pytest."
    )

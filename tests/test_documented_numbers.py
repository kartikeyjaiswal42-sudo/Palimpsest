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
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))


def _artifacts() -> dict:
    """Every artifact under one namespace so a claim can point anywhere."""
    out = {}
    for name in ("evaluation", "detector", "document_detector", "ablation_length"):
        path = ARTIFACTS / f"{name}.json"
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


#: (label, regex capturing one number, artifact path, scale, tolerance)
#:
#: ``scale`` converts the artifact's units to the units the prose uses -- 100 where the docs
#: write a percentage and the artifact stores a fraction. ``tolerance`` is half a display
#: unit, so a claim rounded to one decimal place passes and a claim that is actually wrong
#: does not.
CLAIMS: list[tuple[str, str, str, float, float]] = [
    (
        "sentence AUROC",
        r"[Ss]entence AUROC,? out-of-fold \| \*\*(\d\.\d+)\*\*",
        "detector.metadata.oofSentenceAuroc", 1.0, 0.0005,
    ),
    (
        "document AUROC",
        r"[Dd]ocument AUROC,? out-of-fold \| \*\*(\d\.\d+)\*\*",
        "document_detector.metadata.auroc", 1.0, 0.0005,
    ),
    (
        "localisation AUROC",
        r"\*\*AUROC (\d\.\d+)\*\*, seam found within 2 sentences",
        "evaluation.sets.localisation.sentenceAuroc", 1.0, 0.0005,
    ),
    (
        "seam within two sentences",
        r"seam found within 2 sentences in \*\*(\d+)%\*\*",
        "evaluation.sets.localisation.seam.withinTwoSentences", 100.0, 0.5,
    ),
    (
        "ESL document FPR",
        r"[Ff]alse positives on essays by English-language learners \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.esl.documentFPR", 100.0, 0.05,
    ),
    (
        "domain-shift document FPR",
        r"[Ff]alse positives on out-of-domain human essays \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.domain_shift.documentFPR", 100.0, 0.05,
    ),
    (
        "TOEFL document FPR",
        r"TOEFL essays by non-native writers, wrongly flagged \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.esl.breakdown.liang_toefl.documentFPR", 100.0, 0.05,
    ),
    (
        "prompt-engineered recall",
        r"prompted to evade detection\*\* \| only \*\*(\d+\.\d+)%\*\* of essays caught",
        "evaluation.sets.unseen_prompting.documentRecall", 100.0, 0.05,
    ),
    (
        "ASAP originals flagged",
        r"The 88 original student essays \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.ablation.ASAP essays, GPT-simplified.originalFlagged", 100.0, 0.05,
    ),
    (
        "ASAP rewrites flagged",
        r"The same 88 essays, rewritten by a model \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.ablation.ASAP essays, GPT-simplified.rewrittenFlagged", 100.0, 0.05,
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
    ),
    # The modern-generator results. These are why the detector was refitted, and the GRADIENT
    # across them is the finding -- so all four are pinned, not only the flattering one. If a
    # later retrain lifts `modern_holdout` while `modern_unseen_family` collapses, that is
    # generator memorisation dressed up as progress, and this is where it surfaces.
    (
        "modern recall, generator in training",
        r"in the training pool, these essays are not \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_holdout.documentRecall", 100.0, 0.05,
    ),
    (
        "modern recall, topic control",
        r"no topic steering\*\* \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_control.documentRecall", 100.0, 0.05,
    ),
    (
        "modern recall, unseen checkpoint",
        r"a checkpoint withheld from training entirely \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_unseen.documentRecall", 100.0, 0.05,
    ),
    (
        "modern recall, unseen family",
        r"a different model family, withheld entirely \(n=\d+\) \| \*\*(\d+\.\d+)%\*\*",
        "evaluation.sets.modern_unseen_family.documentRecall", 100.0, 0.05,
    ),
]


@pytest.mark.parametrize("label,pattern,path,scale,tol", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_documented_number_matches_artifact(label, pattern, path, scale, tol):
    artifacts = _artifacts()
    expected = _dig(artifacts, path)
    if expected is None:
        pytest.skip(f"{path} not present -- run scripts/train.py and scripts/evaluate.py")

    found = [
        (doc.name, float(m.group(1)))
        for doc in DOCS
        for m in re.finditer(pattern, doc.read_text(encoding="utf-8"))
    ]
    assert found, f"no documented value for {label!r} matched {pattern!r} in any doc"

    target = expected * scale
    for doc_name, claimed in found:
        assert abs(claimed - target) <= tol, (
            f"{doc_name} claims {label} = {claimed}, but {path} says {target:.4f}. "
            f"Re-run the evaluation and update the prose, or the docs are lying."
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

"""The properties that make the Claude corpus a measurement rather than a demo.

Every claim in docs/08-cross-vendor.md rests on the corpus having four properties, and each
of them is the kind that decays silently. A subject can leak across the train/held-out
boundary when someone adds essays; a length gap can open up when a generator's habits change;
the plan can stop being reproducible the moment an unseeded shuffle creeps in. None of those
announce themselves -- they just quietly turn a generalisation number into a memorisation
number, and the docs go on claiming otherwise.

So they are tests. They run without a network, without a model download and without the
corpus being complete, because the point is to catch the regression on the commit that
introduces it rather than after the next 500-essay run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "data" / "generated" / "plan" / "plan.json"
TRAIN = ROOT / "data" / "generated" / "claude_modern.jsonl"
HELDOUT = ROOT / "data" / "generated" / "claude_modern_heldout.jsonl"

pytestmark = pytest.mark.skipif(not PLAN.exists(), reason="corpus plan not built")


def plan() -> list[dict]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- the plan


def test_plan_is_reproducible():
    """Re-running the planner must reproduce the plan exactly.

    The provenance recorded in every essay -- which subject, which style, which split -- is
    only meaningful if the plan that assigned it can be regenerated. An unseeded shuffle here
    would make every meta field in the corpus unverifiable after the fact.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "plan_claude_corpus", ROOT / "scripts" / "plan_claude_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rebuilt = mod.build(len(plan()))
    on_disk = plan()
    assert len(rebuilt) == len(on_disk)
    for a, b in zip(rebuilt, on_disk, strict=True):
        for field in ("id", "topic", "frame", "style", "model", "split", "prompt_id",
                      "target_words", "batch"):
            assert a[field] == b[field], f"{field} drifted for {a['id']}"


def test_no_subject_crosses_the_split():
    """The property the held-out number depends on entirely.

    If one subject appears in both splits, a held-out essay has a training essay about the
    same kitchen, the same injury, the same shop -- and the detector can score it by
    recognising the subject. The result would still be reported as generalisation.
    """
    sides: dict[str, set[str]] = {}
    for row in plan():
        sides.setdefault(row["topic"], set()).add(row["split"])
    straddling = {t for t, s in sides.items() if len(s) > 1}
    assert not straddling, f"{len(straddling)} subjects appear in both splits: {sorted(straddling)[:3]}"


def test_every_assignment_is_distinct():
    rows = plan()
    assert len({r["id"] for r in rows}) == len(rows), "duplicate essay ids"
    pairs = Counter((r["topic"], r["frame"]) for r in rows)
    assert not [k for k, v in pairs.items() if v > 1], "a subject/frame pair was assigned twice"


def test_unsteered_essays_are_the_majority():
    """`plain` carries no instruction about voice at all.

    docs/04-failures.md #4 is the corpus that measured its own prompt. The defence is that
    most of this one is not steered, so if the majority slice were ever to become a styled
    variant, the per-style table in docs/08-cross-vendor.md would stop meaning what it says.
    """
    styles = Counter(r["style"] for r in plan())
    assert styles.most_common(1)[0][0] == "plain"
    assert styles["plain"] / sum(styles.values()) > 0.40


def test_every_commissioned_essay_clears_the_400_word_floor():
    assert min(r["target_words"] for r in plan()) >= 400


def test_all_four_generators_are_represented_evenly():
    models = Counter(r["model"] for r in plan())
    assert len(models) == 4, f"expected four checkpoints, got {sorted(models)}"
    assert max(models.values()) - min(models.values()) <= 1


def test_planned_lengths_predict_the_human_distribution():
    """Length must not be a shortcut to the answer.

    docs/06-decisions.md #6: when document length was visible, it took a weight of -3.09 and
    the model's strongest belief became "short means machine" -- learned purely from the
    GPT-3.5 essays being shorter. Length was removed from the DOCUMENT model, but the
    sentence features never got that surgery, so a machine corpus with its own length
    silhouette would put the artifact back somewhere nobody is checking.

    The planner asks for less than it wants, because generators overshoot by a measured
    factor. This asserts the corrected ask lands on the human corpus.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "plan_claude_corpus", ROOT / "scripts" / "plan_claude_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    human: list[int] = []
    for name in ("liang_college_human", "jhu"):
        path = ROOT / "data" / "raw" / f"{name}.jsonl"
        if path.exists():
            human += [json.loads(l)["n_words"] for l in path.open(encoding="utf-8")]
    if not human:
        pytest.skip("human corpus not fetched")

    predicted = [r["target_words"] * mod.OVERSHOOT for r in plan()]
    gap = sum(predicted) / len(predicted) - sum(human) / len(human)
    assert abs(gap) < 40, (
        f"the machine corpus is predicted to sit {gap:+.0f} words from the human mean; "
        "at that separation the classifier can reach the right answer by measuring length")


# --------------------------------------------------------------------------- the corpus


@pytest.mark.skipif(not TRAIN.exists() and not HELDOUT.exists(),
                    reason="no assembled corpus yet")
def test_assembled_essays_are_prose_not_transcripts():
    """What a failed generation looks like on disk.

    A worker that refuses, narrates, or emits a markdown title still writes a file. If those
    reach the machine class the detector learns that machine text is apologetic and
    bulleted, and the recall figure describes our pipeline instead of the model.
    """
    import re

    bad = re.compile(r"(^\s{0,3}#{1,6}\s)|(^\s{0,3}[-*+]\s)|(\*\*.+?\*\*)", re.M)
    lead = re.compile(r"^\s*(here(?:'s| is)\b|sure[,!]|certainly[,!]|as an ai\b|i can'?t\b)", re.I)
    for doc in read(TRAIN) + read(HELDOUT):
        assert not bad.search(doc["text"]), f"{doc['id']} contains markdown"
        assert not lead.match(doc["text"]), f"{doc['id']} opens with a preamble"
        assert doc["n_words"] >= 400, f"{doc['id']} is {doc['n_words']} words"
        assert doc["authorship"] == "machine"


@pytest.mark.skipif(not TRAIN.exists() or not HELDOUT.exists(),
                    reason="both splits needed")
def test_no_essay_appears_in_both_splits():
    train = {d["sha256"] for d in read(TRAIN)}
    held = {d["sha256"] for d in read(HELDOUT)}
    assert not (train & held), "the same essay text is in both splits"

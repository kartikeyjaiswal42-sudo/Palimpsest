"""HTTP API and static host for the Palimpsest interface.

    uvicorn palimpsest.api.app:app --reload

One endpoint does the work. It returns the document verdict, every sentence with its
probability, and for each sentence the feature contributions that produced that probability
-- the same numbers the classifier summed, not a paraphrase of them.

There is no generative model behind this API. ``/api/health`` reports what is loaded, and a
test asserts that nothing in the scoring path can call one.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..analyze import Analyzer
from ..features.registry import FEATURES, GROUPS
from ..limitations import render as render_limitations

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"
WEB = ROOT / "web"

#: Essays are ~650 words. This is generous, and it exists so a paste of a novel cannot tie
#: up the single-process scorer for minutes.
MAX_CHARS = 40_000

app = FastAPI(
    title="Palimpsest",
    description="A white-box AI-text detector for college admissions essays.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The essay to analyse.")
    include_tokens: bool = Field(True, description="Include per-token observer statistics.")


#: Which observer serves requests. "remote" uses the 30 B model on Workers AI and loads NO
#: local weights; "gpt2" is the original in-process observer. The detector artifact must
#: match, so the two move together and are never mixed.
OBSERVER = os.environ.get("PALIMPSEST_OBSERVER", "remote")
# Artifact set to serve. Defaults to the observer's own suffix; overridable so an
# ablation detector can be driven through the REAL application rather than a script.
SUFFIX = os.environ.get("PALIMPSEST_SUFFIX") or ("_remote" if OBSERVER == "remote" else "")


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    """Load the fitted artifacts once.

    With the remote observer nothing is downloaded and no model occupies memory; the first
    request pays one network round-trip instead of a weight load.
    """
    started = time.perf_counter()
    analyzer = Analyzer.from_artifacts(ARTIFACTS, observer=OBSERVER, suffix=SUFFIX)
    log.info("analyzer ready in %.1fs (observer=%s)", time.perf_counter() - started, OBSERVER)
    return analyzer


@app.get("/api/health")
def health() -> dict:
    """What is loaded, and what this service will and will not do."""
    try:
        analyzer = get_analyzer()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"artifacts missing: {exc}. Run scripts/train.py.") from exc
    return {
        "status": "ok",
        "observer": analyzer.scorer.model_name,
        "device": analyzer.scorer.device,
        "corpusReferenceLoaded": analyzer.reference is not None,
        "flagThreshold": round(analyzer.detector.flag_threshold, 4),
        "trainedOn": analyzer.detector.metadata,
        # Stated in the payload because it is the central design claim of the project.
        "usesGenerativeModel": False,
        # Whether the essay leaves this machine. This MUST track the observer: the claim
        # "no text is sent anywhere" was true while the observer was GPT-2 in-process and
        # became false the moment the default moved to Workers AI. A privacy claim that is
        # correct only for a configuration nobody is running is worse than no claim.
        "textLeavesMachine": OBSERVER == "remote",
        "note": (
            "The language model is read for token probabilities only -- a single forward "
            "pass over the essay, whose logits are then arithmetic. It is never prompted "
            "and never asked for a verdict."
            + (
                " The observer runs on Cloudflare Workers AI, so THE ESSAY TEXT IS SENT "
                "to that service to be scored. It is not stored by this application, but "
                "it does leave this machine. Set PALIMPSEST_OBSERVER=gpt2 to score "
                "entirely locally instead."
                if OBSERVER == "remote"
                else " The observer runs in this process, so no text is sent anywhere."
            )
        ),
    }


@app.get("/api/features")
def features() -> dict:
    """The feature catalogue, so the interface can explain any feature it displays."""
    return {
        "groups": GROUPS,
        "features": [
            {
                "name": f.name,
                "group": f.group,
                "label": f.label,
                "description": f.description,
                "unit": f.unit,
                "expectedDirection": f.expected_direction,
            }
            for f in FEATURES
        ],
    }


@lru_cache(maxsize=1)
def get_bands() -> dict:
    """Two thresholds defining machine / insufficient-evidence / human. See fit_bands.py."""
    path = ARTIFACTS / f"bands{SUFFIX}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # Without a fitted band file the honest default is to answer nothing: a missing
    # calibration is not a licence to guess.
    return {"tHuman": 0.0, "tMachine": 1.01, "unfitted": True}


@lru_cache(maxsize=1)
def get_genre_gate():
    """The pre-filter that refuses writing this detector was not fitted for."""
    from ..detect.genre import GenreGate

    path = ARTIFACTS / f"genre_gate{SUFFIX}.json"
    return GenreGate.load(path) if path.exists() else GenreGate()


def _band(score: float) -> dict:
    """Map a document score to one of three bands.

    The middle band is the point of the exercise. Both live failures that prompted this
    work -- a Gemini essay at 35% and an Opus statement of purpose at 0% -- were machine
    written, and a two-word interface reported them as human. "This tool cannot tell you
    anything about this document" is a correct answer there; "0% machine" is a false one.

    In particular a LOW score is not evidence of a human author. Frontier prose scores like
    human prose (docs/09-frontier-ceiling.md: Opus document recall 4.4%), so the bottom band
    is deliberately narrow and is worded as "no evidence found", never "human".
    """
    b = get_bands()
    if b.get("unfitted"):
        return {"band": "insufficient_evidence",
                "bandLabel": "Not calibrated",
                "bandDetail": "No operating point has been fitted, so no verdict is offered.",
                "canExonerate": False}
    if score >= b["tMachine"]:
        return {
            "band": "likely_machine",
            "bandLabel": "Likely machine-written",
            "bandDetail": (
                f"Above the threshold calibrated so that at most "
                f"{b['fprBudget']:.0%} of at-risk human essays are flagged "
                f"(observed {b['observedFpr']:.1%} on {b['nHuman']} held-out documents)."),
            "canExonerate": False,
        }
    if score <= b["tHuman"]:
        return {
            "band": "no_evidence",
            "bandLabel": "No evidence of machine writing",
            "bandDetail": (
                "This is not a finding that a person wrote it. It means the signals this "
                "tool measures are absent, which is also true of capable models -- "
                f"{b['observedMiss']:.0%} of known machine essays land here."),
            "canExonerate": False,
        }
    return {
        "band": "insufficient_evidence",
        "bandLabel": "Insufficient evidence",
        "bandDetail": (
            "This document falls between the two calibrated thresholds. The tool declines "
            "to answer rather than guess; it abstains on "
            f"{b['abstainHuman']:.0%} of human and {b['abstainMachine']:.0%} of machine "
            "essays by design."),
        "canExonerate": False,
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    text = request.text or ""
    if not text.strip():
        raise HTTPException(400, "text is empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"text exceeds {MAX_CHARS} characters")

    analyzer = get_analyzer()
    result = analyzer.analyze(text)
    payload = result.to_dict(include_tokens=request.include_tokens)

    # `Analyzer` loads the fitted document model itself and `result.verdict` already carries
    # its output. This block used to re-derive the verdict here, on the stated grounds that
    # "Analyzer.analyze uses the unfitted fallback so the library works without artifacts" --
    # true at the time, and the reason a real defect went unnoticed for so long. Every caller
    # that reported a number patched around it privately (this file, the parity exporter),
    # so the broken path was the only one nobody exercised, and `Analyzer.analyze()` returned
    # `max_p` under a field named `any_machine_probability`. On one essay that read 70.9%
    # where the fitted model reads 14.9%. A repair applied at each call site hides the fault
    # instead of fixing it; the load now happens once, in `from_artifacts`.
    v = result.verdict
    # The genre gate runs AFTER scoring and BEFORE the verdict is worded. Scoring anyway
    # costs nothing and keeps the per-sentence evidence available for a curious reader, but
    # an out-of-domain document must never be handed a band: the thresholds were calibrated
    # on admissions essays and mean nothing here. This is the fix for a measured failure --
    # a real school student's essay called machine-written at 97% confidence.
    from ..detect.genre import document_genre_features

    gate = get_genre_gate()
    # features_for is deterministic and the observer caches, so this re-derivation is cheap
    # and keeps the gate reading exactly the numbers the detector read.
    _, doc_feats, _ = analyzer.features_for(text)
    gfeat = document_genre_features(doc_feats)
    p_in = gate.probability(gfeat)

    if not gate.in_domain(gfeat):
        payload["verdict"].update({
            "band": "out_of_scope",
            "bandLabel": "Outside this tool's scope",
            "bandDetail": (
                "This does not read as a college admissions personal statement, which is "
                "the only kind of writing this detector was fitted and calibrated on. It "
                "refuses rather than guessing: pointed at other genres it fails "
                "confidently, not gracefully -- it once called a school student's "
                "argumentative essay machine-written at 97% confidence. Sentence scores "
                "are still shown as evidence, but no verdict is offered."),
            "canExonerate": False,
            "inDomainProbability": round(p_in, 4),
        })
    else:
        payload["verdict"].update(_band(v.any_machine_probability))
        payload["verdict"]["inDomainProbability"] = round(p_in, 4)
    payload["flagThreshold"] = round(analyzer.detector.flag_threshold, 4)
    payload["limitations"] = _limitations()
    return payload


@lru_cache(maxsize=1)
def _limitations() -> list[str]:
    """Shipped with every response. A detector that hides its error rates is not honest.

    Rendered from the measurements rather than written here as prose. An earlier version
    hard-coded the percentages, the model was retrained, and the interface went on
    confidently displaying the previous run's error rates.

    It then failed the same way a second time for a different reason, and this call site is
    where: it read ``evaluation.json`` unconditionally while every other artifact on this
    page -- detector, bands, genre gate -- was selected by ``SUFFIX``. With the default
    remote observer the application served the ``_remote`` detector and published the GPT-2
    build's error rates under it, unlabelled. Passing ``SUFFIX`` is the fix; sharing
    ``palimpsest.limitations`` with the hosted build is what stops the two drifting apart
    again, which is how the discrepancy survived as long as it did.
    """
    return render_limitations(ARTIFACTS, SUFFIX)


#: Application code is served no-cache.
#:
FAILURES = ARTIFACTS / "confident_failures.json"


class ExplanationRequest(BaseModel):
    doc_id: str = Field(..., description="Which failure the explanation belongs to.")
    text: str = Field(..., max_length=8_000, description="Why the arithmetic failed.")


@app.get("/api/failures")
def failures() -> dict:
    """The detector's worst held-out mistakes, ranked by confident wrongness.

    Served from ``artifacts/confident_failures.json``, written by
    ``scripts/confident_failures.py``. Rendered rather than described, for the reason
    PROJECT.md §2 records: the limitations panel once published a 17.8% false-positive rate
    where the served build measured 10.9%, because the number lived in prose instead of in
    the artifact the run produced.

    Returns 404 with the command to run rather than an empty list, so a missing artifact
    cannot be mistaken for a detector that has no failures.
    """
    if not FAILURES.exists():
        raise HTTPException(
            status_code=404,
            detail="No failure analysis has been run. "
                   "Generate it with: python scripts/confident_failures.py",
        )
    return json.loads(FAILURES.read_text(encoding="utf-8"))


@app.post("/api/failures/explanation")
def save_explanation(req: ExplanationRequest) -> dict:
    """Persist the human account of why a given failure happened.

    This is the one field in the artifact a script must not write. It is stored back into
    the artifact itself rather than a side file, so ``confident_failures.py`` -- which
    already carries explanations across regeneration -- keeps them when the corpus changes.
    """
    if not FAILURES.exists():
        raise HTTPException(status_code=404, detail="No failure analysis to annotate.")
    payload = json.loads(FAILURES.read_text(encoding="utf-8"))
    for entry in payload.get("failures", []):
        if entry.get("docId") == req.doc_id:
            entry["humanExplanation"] = req.text
            # Written via a temporary file and replaced: a half-written artifact would be
            # read as a truncated failure list by the next GET.
            tmp = FAILURES.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(FAILURES)
            return {"ok": True, "docId": req.doc_id, "chars": len(req.text)}
    raise HTTPException(status_code=404, detail=f"No failure with docId {req.doc_id!r}")


#: Not a micro-optimisation in reverse -- a correctness measure. A CSS fix was made, served
#: correctly by this process, and still absent in the browser because the old stylesheet was
#: cached; the screenshot showed an unstyled element and the file on disk showed the rule,
#: which is a debugging trap that costs an hour and teaches nothing. The page is a handful of
#: kilobytes served from localhost, so the cache buys nothing and can only mislead.
_NO_STORE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


class _NoCacheStatic(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers.update(_NO_STORE)
        return response


if WEB.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html", headers=_NO_STORE)

    app.mount("/", _NoCacheStatic(directory=WEB), name="web")

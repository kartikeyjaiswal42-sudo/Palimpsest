"""HTTP API and static host for the Palimpsest interface.

    uvicorn palimpsest.api.app:app --reload

One endpoint does the work. It returns the document verdict, every sentence with its
probability, and for each sentence the feature contributions that produced that probability
-- the same numbers the classifier summed, not a paraphrase of them.

There is no generative model behind this API. ``/api/health`` reports what is loaded, and a
test asserts that nothing in the scoring path can call one.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..analyze import Analyzer
from ..detect.document import DocumentDetector
from ..features.registry import FEATURES, GROUPS

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


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    """Load the fitted artifacts once. First call pulls GPT-2 weights if not cached."""
    started = time.perf_counter()
    analyzer = Analyzer.from_artifacts(ARTIFACTS)
    log.info("analyzer ready in %.1fs", time.perf_counter() - started)
    return analyzer


@lru_cache(maxsize=1)
def get_document_model() -> DocumentDetector:
    path = ARTIFACTS / "document_detector.json"
    return DocumentDetector.load(path) if path.exists() else DocumentDetector()


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
        "note": (
            "The language model is read for token probabilities only. It is never asked "
            "for a verdict, and no text is sent anywhere."
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

    # Recompute the document verdict with the fitted document model. Analyzer.analyze uses
    # the unfitted fallback so the library works without artifacts; the API always has them.
    from ..detect.document import DocumentVerdict, aggregate
    from ..detect.document import SentenceVerdict as SV

    verdicts = [
        SV(index=s.index, start=s.start, end=s.end, text=s.text, probability=s.probability,
           n_words=s.n_words, smoothed=s.smoothed, reliable=s.reliable)
        for s in result.sentences
    ]
    v: DocumentVerdict = aggregate(
        verdicts, threshold=analyzer.detector.flag_threshold, doc_model=get_document_model()
    )
    payload["verdict"].update({
        "machineShare": round(v.machine_share, 4),
        "machineShareLow": round(v.machine_share_low, 4),
        "machineShareHigh": round(v.machine_share_high, 4),
        "anyMachineProbability": round(v.any_machine_probability, 4),
    })
    payload["flagThreshold"] = round(analyzer.detector.flag_threshold, 4)
    payload["limitations"] = _limitations()
    return payload


@lru_cache(maxsize=1)
def _limitations() -> list[str]:
    """Shipped with every response. A detector that hides its error rates is not honest.

    Read from ``artifacts/evaluation.json`` rather than written here as prose. An earlier
    version hard-coded the percentages, the model was retrained, and the interface went on
    confidently displaying the previous run's error rates. A tool whose whole claim is
    honesty cannot afford stale numbers, so these are generated from the measurements.
    """
    generic = (
        "Short passages carry little evidence. Anything under about five sentences is "
        "reported as unreliable rather than scored confidently."
    )
    path = ARTIFACTS / "evaluation.json"
    if not path.exists():
        return ["Error rates have not been measured for this build. Run scripts/evaluate.py.", generic]

    import json

    sets = json.loads(path.read_text(encoding="utf-8")).get("sets", {})
    out: list[str] = []

    esl = sets.get("esl", {})
    toefl = (esl.get("breakdown") or {}).get("liang_toefl", {})
    if toefl.get("documentFPR") is not None:
        out.append(
            f"Measured on held-out data: {toefl['documentFPR']:.1%} of TOEFL essays written "
            f"by non-native speakers were wrongly flagged "
            f"({esl.get('documentFPR', 0):.1%} across all English-language-learner essays). "
            "Do not use this as evidence against a student."
        )
    prompting = sets.get("unseen_prompting", {})
    if prompting.get("documentRecall") is not None:
        out.append(
            "Trained on one generator (GPT-3.5). When that same generator was prompted to "
            f"evade detection, only {prompting['documentRecall']:.1%} of essays were caught. "
            "Other model families are unmeasured."
        )
    adversarial = sets.get("adversarial", {})
    if adversarial.get("documentRecall") is not None:
        out.append(
            "Prose deliberately composed to imitate a model was caught "
            f"{adversarial['documentRecall']:.0%} of the time."
        )
    loc = sets.get("localisation", {})
    if loc.get("recall") is not None:
        out.append(
            f"Inside a part-machine essay we find {loc['recall']:.0%} of the machine "
            f"sentences (precision {loc.get('precision', 0):.0%}), so an unhighlighted "
            "sentence is not evidence of anything."
        )
    out.append(generic)
    return out


if WEB.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    app.mount("/", StaticFiles(directory=WEB), name="web")

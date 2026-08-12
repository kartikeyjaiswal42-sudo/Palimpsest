"""Export the Python pipeline's answers so the JavaScript port can be checked against them.

The port is only worth anything if it is the *same detector*. Every accuracy figure in
`docs/` was measured on the Python implementation; a JavaScript rewrite that is merely
close reproduces the interface and not the evidence. So: run real corpus documents through
the shipped Python serving path, dump the observer's token stream alongside the resulting
sentence probabilities, verdict, gate probability and evidence, and let
`edge/test/parity.test.mjs` demand agreement.

Documents are drawn from the observer's on-disk cache, so this costs no Workers AI neurons
and the JavaScript side is fed byte-identical observer output — which is what isolates the
comparison to the port itself rather than to two different scorings of the same essay.

    .venv/bin/python edge/scripts/export_parity.py --limit 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "cache" / "remote_scores"
OUT = ROOT / "edge" / "test" / "parity-cases.json"
MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
MAX_CHARS = 6000  # the Worker clips here, and the cache key is the UNclipped text

#: A spread of genres and generators, not just the easy cases. The ESL and domain-shift sets
#: are in here deliberately: they are where the gate, the reliability rule and the z-clip all
#: fire, which is exactly the code a naive port gets wrong.
SOURCES = [
    "data/raw/liang_college_human.jsonl",
    "data/raw/liang_college_gpt3.jsonl",
    "data/raw/jhu.jsonl",
    "data/raw/liang_toefl.jsonl",
    "data/raw/ellipse.jsonl",
    "data/raw/persuade.jsonl",
    "data/raw/hamilton.jsonl",
    "data/generated/modern_holdout.jsonl",
    "data/generated/claude_modern.jsonl",
    "data/generated/real_hybrid_essays.jsonl",
    "data/generated/hybrid_essays.jsonl",
]


def cache_path(text: str) -> Path:
    key = hashlib.sha256(f"{MODEL}\x00{text}".encode("utf-8")).hexdigest()
    return CACHE / f"{key}.json"


def flatten(text: str) -> str:
    """`scripts/build_features.py:flatten`. The corpus was scored through this, so it is
    also what the observer cache is keyed on -- without it nothing but the already-flat
    sources matches and the parity sample silently loses every machine and hybrid essay."""
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def load_documents(live: bool) -> list[dict]:
    """Cached documents are flattened (that is how they were scored). ``live`` instead keeps
    the original paragraph breaks and scores them for real, because flattening removes the
    only thing `split_paragraphs` exists to handle -- a port could get paragraph segmentation
    completely wrong and every cached case would still pass."""
    docs: list[dict] = []
    for rel in SOURCES:
        path = ROOT / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("text") or row.get("essay") or ""
            if not raw.strip():
                continue
            if live:
                if "\n" in raw.strip() and len(raw) <= MAX_CHARS:
                    docs.append({"source": rel, "id": row.get("id"), "text": raw, "probe": None})
            else:
                text = flatten(raw)
                probe = cache_path(text)
                if probe.exists():
                    docs.append({"source": rel, "id": row.get("id"), "text": text, "probe": probe})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--live", action="store_true",
                    help="score UNflattened, multi-paragraph documents for real (costs neurons)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = (ROOT / args.out).resolve() if args.out else OUT
    docs = load_documents(args.live)
    if not docs:
        print("no corpus document has a cached probe; nothing to compare", file=sys.stderr)
        return 1

    # Stratify by source so one large collection cannot dominate the sample.
    by_source: dict[str, list[dict]] = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)
    rng = random.Random(args.seed)
    per = max(1, args.limit // len(by_source))
    chosen: list[dict] = []
    for rows in by_source.values():
        rng.shuffle(rows)
        chosen.extend(rows[:per])
    rng.shuffle(chosen)
    chosen = chosen[: args.limit]

    from palimpsest.analyze import Analyzer
    from palimpsest.detect.genre import GenreGate, document_genre_features

    artifacts = ROOT / "artifacts"
    analyzer = Analyzer.from_artifacts(artifacts, observer="remote", suffix="_remote")
    gate = GenreGate.load(artifacts / "genre_gate_remote.json")
    # The expected verdict is read straight off `analyzer.analyze()`. It used to be
    # recomputed here by calling `aggregate` with the document model and threshold passed in
    # by hand, which meant the harness compared the Worker against a document score the
    # library itself never produced -- and so reported exact parity while `Analyzer` was
    # returning `max_p` from an unfitted fallback. A harness that repairs its reference
    # implementation on the way past cannot detect a fault in it.
    if analyzer.document_model is None or analyzer.document_model.coef is None:
        raise SystemExit(
            f"no fitted document model at {artifacts}/document_detector_remote.json; "
            "the exported verdicts would come from the unfitted max_p fallback"
        )

    cases = []
    for i, d in enumerate(chosen, 1):
        text = d["text"]
        result = analyzer.analyze(text)
        payload = result.to_dict(include_tokens=False)

        v = result.verdict
        _, doc_feats, _ = analyzer.features_for(text)
        gfeat = document_genre_features(doc_feats)

        probe = json.loads((d["probe"] or cache_path(text)).read_text(encoding="utf-8"))
        cases.append({
            "source": d["source"],
            "id": d["id"],
            "text": text,
            # Exactly what the Worker's observer call would return for this text.
            "observation": {
                "model": probe.get("model", MODEL),
                "clipped": bool(probe.get("clipped")),
                "tokens": probe["tokens"],
            },
            "expected": {
                "sentences": [
                    {
                        "start": s["start"], "end": s["end"], "text": s["text"],
                        "probability": s["probability"], "smoothed": s["smoothed"],
                        "logit": s["logit"], "reliable": s["reliable"], "nWords": s["nWords"],
                        "intercept": s["intercept"],
                        "evidenceRemainder": s["evidenceRemainder"],
                        "evidence": [
                            {"name": e["name"], "z": e["z"], "contribution": e["contribution"],
                             "measured": e["measured"], "value": e["value"]}
                            for e in s["evidence"]
                        ],
                    }
                    for s in payload["sentences"]
                ],
                "passages": payload["passages"],
                "verdict": {
                    "machineShare": round(v.machine_share, 4),
                    "machineShareLow": round(v.machine_share_low, 4),
                    "machineShareHigh": round(v.machine_share_high, 4),
                    "anyMachineProbability": round(v.any_machine_probability, 4),
                    "nSentences": v.n_sentences,
                    "nWords": v.n_words,
                    "nReliableSentences": v.n_reliable_sentences,
                },
                "genre": {
                    "inDomainProbability": round(gate.probability(gfeat), 4),
                    "inDomain": bool(gate.in_domain(gfeat)),
                    "features": {k: (None if v_ != v_ else v_) for k, v_ in gfeat.items()},
                },
                # Every feature of every sentence, so a divergence names the feature rather
                # than surfacing as an unexplained probability difference.
                "features": [
                    {k: (None if val != val else val) for k, val in f.items()}
                    for f in doc_feats
                ],
            },
        })
        if i % 20 == 0:
            print(f"  {i}/{len(chosen)}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases), encoding="utf-8")
    sentences = sum(len(c["expected"]["sentences"]) for c in cases)
    print(f"{out_path.relative_to(ROOT)}: {len(cases)} documents, {sentences} sentences")
    print("sources:", json.dumps({s: sum(1 for c in cases if c["source"] == s) for s in
                                  sorted({c['source'] for c in cases})}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

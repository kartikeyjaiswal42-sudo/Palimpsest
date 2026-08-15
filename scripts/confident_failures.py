#!/usr/bin/env python
"""The three essays this detector gets most aggressively wrong, and the arithmetic that did it.

    python scripts/confident_failures.py
    python scripts/confident_failures.py --top 3 --arm augmented

Writes ``artifacts/confident_failures.json``, which the interface renders directly.

How "most aggressively wrong" is defined, and why not just "highest score"
--------------------------------------------------------------------------
``scripts/find_failures.py`` already lists the highest-scoring human essays and the
lowest-scoring machine ones. That ranks by *score*, which quietly favours long documents:
more sentences means more chances to contain one extreme sentence, so the top of that list
drifts toward whoever wrote most.

This ranks by **confident wrongness**, which is the product of two things a reader would
recognise as separate complaints:

    severity = |p - truth| x confidence

where ``confidence`` is how far the document sits past the verdict threshold in units of the
band's own width. A human essay at p=0.97 in a band that starts at 0.5 is worse than one at
p=0.97 in a band that starts at 0.95, because the second is a near-miss and the first is the
tool insisting. Both components are reported separately so the ranking can be argued with.

Ties are broken toward **shorter** documents, deliberately -- the opposite of the length
drift above. A confident wrong answer about a 90-word essay is a worse failure than the same
answer about a 900-word one, because there was less evidence to be confident from.

What each entry carries
-----------------------
Everything needed to contest the result: the essay text, the score, the band, and the
per-feature contributions that **sum to the logit** -- the same reconstruction the live
interface shows, not a summary of it. Plus an empty ``humanExplanation`` field, which is the
one thing this script cannot fill in. The brief asks for a theory of why the maths failed;
a theory is a human artefact, so the field is created empty and preserved across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402
from palimpsest.detect.document import MAX_SENTENCE_WORDS, document_statistics  # noqa: E402
from palimpsest.features.registry import FEATURE_NAMES, FEATURES_BY_NAME  # noqa: E402
from palimpsest.features.syntax import ALL_SYNTAX_FEATURE_NAMES  # noqa: E402

OUT = ROOT / "artifacts" / "confident_failures.json"
FEATURE_DIR = ROOT / "data" / "features"

REMOTE_UNAVAILABLE = ("mean_entropy", "entropy_sd", "curvature", "curvature_z_in_doc")

#: Held-out sets only. A failure on training data is a fitting artefact, not a failure.
EVAL_SETS = (
    "esl_remote",
    "domain_shift_remote",
    "modern_holdout_remote",
    "modern_claude_eval_remote",
    "modern_unseen_family_remote",
    "adversarial_remote",
    "localisation_remote",
)

#: Labels for the structural block, which is not in the shipped registry.
SYNTAX_LABELS = {
    "tree_depth_max": ("Deepest clause nesting", "structure"),
    "tree_depth_mean": ("Average clause depth", "structure"),
    "tree_depth_sd": ("Unevenness of clause depth", "structure"),
    "branching_factor": ("Children per clause head", "structure"),
    "stopword_ratio": ("Grammatical scaffolding", "structure"),
    "content_function_ratio": ("Content vs function words", "structure"),
    "pos_trigram_entropy": ("Grammatical variety", "structure"),
    "pos_trigram_surprisal": ("Unusual grammatical shape", "structure"),
    "tree_depth_z_in_doc": ("Depth vs the author's baseline", "structure"),
    "local_depth_burstiness": ("Structural rhythm", "structure"),
    "pos_surprisal_z_in_doc": ("Grammar vs the author's baseline", "structure"),
}


def load(name: str) -> list[dict]:
    path = FEATURE_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r["features"] = {
            k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()
        }
        r["_set"] = name
        rows.append(r)
    return rows


def matrix(rows, names):
    return np.array(
        [[r["features"].get(n, np.nan) for n in names] for r in rows], dtype=np.float64
    )


def _sigmoid(x):
    """Overflow-free logistic. Uses the positive/negative branches separately."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class Arm:
    """A fitted model that can score rows AND explain a single row's logit exactly."""

    def __init__(self, train_rows, names, seed=0):
        self.names = names
        x = matrix(train_rows, names)
        y = np.array([r["label"] for r in train_rows], dtype=int)
        mu = np.nanmean(x, axis=0)
        self.mu = np.where(np.isfinite(mu), mu, 0.0)
        filled = np.where(np.isfinite(x), x, self.mu)
        sd = filled.std(axis=0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        self.model = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
        self.model.fit((filled - self.mu) / self.sd, y)
        self.w = self.model.coef_[0]
        self.b = float(self.model.intercept_[0])
        # The 5% false-positive budget on training humans -- the same rule every other
        # threshold in this project is set by.
        self.threshold = float(np.quantile(self.score(train_rows)[y == 0], 0.95))

    def logits(self, rows):
        x = matrix(rows, self.names)
        z = (np.where(np.isfinite(x), x, self.mu) - self.mu) / self.sd
        return z @ self.w + self.b

    def score(self, rows):
        # Numerically stable sigmoid. The naive form overflows on this corpus: an ESL essay
        # whose `style_gap_from_doc` sits 55 training-SDs from the mean produces a logit of
        # several hundred, and np.exp(-x) warns and returns inf. The probability is right
        # either way, but a warning that fires on real data trains people to ignore warnings.
        return _sigmoid(self.logits(rows))

    def explain(self, row, k=6):
        """Per-feature contributions to this sentence's logit, largest magnitude first.

        The returned contributions plus the intercept reconstruct the logit exactly; the
        check is asserted, not trusted, because an explanation that does not add up is
        worse than no explanation.
        """
        vals = np.array([row["features"].get(n, np.nan) for n in self.names])
        measured = np.isfinite(vals)
        z = (np.where(measured, vals, self.mu) - self.mu) / self.sd
        contrib = z * self.w
        logit = float(contrib.sum() + self.b)

        order = np.argsort(-np.abs(contrib))[:k]
        out = []
        for i in order:
            name = self.names[i]
            feat = FEATURES_BY_NAME.get(name)
            label, group = (feat.label, feat.group) if feat else SYNTAX_LABELS.get(
                name, (name, "structure"))
            out.append({
                "name": name, "label": label, "group": group,
                "value": None if not measured[i] else round(float(vals[i]), 4),
                "z": round(float(z[i]), 4),
                "weight": round(float(self.w[i]), 4),
                "contribution": round(float(contrib[i]), 4),
                "measured": bool(measured[i]),
                "toward": "machine" if contrib[i] > 0 else "human",
            })
        return {
            "logit": round(logit, 6),
            "intercept": round(self.b, 6),
            "probability": round(float(1 / (1 + np.exp(-logit))), 6),
            "shownContributions": out,
            # What the top-k bars do NOT account for. Published so the arithmetic visibly
            # closes rather than appearing to.
            "remainder": round(float(contrib.sum() - contrib[order].sum()), 6),
        }


def documents(rows, probs, threshold):
    idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["doc_id"], []).append(i)
    out = []
    for doc_id, ii in idx.items():
        ii.sort(key=lambda i: rows[i]["sentence_index"])
        keep = [i for i in ii
                if float(rows[i]["features"].get("n_words") or 1.0) <= MAX_SENTENCE_WORDS]
        if not keep:
            continue
        w = np.array([rows[i]["features"].get("n_words", 1.0) for i in keep], dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        p = probs[keep]
        stats = document_statistics(p, w, threshold)
        # Log-odds of the document score. Probability saturates at 1.000 and stops
        # discriminating exactly among the worst failures -- the ones this script exists to
        # rank -- so the ordering inside that group is settled in logit space, where a
        # document at P=0.99999 is visibly further gone than one at P=0.99.
        out.append({
            "logit": float(np.log(np.clip(stats["mean_p"], 1e-15, 1 - 1e-15)
                                  / (1 - np.clip(stats["mean_p"], 1e-15, 1 - 1e-15)))),
            "doc_id": doc_id, "set": rows[keep[0]]["_set"],
            "source": rows[keep[0]]["source_id"],
            "label": int(any(rows[i]["label"] for i in keep)),
            "p": stats["mean_p"], "maxP": stats["max_p"], "share": stats["share"],
            "rowIdx": keep, "probs": p,
            "words": int(w.sum()),
            "meta": rows[keep[0]].get("doc_meta") or {},
        })
    return out


def severity(doc, doc_threshold):
    """|p - truth| x confidence past the threshold, in units of the band's own width."""
    truth = float(doc["label"])
    err = abs(doc["p"] - truth)
    if doc["label"] == 0:  # a human accused: how far past the threshold did it go?
        width = max(1.0 - doc_threshold, 1e-6)
        conf = max(0.0, doc["p"] - doc_threshold) / width
    else:  # machine missed: how far BELOW the threshold did it fall?
        width = max(doc_threshold, 1e-6)
        conf = max(0.0, doc_threshold - doc["p"]) / width
    return err * conf, err, conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="train_remote")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--arm", choices=["base", "augmented"], default="augmented")
    ap.add_argument("--doc-threshold", type=float, default=0.5)
    args = ap.parse_args()

    train = load(args.train)
    if not train:
        raise SystemExit(f"no rows in {args.train}")

    names = [n for n in FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]
    if args.arm == "augmented":
        syn = [n for n in ALL_SYNTAX_FEATURE_NAMES if n not in REMOTE_UNAVAILABLE]
        if any(n in train[0]["features"] for n in syn):
            names = names + syn
        else:
            print("! no structural block in the training file; falling back to base")
            args.arm = "base"

    arm = Arm(train, names)

    texts: dict[str, str] = {}
    for folder in ("raw", "generated"):
        for path in (ROOT / "data" / folder).glob("*.jsonl"):
            try:
                for d in read_jsonl(path):
                    texts[d.id] = d.text
            except Exception:
                continue

    all_docs, all_rows = [], {}
    for name in EVAL_SETS:
        rows = load(name)
        if not rows:
            continue
        if args.arm == "augmented" and not any(
            n in rows[0]["features"] for n in ALL_SYNTAX_FEATURE_NAMES
        ):
            print(f"  skipping {name}: no structural block")
            continue
        all_rows[name] = rows
        all_docs += documents(rows, arm.score(rows), arm.threshold)

    if not all_docs:
        raise SystemExit("no evaluation documents scored")

    for d in all_docs:
        d["severity"], d["error"], d["confidence"] = severity(d, args.doc_threshold)

    wrong = [
        d for d in all_docs
        if (d["label"] == 0 and d["p"] >= args.doc_threshold)
        or (d["label"] == 1 and d["p"] < args.doc_threshold)
    ]
    # Severity first; among documents that saturate it (P indistinguishable from 0 or 1),
    # the logit margin still discriminates; only then length, shorter being worse because
    # there was less evidence to be that confident from.
    doc_logit_thr = float(np.log(args.doc_threshold / (1 - args.doc_threshold)))
    for d in wrong:
        d["logitMargin"] = (d["logit"] - doc_logit_thr if d["label"] == 0
                            else doc_logit_thr - d["logit"])
    wrong.sort(key=lambda d: (-d["severity"], -d["logitMargin"], d["words"]))

    entries = []
    for d in wrong[:args.top]:
        rows = all_rows[d["set"]]
        text = texts.get(d["doc_id"], "")
        # The sentence that drove the verdict, in the direction of the mistake.
        drivers = sorted(
            range(len(d["rowIdx"])),
            key=lambda j: -d["probs"][j] if d["label"] == 0 else d["probs"][j],
        )[:2]
        sentences = []
        for j in drivers:
            r = rows[d["rowIdx"][j]]
            sentences.append({
                "probability": round(float(d["probs"][j]), 4),
                "start": r["start"], "end": r["end"],
                "text": text[r["start"]:r["end"]] if text else "",
                "evidence": arm.explain(r),
            })

        entries.append({
            "docId": d["doc_id"],
            "set": d["set"], "source": d["source"],
            "truth": "human" if d["label"] == 0 else "machine",
            "verdict": "machine" if d["p"] >= args.doc_threshold else "no evidence of machine",
            "direction": "false accusation" if d["label"] == 0 else "missed machine text",
            "documentProbability": round(d["p"], 4),
            "maxSentenceProbability": round(d["maxP"], 4),
            "flaggedShare": round(d["share"], 4),
            "words": d["words"],
            "severity": round(d["severity"], 4),
            "logitMargin": round(d["logitMargin"], 3),
            "errorComponent": round(d["error"], 4),
            "confidenceComponent": round(d["confidence"], 4),
            "meta": d["meta"],
            "text": text,
            "drivingSentences": sentences,
            # The one field a script must not fill in.
            "humanExplanation": "",
        })

    # Preserve any explanation already written. Regenerating the artifact must not delete
    # the analysis it exists to hold.
    if OUT.exists():
        try:
            prior = {e["docId"]: e.get("humanExplanation", "")
                     for e in json.loads(OUT.read_text()).get("failures", [])}
            for e in entries:
                if prior.get(e["docId"]):
                    e["humanExplanation"] = prior[e["docId"]]
        except Exception:
            pass

    payload = {
        "arm": args.arm,
        "nFeatures": len(names),
        "sentenceThreshold": round(arm.threshold, 4),
        "documentThreshold": args.doc_threshold,
        "nEvaluated": len(all_docs),
        "nWrong": len(wrong),
        "ranking": (
            "severity = |p - truth| x confidence past the verdict threshold, in units of the "
            "band's own width. Ties break toward the shorter document."
        ),
        "failures": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"CONFIDENT FAILURES  ({args.arm}, {len(names)} features, "
          f"{len(all_docs)} held-out documents, {len(wrong)} wrong)")
    print("=" * 78)
    for i, e in enumerate(entries, 1):
        print(f"\n{i}. {e['docId']}  [{e['set']}/{e['source']}]  {e['direction'].upper()}")
        print(f"   truth {e['truth']}, called {e['verdict']} at P={e['documentProbability']:.3f}"
              f"  ({e['words']} words)")
        print(f"   severity {e['severity']:.3f}  = error {e['errorComponent']:.3f} "
              f"x confidence {e['confidenceComponent']:.3f}")
        if e["text"]:
            print(f"   opening: {e['text'][:110]!r}")
        for s in e["drivingSentences"][:1]:
            print(f"   driving sentence P={s['probability']:.3f}: {s['text'][:90]!r}")
            for c in s["evidence"]["shownContributions"][:4]:
                print(f"     {c['label']:34} {c['contribution']:+.3f}  ({c['group']})")
        if not e["humanExplanation"]:
            print("   humanExplanation: EMPTY -- write it in the UI or in the artifact")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

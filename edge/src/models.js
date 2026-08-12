/**
 * Assemble the fitted artifacts into the object `analyze()` expects.
 *
 * Kept separate from the Worker entry point so the parity test builds the models the exact
 * same way the deployed request path does. A test that constructs the pipeline differently
 * from production is testing a different pipeline.
 */

import bundle from './artifacts.js';
import { SentenceDetector, DocumentDetector, GenreGate } from './detect.js';
import { NgramReference } from './ngram.js';

export const OBSERVER_MODEL = '@cf/qwen/qwen3-30b-a3b-fp8';

export function buildModels(ngramBuffer) {
  return {
    detector: new SentenceDetector(bundle.detector),
    documentModel: new DocumentDetector(bundle.documentModel),
    gate: new GenreGate(bundle.gate),
    bands: bundle.bands,
    reference: ngramBuffer ? NgramReference.parse(ngramBuffer) : null,
    limitations: bundle.limitations,
    productReport: bundle.productReport,
    suffix: bundle.suffix,
    observerName: OBSERVER_MODEL,
  };
}

export { bundle };

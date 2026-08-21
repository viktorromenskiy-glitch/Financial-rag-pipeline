"""Plan doradotki-2, item 7: runs a cheap retrieval-only pass over the same
250 questions as the committed `error_analysis_250` run, then attributes
each question's outcome to retrieval_failure / reranking_failure /
generation_failure_candidate / success by joining the two.

Two steps, one script (same pattern as scripts/run_migration_and_eval.py):
  1. `eval --retrieval-only` - retrieval + reranking only, no generation or
     judge calls (no Claude Sonnet API cost at all). Writes
     results/<RETRIEVAL_RUN_ID>/retrieval_trace.jsonl.
  2. `pipeline.attribution` - joins that trace against
     results/JUDGED_RUN_ID/eval_results.jsonl (already paid for, already
     committed) by question_id, using each question's gold `context_id`
     from QUESTIONS itself (no gold_inds mapping needed - see
     docs/tehnicheskoe_zadanie.md, section 21, for why). Writes
     results/<RETRIEVAL_RUN_ID>/attribution_results.jsonl and
     attribution_summary.json, and prints the summary.

Estimated cost (docs/tehnicheskoe_zadanie.md, section 21): ~$0.63 for all
250 questions - Voyage-4 query embeddings + Cohere Rerank only, no
generation/judge. Needs the same MongoDB Atlas cluster + indexed corpus
`error_analysis_250` was run against (retrieval must be over the same
corpus for the join to be valid) - if that cluster isn't already up,
bringing it up isn't costed here.

Usage (Colab, after the usual %cd + secrets-loading cells - no migration
or index update needed, this only reads the already-indexed corpus):
    !python scripts/run_retrieval_attribution.py

After it finishes, commit and push the two new results/<RETRIEVAL_RUN_ID>/
files (retrieval_trace.jsonl is the expensive one to redo; the two
attribution_*.json files are cheap to regenerate from it if needed) the
same way every other run in this project has been - see
docs/struktura_repozitoriya.md's rule about not leaving paid-run output
only on the ephemeral Colab disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

RETRIEVAL_RUN_ID = "retrieval_trace_250"  # new - this script creates it
JUDGED_RUN_ID = "error_analysis_250"  # existing, already-committed run to take judge_correct from
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"

import pipeline.attribution as attribution
import pipeline.cli as cli

print("=" * 70)
print("STEP 1/2: eval --retrieval-only (no generation/judge - Voyage + Cohere Rerank only)")
print("=" * 70)
cli.main(["eval", "--questions", QUESTIONS, "--run-id", RETRIEVAL_RUN_ID, "--retrieval-only"])

print("\n" + "=" * 70)
print(f"STEP 2/2: attribute retrieval_trace_250 against {JUDGED_RUN_ID}'s judge results")
print("=" * 70)
attribution.main(
    [
        "--retrieval-run-id",
        RETRIEVAL_RUN_ID,
        "--judged-run-id",
        JUDGED_RUN_ID,
        "--questions",
        QUESTIONS,
    ]
)

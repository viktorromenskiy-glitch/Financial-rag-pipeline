"""Step 2 of the plan-2 item 7 sequence (environment confirmed working in
step 1 - scripts/check_environment.py). This is the first step in the
sequence that actually spends money.

Runs `eval --retrieval-only` over the same 250 questions as the committed
`error_analysis_250` run: retrieval + reranking only, no generation and no
judge calls (no Claude Sonnet API cost at all - only Voyage embeddings and
Cohere Rerank). Writes results/retrieval_trace_250/retrieval_trace.jsonl,
containing, per question, the top-50 pre-rerank candidate pool and the
top-5 post-rerank result, with their gold-relevance info.

Estimated cost (docs/tehnicheskoe_zadanie.md, section 21): ~$0.63 for all
250 questions.

This step deliberately does NOT also run the attribution join (unlike the
earlier combined scripts/run_retrieval_attribution.py, which is no longer
the intended path) - per the current one-script-at-a-time workflow, the
join against results/error_analysis_250/eval_results.jsonl is its own
separate step, run only after this one's output is confirmed sane.

Usage (Colab, after step 1 has passed):
    !python scripts/run_retrieval_only_250.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

RUN_ID = "retrieval_trace_250"
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"

import pipeline.cli as cli

cli.main(["eval", "--questions", QUESTIONS, "--run-id", RUN_ID, "--retrieval-only"])

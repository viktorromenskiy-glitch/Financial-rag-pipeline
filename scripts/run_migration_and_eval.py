"""Runs step 2 (migrate_embedding_routing.py - backfill source_dataset,
switch TAT-DQA documents to their already-computed voyage-finance-2
vectors) followed by step 3 (a full eval run under per-dataset embedding
routing, equivalent to `python -m pipeline.cli eval`), in one script - so
a single Colab cell takes the corpus from "routing implemented but not
live" all the way to an eval_report.md showing the effect of routing.

Prerequisite: scripts/update_atlas_search_indexes.py must have already run and
both indexes reached READY/queryable status - this script's eval step
calls validate_startup_indexes() first (same as `pipeline.cli eval`
always does) and will fail loudly, with a clear message, if the Atlas
index updates haven't completed yet - it will not silently retrieve wrong
candidates.

RUN_ID/COMPARE_TO/QUESTIONS below are the same options
`python -m pipeline.cli eval --run-id ... --compare-to ... --questions
...` takes on the command line - set them here instead of passing CLI
flags, since this script chains two steps together. COMPARE_TO, if set to
a previous run_id (e.g. your last full run before routing), gets you a
per-question regression report in eval_report.md for free - the same
comparison this project has used at every prior architectural change.

Usage (Colab, after the usual %cd + secrets-loading cells, once
scripts/update_atlas_search_indexes.py has finished):
    !python scripts/run_migration_and_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

RUN_ID = None  # None -> auto UTC timestamp (same default as `pipeline.cli eval`); set e.g. "full250_routing" for a memorable name
COMPARE_TO = None  # set to a previous run_id, e.g. "full250_v3", for a regression report against it
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"

import migrate_embedding_routing
import pipeline.cli as cli

print("=" * 70)
print("STEP 1/2: migrate_embedding_routing.py")
print("(backfill source_dataset, switch TAT-DQA embeddings to voyage-finance-2 - no new API calls)")
print("=" * 70)
migrate_embedding_routing.main()

print("\n" + "=" * 70)
print("STEP 2/2: eval run under per-dataset embedding routing")
print("=" * 70)
eval_argv = ["eval", "--questions", QUESTIONS]
if RUN_ID:
    eval_argv += ["--run-id", RUN_ID]
if COMPARE_TO:
    eval_argv += ["--compare-to", COMPARE_TO]
cli.main(eval_argv)

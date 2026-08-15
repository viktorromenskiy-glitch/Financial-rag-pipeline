"""Runs step 3 only (a full eval run) - migration doesn't need to be
repeated, the corpus in Atlas is already correctly migrated
(source_dataset backfilled, TAT-DQA documents already on voyage-finance-2
vectors). This just re-runs eval with the 2026-08-15 filter fix in
pipeline/retrieval.py + pipeline/cli.py (unrouted questions - ConvFinQA,
FinQA - now search across all non-TAT-DQA sources again instead of being
wrongly restricted to only their own source).

COMPARE_TO is set to "20260815T171743Z" - the previous (buggy) routed run
- so eval_report.md includes a per-question regression report showing
exactly which questions changed because of this fix.

Usage (Colab, after the usual %cd + secrets-loading cells, after `git
pull` has picked up the fix):
    !python run_eval.py
"""
from __future__ import annotations

RUN_ID = None  # None -> auto UTC timestamp; set e.g. "full250_routing_fixed" for a memorable name
COMPARE_TO = "20260815T171743Z"  # the buggy routed run - set to None to skip the regression report
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"

import pipeline.cli as cli

eval_argv = ["eval", "--questions", QUESTIONS]
if RUN_ID:
    eval_argv += ["--run-id", RUN_ID]
if COMPARE_TO:
    eval_argv += ["--compare-to", COMPARE_TO]

cli.main(eval_argv)

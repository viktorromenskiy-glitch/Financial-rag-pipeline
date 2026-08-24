"""Фаза 6: run one A/B variant of the generation prompt
(docs/tehnicheskoe_zadanie.md, section 28), then verify and save the
result per "Правила сохранения долгих платных прогонов" (project doc,
2026-08-24) via pipeline.common.persist - not a hand-rolled Drive-copy
cell written from memory (that's exactly how incident 2 in
pipeline/common/persist.py's module docstring happened).

Edit CONFIG_PATH and RUN_ID below for each of the three variants, run
this script three times (baseline / cite_and_check / formula_base) - one
variant per invocation, same "edit constants, re-run" pattern as
scripts/run_eval.py.

Usage (Colab, after the usual %cd + secrets-loading cells, Drive already
mounted at /content/drive - `from google.colab import drive;
drive.mount('/content/drive')`):
    !python scripts/run_phase6_ab_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

CONFIG_PATH = "config/config.yaml"  # baseline. Or: config/config_cite_and_check.yaml / config/config_formula_base.yaml
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"
RUN_ID = "baseline_phase6"  # cite_and_check_phase6 / formula_base_phase6 for the other two variants

import pandas as pd

import pipeline.cli as cli
from config.config_schema import load_config
from pipeline.common.persist import save_run_to_drive, verify_run_files

n_questions = len(pd.read_parquet(QUESTIONS))
print(f"Loaded {n_questions} questions from {QUESTIONS} (expected record count for verification below)")

cli.main(["eval", "--config", CONFIG_PATH, "--questions", QUESTIONS, "--run-id", RUN_ID])

run_dir = Path("results") / RUN_ID

# Exact-match expectation is safe here specifically because retrieval/
# reranking config is IDENTICAL across all three Phase 6 variants (only
# the generation prompt differs), and the production baseline on this
# exact question set has already been verified to skip 0 questions
# (docs/tehnicheskoe_zadanie.md, section 14). If that ever changes for
# some reason, this raises loudly instead of silently accepting whatever
# count happened to come out - that's the point of the rule (verify,
# don't assume).
verify_run_files(
    run_dir,
    {
        "predictions.jsonl": n_questions,
        "eval_results.jsonl": n_questions,
    },
)
print(f"Verified: predictions.jsonl and eval_results.jsonl both have exactly {n_questions} records.")

config = load_config(CONFIG_PATH)
save_run_to_drive(run_dir, config.persistence.google_drive_results_dir, RUN_ID)

print(
    f"\nRun {RUN_ID!r} verified and saved to Drive. Next: review results/{RUN_ID}/eval_report.md, "
    f"then commit results/{RUN_ID}/ to git once you're satisfied it's correct."
)

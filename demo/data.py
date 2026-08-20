"""Static data loader for the minimal demo (plan item 10, "Финальный план
доработки проекта после экспертизы.docx", п.10).

Deliberately does NOT call any external API (MongoDB Atlas, Voyage, Cohere,
Claude) - the demo replays already-computed, already-audited pipeline
output from the committed n=250 run (results/error_analysis_250/), not a
live retrieval+generation+judge call. This is a documented, intentional
scope choice: it lets anyone clone the repo and see representative
input/output pairs without provisioning any credentials or spending API
budget, at the cost of not demonstrating live retrieval. See README,
"Minimal demo" section, for the honest framing.

The question_ids below are a hand-picked, not random, sample - chosen to
show both the pipeline working (judge-correct, deterministic-match cases
across all three source datasets) and the specific, already-documented
failure modes from docs/tehnicheskoe_zadanie.md, section 14 (sign flips,
answer-format mismatches, near-misses, explicit INSUFFICIENT_CONTEXT
refusals) and the judge/deterministic-check disagreement cases covered by
tests/test_is_close_v2_error_analysis.py (plan item 1). Every field shown
is read at runtime from the two committed JSONL files, not retyped or
duplicated here - so the demo can never drift from the audited run data.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "error_analysis_250" / "predictions.jsonl"
EVAL_RESULTS_PATH = REPO_ROOT / "results" / "error_analysis_250" / "eval_results.jsonl"

# Hand-picked question_ids from the committed n=250 run (results/error_analysis_250).
# 6 judge-correct / deterministic-match examples (2 per source_dataset), plus 7
# examples of documented failure/disagreement patterns from TZ section 14.
CURATED_QUESTION_IDS = [
    # Correct, straightforward — two per source_dataset.
    "finqa_train_472",
    "finqa_test_574",
    "convfinqa_358",
    "convfinqa_2138",
    "tatqa_train_3026",
    "tatqa_train_398",
    # Sign flip at the right order of magnitude (TZ §14, taxonomy row 5).
    "tatqa_train_4978",
    # Answer-format mismatch: model returned category names, not a count (TZ §14, row 6).
    "tatqa_train_3621",
    # Close miss, off by one — deterministic check passes (within tolerance),
    # judge correctly flags it wrong for a question expecting an exact count
    # (TZ §14, "близкий промах"; also a judge-stricter-than-deterministic case).
    "finqa_dev_569",
    # Close miss, wrong period pair averaged — same disagreement pattern.
    "tatqa_train_8832",
    # Explicit INSUFFICIENT_CONTEXT refusal despite gold answer existing (TZ §14, row 3).
    "finqa_dev_788",
    "tatqa_train_4974",
    # The 10th "judge softer than deterministic" disagreement case (plan item 1 /
    # tests/test_is_close_v2_error_analysis.py) — judge accepted an answer
    # is_close_v2 rejected (sign+scale mismatch: gold=17.2, answer=-17198).
    "tatqa_train_2447",
]


def _load_jsonl(path: Path) -> dict:
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["question_id"]] = record
    return records


def load_demo_sample(
    predictions_path: Path = PREDICTIONS_PATH,
    eval_results_path: Path = EVAL_RESULTS_PATH,
    question_ids: list[str] = CURATED_QUESTION_IDS,
) -> list[dict]:
    """Join predictions.jsonl + eval_results.jsonl on question_id, filtered to
    `question_ids`, in that order. Raises KeyError if a requested id is missing
    from either file (fail loudly rather than silently dropping a curated
    example)."""
    predictions = _load_jsonl(predictions_path)
    eval_results = _load_jsonl(eval_results_path)

    sample = []
    for qid in question_ids:
        pred = predictions[qid]
        eval_result = eval_results[qid]
        sample.append(
            {
                "question_id": qid,
                "source_dataset": pred["source_dataset"],
                "question": pred["question"],
                "gold_answer": pred["gold_answer"],
                "answer_text": pred["answer_text"],
                "judge_verdict": eval_result["judge_scores"]["verdict"],
                "judge_correct": eval_result["judge_scores"]["judge_correct"],
                "deterministic_match": eval_result["deterministic_match"],
            }
        )
    return sample

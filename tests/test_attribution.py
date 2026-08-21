"""Tests for pipeline/attribution.py - the deterministic retrieval/
generation error-attribution classifier (plan doradotki-2, item 7). See
that module's docstring for why this attributes all three T2-RAGBench
sources, not FinQA-only as originally scoped.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.attribution import (
    GENERATION_FAILURE_CANDIDATE,
    RERANKING_FAILURE,
    RETRIEVAL_FAILURE,
    SUCCESS,
    UNKNOWN_OUTCOME,
    attribute_run,
    classify,
    load_gold_context_ids,
    summarize_attribution,
)


def _cand(context_id: str, rank: int, score: float = 0.5) -> dict:
    return {"context_id": context_id, "rank": rank, "score": score, "content_sha256": "x"}


# --- classify ----------------------------------------------------------


def test_classify_retrieval_failure_when_gold_not_in_top50():
    top50 = [_cand("other_doc", 1)]
    top5 = [_cand("other_doc", 1)]
    assert classify("gold_doc", top50, top5, judge_correct=False) == RETRIEVAL_FAILURE


def test_classify_reranking_failure_when_gold_in_top50_but_not_top5():
    top50 = [_cand("gold_doc", 12)] + [_cand(f"doc{i}", i) for i in range(1, 6)]
    top5 = [_cand(f"doc{i}", i) for i in range(1, 6)]
    assert classify("gold_doc", top50, top5, judge_correct=False) == RERANKING_FAILURE


def test_classify_success_when_gold_in_top5_and_judge_correct():
    top50 = [_cand("gold_doc", 1)]
    top5 = [_cand("gold_doc", 1)]
    assert classify("gold_doc", top50, top5, judge_correct=True) == SUCCESS


def test_classify_generation_failure_candidate_when_gold_in_top5_but_judge_incorrect():
    top50 = [_cand("gold_doc", 2)]
    top5 = [_cand("gold_doc", 2)]
    assert classify("gold_doc", top50, top5, judge_correct=False) == GENERATION_FAILURE_CANDIDATE


def test_classify_unknown_outcome_when_judge_correct_is_none():
    # question_id present in retrieval_trace but not in the judged run's
    # eval_results.jsonl (e.g. mismatched question sets) - must not be
    # silently treated as either success or failure.
    top50 = [_cand("gold_doc", 1)]
    top5 = [_cand("gold_doc", 1)]
    assert classify("gold_doc", top50, top5, judge_correct=None) == UNKNOWN_OUTCOME


def test_classify_retrieval_failure_takes_priority_when_top50_empty():
    assert classify("gold_doc", [], [], judge_correct=True) == RETRIEVAL_FAILURE


# --- load_gold_context_ids ----------------------------------------------


def test_load_gold_context_ids_reads_id_and_context_id_columns(tmp_path):
    path = tmp_path / "questions.parquet"
    pd.DataFrame(
        {
            "id": ["finqa_dev_0", "tatqa_train_1"],
            "context_id": ["finqa_dev_ctx_0", "tatqa_ctx_1"],
            "question": ["q1", "q2"],
        }
    ).to_parquet(path)

    result = load_gold_context_ids(path)

    assert result == {"finqa_dev_0": "finqa_dev_ctx_0", "tatqa_train_1": "tatqa_ctx_1"}


def test_load_gold_context_ids_raises_when_context_id_column_missing(tmp_path):
    path = tmp_path / "questions.parquet"
    pd.DataFrame({"id": ["a"], "question": ["q"]}).to_parquet(path)

    with pytest.raises(ValueError, match="context_id"):
        load_gold_context_ids(path)


# --- attribute_run (file-join integration) -------------------------------


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_attribute_run_joins_trace_and_judge_results_by_question_id(tmp_path):
    questions_path = tmp_path / "questions.parquet"
    pd.DataFrame(
        {
            "id": ["q_success", "q_retrieval_fail", "q_generation_fail"],
            "context_id": ["ctx_a", "ctx_b", "ctx_c"],
            "question": ["q1", "q2", "q3"],
        }
    ).to_parquet(questions_path)

    trace_path = tmp_path / "retrieval_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "question_id": "q_success",
                "source_dataset": "FinQA",
                "candidate_top50": [_cand("ctx_a", 1)],
                "reranked_top5": [_cand("ctx_a", 1)],
            },
            {
                "question_id": "q_retrieval_fail",
                "source_dataset": "ConvFinQA",
                "candidate_top50": [_cand("other", 1)],
                "reranked_top5": [_cand("other", 1)],
            },
            {
                "question_id": "q_generation_fail",
                "source_dataset": "TAT-DQA",
                "candidate_top50": [_cand("ctx_c", 3)],
                "reranked_top5": [_cand("ctx_c", 3)],
            },
        ],
    )

    eval_results_path = tmp_path / "eval_results.jsonl"
    _write_jsonl(
        eval_results_path,
        [
            {"question_id": "q_success", "judge_scores": {"judge_correct": True}, "deterministic_match": True, "judge_agrees": True},
            {"question_id": "q_retrieval_fail", "judge_scores": {"judge_correct": False}, "deterministic_match": False, "judge_agrees": True},
            {"question_id": "q_generation_fail", "judge_scores": {"judge_correct": False}, "deterministic_match": False, "judge_agrees": True},
        ],
    )

    records = attribute_run(trace_path, eval_results_path, questions_path)
    by_qid = {r["question_id"]: r for r in records}

    assert by_qid["q_success"]["failure_stage"] == SUCCESS
    assert by_qid["q_success"]["gold_in_top50"] is True
    assert by_qid["q_success"]["gold_in_top5"] is True

    assert by_qid["q_retrieval_fail"]["failure_stage"] == RETRIEVAL_FAILURE
    assert by_qid["q_retrieval_fail"]["gold_in_top50"] is False
    assert by_qid["q_retrieval_fail"]["gold_in_top5"] is False

    assert by_qid["q_generation_fail"]["failure_stage"] == GENERATION_FAILURE_CANDIDATE
    assert by_qid["q_generation_fail"]["gold_context_id"] == "ctx_c"


def test_attribute_run_raises_on_question_id_not_in_questions_file(tmp_path):
    questions_path = tmp_path / "questions.parquet"
    pd.DataFrame({"id": ["known"], "context_id": ["ctx"], "question": ["q"]}).to_parquet(questions_path)

    trace_path = tmp_path / "retrieval_trace.jsonl"
    _write_jsonl(trace_path, [{"question_id": "unknown_qid", "source_dataset": "FinQA", "candidate_top50": [], "reranked_top5": []}])

    eval_results_path = tmp_path / "eval_results.jsonl"
    _write_jsonl(eval_results_path, [])

    with pytest.raises(ValueError, match="unknown_qid"):
        attribute_run(trace_path, eval_results_path, questions_path)


# --- summarize_attribution -----------------------------------------------


def test_summarize_attribution_counts_overall_and_by_source():
    records = [
        {"source_dataset": "FinQA", "failure_stage": SUCCESS},
        {"source_dataset": "FinQA", "failure_stage": RETRIEVAL_FAILURE},
        {"source_dataset": "TAT-DQA", "failure_stage": SUCCESS},
    ]

    summary = summarize_attribution(records)

    assert summary["n"] == 3
    assert summary["overall"] == {SUCCESS: 2, RETRIEVAL_FAILURE: 1}
    assert summary["by_source_dataset"]["FinQA"] == {SUCCESS: 1, RETRIEVAL_FAILURE: 1}
    assert summary["by_source_dataset"]["TAT-DQA"] == {SUCCESS: 1}

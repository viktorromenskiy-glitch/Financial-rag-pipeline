"""Tests for demo/data.py (plan item 10, minimal demo).

Loads directly from the committed results/error_analysis_250/*.jsonl files -
no mocking, no fixtures - the same rule applied in
tests/test_is_close_v2_error_analysis.py: verify against the real audited
run data, not a retyped copy of it, so these tests would actually catch a
mismatch if that run were ever replaced or a curated question_id typo'd.
"""

from __future__ import annotations

import pytest

from demo.data import CURATED_QUESTION_IDS, load_demo_sample


def test_load_demo_sample_returns_one_row_per_curated_id():
    sample = load_demo_sample()
    assert len(sample) == len(CURATED_QUESTION_IDS)
    assert [row["question_id"] for row in sample] == CURATED_QUESTION_IDS


def test_load_demo_sample_preserves_curated_order():
    sample = load_demo_sample()
    # Order matters for the UI dropdown - must match CURATED_QUESTION_IDS exactly,
    # not whatever order the JSONL files happen to store them in.
    assert [row["question_id"] for row in sample] == list(CURATED_QUESTION_IDS)


def test_load_demo_sample_rows_have_required_fields():
    required = {
        "question_id",
        "source_dataset",
        "question",
        "gold_answer",
        "answer_text",
        "judge_verdict",
        "judge_correct",
        "deterministic_match",
    }
    for row in load_demo_sample():
        assert required <= row.keys()
        assert row["question"], row["question_id"]
        assert row["source_dataset"] in {"FinQA", "ConvFinQA", "TAT-DQA"}


def test_load_demo_sample_covers_all_three_source_datasets():
    sources = {row["source_dataset"] for row in load_demo_sample()}
    assert sources == {"FinQA", "ConvFinQA", "TAT-DQA"}


def test_load_demo_sample_includes_both_agreeing_and_disagreeing_cases():
    # The curated set is meant to show the judge/deterministic-check disagreement
    # documented in TZ section 14 and tests/test_is_close_v2_error_analysis.py,
    # not just clean agreement cases - assert both are actually present.
    sample = load_demo_sample()
    agreeing = [r for r in sample if r["judge_correct"] == r["deterministic_match"]]
    disagreeing = [r for r in sample if r["judge_correct"] != r["deterministic_match"]]
    assert agreeing, "expected at least one judge/deterministic agreement case"
    assert disagreeing, "expected at least one judge/deterministic disagreement case"


def test_load_demo_sample_missing_question_id_raises_key_error():
    with pytest.raises(KeyError):
        load_demo_sample(question_ids=["this_question_id_does_not_exist"])


def test_curated_question_ids_are_unique():
    assert len(CURATED_QUESTION_IDS) == len(set(CURATED_QUESTION_IDS))

"""Tests for pipeline.cli - the orchestrator wiring modules 1-9 together.

Scope, deliberately narrow: this file tests the CLI's OWN logic - the
parts that are not already covered by unit tests in the module they call
into. It does not re-run cmd_index/cmd_eval end-to-end against fake
retrieve/rerank/generate_answer/evaluate_answers - that would mean
re-implementing a second copy of modules 6-9's fakes just to drive this
file, duplicating tests/test_generation.py, tests/test_evaluation.py, and
the other per-module test files instead of complementing them. What IS
tested here, all real bugs or real schema quirks documented in
pipeline/cli.py's own docstrings:

- _extract_text(): the "first content block isn't always the text block"
  bug (a real claude-sonnet-5 response observed 2026-08-15 with a non-text
  block at position 0).
- _infer_source_dataset(): the id-prefix routing table, including the
  ConvFinQA-must-be-checked-before-FinQA ordering requirement the
  docstring calls out explicitly.
- load_eval_questions(): schema fallbacks (id vs question_id vs
  context_id-derived, answer vs program_answer, source_dataset column vs
  inferred).
- generation checkpoint round-trip (the fix for the real 250-question run
  that crashed at question 50/250 and lost all prior answers).
- load_eval_results() / write_eval_report(): report math and the
  regression-vs-previous-run section.
- main()'s argparse wiring for both subcommands (index/eval), including
  the auto-generated run_id when --run-id is omitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd
import pytest

from pipeline.cli import (
    _append_generation_checkpoint,
    _extract_text,
    _infer_source_dataset,
    _load_generation_checkpoint,
    _retrieved_docs_for_prediction,
    load_eval_questions,
    load_eval_results,
    main,
    write_eval_report,
)
from pipeline.evaluation import EvalResult


# --- _extract_text -----------------------------------------------------


@dataclass
class FakeBlock:
    type: str
    text: str = None


@dataclass
class FakeResponse:
    content: list = field(default_factory=list)


def test_extract_text_finds_text_block_not_at_position_zero():
    # the exact real bug: a non-text block (e.g. thinking) ahead of the
    # actual text block - response.content[0].text would be None here
    response = FakeResponse(content=[FakeBlock(type="thinking"), FakeBlock(type="text", text="42")])
    assert _extract_text(response) == "42"


def test_extract_text_works_when_text_block_is_first():
    response = FakeResponse(content=[FakeBlock(type="text", text="hello")])
    assert _extract_text(response) == "hello"


def test_extract_text_raises_when_no_text_block_present():
    response = FakeResponse(content=[FakeBlock(type="thinking")])
    with pytest.raises(RuntimeError):
        _extract_text(response)


# --- _infer_source_dataset -----------------------------------------------


def test_infer_source_dataset_finqa():
    assert _infer_source_dataset("finqa_train_2917") == "FinQA"


def test_infer_source_dataset_convfinqa_checked_before_finqa():
    # "convfinqa_" is not a superstring of "finqa_" but both start with
    # letters that could collide under a careless check - this is the
    # exact ordering requirement called out in the module docstring
    assert _infer_source_dataset("convfinqa_dev_12") == "ConvFinQA"


def test_infer_source_dataset_tatqa_and_tatdqa_both_map_to_tat_dqa():
    assert _infer_source_dataset("tat-dqa_train_5") == "TAT-DQA"
    assert _infer_source_dataset("tatqa_train_5") == "TAT-DQA"


def test_infer_source_dataset_case_insensitive():
    assert _infer_source_dataset("FinQA_Train_1") == "FinQA"


def test_infer_source_dataset_falls_back_through_candidates():
    # first candidate (id) is None/unrecognized, second (context_id) matches
    assert _infer_source_dataset(None, "finqa_train_9") == "FinQA"


def test_infer_source_dataset_unknown_when_nothing_matches():
    assert _infer_source_dataset("some_other_prefix_1") == "unknown"


# --- load_eval_questions --------------------------------------------------


def test_load_eval_questions_uses_id_and_program_answer(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["finqa_train_1", "convfinqa_dev_2"],
            "context_id": ["ctx1", "ctx2"],
            "question": ["What was revenue?", "What was growth?"],
            "program_answer": ["100", "5.2"],
        }
    )
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)

    items = load_eval_questions(path)

    assert len(items) == 2
    assert items[0]["question_id"] == "finqa_train_1"
    assert items[0]["gold_answer"] == "100"
    assert items[0]["source_dataset"] == "FinQA"  # inferred from id, no source_dataset column
    assert items[1]["source_dataset"] == "ConvFinQA"


def test_load_eval_questions_uses_explicit_source_dataset_when_present(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["finqa_train_1"],
            "question": ["Q?"],
            "answer": ["42"],
            "source_dataset": ["FinQA"],
        }
    )
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)
    items = load_eval_questions(path)
    assert items[0]["source_dataset"] == "FinQA"
    assert items[0]["gold_answer"] == "42"


def test_load_eval_questions_falls_back_to_context_id_and_row_index(tmp_path):
    df = pd.DataFrame({"context_id": ["ctx1"], "question": ["Q?"], "answer": ["1"]})
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)
    items = load_eval_questions(path)
    assert items[0]["question_id"] == "ctx1::0"


def test_load_eval_questions_falls_back_to_bare_row_index(tmp_path):
    df = pd.DataFrame({"question": ["Q?"], "answer": ["1"]})
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)
    items = load_eval_questions(path)
    assert items[0]["question_id"] == "q0"


def test_load_eval_questions_requires_question_column(tmp_path):
    df = pd.DataFrame({"answer": ["1"]})
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError):
        load_eval_questions(path)


def test_load_eval_questions_requires_an_answer_column(tmp_path):
    df = pd.DataFrame({"question": ["Q?"]})
    path = tmp_path / "eval.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError):
        load_eval_questions(path)


# --- generation checkpoint round-trip --------------------------------------


def test_generation_checkpoint_missing_file_returns_empty(tmp_path):
    assert _load_generation_checkpoint(tmp_path / "nope.jsonl") == {}


def test_generation_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "generation_checkpoint.jsonl"
    # Compact schema (plan_dorabotki_2.md item 1) - no full_indexed_content,
    # just context_id/rank/score/content_sha256.
    retrieved_docs = [{"context_id": "doc1", "rank": 1, "score": 0.9, "content_sha256": "abc123"}]
    _append_generation_checkpoint(path, "q1", "What was revenue?", "FinQA", "100", "100", "ctx text", retrieved_docs)

    loaded = _load_generation_checkpoint(path)

    assert "q1" in loaded
    assert loaded["q1"]["answer_text"] == "100"
    assert loaded["q1"]["source_dataset"] == "FinQA"
    assert loaded["q1"]["context"] == "ctx text"
    assert loaded["q1"]["retrieved_docs"] == retrieved_docs


def test_generation_checkpoint_resumes_across_multiple_appends(tmp_path):
    path = tmp_path / "generation_checkpoint.jsonl"
    _append_generation_checkpoint(path, "q1", "Q1", "FinQA", "1", "1", "ctx1", [{"context_id": "d1", "rank": 1, "score": 1.1, "content_sha256": "h1"}])
    _append_generation_checkpoint(path, "q2", "Q2", "ConvFinQA", "2", "2", "ctx2", [{"context_id": "d2", "rank": 1, "score": 1.2, "content_sha256": "h2"}])
    loaded = _load_generation_checkpoint(path)
    assert set(loaded) == {"q1", "q2"}


def test_generation_checkpoint_missing_retrieved_docs_defaults_to_empty_list(tmp_path):
    # Backward compatibility: a checkpoint written by a pipeline.cli version
    # before retrieved_docs existed (tehnicheskoe_zadanie.md, section 14,
    # "Закрыто в коде 2026-08-20") must still be resumable, per
    # _load_generation_checkpoint's docstring.
    path = tmp_path / "generation_checkpoint.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Q1",
                "source_dataset": "FinQA",
                "gold_answer": "1",
                "answer_text": "1",
                "context": "ctx1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = _load_generation_checkpoint(path)
    assert loaded["q1"]["retrieved_docs"] == []


# --- _retrieved_docs_for_prediction -----------------------------------------
#
# tehnicheskoe_zadanie.md, section 14, "Ограничение, обязательное к
# указанию": predictions.jsonl previously didn't record which documents
# were actually retrieved, so retrieval vs. generation errors couldn't be
# told apart. This helper is what closes that gap for future runs (see the
# "Закрыто в коде 2026-08-20" note added to that section).


@dataclass
class FakeRerankedCandidate:
    context_id: str
    full_indexed_content: str
    relevance_score: float = 0.0


@dataclass
class FakeRetrievalCandidate:
    context_id: str
    full_indexed_content: str
    score: float = 0.0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_retrieved_docs_for_prediction_from_reranked_candidates():
    # reranker-enabled branch: cmd_eval's `ranked` holds
    # pipeline.reranking.RerankedCandidate objects (relevance_score, not
    # score) - the helper must not depend on that field name, and must not
    # copy full_indexed_content into the result (plan_dorabotki_2.md item 1
    # - compact evidence trace, MongoDB is the authoritative text store).
    ranked = [
        FakeRerankedCandidate("doc1", "chunk one text", relevance_score=0.9),
        FakeRerankedCandidate("doc2", "chunk two text", relevance_score=0.5),
    ]
    assert _retrieved_docs_for_prediction(ranked) == [
        {"context_id": "doc1", "rank": 1, "score": 0.9, "content_sha256": _sha256("chunk one text")},
        {"context_id": "doc2", "rank": 2, "score": 0.5, "content_sha256": _sha256("chunk two text")},
    ]


def test_retrieved_docs_for_prediction_from_plain_retrieval_candidates():
    # reranker-disabled branch: cmd_eval's `ranked` is a plain slice of
    # pipeline.retrieval.Candidate objects (score, not relevance_score).
    ranked = [FakeRetrievalCandidate("doc1", "chunk one text", score=1.2)]
    assert _retrieved_docs_for_prediction(ranked) == [
        {"context_id": "doc1", "rank": 1, "score": 1.2, "content_sha256": _sha256("chunk one text")}
    ]


def test_retrieved_docs_for_prediction_preserves_rank_order():
    # Order matters here - it's the post-rerank order used to build
    # `context` for generation, so retrieved_docs[0] should be the
    # top-ranked document (rank=1), not just any of the top-N.
    ranked = [
        FakeRerankedCandidate("first", "a", relevance_score=0.9),
        FakeRerankedCandidate("second", "b", relevance_score=0.8),
        FakeRerankedCandidate("third", "c", relevance_score=0.1),
    ]
    result = _retrieved_docs_for_prediction(ranked)
    assert [(d["context_id"], d["rank"]) for d in result] == [("first", 1), ("second", 2), ("third", 3)]


def test_retrieved_docs_for_prediction_content_hash_detects_different_text():
    # Two candidates with the same context_id-shaped identity but different
    # text must not hash the same - the whole point of content_sha256 is
    # catching silent drift between what was retrieved and what's in
    # MongoDB now.
    ranked = [FakeRerankedCandidate("doc1", "version A", relevance_score=0.9)]
    other = [FakeRerankedCandidate("doc1", "version B", relevance_score=0.9)]
    assert _retrieved_docs_for_prediction(ranked)[0]["content_sha256"] != _retrieved_docs_for_prediction(other)[0]["content_sha256"]


def test_retrieved_docs_for_prediction_empty_when_no_candidates():
    assert _retrieved_docs_for_prediction([]) == []


# --- load_eval_results -----------------------------------------------------


def test_load_eval_results_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_results(tmp_path / "nope.jsonl")


def test_load_eval_results_roundtrip(tmp_path):
    path = tmp_path / "eval_results.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "question_id": "q1",
                    "judge_scores": {"verdict": "CORRECT", "judge_correct": True},
                    "deterministic_match": True,
                    "judge_agrees": True,
                }
            )
            + "\n"
        )
    results = load_eval_results(path)
    assert len(results) == 1
    assert isinstance(results[0], EvalResult)
    assert results[0].question_id == "q1"
    assert results[0].judge_scores["judge_correct"] is True


# --- write_eval_report -------------------------------------------------


def _eval_result(question_id: str, judge_correct: bool, deterministic_match: bool = None) -> EvalResult:
    if deterministic_match is None:
        deterministic_match = judge_correct
    return EvalResult(
        question_id=question_id,
        judge_scores={"verdict": "CORRECT" if judge_correct else "INCORRECT", "judge_correct": judge_correct},
        deterministic_match=deterministic_match,
        judge_agrees=True,
    )


def test_write_eval_report_computes_aggregate_and_per_source_accuracy(tmp_path):
    items = [
        {"question_id": "q1", "source_dataset": "FinQA"},
        {"question_id": "q2", "source_dataset": "FinQA"},
        {"question_id": "q3", "source_dataset": "ConvFinQA"},
    ]
    results = [
        _eval_result("q1", judge_correct=True),
        _eval_result("q2", judge_correct=False),
        _eval_result("q3", judge_correct=True),
    ]
    report_path = tmp_path / "eval_report.md"

    write_eval_report(report_path, results, items, run_id="test_run")

    text = report_path.read_text(encoding="utf-8")
    assert "Questions evaluated: 3" in text
    assert "Judge accuracy: 2/3" in text
    assert "| FinQA | 2 |" in text
    assert "| ConvFinQA | 1 |" in text


def test_write_eval_report_includes_regression_section_when_previous_given(tmp_path):
    items = [{"question_id": "q1", "source_dataset": "FinQA"}]
    previous = [_eval_result("q1", judge_correct=False)]
    current = [_eval_result("q1", judge_correct=True)]
    report_path = tmp_path / "eval_report.md"

    write_eval_report(report_path, current, items, run_id="new_run", previous_results=previous, compare_to="old_run")

    text = report_path.read_text(encoding="utf-8")
    assert "Regression vs run old_run" in text
    assert "Improved (wrong -> correct): 1" in text


def test_write_eval_report_omits_regression_section_without_previous(tmp_path):
    items = [{"question_id": "q1", "source_dataset": "FinQA"}]
    results = [_eval_result("q1", judge_correct=True)]
    report_path = tmp_path / "eval_report.md"

    write_eval_report(report_path, results, items, run_id="test_run")

    assert "Regression vs run" not in report_path.read_text(encoding="utf-8")


# --- main() argparse wiring --------------------------------------------


def test_main_index_parses_args_and_dispatches(monkeypatch):
    captured = {}

    def fake_cmd_index(args):
        captured["args"] = args

    monkeypatch.setattr("pipeline.cli.cmd_index", fake_cmd_index)
    main(["index", "--data-dir", "my_data_dir"])

    assert captured["args"].data_dir == "my_data_dir"
    assert captured["args"].config == "config/config.yaml"
    assert captured["args"].checkpoint == "checkpoints/enrichment_checkpoint.jsonl"


def test_main_eval_generates_run_id_when_omitted(monkeypatch):
    captured = {}

    def fake_cmd_eval(args):
        captured["args"] = args

    monkeypatch.setattr("pipeline.cli.cmd_eval", fake_cmd_eval)
    main(["eval", "--questions", "data/t2-ragbench/eval_subset_250.parquet"])

    args = captured["args"]
    assert args.run_id is not None
    assert args.run_id.endswith("Z")  # UTC timestamp format, e.g. 20260818T120000Z
    assert args.limit is None
    assert args.compare_to is None


def test_main_eval_respects_explicit_run_id_and_options(monkeypatch):
    captured = {}

    def fake_cmd_eval(args):
        captured["args"] = args

    monkeypatch.setattr("pipeline.cli.cmd_eval", fake_cmd_eval)
    main(
        [
            "eval",
            "--questions",
            "q.parquet",
            "--run-id",
            "my_run",
            "--limit",
            "10",
            "--compare-to",
            "baseline_run",
        ]
    )

    args = captured["args"]
    assert args.run_id == "my_run"
    assert args.limit == 10
    assert args.compare_to == "baseline_run"

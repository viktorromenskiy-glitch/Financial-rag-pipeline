"""Tests for pipeline.evaluation (module 9).

Covers: _extract_verdict (same "let it reason, delimit the marker" parsing
pattern as generation.py's _extract_final_answer - see that revision
history in pipeline/evaluation.py's module docstring), cache_key's
prompt-version sensitivity (a stale-cache bug here would silently reuse
scores from an old JUDGE_PROMPT - tehnicheskoe_zadanie.md, section 8),
JudgeCache's on-disk roundtrip, evaluate_answer's deterministic_match /
judge_agrees wiring, evaluate_answers' cache-hit skip behavior, and
regression_report's improved/regressed/unchanged classification (the
comparison that caught the real bge-reranker regression documented in
pipeline/evaluation.py: 13 fixed but 44 broken behind a neutral
aggregate). Not a real Claude call anywhere - a FakeJudge stands in for
JudgeProtocol.
"""

from __future__ import annotations

import json

import pytest

from pipeline.evaluation import (
    EvalResult,
    JudgeCache,
    cache_key,
    evaluate_answer,
    evaluate_answers,
    regression_report,
    _extract_verdict,
)


class FakeJudge:
    """Returns responses from a fixed queue, one per call - lets a test
    control exactly what the "model" says without a network call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    def judge(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._responses.pop(0)


# --- _extract_verdict ------------------------------------------------------


def test_extract_verdict_same_line():
    assert _extract_verdict("looks right\nVERDICT: CORRECT") == "CORRECT"


def test_extract_verdict_marker_then_newline():
    assert _extract_verdict("reasoning\nVERDICT:\nINCORRECT") == "INCORRECT"


def test_extract_verdict_case_insensitive():
    assert _extract_verdict("verdict: correct") == "correct"


def test_extract_verdict_last_marker_wins_if_more_than_one():
    raw = "VERDICT: INCORRECT\nactually wait\nVERDICT: CORRECT"
    assert _extract_verdict(raw) == "CORRECT"


def test_extract_verdict_falls_back_to_last_line_when_marker_missing():
    raw = "Comparing 5413606 to 5413606000 - same value, different units.\n\nCORRECT"
    assert _extract_verdict(raw) == "CORRECT"


# --- cache_key ---------------------------------------------------------


def test_cache_key_stable_for_identical_inputs():
    k1 = cache_key("q", "ctx", "answer", "v3")
    k2 = cache_key("q", "ctx", "answer", "v3")
    assert k1 == k2


def test_cache_key_changes_with_prompt_version():
    # the whole point of including prompt_version in the key - a JUDGE_PROMPT
    # bump must invalidate old cached scores rather than silently reusing them
    k_v2 = cache_key("q", "ctx", "answer", "v2")
    k_v3 = cache_key("q", "ctx", "answer", "v3")
    assert k_v2 != k_v3


def test_cache_key_changes_with_context():
    # context (the actual retrieved text) is part of the key so a
    # retrieval/reranking change also invalidates stale judge results
    k1 = cache_key("q", "context A", "answer", "v3")
    k2 = cache_key("q", "context B", "answer", "v3")
    assert k1 != k2


# --- JudgeCache -----------------------------------------------------------


def test_judge_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JudgeCache(cache_path)
    assert cache.load() == {}

    result = EvalResult(
        question_id="q1",
        judge_scores={"verdict": "CORRECT", "judge_correct": True},
        deterministic_match=True,
        judge_agrees=True,
    )
    cache.append("key1", result)

    reloaded = JudgeCache(cache_path).load()
    assert "key1" in reloaded
    assert reloaded["key1"]["question_id"] == "q1"
    assert reloaded["key1"]["judge_scores"]["judge_correct"] is True


def test_judge_cache_survives_multiple_appends(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JudgeCache(cache_path)
    for i in range(3):
        cache.append(
            f"key{i}",
            EvalResult(
                question_id=f"q{i}",
                judge_scores={"verdict": "CORRECT", "judge_correct": True},
                deterministic_match=True,
                judge_agrees=True,
            ),
        )
    reloaded = cache.load()
    assert set(reloaded) == {"key0", "key1", "key2"}


# --- evaluate_answer -------------------------------------------------------


def test_evaluate_answer_deterministic_and_judge_agree():
    judge = FakeJudge(["VERDICT: CORRECT"])
    result = evaluate_answer(judge, "q1", "What was revenue?", "ctx", "100", "100")
    assert result.deterministic_match is True
    assert result.judge_scores["judge_correct"] is True
    assert result.judge_agrees is True


def test_evaluate_answer_judge_and_deterministic_disagree():
    # deterministic check says no match (77 != 100 within tolerance), but
    # the judge says CORRECT - judge_agrees must reflect the disagreement,
    # not just echo the judge's own verdict
    judge = FakeJudge(["VERDICT: CORRECT"])
    result = evaluate_answer(judge, "q1", "What was revenue?", "ctx", "77", "100")
    assert result.deterministic_match is False
    assert result.judge_scores["judge_correct"] is True
    assert result.judge_agrees is False


def test_evaluate_answer_with_deterministic_check_disabled():
    judge = FakeJudge(["VERDICT: INCORRECT"])
    result = evaluate_answer(
        judge, "q1", "q", "ctx", "100", "100", deterministic_check_enabled=False
    )
    # nothing to disagree with when the check didn't run - vacuously True,
    # not a measured comparison (see module docstring on evaluate_answer)
    assert result.deterministic_match is False
    assert result.judge_agrees is True


# --- evaluate_answers (batch + cache) --------------------------------------


def _item(question_id, question="q", context="ctx", generated="100", gold="100"):
    return {
        "question_id": question_id,
        "question": question,
        "context": context,
        "generated_answer": generated,
        "gold_answer": gold,
    }


def test_evaluate_answers_skips_cached_items(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JudgeCache(cache_path)
    item = _item("q1")
    key = cache_key(item["question"], item["context"], item["generated_answer"])
    cache.append(
        key,
        EvalResult(
            question_id="q1",
            judge_scores={"verdict": "CORRECT", "judge_correct": True},
            deterministic_match=True,
            judge_agrees=True,
        ),
    )

    judge = FakeJudge([])  # no responses queued - must not be called at all
    results = evaluate_answers(judge, [item], cache=cache)

    assert judge.call_count == 0
    assert results[0].question_id == "q1"
    assert results[0].judge_scores["judge_correct"] is True


def test_evaluate_answers_judges_and_caches_new_items(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JudgeCache(cache_path)
    judge = FakeJudge(["VERDICT: CORRECT"])

    results = evaluate_answers(judge, [_item("q1")], cache=cache)

    assert judge.call_count == 1
    assert results[0].judge_scores["judge_correct"] is True
    # now actually persisted, not just held in memory
    assert len(cache.load()) == 1


def test_evaluate_answers_without_cache_always_judges():
    judge = FakeJudge(["VERDICT: CORRECT", "VERDICT: INCORRECT"])
    results = evaluate_answers(judge, [_item("q1"), _item("q2", generated="999")], cache=None)
    assert judge.call_count == 2
    assert len(results) == 2


# --- regression_report -----------------------------------------------------


def _result(question_id: str, judge_correct: bool) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        judge_scores={"verdict": "CORRECT" if judge_correct else "INCORRECT", "judge_correct": judge_correct},
        deterministic_match=judge_correct,
        judge_agrees=True,
    )


def test_regression_report_classifies_all_four_buckets():
    previous = [
        _result("improved_q", judge_correct=False),
        _result("regressed_q", judge_correct=True),
        _result("stable_correct_q", judge_correct=True),
        _result("stable_incorrect_q", judge_correct=False),
    ]
    current = [
        _result("improved_q", judge_correct=True),
        _result("regressed_q", judge_correct=False),
        _result("stable_correct_q", judge_correct=True),
        _result("stable_incorrect_q", judge_correct=False),
    ]
    report = regression_report(previous, current)
    assert report["improved"] == ["improved_q"]
    assert report["regressed"] == ["regressed_q"]
    assert report["unchanged_correct"] == ["stable_correct_q"]
    assert report["unchanged_incorrect"] == ["stable_incorrect_q"]


def test_regression_report_skips_questions_new_to_current_run():
    previous = [_result("q1", judge_correct=True)]
    current = [_result("q1", judge_correct=True), _result("q_new", judge_correct=True)]
    report = regression_report(previous, current)
    assert "q_new" not in sum(report.values(), [])

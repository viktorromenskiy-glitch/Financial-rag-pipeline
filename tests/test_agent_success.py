"""Tests for agent/success.py - the harness-level "successful completion
rate" definition (День 2, Kimi's followup response on primary metric).

FakeJudge mirrors tests/test_evaluation.py's exactly (per this project's
per-module test-file convention, no cross-file fake reuse) - no real
Claude call anywhere in this file.
"""

from __future__ import annotations

from agent.success import (
    INSUFFICIENCY_JUDGE_PROMPT_VERSION,
    SuccessResult,
    _extract_insufficiency_verdict,
    evaluate_agent_success,
    judge_context_insufficiency,
)


class FakeJudge:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    def judge(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._responses.pop(0)


# --- _extract_insufficiency_verdict ---------------------------------------


def test_extract_insufficiency_verdict_same_line():
    assert _extract_insufficiency_verdict("reasoning\nVERDICT: JUSTIFIED_REFUSAL") == "JUSTIFIED_REFUSAL"


def test_extract_insufficiency_verdict_marker_then_newline():
    assert _extract_insufficiency_verdict("reasoning\nVERDICT:\nSHOULD_HAVE_ANSWERED") == "SHOULD_HAVE_ANSWERED"


def test_extract_insufficiency_verdict_case_insensitive():
    assert _extract_insufficiency_verdict("verdict: justified_refusal") == "justified_refusal"


def test_extract_insufficiency_verdict_last_marker_wins():
    raw = "VERDICT: SHOULD_HAVE_ANSWERED\nwait, reconsidering...\nVERDICT: JUSTIFIED_REFUSAL"
    assert _extract_insufficiency_verdict(raw) == "JUSTIFIED_REFUSAL"


def test_extract_insufficiency_verdict_falls_back_to_last_line_when_marker_missing():
    assert _extract_insufficiency_verdict("no marker here\njust a stray line") == "just a stray line"


# --- judge_context_insufficiency ------------------------------------------


def test_judge_context_insufficiency_true_on_justified_refusal():
    judge = FakeJudge(["The context has no revenue figure at all.\nVERDICT: JUSTIFIED_REFUSAL"])
    assert judge_context_insufficiency(judge, "q?", "ctx", "42") is True


def test_judge_context_insufficiency_false_on_should_have_answered():
    judge = FakeJudge(["The number 42 is right there in the context.\nVERDICT: SHOULD_HAVE_ANSWERED"])
    assert judge_context_insufficiency(judge, "q?", "ctx", "42") is False


def test_judge_context_insufficiency_fails_safe_to_not_justified_on_malformed_response():
    # No parseable VERDICT: marker at all and the fallback last-line isn't
    # one of the two known values - must not accidentally count as
    # justified (same fail-safe direction as agent/loop.py's
    # _parse_assessment for a malformed SUFFICIENT: response).
    judge = FakeJudge(["I'm not sure how to answer this."])
    assert judge_context_insufficiency(judge, "q?", "ctx", "42") is False


def test_judge_context_insufficiency_hedged_verdict_is_not_justified():
    # находка 3 (claude/status_agent_rezultaty_4_nahodki_kod.md,
    # claude/verifikaciya_qwen_substring_bug.md): a hedged/malformed
    # verdict like "NOT JUSTIFIED_REFUSAL" contains "JUSTIFIED_REFUSAL" as
    # a literal substring and does not contain "SHOULD_HAVE_ANSWERED" - a
    # substring-based check would wrongly return True here (the one place
    # a substring check pointed toward a false success instead of the
    # fail-safe "not justified" direction used everywhere else in this
    # module). Exact-match comparison must return False.
    judge = FakeJudge(["On reflection this is NOT JUSTIFIED_REFUSAL after all.\nVERDICT: NOT JUSTIFIED_REFUSAL"])
    assert judge_context_insufficiency(judge, "q?", "ctx", "42") is False


def test_judge_context_insufficiency_builds_prompt_with_question_context_and_gold():
    judge = FakeJudge(["VERDICT: JUSTIFIED_REFUSAL"])
    judge_context_insufficiency(judge, "What was net income?", "[Document 1]\nsome text", "500")
    assert "What was net income?" in judge.last_prompt
    assert "[Document 1]\nsome text" in judge.last_prompt
    assert "500" in judge.last_prompt


# --- evaluate_agent_success -------------------------------------------------


def test_gold_insufficient_and_answer_insufficient_is_success_no_judge_call():
    judge = FakeJudge([])  # must never be called - popping an empty list raises
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "INSUFFICIENT_CONTEXT", "INSUFFICIENT_CONTEXT")
    assert isinstance(result, SuccessResult)
    assert result.success is True
    assert result.reason == "insufficient_context_matches_gold"
    assert result.judge_scores is None
    assert judge.call_count == 0


def test_gold_insufficient_but_agent_answered_confidently_is_failure_no_judge_call():
    judge = FakeJudge([])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "999", "INSUFFICIENT_CONTEXT")
    assert result.success is False
    assert result.reason == "confidently_answered_unanswerable_question"
    assert judge.call_count == 0


def test_gold_insufficient_check_is_case_and_whitespace_insensitive():
    judge = FakeJudge([])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "insufficient_context", "  Insufficient_Context  ")
    assert result.success is True
    assert judge.call_count == 0


def test_agent_refused_on_answerable_question_and_judge_says_justified_is_success():
    judge = FakeJudge(["VERDICT: JUSTIFIED_REFUSAL"])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "INSUFFICIENT_CONTEXT", "500")
    assert result.success is True
    assert result.reason == "justified_refusal"
    assert result.judge_scores is None
    assert judge.call_count == 1


def test_agent_refused_on_answerable_question_and_judge_says_should_have_answered_is_failure():
    judge = FakeJudge(["VERDICT: SHOULD_HAVE_ANSWERED"])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "INSUFFICIENT_CONTEXT", "500")
    assert result.success is False
    assert result.reason == "unjustified_refusal"
    assert judge.call_count == 1


def test_agent_answered_and_judge_says_correct_is_success():
    # evaluate_answer() runs its own deterministic is_close_v2 check plus
    # one judge call, same as the baseline pipeline (pipeline.evaluation) -
    # this only checks evaluate_agent_success wires that through correctly,
    # not evaluate_answer's own internals (see test_evaluation.py).
    judge = FakeJudge(["VERDICT: CORRECT"])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "500", "500")
    assert result.success is True
    assert result.reason == "correct_answer"
    assert result.judge_scores is not None
    assert result.judge_scores["judge_correct"] is True


def test_agent_answered_and_judge_says_incorrect_is_failure():
    judge = FakeJudge(["VERDICT: INCORRECT"])
    result = evaluate_agent_success(judge, "q1", "question", "ctx", "999", "500")
    assert result.success is False
    assert result.reason == "incorrect_answer"


def test_prompt_version_constant_is_a_non_empty_string():
    # Reproducibility requirement (День 2): a bare version marker must
    # exist and be bumpable independently of pipeline.evaluation's own
    # PROMPT_VERSION - see agent/run_metadata.py.
    assert isinstance(INSUFFICIENCY_JUDGE_PROMPT_VERSION, str) and INSUFFICIENCY_JUDGE_PROMPT_VERSION

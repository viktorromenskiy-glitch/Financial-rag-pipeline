"""Tests for agent/safety.py - the run-level (whole evaluation run, not one
question) safety budget: RunSafetyLimits/RunBudgetTracker/BudgetExceededError
and the BudgetedGenerator/BudgetedJudge wrappers. All offline - fakes stand
in for GeneratorProtocol/JudgeProtocol, no real API keys or network access.
"""

from __future__ import annotations

import pytest

from agent.safety import BudgetedGenerator, BudgetedJudge, BudgetExceededError, RunBudgetTracker, RunSafetyLimits


class FakeGenerator:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


class FakeJudge:
    def __init__(self, response: str = "VERDICT: CORRECT"):
        self.response = response
        self.call_count = 0

    def judge(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


class _FakeClock:
    def __init__(self, step: float = 1.0):
        self.step = step
        self.value = 0.0

    def __call__(self) -> float:
        self.value += self.step
        return self.value


# --- RunSafetyLimits ---------------------------------------------------


def test_run_safety_limits_all_none_by_default():
    limits = RunSafetyLimits()
    assert limits.max_llm_calls is None
    assert limits.max_wall_clock_seconds is None
    assert limits.max_estimated_cost_usd is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_llm_calls": -1},
        {"max_wall_clock_seconds": -0.1},
        {"max_estimated_cost_usd": -5.0},
    ],
)
def test_run_safety_limits_rejects_negative_values(kwargs):
    with pytest.raises(ValueError):
        RunSafetyLimits(**kwargs)


def test_run_safety_limits_accepts_zero():
    # 0 is a valid, deliberate value (e.g. max_llm_calls=0 - a dry run
    # that should never actually call an LLM).
    limits = RunSafetyLimits(max_llm_calls=0, max_wall_clock_seconds=0, max_estimated_cost_usd=0)
    assert limits.max_llm_calls == 0


# --- RunBudgetTracker ----------------------------------------------------


def test_tracker_never_raises_with_no_limits_set():
    tracker = RunBudgetTracker(RunSafetyLimits())
    for _ in range(100):
        tracker.record_llm_call()
    assert tracker.llm_calls == 100


def test_tracker_raises_when_call_count_limit_exceeded():
    tracker = RunBudgetTracker(RunSafetyLimits(max_llm_calls=2))
    tracker.record_llm_call()
    tracker.record_llm_call()
    with pytest.raises(BudgetExceededError):
        tracker.record_llm_call()
    # The call that tripped the limit still counts - reflects what was
    # actually spent, not one call short of it.
    assert tracker.llm_calls == 3


def test_tracker_raises_when_wall_clock_limit_exceeded():
    clock = _FakeClock(step=1.0)
    tracker = RunBudgetTracker(RunSafetyLimits(max_wall_clock_seconds=2.5), clock=clock)
    tracker.record_llm_call()  # elapsed ~1s, ok
    tracker.record_llm_call()  # elapsed ~2s, ok
    with pytest.raises(BudgetExceededError):
        tracker.record_llm_call()  # elapsed ~3s > 2.5s


def test_tracker_raises_when_cost_limit_exceeded():
    tracker = RunBudgetTracker(RunSafetyLimits(max_estimated_cost_usd=0.05), cost_per_llm_call_usd=0.02)
    tracker.record_llm_call()  # $0.02
    tracker.record_llm_call()  # $0.04
    with pytest.raises(BudgetExceededError):
        tracker.record_llm_call()  # $0.06 > $0.05
    assert tracker.estimated_cost_usd == pytest.approx(0.06)


def test_tracker_rejects_negative_cost_per_call():
    with pytest.raises(ValueError):
        RunBudgetTracker(RunSafetyLimits(), cost_per_llm_call_usd=-0.01)


def test_tracker_elapsed_seconds_reflects_injected_clock():
    clock = _FakeClock(step=2.0)
    tracker = RunBudgetTracker(RunSafetyLimits(), clock=clock)
    assert tracker.elapsed_seconds == pytest.approx(2.0)  # one clock() call inside elapsed_seconds
    assert tracker.elapsed_seconds == pytest.approx(4.0)  # advances again on each read


# --- BudgetedGenerator / BudgetedJudge ------------------------------------


def test_budgeted_generator_records_and_delegates():
    inner = FakeGenerator(response="FINAL ANSWER: 42")
    tracker = RunBudgetTracker(RunSafetyLimits())
    budgeted = BudgetedGenerator(inner, tracker)

    result = budgeted.generate("some prompt")

    assert result == "FINAL ANSWER: 42"
    assert inner.call_count == 1
    assert tracker.llm_calls == 1


def test_budgeted_generator_stops_delegating_once_budget_exceeded():
    inner = FakeGenerator()
    tracker = RunBudgetTracker(RunSafetyLimits(max_llm_calls=1))
    budgeted = BudgetedGenerator(inner, tracker)

    budgeted.generate("first")
    with pytest.raises(BudgetExceededError):
        budgeted.generate("second")
    # The real generator is never called for the request that pushed the
    # tracker over budget - the check happens before delegating.
    assert inner.call_count == 1


def test_budgeted_judge_records_and_delegates():
    inner = FakeJudge(response="VERDICT: INCORRECT")
    tracker = RunBudgetTracker(RunSafetyLimits())
    budgeted = BudgetedJudge(inner, tracker)

    result = budgeted.judge("some prompt")

    assert result == "VERDICT: INCORRECT"
    assert inner.call_count == 1
    assert tracker.llm_calls == 1


def test_budgeted_generator_and_judge_share_one_tracker():
    tracker = RunBudgetTracker(RunSafetyLimits(max_llm_calls=3))
    gen = BudgetedGenerator(FakeGenerator(), tracker)
    judge = BudgetedJudge(FakeJudge(), tracker)

    gen.generate("q1 answer")
    judge.judge("q1 judge")
    gen.generate("q2 answer")
    assert tracker.llm_calls == 3
    with pytest.raises(BudgetExceededError):
        judge.judge("q2 judge")

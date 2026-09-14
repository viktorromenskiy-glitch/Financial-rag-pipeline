"""Run-level safety budget for a whole agent evaluation harness run - День 2
of plan_rabot_posle_ekspertizy_agent_profil.md: "Глобальные предохранители:
максимум токенов / wall-clock / стоимость на прогон".

Scope: this tracks the WHOLE evaluation run (e.g. scripts/run_agent_eval.py's
~20-30 questions, both baseline and agent, plus judge calls), not a single
question's loop - that is agent/loop.py's `max_wall_clock_seconds` param
(config.agent.max_wall_clock_seconds), a separate, narrower safety limit
already added there. A single question finishing within its own loop-level
limit says nothing about whether the run as a whole has gone over its
overall budget; both checks are needed, at their own scope.

Token accounting: pipeline.generation.GeneratorProtocol/pipeline.evaluation.
JudgeProtocol only expose `generate(prompt) -> str` / `judge(prompt) -> str`
- no usage/token-count metadata reaches this layer (pipeline/cli.py's real
adapters extract only the text block from the Anthropic response, see
_extract_text there). Rather than plumb real token counts through every
adapter for this one safety feature, this module counts LLM *calls* and
converts to an estimated dollar cost via a configurable per-call price
(config.agent_eval.cost_per_llm_call_usd) - coarser than a real token
count, but exactly the quantity "Зафиксировать заранее бюджет
evaluation-прогона (число LLM-вызовов × стоимость)" (План, День 2) asks
to fix in advance, and a call-count ceiling is itself a meaningful,
directly-enforceable safety limit independent of the cost estimate's
accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from pipeline.evaluation import JudgeProtocol
from pipeline.generation import GeneratorProtocol


class BudgetExceededError(RuntimeError):
    """Raised by RunBudgetTracker.record_llm_call() the moment any
    configured limit is exceeded - the caller (scripts/run_agent_eval.py)
    is expected to catch this, stop issuing new LLM calls, and write out
    whatever has already been checkpointed, rather than let a runaway loop
    keep spending real API budget (see pipeline/common/persist.py's
    "never lose a long paid run" convention - this is the same principle
    applied to *starting* an unbounded spend, not just to not losing one
    already made)."""


@dataclass(frozen=True)
class RunSafetyLimits:
    """None disables the corresponding limit - every field is opt-in, so a
    caller that only wants a wall-clock ceiling (say) isn't forced to also
    guess a call-count or cost ceiling."""

    max_llm_calls: int | None = None
    max_wall_clock_seconds: float | None = None
    max_estimated_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_llm_calls", self.max_llm_calls),
            ("max_wall_clock_seconds", self.max_wall_clock_seconds),
            ("max_estimated_cost_usd", self.max_estimated_cost_usd),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None, got {value}")


class RunBudgetTracker:
    """Call record_llm_call() once per real LLM call (generation,
    assessment, or judging) anywhere in a harness run. Raises
    BudgetExceededError the instant any configured limit is first crossed
    - the call that trips the limit still counts (so `llm_calls`/
    `estimated_cost_usd` on the tracker after the exception accurately
    reflect what was actually spent, not one call short of it).

    `clock`: injectable time source (default time.monotonic), purely for
    deterministic tests - see test_agent_safety.py.
    """

    def __init__(
        self,
        limits: RunSafetyLimits,
        cost_per_llm_call_usd: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if cost_per_llm_call_usd < 0:
            raise ValueError(f"cost_per_llm_call_usd must be >= 0, got {cost_per_llm_call_usd}")
        self.limits = limits
        self.cost_per_llm_call_usd = cost_per_llm_call_usd
        self._clock = clock
        self._start = clock()
        self._llm_calls = 0
        self._estimated_cost_usd = 0.0

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def estimated_cost_usd(self) -> float:
        return self._estimated_cost_usd

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._start

    def record_llm_call(self) -> None:
        self._llm_calls += 1
        self._estimated_cost_usd += self.cost_per_llm_call_usd
        self._check()

    def _check(self) -> None:
        limits = self.limits
        if limits.max_llm_calls is not None and self._llm_calls > limits.max_llm_calls:
            raise BudgetExceededError(
                f"LLM call budget exceeded: {self._llm_calls} calls > limit {limits.max_llm_calls}"
            )
        elapsed = self.elapsed_seconds
        if limits.max_wall_clock_seconds is not None and elapsed > limits.max_wall_clock_seconds:
            raise BudgetExceededError(
                f"wall-clock budget exceeded: {elapsed:.1f}s > limit {limits.max_wall_clock_seconds}s"
            )
        if limits.max_estimated_cost_usd is not None and self._estimated_cost_usd > limits.max_estimated_cost_usd:
            raise BudgetExceededError(
                f"estimated cost budget exceeded: ${self._estimated_cost_usd:.4f} > "
                f"limit ${limits.max_estimated_cost_usd:.4f}"
            )


class BudgetedGenerator:
    """Wraps any GeneratorProtocol-shaped client (pipeline.cli.ClaudeGenerator,
    or agent/loop.py's ClaudeEvidenceAssessor's inner client) so every
    generate() call is recorded on a shared RunBudgetTracker before the
    real call is made - one instance can wrap the generator used for
    baseline answers, the agent's final answers, and the agent's evidence
    assessments, all counting against the same run-level budget."""

    def __init__(self, inner: GeneratorProtocol, tracker: RunBudgetTracker):
        self.inner = inner
        self.tracker = tracker

    def generate(self, prompt: str) -> str:
        self.tracker.record_llm_call()
        return self.inner.generate(prompt)


class BudgetedJudge:
    """Same wrapping as BudgetedGenerator, for JudgeProtocol (pipeline.cli.
    ClaudeJudge / agent/success.py's insufficiency-judge calls, which share
    the JudgeProtocol shape) - a separate class rather than a shared base
    only because JudgeProtocol's method is named judge(), not generate()."""

    def __init__(self, inner: JudgeProtocol, tracker: RunBudgetTracker):
        self.inner = inner
        self.tracker = tracker

    def judge(self, prompt: str) -> str:
        self.tracker.record_llm_call()
        return self.inner.judge(prompt)

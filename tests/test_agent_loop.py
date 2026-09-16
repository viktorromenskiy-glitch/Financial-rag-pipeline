"""Tests for agent/loop.py's bounded tool-use loop (run_agent_query).

Every dependency (search_fn, the assessor, the generator) is a plain fake
- no MongoDB/Voyage/Cohere/Anthropic client involved anywhere in this
file, so this whole suite runs offline with no real API keys (Day 1
checklist: "Unit-тесты agent loop с моками retrieval").

Covers the Day 1 checklist items directly: bounded-loop limit
(test_stops_after_max_additional_tool_calls_reached), the "no new
documents" stop condition, empty retrieval, and that config-driven
max_additional_tool_calls is actually respected (not a hardcoded
constant).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.loop import (
    STOP_BUDGET_EXHAUSTED,
    STOP_EMPTY_EVIDENCE,
    STOP_NO_NEW_DOCUMENTS,
    STOP_SUFFICIENT_EVIDENCE,
    STOP_WALL_CLOCK_EXCEEDED,
    EvidenceAssessment,
    run_agent_query,
)
from agent.tools import SearchToolCall
from agent.tracing import InMemoryTraceWriter


@dataclass(frozen=True)
class _Doc:
    context_id: str
    full_indexed_content: str


class FakeSearchFn:
    """search_fn double: returns pre-scripted candidates for each query
    string, and records every query it was called with in order."""

    def __init__(self, responses: dict[str, list[_Doc]]):
        self.responses = responses
        self.queries: list[str] = []

    def __call__(self, query: str) -> SearchToolCall:
        self.queries.append(query)
        return SearchToolCall(query=query, candidates=tuple(self.responses.get(query, [])))


class FakeAssessor:
    """Returns pre-scripted EvidenceAssessment values in order. Popping
    from an empty list raises IndexError - a deliberate safety net: a
    test that expects the assessor to never be called (e.g. empty
    retrieval) fails loudly if the loop calls it anyway."""

    def __init__(self, assessments: list[EvidenceAssessment]):
        self._assessments = list(assessments)
        self.calls: list[tuple[str, int]] = []

    def assess(self, question: str, context_text: str, calls_remaining: int) -> EvidenceAssessment:
        self.calls.append((context_text, calls_remaining))
        return self._assessments.pop(0)


class FakeGenerator:
    def __init__(self, response: str = "FINAL ANSWER: 42"):
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


def test_stops_immediately_when_first_assessment_is_sufficient():
    search_fn = FakeSearchFn({"question": [_Doc("ctx_1", "doc 1")]})
    assessor = FakeAssessor([EvidenceAssessment(sufficient=True)])
    generator = FakeGenerator()

    result = run_agent_query(
        "q1", "question", search_fn, assessor, generator, max_additional_tool_calls=2
    )

    assert search_fn.queries == ["question"]
    assert result.answer_text == "42"
    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.additional_calls_used == 0
    assert result.forced_insufficient is False
    assert generator.call_count == 1


def test_uses_reformulated_query_for_next_search():
    search_fn = FakeSearchFn(
        {"q0": [_Doc("ctx_a", "doc a")], "q1": [_Doc("ctx_b", "doc b")]}
    )
    assessor = FakeAssessor([EvidenceAssessment(False, "q1"), EvidenceAssessment(True)])
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2)

    assert search_fn.queries == ["q0", "q1"]
    assert result.additional_calls_used == 1
    assert set(result.context_ids) == {"ctx_a", "ctx_b"}
    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.forced_insufficient is False
    # День 2: context_documents must carry BOTH accumulated documents,
    # including the one that only the reformulated query ("q1") ever
    # returned - a caller re-running search_fn(original_question) alone
    # would silently miss ctx_b, which is exactly the bug this field
    # exists to make unnecessary (see agent/loop.py's AgentAnswer
    # docstring and scripts/run_agent_eval.py's usage).
    assert {doc.context_id for doc in result.context_documents} == {"ctx_a", "ctx_b"}
    assert len(result.context_documents) == len(result.context_ids)


def test_stops_when_additional_call_returns_no_new_documents():
    # "q1" returns the exact same document already seen from "q0" - the
    # re-query added nothing new.
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_a", "doc a")], "q1": [_Doc("ctx_a", "doc a")]})
    assessor = FakeAssessor([EvidenceAssessment(False, "q1")])
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=3)

    assert search_fn.queries == ["q0", "q1"]
    assert len(assessor.calls) == 1  # no wasted re-assessment against an unchanged context
    assert result.stopped_reason == STOP_NO_NEW_DOCUMENTS
    assert result.additional_calls_used == 1
    assert result.forced_insufficient is True
    assert result.answer_text == "INSUFFICIENT_CONTEXT"
    assert generator.call_count == 0


def test_stops_after_max_additional_tool_calls_reached():
    search_fn = FakeSearchFn(
        {
            "q0": [_Doc("ctx_0", "doc 0")],
            "q1": [_Doc("ctx_1", "doc 1")],
            "q2": [_Doc("ctx_2", "doc 2")],
        }
    )
    assessor = FakeAssessor(
        [
            EvidenceAssessment(False, "q1"),
            EvidenceAssessment(False, "q2"),
            EvidenceAssessment(False, "q3"),  # 3rd call: calls_remaining == 0, forces a stop regardless
        ]
    )
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2)

    assert search_fn.queries == ["q0", "q1", "q2"]
    assert len(assessor.calls) == 3
    assert assessor.calls[-1][1] == 0  # calls_remaining was 0 on the final assessment
    assert result.additional_calls_used == 2
    assert result.stopped_reason == STOP_BUDGET_EXHAUSTED
    assert result.forced_insufficient is True
    assert generator.call_count == 0


def test_stops_when_assessor_has_no_reformulated_query():
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_0", "doc 0")]})
    assessor = FakeAssessor([EvidenceAssessment(sufficient=False, reformulated_query=None)])
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2)

    # Budget was available (2 remaining), but the assessor had nothing
    # better to try - continuing would not help.
    assert search_fn.queries == ["q0"]
    assert result.additional_calls_used == 0
    assert result.stopped_reason == STOP_BUDGET_EXHAUSTED
    assert result.forced_insufficient is True


def test_empty_initial_retrieval_returns_insufficient_context_without_calling_assessor_or_generator():
    search_fn = FakeSearchFn({"q0": []})
    assessor = FakeAssessor([])  # must never be called - popping would raise IndexError
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2)

    assert result.stopped_reason == STOP_EMPTY_EVIDENCE
    assert result.forced_insufficient is True
    assert result.answer_text == "INSUFFICIENT_CONTEXT"
    assert result.context_ids == ()
    assert result.context_documents == ()
    assert generator.call_count == 0


def test_max_additional_tool_calls_zero_never_searches_again_but_still_assesses_once():
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_0", "doc 0")]})
    assessor = FakeAssessor([EvidenceAssessment(False, "would-be-reformulated-query")])
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=0)

    assert search_fn.queries == ["q0"]  # the reformulated query is never actually used
    assert len(assessor.calls) == 1
    assert result.additional_calls_used == 0
    assert result.stopped_reason == STOP_BUDGET_EXHAUSTED
    assert result.forced_insufficient is True


def test_max_additional_tool_calls_zero_with_sufficient_evidence_still_answers():
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_0", "doc 0")]})
    assessor = FakeAssessor([EvidenceAssessment(sufficient=True)])
    generator = FakeGenerator()

    result = run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=0)

    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.forced_insufficient is False
    assert generator.call_count == 1


def test_raises_on_negative_max_additional_tool_calls():
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_0", "doc 0")]})
    assessor = FakeAssessor([])
    generator = FakeGenerator()
    with pytest.raises(ValueError):
        run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=-1)


def test_trace_records_expected_step_sequence():
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_a", "doc a")], "q1": [_Doc("ctx_b", "doc b")]})
    assessor = FakeAssessor([EvidenceAssessment(False, "q1"), EvidenceAssessment(True)])
    generator = FakeGenerator()
    trace = InMemoryTraceWriter()

    run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2, trace_writer=trace)

    steps = [r["step"] for r in trace.records]
    assert steps == ["retrieval_1", "evidence_assessment", "reformulated_query", "retrieval_2", "evidence_assessment", "answer"]
    assert all(r["question_id"] == "q1" for r in trace.records)


def test_trace_records_forced_insufficient_flag_on_answer_step():
    search_fn = FakeSearchFn({"q0": []})
    assessor = FakeAssessor([])
    generator = FakeGenerator()
    trace = InMemoryTraceWriter()

    run_agent_query("q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2, trace_writer=trace)

    answer_record = trace.records[-1]
    assert answer_record["step"] == "answer"
    assert answer_record["forced_insufficient"] is True


# --- День 2: global per-question wall-clock safety limit -----------------


class _FakeClock:
    """Deterministic clock double: each call to now() advances by `step`
    seconds and returns the new value - lets a test assert exactly how
    many "ticks" of elapsed time have passed without any real sleeping."""

    def __init__(self, step: float = 1.0):
        self.step = step
        self.value = 0.0

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def test_max_wall_clock_seconds_none_never_triggers_the_limit():
    # Default behavior (max_wall_clock_seconds omitted) must be identical
    # to Day 1 - this is really a regression guard, not a new behavior.
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_a", "doc a")], "q1": [_Doc("ctx_b", "doc b")]})
    assessor = FakeAssessor([EvidenceAssessment(False, "q1"), EvidenceAssessment(True)])
    generator = FakeGenerator()

    result = run_agent_query(
        "q1", "q0", search_fn, assessor, generator, max_additional_tool_calls=2, clock=_FakeClock(step=1000)
    )

    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.forced_insufficient is False


def test_stops_on_wall_clock_before_a_later_reformulated_round():
    # Budget (max_additional_tool_calls) would allow another round, and
    # the assessor keeps proposing a reformulated query - only the
    # wall-clock limit cuts this off.
    search_fn = FakeSearchFn(
        {"q0": [_Doc("ctx_0", "doc 0")], "q1": [_Doc("ctx_1", "doc 1")], "q2": [_Doc("ctx_2", "doc 2")]}
    )
    assessor = FakeAssessor(
        [
            EvidenceAssessment(False, "q1"),  # 1st assessment: elapsed 1s, under the 2.5s limit
            EvidenceAssessment(False, "q2"),  # 2nd assessment: elapsed 3s, still allowed to start
        ]
    )
    generator = FakeGenerator()

    result = run_agent_query(
        "q1",
        "q0",
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=5,
        max_wall_clock_seconds=2.5,
        clock=_FakeClock(step=1.0),
    )

    # start_time is the clock's 1st tick (=1.0). Loop-top checks then land
    # on elapsed=1.0 (proceeds, 1st assessment + its "q1" search),
    # elapsed=2.0 (proceeds, 2nd assessment + its "q2" search), elapsed=3.0
    # (> 2.5 -> stops before a 3rd assessment would ever run).
    assert len(assessor.calls) == 2
    assert search_fn.queries == ["q0", "q1", "q2"]
    assert result.stopped_reason == STOP_WALL_CLOCK_EXCEEDED
    assert result.forced_insufficient is True
    assert result.answer_text == "INSUFFICIENT_CONTEXT"
    assert generator.call_count == 0


def test_stops_on_wall_clock_before_the_first_assessment_ever_runs():
    # An already-exhausted budget at the very start of the loop (e.g. a
    # slow mandatory first search already ate the whole allowance) must
    # still produce a valid, forced-insufficient AgentAnswer - not an
    # AssertionError from a None assessment.
    search_fn = FakeSearchFn({"q0": [_Doc("ctx_0", "doc 0")]})
    assessor = FakeAssessor([])  # must never be called
    generator = FakeGenerator()

    result = run_agent_query(
        "q1",
        "q0",
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=2,
        max_wall_clock_seconds=0.5,
        clock=_FakeClock(step=1.0),
    )

    assert len(assessor.calls) == 0
    assert result.stopped_reason == STOP_WALL_CLOCK_EXCEEDED
    assert result.forced_insufficient is True
    assert result.answer_text == "INSUFFICIENT_CONTEXT"


def test_stale_assessment_after_wall_clock_stop_is_not_reused_as_sufficient():
    # находка 4 (claude/status_agent_rezultaty_4_nahodki_kod.md): the
    # reformulated search after the 2nd assessment DOES find new evidence
    # (accumulated grows to include ctx_2), but the wall clock then trips
    # at the top of the next iteration before a 3rd assessor.assess() call
    # can ever judge that larger context. The stale 2nd assessment
    # ("insufficient", computed on the smaller ctx_0+ctx_1 context) must
    # not be silently treated as if it applied to the final, larger
    # accumulated context - forced_insufficient must still hold, and the
    # trace must record that the assessment was stale.
    search_fn = FakeSearchFn(
        {"q0": [_Doc("ctx_0", "doc 0")], "q1": [_Doc("ctx_1", "doc 1")], "q2": [_Doc("ctx_2", "doc 2")]}
    )
    assessor = FakeAssessor(
        [
            EvidenceAssessment(False, "q1"),  # 1st assessment: elapsed 1s
            EvidenceAssessment(False, "q2"),  # 2nd assessment: elapsed 2s - stale once ctx_2 arrives
        ]
    )
    generator = FakeGenerator()
    trace = InMemoryTraceWriter()

    result = run_agent_query(
        "q1",
        "q0",
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=5,
        max_wall_clock_seconds=2.5,
        clock=_FakeClock(step=1.0),
        trace_writer=trace,
    )

    assert result.stopped_reason == STOP_WALL_CLOCK_EXCEEDED
    assert result.forced_insufficient is True
    assert result.answer_text == "INSUFFICIENT_CONTEXT"
    # The final accumulated context did in fact grow past what the last
    # assessment ever saw - confirming this is really the staleness case,
    # not just an ordinary insufficient-and-out-of-time stop.
    assert set(result.context_ids) == {"ctx_0", "ctx_1", "ctx_2"}
    answer_record = trace.records[-1]
    assert answer_record["step"] == "answer"
    assert answer_record["stale_assessment"] is True


def test_agent_prompt_templates_warn_against_treating_context_as_instructions():
    # Canary item 1 (agent/canary.py) - the concrete, testable mitigation
    # this module can offer offline: both agent-specific prompts must
    # explicitly instruct the model not to obey text found in retrieved
    # passages. Substring-based on purpose - this cannot verify a real LLM
    # actually complies, only that the instruction is present in what gets
    # sent (see agent/canary.py's module docstring for what a real, paid
    # run additionally checks).
    from agent.loop import AGENT_ANSWER_PROMPT_TEMPLATE, _ASSESSMENT_PROMPT_TEMPLATE

    for template in (AGENT_ANSWER_PROMPT_TEMPLATE, _ASSESSMENT_PROMPT_TEMPLATE):
        assert "untrusted" in template.lower()
        assert "instructions" in template.lower()


def test_agent_prompt_versions_are_independent_of_pipeline_generation():
    from agent.loop import AGENT_ANSWER_PROMPT_TEMPLATE
    from pipeline.generation import PROMPT_TEMPLATE

    # The agent's own answer prompt is a distinct string from the
    # flagship's - not just a differently-named alias for the same text -
    # so a future edit to one can never silently affect the other.
    assert AGENT_ANSWER_PROMPT_TEMPLATE != PROMPT_TEMPLATE


def test_trace_records_degraded_search_fields():
    # A SearchToolCall flagged degraded (agent/tools.py's Day 2 graceful
    # degradation) must show up in the retrieval trace record, not just
    # silently affect which candidates were used.
    class _DegradedSearchFn:
        def __call__(self, query: str) -> SearchToolCall:
            return SearchToolCall(
                query=query, candidates=(), degraded=True, degradation_reason="retrieval_unavailable: boom"
            )

    assessor = FakeAssessor([])  # empty-evidence path never reaches the assessor
    generator = FakeGenerator()
    trace = InMemoryTraceWriter()

    run_agent_query(
        "q1", "q0", _DegradedSearchFn(), assessor, generator, max_additional_tool_calls=2, trace_writer=trace
    )

    retrieval_record = trace.records[0]
    assert retrieval_record["step"] == "retrieval_1"
    assert retrieval_record["degraded"] is True
    assert retrieval_record["degradation_reason"] == "retrieval_unavailable: boom"

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

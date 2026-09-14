"""Tests for agent/debug_view.py - the Day 3 human-readable trace dump.

Uses hand-built trace records matching exactly the field names
agent/loop.py's `_trace`/`_trace_retrieval` closures actually emit (see
that module), not a separate invented schema - so a real
agent_trace.jsonl line (or an InMemoryTraceWriter.records list from
test_agent_loop.py) renders correctly, not just an idealized fixture.
"""

from __future__ import annotations

from agent.debug_view import (
    format_step,
    group_by_question,
    render_question_trace,
    render_trace_dump,
)


def test_group_by_question_preserves_order_within_and_across_questions():
    records = [
        {"question_id": "q1", "step": "retrieval_1"},
        {"question_id": "q2", "step": "retrieval_1"},
        {"question_id": "q1", "step": "answer"},
        {"question_id": "q2", "step": "answer"},
    ]
    grouped = group_by_question(records)
    assert list(grouped.keys()) == ["q1", "q2"]
    assert [r["step"] for r in grouped["q1"]] == ["retrieval_1", "answer"]
    assert [r["step"] for r in grouped["q2"]] == ["retrieval_1", "answer"]


def test_group_by_question_skips_records_without_question_id():
    records = [{"step": "answer"}, {"question_id": "q1", "step": "answer"}]
    grouped = group_by_question(records)
    assert list(grouped.keys()) == ["q1"]


def test_format_step_retrieval_1_shows_query_and_context_ids():
    record = {
        "question_id": "q1",
        "step": "retrieval_1",
        "query": "What was revenue?",
        "context_ids": ["doc_2", "doc_1"],
        "degraded": False,
        "degradation_reason": None,
    }
    rendered = format_step(record, 1)
    assert "[1]" in rendered
    assert "search_documents(query='What was revenue?')" in rendered
    assert "doc_1" in rendered and "doc_2" in rendered
    assert "DEGRADED" not in rendered


def test_format_step_retrieval_marks_degraded_with_reason():
    record = {
        "question_id": "q1",
        "step": "retrieval_1",
        "query": "q",
        "context_ids": [],
        "degraded": True,
        "degradation_reason": "retrieval_unavailable: ConnectionError: boom",
    }
    rendered = format_step(record, 1)
    assert "DEGRADED" in rendered
    assert "retrieval_unavailable: ConnectionError: boom" in rendered


def test_format_step_later_retrieval_shows_new_context_ids():
    record = {
        "question_id": "q1",
        "step": "retrieval_2",
        "query": "reformulated",
        "context_ids": ["doc_1", "doc_2", "doc_3"],
        "new_context_ids": ["doc_3"],
        "degraded": False,
        "degradation_reason": None,
    }
    rendered = format_step(record, 3)
    assert "1 new" in rendered
    assert "['doc_3']" in rendered


def test_format_step_evidence_assessment_sufficient():
    record = {
        "question_id": "q1",
        "step": "evidence_assessment",
        "sufficient": True,
        "reformulated_query": None,
        "calls_remaining": 1,
    }
    rendered = format_step(record, 2)
    assert "sufficient" in rendered
    assert "NOT sufficient" not in rendered
    assert "calls_remaining=1" in rendered


def test_format_step_evidence_assessment_insufficient_shows_reformulated_query():
    record = {
        "question_id": "q1",
        "step": "evidence_assessment",
        "sufficient": False,
        "reformulated_query": "ACME 2023 segment revenue",
        "calls_remaining": 2,
    }
    rendered = format_step(record, 2)
    assert "NOT sufficient" in rendered
    assert "ACME 2023 segment revenue" in rendered


def test_format_step_reformulated_query():
    record = {"question_id": "q1", "step": "reformulated_query", "query": "new query", "call_number": 2}
    rendered = format_step(record, 3)
    assert "call #2" in rendered
    assert "new query" in rendered


def test_format_step_answer_plain():
    record = {"question_id": "q1", "step": "answer", "answer_text": "77143", "forced_insufficient": False}
    rendered = format_step(record, 5)
    assert "ANSWER" in rendered
    assert "77143" in rendered
    assert "forced" not in rendered


def test_format_step_answer_forced_insufficient_is_flagged():
    record = {
        "question_id": "q1",
        "step": "answer",
        "answer_text": "INSUFFICIENT_CONTEXT",
        "forced_insufficient": True,
    }
    rendered = format_step(record, 4)
    assert "forced - give-up policy" in rendered


def test_format_step_unknown_step_falls_back_without_raising():
    record = {"question_id": "q1", "step": "some_future_step", "foo": "bar"}
    rendered = format_step(record, 1)
    assert "some_future_step" in rendered
    assert "foo" in rendered


def test_render_question_trace_with_no_records():
    assert "no trace records" in render_question_trace("q1", [])


def test_render_question_trace_includes_header_and_numbered_steps():
    records = [
        {"question_id": "q1", "step": "retrieval_1", "query": "q", "context_ids": ["d1"], "degraded": False},
        {"question_id": "q1", "step": "answer", "answer_text": "42", "forced_insufficient": False},
    ]
    rendered = render_question_trace("q1", records)
    assert "=== Question q1 ===" in rendered
    assert "[1]" in rendered and "[2]" in rendered


def test_render_trace_dump_empty_input():
    assert render_trace_dump([]) == "(no trace records)"


def test_render_trace_dump_multiple_questions_in_first_seen_order():
    records = [
        {"question_id": "q2", "step": "answer", "answer_text": "a", "forced_insufficient": False},
        {"question_id": "q1", "step": "answer", "answer_text": "b", "forced_insufficient": False},
    ]
    rendered = render_trace_dump(records)
    q2_pos = rendered.index("Question q2")
    q1_pos = rendered.index("Question q1")
    assert q2_pos < q1_pos  # first-seen order, not sorted order


def test_render_trace_dump_full_realistic_run(tmp_path):
    """End-to-end-ish: a full question trace shaped exactly like what
    agent/loop.py's run_agent_query would append to an InMemoryTraceWriter
    for a two-round run (initial retrieval -> insufficient -> reformulated
    retrieval -> sufficient -> answer)."""
    records = [
        {"question_id": "q1", "step": "retrieval_1", "query": "orig", "context_ids": ["d1"], "degraded": False, "degradation_reason": None},
        {"question_id": "q1", "step": "evidence_assessment", "sufficient": False, "reformulated_query": "better query", "calls_remaining": 2},
        {"question_id": "q1", "step": "reformulated_query", "query": "better query", "call_number": 1},
        {"question_id": "q1", "step": "retrieval_2", "query": "better query", "context_ids": ["d1", "d2"], "new_context_ids": ["d2"], "degraded": False, "degradation_reason": None},
        {"question_id": "q1", "step": "evidence_assessment", "sufficient": True, "reformulated_query": None, "calls_remaining": 1},
        {"question_id": "q1", "step": "answer", "answer_text": "42", "forced_insufficient": False},
    ]
    rendered = render_trace_dump(records)
    assert rendered.count("[") >= 6  # every step numbered
    assert "better query" in rendered
    assert "ANSWER" in rendered and "42" in rendered

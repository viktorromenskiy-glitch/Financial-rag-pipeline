"""Tests for agent/demonstration.py's pre-registered demonstration-case
selection rule - see that module's docstring for the full rule text.

Fixtures use the exact record shape scripts/run_agent_eval.py's
_load_jsonl_checkpoint produces: {question_id: {..., "success": bool}}
for agent_records and {question_id: {..., "judge_correct": bool}} for
baseline_records.
"""

from __future__ import annotations

import pytest

from agent.demonstration import (
    ROLE_AGENT_DID_NOT_HELP,
    ROLE_AGENT_HELPED,
    ROLE_AGENT_HURT,
    select_demonstration_cases,
)


def _baseline(**overrides: bool) -> dict[str, dict]:
    """Builds {qid: {"judge_correct": bool}} from qid=bool kwargs."""
    return {qid: {"judge_correct": correct} for qid, correct in overrides.items()}


def _agent(**overrides: bool) -> dict[str, dict]:
    return {qid: {"success": correct} for qid, correct in overrides.items()}


def test_raises_on_no_overlap_at_all():
    baseline = _baseline(q1=True)
    agent = _agent(q2=True)
    with pytest.raises(ValueError, match="No question_id"):
        select_demonstration_cases(baseline, agent)


def test_picks_lowest_question_id_from_a_only_as_positive_case():
    baseline = _baseline(q3=False, q1=False, q2=True)
    agent = _agent(q3=True, q1=True, q2=True)
    # q3, q1: agent right, baseline wrong -> a_only = {q1, q3}; lowest is q1
    cases = select_demonstration_cases(baseline, agent)
    positive = next(c for c in cases if c.role == ROLE_AGENT_HELPED)
    assert positive.question_id == "q1"


def test_picks_lowest_question_id_from_b_only_as_required_negative_case():
    baseline = _baseline(q1=True, q5=True, q2=False)
    agent = _agent(q1=False, q5=False, q2=False)
    # q1, q5: baseline right, agent wrong -> b_only = {q1, q5}; lowest is q1
    cases = select_demonstration_cases(baseline, agent)
    negative = next(c for c in cases if c.role == ROLE_AGENT_HURT)
    assert negative.question_id == "q1"


def test_falls_back_to_both_wrong_when_b_only_is_empty():
    baseline = _baseline(q1=True, q2=False, q9=False)
    agent = _agent(q1=True, q2=True, q9=False)
    # b_only empty (agent never loses to baseline); both_wrong = {q9}
    cases = select_demonstration_cases(baseline, agent)
    negative = next(c for c in cases if c.role == ROLE_AGENT_DID_NOT_HELP)
    assert negative.question_id == "q9"
    assert "not a case where the agent was strictly worse" in negative.note


def test_omits_positive_case_with_explicit_note_when_a_only_is_empty():
    baseline = _baseline(q1=True, q2=False)
    agent = _agent(q1=True, q2=False)
    # No question where agent is right and baseline is wrong.
    cases = select_demonstration_cases(baseline, agent)
    positive = next(c for c in cases if c.role == ROLE_AGENT_HELPED)
    assert positive.question_id == ""
    assert "OMITTED" in positive.note


def test_omits_negative_case_with_explicit_note_when_agent_never_worse_or_tied_wrong():
    baseline = _baseline(q1=False, q2=True)
    agent = _agent(q1=True, q2=True)
    # a_only = {q1}; b_only empty; both_wrong empty (agent got everything
    # baseline got, plus more) - no negative case exists in this data.
    cases = select_demonstration_cases(baseline, agent)
    negative = next(c for c in cases if c.role == ROLE_AGENT_HURT)
    assert negative.question_id == ""
    assert "OMITTED" in negative.note
    assert "matched or beat the baseline on every matched question" in negative.note


def test_never_backfills_negative_case_with_a_both_correct_question():
    """Regression guard for the anti-cherry-picking requirement: even
    though q3 (both correct) exists and could superficially pad the
    result to "two real cases", it must never be used for the negative
    slot - only b_only or both_wrong may be."""
    baseline = _baseline(q1=False, q3=True)
    agent = _agent(q1=True, q3=True)
    cases = select_demonstration_cases(baseline, agent)
    negative = next(c for c in cases if c.role == ROLE_AGENT_HURT)
    assert negative.question_id != "q3"
    assert negative.question_id == ""


def test_only_matched_question_ids_are_ever_considered():
    baseline = _baseline(q1=False, q_unmatched_baseline_only=True)
    agent = _agent(q1=True, q_unmatched_agent_only=False)
    cases = select_demonstration_cases(baseline, agent)
    positive = next(c for c in cases if c.role == ROLE_AGENT_HELPED)
    assert positive.question_id == "q1"


def test_custom_field_names_are_respected():
    baseline = {"q1": {"was_correct": False}}
    agent = {"q1": {"is_success": True}}
    cases = select_demonstration_cases(
        baseline, agent, agent_success_field="is_success", baseline_success_field="was_correct"
    )
    positive = next(c for c in cases if c.role == ROLE_AGENT_HELPED)
    assert positive.question_id == "q1"


def test_result_always_has_exactly_two_case_slots():
    baseline = _baseline(q1=True)
    agent = _agent(q1=True)
    cases = select_demonstration_cases(baseline, agent)
    assert len(cases) == 2
    assert {c.role for c in cases} == {ROLE_AGENT_HELPED, ROLE_AGENT_HURT}

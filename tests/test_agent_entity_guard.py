"""Tests for the deictic/entity guard (agent/loop.py's
_deictic_entity_guard_should_block and _deictic_entity_guard_has_named_entity)
and its wiring into run_agent_query via enable_deictic_entity_guard.

See claude/itog_ekspertizy_cuad_overrefusal_fix.md, "Что осталось сделать",
item 1, for the design rationale this guard implements: a question that
combines a deictic reference to "this/that/the contract/agreement/document/
filing" with nothing else in the text that could identify WHICH document is
meant, out of a pool of several, is exactly the pattern that made the CUAD
smoke test's agent answer confidently from the wrong document instead of
declining. Blocking it here - before the mandatory first search - costs
nothing on ordinary questions, which is what the false-positive checks below
exist to demonstrate on real project data, not synthetic examples.

Every check here runs offline against data files already committed to this
repository (data/cuad_smoke/cuad_smoke_questions.json,
data/financial_entity_ambiguity/fixture.json,
data/t2-ragbench/eval_subset_250.parquet) - no paid API calls, matching this
project's "test locally before Colab" discipline for a change with zero
LLM/network dependency of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.loop import (
    INSUFFICIENT_CONTEXT_MARKER,
    STOP_DEICTIC_ENTITY_GUARD,
    STOP_SUFFICIENT_EVIDENCE,
    EvidenceAssessment,
    _deictic_entity_guard_has_named_entity,
    _deictic_entity_guard_should_block,
    run_agent_query,
)
from agent.tools import SearchToolCall

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Unit tests: the pure text-matching functions, no run_agent_query involved ---


def _load_cuad_smoke_questions() -> list[str]:
    path = REPO_ROOT / "data" / "cuad_smoke" / "cuad_smoke_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [q["question"] for q in data["questions"]]


def _load_financial_fixture_questions() -> list[str]:
    path = REPO_ROOT / "data" / "financial_entity_ambiguity" / "fixture.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    # question_original is the real, non-anonymized question text - the
    # harder false-positive case, since it always names a real company.
    return [item["question_original"] for item in items]


def _load_t2_ragbench_eval_subset_questions() -> list[str]:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    path = REPO_ROOT / "data" / "t2-ragbench" / "eval_subset_250.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    df = pd.read_parquet(path)
    return df["question"].tolist()


def test_blocks_all_real_cuad_smoke_questions():
    """The exact failure mode this guard exists to catch: CUAD's questions
    all say "this contract" and never name which of the 4 indexed
    documents they mean - see the shared "Highlight the parts... of this
    contract..." template in data/cuad_smoke/cuad_smoke_questions.json."""
    questions = _load_cuad_smoke_questions()
    assert len(questions) == 5  # sanity check on the fixture itself
    for question in questions:
        assert _deictic_entity_guard_should_block(question), (
            f"guard failed to block a real CUAD question it was designed for: {question!r}"
        )


def test_no_false_positives_on_real_financial_fixture_questions():
    """These 28 real financial questions all name a real company
    (question_original, not question_anonymized) - none of them should
    ever be blocked, since naming the company is exactly the information
    the guard checks for."""
    questions = _load_financial_fixture_questions()
    assert len(questions) == 28
    false_positives = [q for q in questions if _deictic_entity_guard_should_block(q)]
    assert false_positives == []


def test_no_false_positives_on_full_t2_ragbench_eval_subset():
    """Broader false-positive check across the full 250-question real
    evaluation subset, including the ~14 questions known to contain the
    literal phrase "the company" (see
    claude/nahodka_deiktichnost_round3_dlya_ekspertov.md) - most of these
    still name the company elsewhere in the question, which is exactly
    why "company"/"the company" was deliberately left out of the guard's
    marker list (_DEICTIC_MARKER_RE only matches contract/agreement/
    document/filing, never "company")."""
    questions = _load_t2_ragbench_eval_subset_questions()
    assert len(questions) == 250
    false_positives = [q for q in questions if _deictic_entity_guard_should_block(q)]
    assert false_positives == []


def test_deictic_marker_alone_without_entity_blocks():
    assert _deictic_entity_guard_should_block(
        'Highlight the parts (if any) of this contract related to "Termination" that should be reviewed by a lawyer.'
    )


def test_named_entity_present_prevents_block_even_with_deictic_marker():
    assert not _deictic_entity_guard_should_block(
        "What does this agreement between Aon and Duke Realty Corporation say about termination?"
    )


def test_no_deictic_marker_never_blocks_regardless_of_entities():
    assert not _deictic_entity_guard_should_block(
        "What was the revenue difference between 2010 and 2009?"
    )


def test_quoted_clause_labels_are_not_mistaken_for_entities():
    # CUAD-style quoted category labels ("Document Name", "Governing Law")
    # must not themselves satisfy the named-entity check - they're quoted
    # boilerplate, not evidence of which document is meant.
    assert not _deictic_entity_guard_has_named_entity(
        'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer.'
    )


def test_second_sentence_leading_word_is_not_mistaken_for_entity():
    # Regression test for the bug found during offline validation: "The"
    # opening the second sentence ("Details: The name of the contract")
    # was originally miscounted as a named entity because only the first
    # word of the WHOLE question was excluded, not the first word of each
    # sentence.
    assert not _deictic_entity_guard_has_named_entity(
        'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by '
        "a lawyer. Details: The name of the contract"
    )


def test_capitalized_month_or_form_word_is_not_mistaken_for_entity():
    assert not _deictic_entity_guard_has_named_entity("Does this filing mention December or Form 10-K?")


def test_curly_quoted_span_is_not_mistaken_for_unquoted_entity():
    """Regression test for a gap found during round-1 external code review:
    a quoted entity using curly/typographic quotes (common when a question
    is copy-pasted from a word processor) was not stripped by the original
    ASCII-only _QUOTED_SPAN_RE, so it would have leaked through as if it
    were unquoted prose. Mirrors test_quoted_clause_labels_are_not_mistaken_for_entities
    above, but with curly double quotes instead of straight ones."""
    assert not _deictic_entity_guard_has_named_entity(
        "Highlight the parts (if any) of this contract related to “Governing Law” that should be reviewed by a lawyer."
    )


# --- Documented known limitations (round-1 external code review) ---
#
# These are NOT regression tests for a fix - they document, with a passing
# assertion, false-negative gaps in the entity heuristic that are known and
# accepted for now (see _deictic_entity_guard_has_named_entity's docstring,
# "Known false-negative gaps"), because none of them occur in this
# project's actual validated data. If any of these ever starts returning
# True (heuristic improved) or the guard is enabled on a corpus where one
# of these patterns is common, revisit the docstring note alongside this
# test.


def test_known_limitation_digit_leading_company_name_not_detected():
    assert not _deictic_entity_guard_has_named_entity("What does this agreement with 3M say about termination?")


def test_known_limitation_lowercase_leading_company_name_not_detected():
    assert not _deictic_entity_guard_has_named_entity("What does this agreement with eBay say about fees?")


def test_known_limitation_sentence_initial_company_name_not_detected():
    assert not _deictic_entity_guard_has_named_entity(
        "Aon is a party to this agreement. What does it say about termination?"
    )


def test_known_limitation_single_word_company_right_after_colon_not_detected():
    assert not _deictic_entity_guard_has_named_entity(
        "Details: Aon is a party. What does this agreement say about termination?"
    )


def test_known_limitation_stopword_collision_company_name_not_detected():
    # A company literally named "May" (e.g. May Department Stores) collides
    # with the calendar-month stopword entry.
    assert not _deictic_entity_guard_has_named_entity("What does this agreement with May say about termination?")


# --- Integration tests: run_agent_query with the guard enabled/disabled ---


@dataclass(frozen=True)
class _Doc:
    context_id: str
    full_indexed_content: str


class FakeSearchFn:
    def __init__(self):
        self.queries: list[str] = []

    def __call__(self, query: str) -> SearchToolCall:
        self.queries.append(query)
        return SearchToolCall(query=query, candidates=(_Doc("ctx_1", "doc 1"),))


class FakeAssessor:
    def __init__(self):
        self.calls = 0

    def assess(self, question: str, context_text: str, calls_remaining: int) -> EvidenceAssessment:
        self.calls += 1
        return EvidenceAssessment(sufficient=True)


class FakeGenerator:
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return "FINAL ANSWER: 42"


_CUAD_LIKE_QUESTION = (
    'Highlight the parts (if any) of this contract related to "Document Name" '
    "that should be reviewed by a lawyer. Details: The name of the contract"
)
_ORDINARY_QUESTION = "What was Aon's revenue in 2010?"


def test_guard_enabled_blocks_before_any_search_call():
    search_fn = FakeSearchFn()
    assessor = FakeAssessor()
    generator = FakeGenerator()

    result = run_agent_query(
        "q1",
        _CUAD_LIKE_QUESTION,
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=2,
        enable_deictic_entity_guard=True,
    )

    assert search_fn.queries == []  # search_fn must never be called
    assert assessor.calls == 0
    assert generator.call_count == 0
    assert result.answer_text == INSUFFICIENT_CONTEXT_MARKER
    assert result.stopped_reason == STOP_DEICTIC_ENTITY_GUARD
    assert result.forced_insufficient is True
    assert result.context_ids == ()
    assert result.additional_calls_used == 0
    assert result.context_documents == ()


def test_guard_enabled_does_not_block_ordinary_question():
    search_fn = FakeSearchFn()
    assessor = FakeAssessor()
    generator = FakeGenerator()

    result = run_agent_query(
        "q1",
        _ORDINARY_QUESTION,
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=2,
        enable_deictic_entity_guard=True,
    )

    assert search_fn.queries == [_ORDINARY_QUESTION]
    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.forced_insufficient is False


def test_guard_disabled_by_default_leaves_behavior_unchanged():
    """Critical regression check: enable_deictic_entity_guard defaults to
    False, so a CUAD-like question must still reach search_fn exactly as
    it did before this guard existed, for every caller that doesn't pass
    the new argument at all."""
    search_fn = FakeSearchFn()
    assessor = FakeAssessor()
    generator = FakeGenerator()

    result = run_agent_query(
        "q1", _CUAD_LIKE_QUESTION, search_fn, assessor, generator, max_additional_tool_calls=2
    )

    assert search_fn.queries == [_CUAD_LIKE_QUESTION]
    assert assessor.calls == 1
    assert generator.call_count == 1
    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE
    assert result.forced_insufficient is False


def test_guard_explicitly_disabled_also_leaves_behavior_unchanged():
    search_fn = FakeSearchFn()
    assessor = FakeAssessor()
    generator = FakeGenerator()

    result = run_agent_query(
        "q1",
        _CUAD_LIKE_QUESTION,
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=2,
        enable_deictic_entity_guard=False,
    )

    assert search_fn.queries == [_CUAD_LIKE_QUESTION]
    assert result.stopped_reason == STOP_SUFFICIENT_EVIDENCE

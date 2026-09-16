"""Tests for agent/loop.py's assessment-response parsing (_parse_assessment)
and its real adapter (ClaudeEvidenceAssessor).

This project's own generation.py docstring documents, at length, that
models sometimes don't comply with a requested output format even when
told to - the same risk applies to the "SUFFICIENT: .../REFORMULATED
QUERY: ..." marker format here, so the parser's edge cases get their own
tests, independent of the loop's control-flow tests in test_agent_loop.py.
"""

from __future__ import annotations

from agent.loop import ClaudeEvidenceAssessor, EvidenceAssessment, _parse_assessment


class FakeGenerator:
    """Implements pipeline.generation.GeneratorProtocol (generate(prompt) -> str)."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_parse_assessment_sufficient_yes_with_no_query():
    result = _parse_assessment("SUFFICIENT: yes\nREFORMULATED QUERY: NONE\n")
    assert result == EvidenceAssessment(sufficient=True, reformulated_query=None)


def test_parse_assessment_insufficient_with_reformulated_query():
    result = _parse_assessment("SUFFICIENT: no\nREFORMULATED QUERY: total operating expenses 2020\n")
    assert result == EvidenceAssessment(sufficient=False, reformulated_query="total operating expenses 2020")


def test_parse_assessment_is_case_insensitive():
    result = _parse_assessment("sufficient: YES\nreformulated query: none")
    assert result.sufficient is True
    assert result.reformulated_query is None


def test_parse_assessment_tolerates_reasoning_before_the_marker():
    raw = "Let me think about this...\n\nSUFFICIENT: no\nREFORMULATED QUERY: net income by segment\n"
    result = _parse_assessment(raw)
    assert result == EvidenceAssessment(sufficient=False, reformulated_query="net income by segment")


def test_parse_assessment_missing_marker_fails_safe_to_insufficient():
    # The model ignored the format entirely - must never be silently
    # treated as "sufficient", which would end the loop on unconfirmed
    # evidence.
    result = _parse_assessment("I think the answer is 42.")
    assert result == EvidenceAssessment(sufficient=False, reformulated_query=None)


def test_parse_assessment_empty_reformulated_query_value_is_none():
    result = _parse_assessment("SUFFICIENT: no\nREFORMULATED QUERY: \n")
    assert result.reformulated_query is None


def test_parse_assessment_uses_last_marker_not_first():
    # находка 1 (claude/status_agent_rezultaty_4_nahodki_kod.md): a model
    # that reasons out loud before committing can restate or reconsider
    # the marker line more than once. The FIRST occurrence here is a
    # draft the model explicitly reconsiders ("actually...") - only the
    # LAST one is the real verdict, and _parse_assessment must return
    # that one, mirroring agent/success.py's _extract_insufficiency_verdict
    # (matches[-1]).
    raw = (
        "SUFFICIENT: no\n"
        "REFORMULATED QUERY: total operating expenses 2020\n"
        "actually, re-reading the passages, that number is already there.\n"
        "SUFFICIENT: yes\n"
        "REFORMULATED QUERY: NONE\n"
    )
    result = _parse_assessment(raw)
    assert result == EvidenceAssessment(sufficient=True, reformulated_query=None)


def test_claude_evidence_assessor_parses_the_underlying_client_response():
    client = FakeGenerator("SUFFICIENT: no\nREFORMULATED QUERY: total liabilities 2019\n")
    assessor = ClaudeEvidenceAssessor(client)

    result = assessor.assess("What were total liabilities in 2019?", "[Document 1]\nsome text", calls_remaining=1)

    assert result == EvidenceAssessment(sufficient=False, reformulated_query="total liabilities 2019")
    assert len(client.prompts) == 1
    assert "calls_remaining" not in client.prompts[0]  # sanity: template placeholder was substituted, not left literal
    assert "1 additional search call" in client.prompts[0]

"""Tests for pipeline.generation (module 8).

Covers the two things worth testing in isolation from a real API call:
build_context_block/build_prompt (pure string assembly) and
_extract_final_answer (the parsing logic protecting module 9's
is_close_v2 comparison from prose/marker leaks - see the "FINAL ANSWER:"
delimiter-pattern rationale in pipeline/generation.py's module docstring,
and the real leaked-answer example on finqa_train_1575 documented there).
generate_answer() itself is covered end-to-end with a fake generator, not
a real Claude call - retry-on-transient-error behavior belongs to
pipeline.common.retry, not this module, and is not re-tested here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline.generation import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_CITE_AND_CHECK,
    PROMPT_TEMPLATE_FORMULA_BASE,
    PROMPT_TEMPLATE_VARIANTS,
    GeneratedAnswer,
    build_context_block,
    build_prompt,
    generate_answer,
    _extract_final_answer,
)


@dataclass
class FakeCandidate:
    full_indexed_content: str


class FakeGenerator:
    """Records the prompt it was called with and returns a fixed response
    - no network call, no retry involved (see module docstring)."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


# --- build_context_block ----------------------------------------------


def test_build_context_block_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        build_context_block([])


def test_build_context_block_numbers_documents_in_given_order():
    candidates = [FakeCandidate("first doc"), FakeCandidate("second doc")]
    block = build_context_block(candidates)
    assert "[Document 1]\nfirst doc" in block
    assert "[Document 2]\nsecond doc" in block
    # not re-sorted - order given is order rendered (module 8 does not
    # re-rank; that's modules 6/7's job)
    assert block.index("[Document 1]") < block.index("[Document 2]")


# --- build_prompt --------------------------------------------------------


def test_build_prompt_includes_question_and_context():
    candidates = [FakeCandidate("revenue was $100 million")]
    prompt = build_prompt("What was the revenue?", candidates)
    assert "What was the revenue?" in prompt
    assert "revenue was $100 million" in prompt
    assert "FINAL ANSWER:" in prompt  # format instruction must reach the model


def test_build_prompt_defaults_to_baseline_template():
    candidates = [FakeCandidate("doc")]
    assert build_prompt("q?", candidates) == build_prompt("q?", candidates, template=PROMPT_TEMPLATE)


def test_build_prompt_accepts_a_variant_template():
    candidates = [FakeCandidate("revenue was $100 million")]
    prompt = build_prompt("What was the revenue?", candidates, template=PROMPT_TEMPLATE_CITE_AND_CHECK)
    assert "What was the revenue?" in prompt
    assert "revenue was $100 million" in prompt
    assert "FINAL ANSWER:" in prompt


# --- Фаза 5 prompt variants (docs/tehnicheskoe_zadanie.md, section 28) ---


def test_prompt_template_variants_baseline_is_unmodified_production_template():
    # PROMPT_TEMPLATE_VARIANTS["baseline"] must be the exact same object
    # as PROMPT_TEMPLATE - a Фаза 5 variant must never silently become
    # the default (see the module docstring's "kept separate on purpose"
    # rationale).
    assert PROMPT_TEMPLATE_VARIANTS["baseline"] is PROMPT_TEMPLATE


def test_prompt_template_variants_has_exactly_the_three_known_keys():
    assert set(PROMPT_TEMPLATE_VARIANTS) == {"baseline", "cite_and_check", "formula_base"}


@pytest.mark.parametrize(
    "template",
    [PROMPT_TEMPLATE, PROMPT_TEMPLATE_CITE_AND_CHECK, PROMPT_TEMPLATE_FORMULA_BASE],
)
def test_every_prompt_variant_keeps_the_final_answer_contract(template):
    # Every variant must still ask for the same "FINAL ANSWER: <value>"
    # marker - _extract_final_answer() is not variant-aware, so a variant
    # that dropped or renamed the marker would silently break extraction.
    candidates = [FakeCandidate("doc")]
    prompt = build_prompt("q?", candidates, template=template)
    assert "FINAL ANSWER: <value>" in prompt
    assert "INSUFFICIENT_CONTEXT" in prompt


def test_cite_and_check_variant_requires_citing_the_source_row():
    candidates = [FakeCandidate("doc")]
    prompt = build_prompt("q?", candidates, template=PROMPT_TEMPLATE_CITE_AND_CHECK)
    assert "table row/column" in prompt
    assert "Recompute" in prompt


def test_formula_base_variant_requires_stating_numerator_and_denominator():
    candidates = [FakeCandidate("doc")]
    prompt = build_prompt("q?", candidates, template=PROMPT_TEMPLATE_FORMULA_BASE)
    assert "numerator and the denominator" in prompt
    assert "a decrease is negative" in prompt


# --- _extract_final_answer -----------------------------------------------


def test_extract_final_answer_same_line():
    assert _extract_final_answer("some reasoning\nFINAL ANSWER: 12.5") == "12.5"


def test_extract_final_answer_marker_then_newline():
    assert _extract_final_answer("reasoning here\nFINAL ANSWER:\n77143") == "77143"


def test_extract_final_answer_case_insensitive():
    assert _extract_final_answer("final answer: apple inc") == "apple inc"


def test_extract_final_answer_last_marker_wins_if_more_than_one():
    raw = "FINAL ANSWER: draft_value\nmore reasoning\nFINAL ANSWER: real_value"
    assert _extract_final_answer(raw) == "real_value"


def test_extract_final_answer_falls_back_to_last_line_when_marker_missing():
    # the exact leak pattern documented in the module docstring
    # (finqa_train_1575): reasoning spelled out, no marker, answer left on
    # the last line.
    raw = "2004 minus 2005 = -1, so NY covers 1 fewer year than U.S. federal.\n\n1"
    assert _extract_final_answer(raw) == "1"


def test_extract_final_answer_insufficient_context_passthrough():
    assert _extract_final_answer("FINAL ANSWER: INSUFFICIENT_CONTEXT") == "INSUFFICIENT_CONTEXT"


# --- generate_answer (end-to-end with a fake generator) ------------------


def test_generate_answer_wires_prompt_and_extraction_together():
    candidates = [FakeCandidate("cash flow was 500")]
    fake = FakeGenerator(response="Let me check.\nFINAL ANSWER: 500")

    result = generate_answer(fake, "q1", "What was cash flow?", candidates)

    assert isinstance(result, GeneratedAnswer)
    assert result.question_id == "q1"
    assert result.answer_text == "500"
    assert result.raw_response == "Let me check.\nFINAL ANSWER: 500"
    assert fake.call_count == 1
    assert "What was cash flow?" in fake.last_prompt
    assert "cash flow was 500" in fake.last_prompt


def test_generate_answer_threads_template_through_to_build_prompt():
    candidates = [FakeCandidate("cash flow was 500")]
    fake = FakeGenerator(response="FINAL ANSWER: 500")

    generate_answer(fake, "q1", "What was cash flow?", candidates, template=PROMPT_TEMPLATE_FORMULA_BASE)

    assert "numerator and the denominator" in fake.last_prompt

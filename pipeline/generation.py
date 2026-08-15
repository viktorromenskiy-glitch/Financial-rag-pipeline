"""Module 8 - Generation.

Generates a direct textual answer to the question from the top-N context
documents (module 7's reranked output, or module 6's if the reranker is
disabled). See docs/specifikatsiya_moduley.md, module 8.

Format: Direct (a plain text answer), not Program-of-Thought - see
docs/tehnicheskoe_zadanie.md, section 7 (fork 9): on a 30 gold-context
question sample, PoT and Direct produced the same number of correct
answers (0.733 = 0.733); PoT's extra code-execution risk was not justified
by a measured gain on this data.

Sign/scale invariance (tehnicheskoe_zadanie.md, section 7): the prompt
explicitly instructs the model to express percentages as a plain 0-100
number, not a 0-1 fraction, and currency amounts without a currency
symbol. This is spelled out in the prompt rather than left for the model
to guess, since the downstream comparison (module 9, is_close_v2) depends
on a predictable answer format.

PROMPT_TEMPLATE revised 2026-08-15 against real claude-sonnet-5 output on
data/t2-ragbench/eval_subset_250.parquet (10-question smoke run): the
original, softer wording ("a single number, short phrase, or sentence")
was not enough - claude-sonnet-5 reliably answered correctly in substance
but wrapped the number in explanatory prose, markdown bold, currency
symbols, or unit words ("$77,143 million", "-103.57 (E*TRADE's cumulative
return was...)", "Based on the data provided, ... **-248 million**").
is_close_v2 is a literal transplant (see pipeline/common/is_close_v2.py)
and deliberately does not strip that kind of text - the fix has to be a
stricter prompt, not a looser deterministic check. 7 of 10 answers in that
smoke run were judge-correct but is_close_v2-incorrect purely because of
this formatting gap, not a reasoning error - see the smoke run's
predictions.jsonl/eval_results.jsonl for the concrete before/after
evidence this rewrite is based on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.common.retry import retryable

MODEL = "claude-sonnet-5"
TEMPERATURE = 0.0

PROMPT_TEMPLATE = """You are answering a question about a company's financial report using the context documents below.

Context:
{context}

Question: {question}

Respond with ONLY the answer value itself - nothing else. No explanation, no reasoning, no restating the question, no markdown formatting (no bold, no bullet points), no parenthetical notes.

Formatting rules for the answer:
- If the answer is numeric, express it as a plain number with no currency symbol, no thousands separator (comma), and no unit word like "million"/"billion"/"thousand" - e.g. "77143", not "$77,143 million".
- Express any percentage as a plain number from 0 to 100, not a 0-1 fraction and not with a "%" sign - e.g. "12.5", not "0.125" and not "12.5%".
- If the answer is a short phrase (not a number) - e.g. a company name or date - give just that phrase, nothing appended.
- If the context does not contain enough information to answer, respond with exactly: INSUFFICIENT_CONTEXT

Your entire response must be just the answer value - a bare number or short phrase - and nothing else.
"""


@dataclass(frozen=True)
class GeneratedAnswer:
    question_id: str
    answer_text: str
    raw_response: str


class GeneratorProtocol(Protocol):
    """Abstraction over the Claude API - allows a fake generator to be
    substituted in tests without a real network call."""

    def generate(self, prompt: str) -> str: ...


def build_context_block(candidates: list) -> str:
    """candidates: module 7 RerankedCandidate list (or module 6 Candidate
    list if the reranker is disabled) - both expose full_indexed_content.
    Concatenates the documents with an explicit boundary, in the order
    given (already ranked by relevance upstream - not re-sorted here)."""
    if not candidates:
        raise ValueError("generate_answer() requires at least one context document")
    blocks = [f"[Document {i}]\n{c.full_indexed_content}" for i, c in enumerate(candidates, start=1)]
    return "\n\n".join(blocks)


def build_prompt(question: str, candidates: list) -> str:
    return PROMPT_TEMPLATE.format(context=build_context_block(candidates), question=question)


@retryable()
def _generate_with_retry(generator: GeneratorProtocol, prompt: str) -> str:
    return generator.generate(prompt)


def generate_answer(
    generator: GeneratorProtocol,
    question_id: str,
    question: str,
    candidates: list,
) -> GeneratedAnswer:
    """candidates: top-N context documents, already ranked (module 7's
    RerankedCandidate list, or module 6's Candidate list if
    reranker.enabled is false - see specifikatsiya_moduley.md, module 8,
    "Зависимости"). Not re-sliced or re-sorted here - the caller decides
    how many documents to include."""
    prompt = build_prompt(question, candidates)
    raw_response = _generate_with_retry(generator, prompt)
    answer_text = raw_response.strip()
    return GeneratedAnswer(question_id=question_id, answer_text=answer_text, raw_response=raw_response)

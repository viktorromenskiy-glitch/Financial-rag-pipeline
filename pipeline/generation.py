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

Instructions:
- Give a direct, concise answer - a single number, short phrase, or sentence. Do not show your reasoning steps or any code.
- Express any percentage as a plain number from 0 to 100 (e.g. "12.5", not "0.125" and not "12.5%").
- Express currency amounts as a plain number, without a currency symbol, unless the question explicitly asks for the currency.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
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

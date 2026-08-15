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

PROMPT_TEMPLATE revision history:

2026-08-15 (v2 of the prompt, no explicit version const - see note below):
the original, softer wording ("a single number, short phrase, or
sentence") was not enough - claude-sonnet-5 reliably answered correctly in
substance but wrapped the number in explanatory prose, markdown bold,
currency symbols, or unit words ("$77,143 million", "Based on the data
provided, ... **-248 million**"). Rewritten to flatly forbid any
explanation.

2026-08-15 (this revision, "FINAL ANSWER:" delimiter pattern): the flat
"never explain" prohibition from the previous revision still leaked on
rare, harder questions - observed directly on the real 250-question eval
(pipeline/cli.py's ClaudeGenerator has thinking disabled, so the model has
no private scratch space; on at least one question it wrote its arithmetic
out loud anyway despite the explicit prohibition: "2004 minus 2005 = -1,
so U.S. federal (2005) covers 1 fewer year than NY (2004).\n\n1" -
question finqa_train_1575, results/full250_v2judge). That leak cost a
judge-correct verdict even though the right value (1) was technically
present, because the judge could no longer cleanly identify which part of
the response was "the answer." Fighting this with an even stricter
prohibition has diminishing returns - the fix here is not to forbid
reasoning but to make extraction robust to it either way: the model may
reason freely, but must end with a fixed-format marker line
("FINAL ANSWER: <value>"), and _extract_final_answer() below always parses
from that marker (or, if the model forgets it, falls back to the last
non-empty line - the pattern actually observed when a leak happens). This
is the same "let it think, but delimit the final output" pattern as
<scratchpad>/<answer> tag conventions - it removes the tension that seems
to cause the occasional non-compliant leak, rather than just re-stating
the same prohibition more forcefully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from pipeline.common.retry import retryable

MODEL = "claude-sonnet-5"
TEMPERATURE = 0.0

PROMPT_TEMPLATE = """You are answering a question about a company's financial report using the context documents below.

Context:
{context}

Question: {question}

You may briefly work through the calculation or reasoning if it helps you get the right answer - that's fine. When you are done, output your final answer on its own line, in exactly this format, with nothing else on that line:

FINAL ANSWER: <value>

Formatting rules for <value>:
- If the answer is numeric, express it as a plain number with no currency symbol, no thousands separator (comma), and no unit word like "million"/"billion"/"thousand" - e.g. "77143", not "$77,143 million".
- Express any percentage as a plain number from 0 to 100, not a 0-1 fraction and not with a "%" sign - e.g. "12.5", not "0.125" and not "12.5%".
- If the answer is a short phrase (not a number) - e.g. a company name or date - give just that phrase, nothing appended.
- If the context does not contain enough information to answer, use: FINAL ANSWER: INSUFFICIENT_CONTEXT
"""

# Matches "final answer:" case-insensitively, wherever it appears in the
# response (there should be exactly one, on its own line, per the prompt
# above - but if the model produces more than one for some reason, the
# LAST occurrence wins, since that is the one intended as the actual
# final answer).
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*", re.IGNORECASE)


def _extract_final_answer(raw_response: str) -> str:
    """Pulls the value out of a "...FINAL ANSWER: <value>..." response.

    Takes the first non-empty line after the last "FINAL ANSWER:" marker
    (handles both "FINAL ANSWER: 100" and the marker followed by a
    newline then the value). If the marker is missing entirely - the
    model ignored the format instruction - falls back to the last
    non-empty line of the whole response, which is where a bare leaked
    answer has been observed to land in practice (see module docstring).
    """
    matches = list(_FINAL_ANSWER_RE.finditer(raw_response))
    if not matches:
        lines = [line.strip() for line in raw_response.strip().splitlines() if line.strip()]
        return lines[-1] if lines else raw_response.strip()
    tail = raw_response[matches[-1].end() :]
    lines = [line.strip() for line in tail.strip().splitlines() if line.strip()]
    return lines[0] if lines else tail.strip()


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
    answer_text = _extract_final_answer(raw_response)
    return GeneratedAnswer(question_id=question_id, answer_text=answer_text, raw_response=raw_response)

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

2026-08-23 (Фаза 5, two new named variants - PROMPT_TEMPLATE itself
unchanged): docs/tehnicheskoe_zadanie.md section 27 taxonomized all 25
confirmed generation_failure_candidate cases by root cause. Two clusters
dominated and are targeted here by separate prompt variants, kept
separate on purpose so a Phase 6 A/B run (see plan_generation_error_analysis.md
in the project, "Фаза 6") can attribute any effect to a specific
intervention rather than a bundled change ("один вариант вмешательства -
один скрипт - одна проверка"):

- PROMPT_TEMPLATE_CITE_AND_CHECK targets categories A (wrong table
  row/column/entity, 10/25), D (incomplete computation, 5/25), B
  (multi-entity ambiguity, 2/25), H (a correct intermediate result
  overridden by further reasoning, 1/25) and J (unwarranted refusal
  despite an unambiguous answer, 1/25) - 19/25 (76%) of the taxonomy.
  Adds an explicit "cite the row/column/period/entity you're using, then
  recompute once and confirm before answering" requirement. Two
  concrete real cases motivate this: `finqa_train_6044` cited "Document
  2 (2016 report)" out loud while the question asked about 2018 - the
  model's own words would have caught this if it had been required to
  name the period *before* using the number; `finqa_train_2917` computed
  the correct value (0.5) explicitly, then talked itself into a wrong
  answer (0) with no new fact - a required final "does my last computed
  number still hold" check targets exactly that pattern.
- PROMPT_TEMPLATE_FORMULA_BASE targets category C (sign or formula-base
  error, 3/25, 12%) - confirmed by exact arithmetic, not inference, in
  section 27 (e.g. `finqa_test_579`: model divided by the pre-tax gain
  and got 15.38, gold divides by the after-tax gain and is 18.18 - both
  numbers were already correct in the model's own working, only the
  denominator choice was wrong). Adds an explicit requirement to state
  the numerator and denominator of any ratio in words, and to preserve
  the sign of an "increase"/"decrease" in the final value.

Both variants keep the "FINAL ANSWER:" marker contract unchanged -
_extract_final_answer() below applies identically regardless of which
template produced raw_response. PROMPT_TEMPLATE (no suffix) remains the
production baseline and default for build_prompt()/generate_answer();
callers opt into a variant explicitly via the new `template` parameter
(pipeline/cli.py's `eval` subcommand exposes this as `--prompt-variant`).
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

# Фаза 5 variant targeting taxonomy categories A/D/B/H/J (see module
# docstring above) - same context/question slots and same FINAL ANSWER
# contract as PROMPT_TEMPLATE, with an explicit citation-then-verify
# requirement inserted before the formatting rules.
PROMPT_TEMPLATE_CITE_AND_CHECK = """You are answering a question about a company's financial report using the context documents below.

Context:
{context}

Question: {question}

Work through this step by step before giving your final answer:

1. For every number you use, state exactly where it comes from - which document, which table row/column label, and which period. If the context contains more than one company, segment, or level of aggregation (e.g. a single segment vs. the consolidated total) that could plausibly answer the question, say explicitly which one you are using and why - do not silently default to whichever number appears first or is largest.
2. Do the calculation.
3. Recompute it once more from the numbers you cited in step 1 and confirm the result still holds before reporting it. Do not discard or change a value you already computed correctly unless you find a specific new fact in the context that contradicts it - "on reflection", "to be safe", or re-reading the question again is not on its own a valid reason to change a correct calculation.
4. If a specific number needed to answer the question is genuinely not present anywhere in the context after this check, answer INSUFFICIENT_CONTEXT. But if the number IS present in the context and is the only plausible candidate, give that answer - do not refuse just because a secondary detail in the question's wording (e.g. an approximate date) does not exactly match the text.

When you are done, output your final answer on its own line, in exactly this format, with nothing else on that line:

FINAL ANSWER: <value>

Formatting rules for <value>:
- If the answer is numeric, express it as a plain number with no currency symbol, no thousands separator (comma), and no unit word like "million"/"billion"/"thousand" - e.g. "77143", not "$77,143 million".
- Express any percentage as a plain number from 0 to 100, not a 0-1 fraction and not with a "%" sign - e.g. "12.5", not "0.125" and not "12.5%".
- If the answer is a short phrase (not a number) - e.g. a company name or date - give just that phrase, nothing appended.
- If the context does not contain enough information to answer, use: FINAL ANSWER: INSUFFICIENT_CONTEXT
"""

# Фаза 5 variant targeting taxonomy category C (sign/formula-base error).
PROMPT_TEMPLATE_FORMULA_BASE = """You are answering a question about a company's financial report using the context documents below.

Context:
{context}

Question: {question}

You may briefly work through the calculation or reasoning if it helps you get the right answer - that's fine. Before computing any ratio or percentage, explicitly state the numerator and the denominator in words (e.g. "tax rate = tax paid / after-tax gain", not just "/ pre-tax gain" by default) - re-read the question to check which quantity it is actually asking you to divide by, rather than assuming the more obvious or larger base. If the question asks for an "increase" or "decrease", keep the sign of that direction in your final numeric answer (a decrease is negative). When you are done, output your final answer on its own line, in exactly this format, with nothing else on that line:

FINAL ANSWER: <value>

Formatting rules for <value>:
- If the answer is numeric, express it as a plain number with no currency symbol, no thousands separator (comma), and no unit word like "million"/"billion"/"thousand" - e.g. "77143", not "$77,143 million".
- Express any percentage as a plain number from 0 to 100, not a 0-1 fraction and not with a "%" sign - e.g. "12.5", not "0.125" and not "12.5%". Preserve the sign: a decrease is negative (e.g. "-12.5"), an increase is positive.
- If the answer is a short phrase (not a number) - e.g. a company name or date - give just that phrase, nothing appended.
- If the context does not contain enough information to answer, use: FINAL ANSWER: INSUFFICIENT_CONTEXT
"""

# Name -> template, for CLI/script selection (pipeline/cli.py's
# `eval --prompt-variant`). "baseline" is the production default.
PROMPT_TEMPLATE_VARIANTS = {
    "baseline": PROMPT_TEMPLATE,
    "cite_and_check": PROMPT_TEMPLATE_CITE_AND_CHECK,
    "formula_base": PROMPT_TEMPLATE_FORMULA_BASE,
}

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


def build_prompt(question: str, candidates: list, template: str = PROMPT_TEMPLATE) -> str:
    """template defaults to the production baseline PROMPT_TEMPLATE - pass
    one of the PROMPT_TEMPLATE_VARIANTS values (or PROMPT_TEMPLATE_CITE_AND_CHECK
    / PROMPT_TEMPLATE_FORMULA_BASE directly) to run a Фаза 5 intervention
    instead. Any template must accept the same {context}/{question} slots."""
    return template.format(context=build_context_block(candidates), question=question)


@retryable()
def _generate_with_retry(generator: GeneratorProtocol, prompt: str) -> str:
    return generator.generate(prompt)


def generate_answer(
    generator: GeneratorProtocol,
    question_id: str,
    question: str,
    candidates: list,
    template: str = PROMPT_TEMPLATE,
) -> GeneratedAnswer:
    """candidates: top-N context documents, already ranked (module 7's
    RerankedCandidate list, or module 6's Candidate list if
    reranker.enabled is false - see specifikatsiya_moduley.md, module 8,
    "Зависимости"). Not re-sliced or re-sorted here - the caller decides
    how many documents to include.

    template defaults to the production baseline PROMPT_TEMPLATE - see
    build_prompt() for how to select a Фаза 5 variant instead."""
    prompt = build_prompt(question, candidates, template=template)
    raw_response = _generate_with_retry(generator, prompt)
    answer_text = _extract_final_answer(raw_response)
    return GeneratedAnswer(question_id=question_id, answer_text=answer_text, raw_response=raw_response)

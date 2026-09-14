"""agent/ core - the bounded tool-use loop.

Design basis: itog_ekspertizy_agent_profil.md (4 independent experts,
consensus reached and verified across all 4 rounds) and
plan_rabot_posle_ekspertizy_agent_profil.md (the resulting V1 work plan -
this module is День 1, "Ядро").

Flow per question (matches the trace shape used throughout the expert
review - "retrieval_1 -> evidence_assessment -> reformulated_query ->
retrieval_2 -> ... -> answer"):

  1. Mandatory first search_documents(question_text) - consensus item 1:
     every question gets at least one retrieval, same as the baseline
     pipeline.
  2. Evidence assessment: an LLM call decides whether the accumulated
     evidence is sufficient to answer, and if not (and budget remains),
     proposes a reformulated query.
  3. If insufficient and budget remains: one more search_documents call
     with the reformulated query, merged into the accumulated evidence.
     Bounded loop (consensus item 3): at most
     config.agent.max_additional_tool_calls additional search_documents
     calls - a config value, never a hardcoded constant - plus an early
     stop if a re-query adds no new context_ids (further calls are very
     unlikely to help, and every call has real latency/cost).
  4. Once the loop stops (sufficient evidence, budget exhausted, or no
     new documents), generate the final answer - unless the last
     assessment still says insufficient, in which case the answer is
     forced to INSUFFICIENT_CONTEXT (round-4 closing-expert addition,
     see itog_ekspertizy_agent_profil.md, "Новые находки закрывающего
     раунда", item 2: "явная политика «когда агент должен сдаться»" - a
     model given thin evidence and no other constraint will tend to
     "додумывать" rather than admit it doesn't know).

This module never talks to MongoDB/Voyage/Cohere/Anthropic directly - it
depends only on Protocols (search_fn, EvidenceAssessorProtocol,
pipeline.generation.GeneratorProtocol, TraceWriterProtocol), so the whole
loop is unit-testable with plain fakes and needs no real API keys or
network access (see test_agent_loop.py and test_agent_smoke.py) - the
offline/mocked-smoke-test requirement from the design review's Day 1
scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from agent.tools import SearchToolCall
from agent.tracing import TraceWriterProtocol
from pipeline.generation import PROMPT_TEMPLATE, GeneratorProtocol, build_context_block, generate_answer

# Same literal token pipeline.generation.PROMPT_TEMPLATE already instructs
# the baseline model to output when context is insufficient - reusing it
# (rather than inventing a second "no answer" sentinel) means is_close_v2
# and the LLM judge already know how to handle this value, unchanged.
INSUFFICIENT_CONTEXT_MARKER = "INSUFFICIENT_CONTEXT"

STOP_SUFFICIENT_EVIDENCE = "sufficient_evidence"
STOP_NO_NEW_DOCUMENTS = "no_new_documents"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_EMPTY_EVIDENCE = "empty_evidence"

# Versioned separately from pipeline.generation's PROMPT_TEMPLATE/prompt_variant
# (round-4 closing-expert note, itog_ekspertizy_agent_profil.md: "Версионирование
# agent-промпта отдельно от pipeline-промпта - иначе baseline и agent становятся
# несравнимыми при любом изменении system prompt"). Bump this whenever
# _ASSESSMENT_PROMPT_TEMPLATE changes, independently of pipeline.generation's
# own versioning.
AGENT_ASSESSMENT_PROMPT_VERSION = "v1"

_ASSESSMENT_PROMPT_TEMPLATE = """You are deciding whether enough evidence has been retrieved to answer a question about a company's financial report.

Question: {question}

Retrieved passages so far:
{context}

You have {calls_remaining} additional search call(s) left after this decision, if you need them.

Decide:
1. Is the evidence above sufficient to answer the question completely and precisely? Judge only from the retrieved passages above - do not use outside knowledge.
2. If not sufficient AND you still have calls remaining, propose ONE reformulated search query that is more likely to find the missing information (e.g. naming a specific line item, period, or company the current passages are missing). If you have no better query to try, or no calls remain, say NONE.

Respond in exactly this format, nothing else:

SUFFICIENT: <yes or no>
REFORMULATED QUERY: <a new search query, or NONE>
"""


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reformulated_query: str | None = None


class EvidenceAssessorProtocol(Protocol):
    def assess(self, question: str, context_text: str, calls_remaining: int) -> EvidenceAssessment: ...


_SUFFICIENT_RE = re.compile(r"sufficient\s*:\s*(yes|no)", re.IGNORECASE)
_REFORMULATED_RE = re.compile(r"reformulated\s*query\s*:\s*(.*)", re.IGNORECASE)


def _parse_assessment(raw_response: str) -> EvidenceAssessment:
    """Parses a "SUFFICIENT: .../REFORMULATED QUERY: ..." response - the
    same marker-line convention pipeline.generation uses for "FINAL
    ANSWER:" (see that module's docstring for why this project prefers a
    parseable marker over a strict format the model sometimes ignores
    anyway).

    Fails safe: a response missing the SUFFICIENT marker entirely (the
    model ignored the format) is treated as insufficient, not as "yes" -
    a malformed response must never accidentally end the loop early and
    let the agent answer from evidence it never actually confirmed was
    enough.
    """
    match = _SUFFICIENT_RE.search(raw_response)
    if not match:
        return EvidenceAssessment(sufficient=False, reformulated_query=None)
    sufficient = match.group(1).strip().lower() == "yes"

    reformulated_query: str | None = None
    reformulated_match = _REFORMULATED_RE.search(raw_response)
    if reformulated_match:
        value = reformulated_match.group(1).strip()
        if value and value.upper() != "NONE":
            reformulated_query = value
    return EvidenceAssessment(sufficient=sufficient, reformulated_query=reformulated_query)


class ClaudeEvidenceAssessor:
    """Real adapter: wraps any pipeline.generation.GeneratorProtocol-shaped
    client (e.g. pipeline.cli.ClaudeGenerator, or a dedicated instance
    with its own model/temperature) to implement EvidenceAssessorProtocol.

    Deliberately not wired into pipeline/cli.py yet - that belongs to Day
    2/3 of plan_rabot_posle_ekspertizy_agent_profil.md, alongside the
    evaluation harness that needs to invoke both baseline and agent. Kept
    here, separate from run_agent_query's control flow, so the loop
    itself is tested against a plain fake (see test_agent_loop.py) and
    this adapter's prompt-parsing is tested independently (see
    test_agent_claude_adapter.py) - the same separation-of-concerns this
    codebase already uses for pipeline.cli's ClaudeGenerator/ClaudeJudge
    vs. pipeline.generation/pipeline.evaluation.
    """

    def __init__(self, client: GeneratorProtocol):
        self.client = client

    def assess(self, question: str, context_text: str, calls_remaining: int) -> EvidenceAssessment:
        prompt = _ASSESSMENT_PROMPT_TEMPLATE.format(
            question=question, context=context_text, calls_remaining=calls_remaining
        )
        raw = self.client.generate(prompt)
        return _parse_assessment(raw)


@dataclass(frozen=True)
class AgentAnswer:
    question_id: str
    answer_text: str
    context_ids: tuple[str, ...]
    additional_calls_used: int
    stopped_reason: str
    forced_insufficient: bool


def run_agent_query(
    question_id: str,
    question_text: str,
    search_fn: Callable[[str], SearchToolCall],
    assessor: EvidenceAssessorProtocol,
    generator: GeneratorProtocol,
    *,
    max_additional_tool_calls: int,
    prompt_template: str = PROMPT_TEMPLATE,
    trace_writer: TraceWriterProtocol | None = None,
) -> AgentAnswer:
    """Runs the full bounded loop for one question.

    `search_fn`: a callable taking a query string and returning a
    SearchToolCall - agent.tools.search_documents bound to real
    voyage/collection/cohere clients and config-driven parameters (via
    functools.partial or a small closure), or a fake in tests. This
    keeps the loop itself free of any dependency on MongoDB/Voyage/Cohere
    - it only ever calls search_fn(query).

    `max_additional_tool_calls`: config.agent.max_additional_tool_calls -
    read by the caller from config/config.yaml, not hardcoded here (see
    config.config_schema.AgentConfig).

    Raises:
        ValueError: if max_additional_tool_calls is negative - config
            validation (AgentConfig's Field(ge=0)) already prevents this
            for any real config file, but a caller constructing the value
            by hand should still get a clear, immediate error rather than
            silently odd loop behavior.
    """
    if max_additional_tool_calls < 0:
        raise ValueError(f"max_additional_tool_calls must be >= 0, got {max_additional_tool_calls}")

    def _trace(step: str, **fields: object) -> None:
        if trace_writer is not None:
            trace_writer.append({"question_id": question_id, "step": step, **fields})

    call = search_fn(question_text)
    accumulated: dict[str, object] = {c.context_id: c for c in call.candidates}
    _trace("retrieval_1", query=call.query, context_ids=sorted(call.context_ids))

    additional_calls_used = 0
    assessment: EvidenceAssessment | None = None
    stopped_reason: str

    if not accumulated:
        # Empty retrieval on the very first (mandatory) search: nothing to
        # reformulate a query against yet, and generate_answer() requires
        # at least one context document - spending an assessment call on
        # zero evidence would only ever come back "insufficient" anyway.
        # Answer INSUFFICIENT_CONTEXT directly rather than crashing or
        # guessing (Day 1 checklist item: graceful handling of empty
        # retrieval).
        stopped_reason = STOP_EMPTY_EVIDENCE
        answer_text = INSUFFICIENT_CONTEXT_MARKER
        _trace("answer", answer_text=answer_text, forced_insufficient=True)
        return AgentAnswer(
            question_id=question_id,
            answer_text=answer_text,
            context_ids=(),
            additional_calls_used=0,
            stopped_reason=stopped_reason,
            forced_insufficient=True,
        )

    while True:
        calls_remaining = max_additional_tool_calls - additional_calls_used
        context_text = build_context_block(list(accumulated.values()))
        assessment = assessor.assess(question_text, context_text, calls_remaining)
        _trace(
            "evidence_assessment",
            sufficient=assessment.sufficient,
            reformulated_query=assessment.reformulated_query,
            calls_remaining=calls_remaining,
        )

        if assessment.sufficient:
            stopped_reason = STOP_SUFFICIENT_EVIDENCE
            break
        if calls_remaining <= 0 or not assessment.reformulated_query:
            stopped_reason = STOP_BUDGET_EXHAUSTED
            break

        additional_calls_used += 1
        _trace("reformulated_query", query=assessment.reformulated_query, call_number=additional_calls_used + 1)
        call = search_fn(assessment.reformulated_query)
        new_ids = call.context_ids - accumulated.keys()
        _trace(
            f"retrieval_{additional_calls_used + 1}",
            query=call.query,
            context_ids=sorted(call.context_ids),
            new_context_ids=sorted(new_ids),
        )
        if not new_ids:
            # Consensus item 3: a re-query that adds nothing new is very
            # unlikely to be fixed by yet another call, and every call has
            # real latency/cost - stop now rather than spend the rest of
            # the budget. `assessment` still holds the verdict that
            # triggered this search (insufficient), which is what decides
            # forced_insufficient below - correct, since nothing changed.
            stopped_reason = STOP_NO_NEW_DOCUMENTS
            break
        accumulated.update({c.context_id: c for c in call.candidates})
        # loop back: re-assess against the newly enlarged context

    assert assessment is not None  # the `if not accumulated` branch above already returned
    forced_insufficient = not assessment.sufficient

    if forced_insufficient:
        # Give-up policy (round-4 closing-expert addition): still
        # insufficient when the loop had to stop - force
        # INSUFFICIENT_CONTEXT rather than letting the generator guess
        # from evidence it (via the assessor) already judged inadequate.
        # Counted as success downstream only if the judge independently
        # agrees the question is unanswerable - see
        # plan_rabot_posle_ekspertizy_agent_profil.md, День 2.
        answer_text = INSUFFICIENT_CONTEXT_MARKER
        _trace("answer", answer_text=answer_text, forced_insufficient=True)
    else:
        generated = generate_answer(
            generator, question_id, question_text, list(accumulated.values()), template=prompt_template
        )
        answer_text = generated.answer_text
        _trace("answer", answer_text=answer_text, forced_insufficient=False)

    return AgentAnswer(
        question_id=question_id,
        answer_text=answer_text,
        context_ids=tuple(accumulated.keys()),
        additional_calls_used=additional_calls_used,
        stopped_reason=stopped_reason,
        forced_insufficient=forced_insufficient,
    )

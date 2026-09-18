"""Human-readable "thought -> tool -> observation" trace dump - Day 3 of
plan_rabot_posle_ekspertizy_agent_profil.md, "Human-readable debug mode
("thought -> tool -> observation" dump), separate from the machine-readable
JSONL - makes it easier to review discordant pairs before publication."

agent/tracing.py's TraceWriter already writes the machine-readable form
(one JSON object per line, results/<run_id>/agent_trace.jsonl - see that
module's docstring). This module is a pure, offline-testable renderer on
top of that same record shape (agent/loop.py's `_trace`/`_trace_retrieval`
closures): it turns a flat list of trace dicts into readable text for a
person reviewing a discordant pair or a demonstration case, without
touching agent/loop.py's control flow or record shape at all.

Deliberately a separate, standalone module rather than a method on
TraceWriter/InMemoryTraceWriter - it operates on records ALREADY
collected (e.g. read back from agent_trace.jsonl, or an
InMemoryTraceWriter's .records list in a test), not on the live write
path, and has no reason to know how those records got there.

Renders every step agent/loop.py currently emits (retrieval_N,
evidence_assessment, reformulated_query, answer) with a dedicated,
readable format, but never raises on an unrecognized step name or a
record missing an expected field - a generic fallback line (the full record
as-is) is used instead, so this module can never crash a report just
because a future agent/loop.py change added a new step type or field this
file doesn't know about yet.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable


def group_by_question(records: Iterable[dict]) -> "OrderedDict[str, list[dict]]":
    """Groups a flat sequence of trace records (as read from one
    agent_trace.jsonl, or one InMemoryTraceWriter.records list covering
    several questions/canaries) by their "question_id" field, preserving
    both the original per-question step order and the order in which each
    question_id was first seen - so rendering the result in insertion
    order matches the order questions actually ran in, not an arbitrary
    dict/sort order.

    A record with no "question_id" key is skipped rather than raising -
    every real record agent/loop.py emits always has one, so this only
    guards against a hand-built/corrupted fixture in a test.
    """
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for record in records:
        question_id = record.get("question_id")
        if question_id is None:
            continue
        grouped.setdefault(question_id, []).append(record)
    return grouped


def _format_retrieval(record: dict) -> str:
    context_ids = record.get("context_ids", [])
    query = record.get("query", "")
    degraded = record.get("degraded", False)
    lines = [f'TOOL CALL   search_documents(query={query!r})']
    if degraded:
        reason = record.get("degradation_reason", "unknown reason")
        lines.append(f"OBSERVATION [DEGRADED] {reason}")
    new_ids = record.get("new_context_ids")
    if new_ids is not None:
        lines.append(
            f"OBSERVATION {len(context_ids)} candidate(s) so far, {len(new_ids)} new: {sorted(new_ids)}"
        )
    else:
        lines.append(f"OBSERVATION {len(context_ids)} candidate(s) retrieved: {sorted(context_ids)}")
    return "\n    ".join(lines)


def _format_evidence_assessment(record: dict) -> str:
    sufficient = record.get("sufficient")
    calls_remaining = record.get("calls_remaining")
    reformulated = record.get("reformulated_query")
    verdict = "sufficient" if sufficient else "NOT sufficient"
    lines = [f"THOUGHT     evidence_assessment -> {verdict} (calls_remaining={calls_remaining})"]
    if reformulated:
        lines.append(f"            proposed reformulated query: {reformulated!r}")
    return "\n    ".join(lines)


def _format_reformulated_query(record: dict) -> str:
    call_number = record.get("call_number")
    query = record.get("query", "")
    return f"THOUGHT     decided to search again (call #{call_number}) with query={query!r}"


def _format_answer(record: dict) -> str:
    answer_text = record.get("answer_text", "")
    forced = record.get("forced_insufficient")
    suffix = "  [forced - give-up policy]" if forced else ""
    return f"ANSWER      {answer_text!r}{suffix}"


_STEP_FORMATTERS = {
    "evidence_assessment": _format_evidence_assessment,
    "reformulated_query": _format_reformulated_query,
    "answer": _format_answer,
}


def format_step(record: dict, index: int) -> str:
    """Renders one trace record as one (possibly multi-line) human-readable
    entry, prefixed with its 1-based position in the question's trace."""
    step = record.get("step", "")
    if step.startswith("retrieval_"):
        body = _format_retrieval(record)
    else:
        formatter = _STEP_FORMATTERS.get(step)
        # Fallback for any step this module doesn't recognize (forward
        # compatibility - see module docstring): dump the record as-is
        # rather than raising or silently dropping it.
        body = f"STEP={step!r}  {record}" if formatter is None else formatter(record)
    return f"[{index}] {body}"


def render_question_trace(question_id: str, records: list[dict]) -> str:
    """Renders every step of a single question's trace, in the order the
    records appear in `records` (the caller is responsible for passing
    them in run order - group_by_question preserves this automatically)."""
    header = f"=== Question {question_id} ==="
    if not records:
        return f"{header}\n(no trace records)"
    body = "\n".join(format_step(record, i) for i, record in enumerate(records, start=1))
    return f"{header}\n{body}"


def render_trace_dump(records: Iterable[dict]) -> str:
    """Renders a full human-readable dump of every question found in
    `records` (e.g. a whole agent_trace.jsonl file, read line-by-line and
    json.loads'd into dicts), one section per question in first-seen
    order, separated by a blank line - the report this module exists to
    produce (see module docstring)."""
    grouped = group_by_question(records)
    if not grouped:
        return "(no trace records)"
    return "\n\n".join(render_question_trace(qid, recs) for qid, recs in grouped.items())

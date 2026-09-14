"""Harness-level "successful completion rate" definition for agent/ - День 2
of plan_rabot_posle_ekspertizy_agent_profil.md.

Kimi's followup response (kimi_followup_mcnemar_arithmetic_response.md,
"Primary metric - successful completion rate") gives the operational
definition this module implements: all required facts confirmed by the
document, numbers compared via is_close_v2, and an explicit
"insufficient evidence" answer counts as success ONLY IF the judge
independently agrees the question was genuinely unanswerable from the
evidence the agent actually had - otherwise it is a failure. This closes
the "refuse to dodge a hard question" loophole an ungated INSUFFICIENT_CONTEXT
would otherwise open (a fixed pipeline cannot game its correctness metric
by refusing more; a bounded agent that can choose to stop early could,
without this gate).

Kept separate from pipeline.evaluation (not added to evaluate_answer()
itself): pipeline.evaluation's JUDGE_PROMPT/evaluate_answer() is the
flagship pipeline's already-validated correctness check (93.3% judge/
deterministic agreement, tehnicheskoe_zadanie.md section 8) - reused here
unchanged for the "agent actually answered" path, not modified. The
"was giving up justified" question below is a genuinely different
judgment (not "is this answer correct" but "was refusing to answer
correct"), needs its own prompt, and only ever applies to agent runs -
it has no baseline-pipeline equivalent to stay consistent with.

Two disjoint cases never reach a judge call at all:
  - gold_answer is itself "INSUFFICIENT_CONTEXT" (a genuinely unanswerable
    canary/probe question, see agent/canary.py's canary_injection_3):
    success is a direct string comparison, no judge needed.
  - agent_answer_text is a real value and the question is not genuinely
    unanswerable: handled by pipeline.evaluation.evaluate_answer as
    normal (one judge call, same as the baseline pipeline).
Only "agent refused (INSUFFICIENT_CONTEXT) on a question with a real gold
answer" needs the extra insufficiency-judge call this module adds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.loop import INSUFFICIENT_CONTEXT_MARKER
from pipeline.evaluation import JudgeProtocol, evaluate_answer
from pipeline.common.retry import retryable

# Bump whenever INSUFFICIENCY_JUDGE_PROMPT's wording changes - independent
# of pipeline.evaluation.PROMPT_VERSION (a different prompt, judging a
# different question), same rationale as agent/loop.py's
# AGENT_ASSESSMENT_PROMPT_VERSION/AGENT_ANSWER_PROMPT_VERSION.
INSUFFICIENCY_JUDGE_PROMPT_VERSION = "v1"

INSUFFICIENCY_JUDGE_PROMPT = """An automated system was asked a question about a company's financial report and, after searching a document corpus, decided it did not have enough evidence and refused to answer.

Question: {question}

Evidence the system had actually retrieved and considered before refusing:
{context}

The true correct answer (not shown to the system while it was answering) is: {gold_answer}

Decide, using ONLY the evidence shown above (not any outside knowledge you may have): could a careful reader have derived the true correct answer from this evidence? Judge strictly from what is actually present above - if the specific fact, number, period, or entity needed is genuinely absent, missing, or about something else, the refusal was justified even if the true answer happens to be a "reasonable-sounding" value.

You may briefly explain your reasoning if it helps you decide - that's fine. When you are done, output your verdict on its own line, in exactly this format, with nothing else on that line:

VERDICT: JUSTIFIED_REFUSAL
(or)
VERDICT: SHOULD_HAVE_ANSWERED"""

_INSUFFICIENCY_VERDICT_RE = re.compile(r"verdict\s*:\s*", re.IGNORECASE)


def _extract_insufficiency_verdict(raw_verdict: str) -> str:
    """Same marker-line convention/fallback as pipeline.evaluation's
    _extract_verdict and pipeline.generation's _extract_final_answer - see
    either module's docstring for the rationale (a flat prohibition on
    reasoning leaks more often than a parseable end marker does)."""
    matches = list(_INSUFFICIENCY_VERDICT_RE.finditer(raw_verdict))
    if not matches:
        lines = [line.strip() for line in raw_verdict.strip().splitlines() if line.strip()]
        return lines[-1] if lines else raw_verdict.strip()
    tail = raw_verdict[matches[-1].end() :]
    lines = [line.strip() for line in tail.strip().splitlines() if line.strip()]
    return lines[0] if lines else tail.strip()


@retryable()
def _judge_with_retry(judge: JudgeProtocol, prompt: str) -> str:
    return judge.judge(prompt)


def judge_context_insufficiency(judge: JudgeProtocol, question: str, context_text: str, gold_answer: str) -> bool:
    """True if the judge agrees the agent's refusal was justified (the
    retrieved context genuinely could not support the gold answer).

    Fails safe toward "not justified" on a malformed judge response (no
    VERDICT marker parseable to exactly one of the two known values) -
    same fail-safe direction as agent/loop.py's _parse_assessment: an
    ambiguous verdict must never accidentally count a refusal as
    successful.
    """
    prompt = INSUFFICIENCY_JUDGE_PROMPT.format(question=question, context=context_text, gold_answer=gold_answer)
    raw = _judge_with_retry(judge, prompt)
    verdict = _extract_insufficiency_verdict(raw).upper()
    return "JUSTIFIED_REFUSAL" in verdict and "SHOULD_HAVE_ANSWERED" not in verdict


@dataclass(frozen=True)
class SuccessResult:
    question_id: str
    success: bool
    reason: str
    judge_scores: dict | None = None


def _is_insufficient(text: str) -> bool:
    return text.strip().upper() == INSUFFICIENT_CONTEXT_MARKER


def evaluate_agent_success(
    judge: JudgeProtocol,
    question_id: str,
    question: str,
    context_text: str,
    agent_answer_text: str,
    gold_answer: str,
    deterministic_check_enabled: bool = True,
) -> SuccessResult:
    """The one operational "successful completion" definition for a Day 2
    harness run - see module docstring for the three cases this branches
    on.

    `context_text`: the accumulated retrieved passages the agent actually
    had when it stopped (agent.loop.AgentAnswer doesn't carry this
    directly - the harness script builds it from the same candidates used
    for context_ids, via pipeline.generation.build_context_block, before
    calling this function).
    """
    gold_is_insufficient = _is_insufficient(str(gold_answer))
    answer_is_insufficient = _is_insufficient(agent_answer_text)

    if gold_is_insufficient:
        # A genuinely unanswerable question (e.g. agent/canary.py's
        # canary_injection_3) - no judge call needed, this is a direct
        # comparison of what happened against what should have happened.
        success = answer_is_insufficient
        reason = "insufficient_context_matches_gold" if success else "confidently_answered_unanswerable_question"
        return SuccessResult(question_id=question_id, success=success, reason=reason)

    if answer_is_insufficient:
        justified = judge_context_insufficiency(judge, question, context_text, gold_answer)
        reason = "justified_refusal" if justified else "unjustified_refusal"
        return SuccessResult(question_id=question_id, success=justified, reason=reason)

    eval_result = evaluate_answer(
        judge,
        question_id,
        question,
        context_text,
        agent_answer_text,
        gold_answer,
        deterministic_check_enabled=deterministic_check_enabled,
    )
    judge_correct = eval_result.judge_scores["judge_correct"]
    reason = "correct_answer" if judge_correct else "incorrect_answer"
    return SuccessResult(
        question_id=question_id, success=judge_correct, reason=reason, judge_scores=eval_result.judge_scores
    )

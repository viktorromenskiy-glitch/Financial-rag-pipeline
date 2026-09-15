"""Formal, pre-registered selection of demonstration cases - Day 3 of
plan_rabot_posle_ekspertizy_agent_profil.md:

  "1-2 demonstration cases с полным трейсом; отбор по формальному
  критерию, зафиксированному заранее (например, «вопросы, на которых
  baseline failed»), не по красоте трейса агента."
  "Обязательно включить минимум один случай, где агент не помог или
  помог хуже - против chery-picking."

The criterion below is written and committed BEFORE this module has ever
been run against real results/agent_eval_day2/ data (that harness run
itself needs paid API credentials this environment does not have - see
scripts/run_agent_eval.py's module docstring). Writing the rule first and
only then looking at data is the entire point: it is what makes the
selection "formal" rather than "pick whichever trace looks best once you
can already see all of them", which is exactly the cherry-picking risk
the plan calls out.

The rule, in full (deterministic - no random or LLM-based choice
anywhere):

  Let `matched` be the sorted (by question_id, ascending - an arbitrary
  but fixed, reproducible order, never "whichever the harness happened to
  process first") list of question_ids present in BOTH baseline_records
  and agent_records.

  Classify each matched question into one of four buckets using the same
  a/b naming scripts/run_agent_eval.py's own 2x2 table already uses (agent
  = "a", baseline = "b"):
    - a_only:      agent succeeded, baseline failed  -> candidate for the
                   POSITIVE case ("agent_helped")
    - b_only:      baseline succeeded, agent failed  -> candidate for the
                   REQUIRED NEGATIVE case ("agent_hurt" - strictly worse
                   than baseline on this question)
    - both_wrong:  both failed                        -> fallback
                   candidate for the negative case ("agent_did_not_help",
                   used only if b_only is empty - agent didn't help here
                   either, but this is NOT evidence agent is worse than
                   baseline, and the returned case says so explicitly)
    - both_correct: not used for either case (neither demonstrates a
                   difference worth showing)

  Selection:
    1. Positive case: the first (lowest question_id) entry of a_only, if
       any.
    2. Negative case: the first entry of b_only, if any; otherwise the
       first entry of both_wrong, if any; otherwise omitted.

  If a bucket used above is empty, the corresponding case is omitted from
  the result, WITH AN EXPLICIT NOTE explaining why (e.g. "no question in
  this run had the agent fail where baseline succeeded") - never silently
  padded with a both_correct question just to force a count of exactly
  two, since that would defeat the anti-cherry-picking purpose of the
  negative case entirely.

This module only picks WHICH question_ids to show; it does not know how
to render a trace (see agent/debug_view.py) or where the result files
live (see scripts/render_demonstration_cases.py, the thin real-data
orchestrator that calls this function and agent/debug_view together).
"""

from __future__ import annotations

from dataclasses import dataclass

ROLE_AGENT_HELPED = "agent_helped"
ROLE_AGENT_HURT = "agent_hurt"
ROLE_AGENT_DID_NOT_HELP = "agent_did_not_help"


@dataclass(frozen=True)
class DemonstrationCase:
    question_id: str
    role: str
    note: str


@dataclass(frozen=True)
class MatchedQuestionClassification:
    """The same four buckets `select_demonstration_cases` classifies
    matched questions into, exposed on their own so a caller can inspect
    ALL discordant questions (a_only + b_only), not only the one
    pre-registered representative of each bucket that
    `select_demonstration_cases` shows - see
    scripts/render_demonstration_cases.py's --all-discordant flag."""

    a_only: tuple[str, ...]
    b_only: tuple[str, ...]
    both_wrong: tuple[str, ...]
    both_correct: tuple[str, ...]


def _matched_question_ids(baseline_records: dict[str, dict], agent_records: dict[str, dict]) -> list[str]:
    return sorted(set(baseline_records) & set(agent_records))


def _as_bool(value: object, *, field: str, question_id: str) -> bool:
    """Rejects anything that isn't already a real bool, instead of doing
    Python's usual truthy coercion.

    `bool("False")` is `True` - a string value of "False" (e.g. from a
    result record that was round-tripped through something that stringifies
    fields) would silently coerce to a correct-looking bool and could move
    a question into the wrong bucket without any error. Since which bucket
    a question lands in directly decides which demonstration case gets
    shown, a silent miscoercion here is exactly the kind of thing the
    anti-cherry-picking guarantee is supposed to rule out - so this fails
    loudly instead.
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"{field!r} for question_id={question_id!r} is {value!r} ({type(value).__name__}), not a bool - "
            "refusing to silently coerce with bool(...) (e.g. bool('False') is True in Python), since that could "
            "silently move this question into the wrong bucket"
        )
    return value


def classify_matched_questions(
    baseline_records: dict[str, dict],
    agent_records: dict[str, dict],
    *,
    agent_success_field: str = "success",
    baseline_success_field: str = "judge_correct",
) -> MatchedQuestionClassification:
    """Applies the pre-registered bucket rule (see module docstring) to two
    loaded results sets and returns all four buckets in full - the part of
    `select_demonstration_cases` that picks ONE representative per bucket
    is deliberately separate (below), so a caller that wants to see every
    discordant question, not just the pre-registered representative, can
    call this directly instead of re-deriving the buckets itself.

    `baseline_records`/`agent_records`: {question_id: record}, matching
    exactly the shape scripts/run_agent_eval.py's `_load_jsonl_checkpoint`
    already produces from baseline_results.jsonl/agent_results.jsonl (see
    that script) - this function takes plain dicts, not file paths, so it
    stays independently testable with fabricated fixtures (see
    tests/test_agent_demonstration.py) and has no file I/O of its own.

    Raises:
        ValueError: if there is no question_id present in both inputs at
            all - there is nothing to classify, and returning empty
            buckets silently would look like "ran fine, found nothing"
            rather than "the two result sets don't overlap", which is
            almost certainly a bug in whatever produced them.
    """
    matched = _matched_question_ids(baseline_records, agent_records)
    if not matched:
        raise ValueError("No question_id is present in both baseline_records and agent_records - nothing to select from")

    a_only: list[str] = []
    b_only: list[str] = []
    both_wrong: list[str] = []
    both_correct: list[str] = []
    for question_id in matched:
        a_correct = _as_bool(
            agent_records[question_id][agent_success_field], field=agent_success_field, question_id=question_id
        )
        b_correct = _as_bool(
            baseline_records[question_id][baseline_success_field], field=baseline_success_field, question_id=question_id
        )
        if a_correct and not b_correct:
            a_only.append(question_id)
        elif not a_correct and b_correct:
            b_only.append(question_id)
        elif not a_correct and not b_correct:
            both_wrong.append(question_id)
        else:
            both_correct.append(question_id)

    return MatchedQuestionClassification(
        a_only=tuple(a_only), b_only=tuple(b_only), both_wrong=tuple(both_wrong), both_correct=tuple(both_correct)
    )


def select_demonstration_cases(
    baseline_records: dict[str, dict],
    agent_records: dict[str, dict],
    *,
    agent_success_field: str = "success",
    baseline_success_field: str = "judge_correct",
) -> list[DemonstrationCase]:
    """Applies the pre-registered rule above to two loaded results sets,
    picking exactly one representative question_id per case slot (the
    lowest-question_id entry of the relevant bucket - see
    `classify_matched_questions` above for the full, unfiltered buckets).

    Raises:
        ValueError: see `classify_matched_questions` - propagated from
            there when the two result sets don't overlap at all.
    """
    classification = classify_matched_questions(
        baseline_records,
        agent_records,
        agent_success_field=agent_success_field,
        baseline_success_field=baseline_success_field,
    )
    a_only = list(classification.a_only)
    b_only = list(classification.b_only)
    both_wrong = list(classification.both_wrong)

    cases: list[DemonstrationCase] = []

    if a_only:
        cases.append(
            DemonstrationCase(
                question_id=a_only[0],
                role=ROLE_AGENT_HELPED,
                note=(
                    "Baseline answered this incorrectly; the agent's final answer was judged correct. The selector "
                    "only compares these two final outcomes - it does not know, and does not claim, that the "
                    "agent's extra retrieval/assessment round(s) specifically were the reason for the difference."
                ),
            )
        )
    else:
        cases.append(
            DemonstrationCase(
                question_id="",
                role=ROLE_AGENT_HELPED,
                note="OMITTED: no matched question in this run had the agent succeed where the baseline failed.",
            )
        )

    if b_only:
        cases.append(
            DemonstrationCase(
                question_id=b_only[0],
                role=ROLE_AGENT_HURT,
                note="Required negative case (anti-cherry-picking): the baseline answered this correctly, the agent did not.",
            )
        )
    elif both_wrong:
        cases.append(
            DemonstrationCase(
                question_id=both_wrong[0],
                role=ROLE_AGENT_DID_NOT_HELP,
                note=(
                    "Required negative case (anti-cherry-picking) - fallback bucket: no question in this run had "
                    "the agent fail where the baseline succeeded (b_only is empty), so this is a question both "
                    "systems got wrong, not a case where the agent was strictly worse than the baseline."
                ),
            )
        )
    else:
        cases.append(
            DemonstrationCase(
                question_id="",
                role=ROLE_AGENT_HURT,
                note=(
                    "OMITTED: neither b_only nor both_wrong had any question in this run - the agent matched or "
                    "beat the baseline on every matched question. This is reported honestly, not backfilled with "
                    "a both-correct question, since that would defeat the point of the required negative case."
                ),
            )
        )

    return cases

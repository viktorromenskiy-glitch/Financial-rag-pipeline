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


def _matched_question_ids(baseline_records: dict[str, dict], agent_records: dict[str, dict]) -> list[str]:
    return sorted(set(baseline_records) & set(agent_records))


def select_demonstration_cases(
    baseline_records: dict[str, dict],
    agent_records: dict[str, dict],
    *,
    agent_success_field: str = "success",
    baseline_success_field: str = "judge_correct",
) -> list[DemonstrationCase]:
    """Applies the pre-registered rule above to two loaded results sets.

    `baseline_records`/`agent_records`: {question_id: record}, matching
    exactly the shape scripts/run_agent_eval.py's `_load_jsonl_checkpoint`
    already produces from baseline_results.jsonl/agent_results.jsonl (see
    that script) - this function takes plain dicts, not file paths, so it
    stays independently testable with fabricated fixtures (see
    tests/test_agent_demonstration.py) and has no file I/O of its own.

    Raises:
        ValueError: if there is no question_id present in both inputs at
            all - there is nothing to select a demonstration case from,
            and returning an empty list silently would look like "ran
            fine, found nothing worth showing" rather than "the two
            result sets don't overlap", which is almost certainly a bug
            in whatever produced them.
    """
    matched = _matched_question_ids(baseline_records, agent_records)
    if not matched:
        raise ValueError("No question_id is present in both baseline_records and agent_records - nothing to select from")

    a_only: list[str] = []
    b_only: list[str] = []
    both_wrong: list[str] = []
    for question_id in matched:
        a_correct = bool(agent_records[question_id][agent_success_field])
        b_correct = bool(baseline_records[question_id][baseline_success_field])
        if a_correct and not b_correct:
            a_only.append(question_id)
        elif not a_correct and b_correct:
            b_only.append(question_id)
        elif not a_correct and not b_correct:
            both_wrong.append(question_id)
        # both_correct: neither bucket - not a candidate for either case.

    cases: list[DemonstrationCase] = []

    if a_only:
        cases.append(
            DemonstrationCase(
                question_id=a_only[0],
                role=ROLE_AGENT_HELPED,
                note="Baseline answered this incorrectly; the agent's extra retrieval/assessment round(s) got it right.",
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

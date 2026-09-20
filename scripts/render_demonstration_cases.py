"""Day 3 demonstration-case report - plan_rabot_posle_ekspertizy_agent_profil.md,
Day 3: "1-2 demonstration cases with the full trace" + the required negative
case, selected by agent/demonstration.py's pre-registered formal rule (see
that module's docstring for the rule itself, written before this script
was ever run against real data).

Thin real-data orchestration only - same category as
scripts/mcnemar_agent_eval.py, not scripts/run_agent_eval.py: this script
makes NO API calls of its own, it only reads the already-committed
results/<run_id>/{baseline_results.jsonl,agent_results.jsonl,agent_trace.jsonl}
that scripts/run_agent_eval.py produced, so it has no test file (matching
this project's convention that thin scripts/*.py orchestration is
untested while the library code it calls - agent/demonstration.py,
agent/debug_view.py - is). Selection logic and trace rendering each have
their own offline test suite; this script is just wiring.

Usage (after scripts/run_agent_eval.py has produced results for RUN_ID):
    !python scripts/render_demonstration_cases.py
    !python scripts/render_demonstration_cases.py --all-discordant

--all-discordant is purely additive: the two pre-registered demonstration
cases above are always selected and rendered exactly the same way,
regardless of this flag - it only appends a short summary table of every
OTHER discordant question (every a_only/b_only question_id besides the one
each already shows), so the two headline cases can't be read as "the only
discordant questions in this run" when the run actually had more of them.
Added per the Day 3 expertise's round 4 finding (operational-readiness
angle): with only n=30+3, two cases are an illustration of the selection
rule, not a statistical summary - McNemar (scripts/mcnemar_agent_eval.py)
is still the source for the actual effect-size conclusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.debug_view import render_question_trace  # noqa: E402
from agent.demonstration import classify_matched_questions, select_demonstration_cases  # noqa: E402
from pipeline.common.run_manifest import verify_manifest_coverage  # noqa: E402

RUN_ID = "agent_eval_day2"  # must match scripts/run_agent_eval.py's RUN_ID
ROOT = Path(__file__).resolve().parent.parent / "results" / RUN_ID
OUTPUT_PATH = ROOT / "demonstration_cases.md"


def _load_jsonl(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - run scripts/run_agent_eval.py first (needs real API credentials, "
            f"see that script's module docstring)"
        )
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec[key]] = rec
    return out


def _render_case_section(role_title: str, case, baseline: dict, agent: dict, trace_by_question: dict) -> str:
    if not case.question_id:
        return f"## {role_title}\n\n{case.note}\n"

    qid = case.question_id
    b = baseline.get(qid, {})
    a = agent.get(qid, {})
    lines = [
        f"## {role_title}: `{qid}`",
        "",
        case.note,
        "",
        f"- gold answer: `{a.get('gold_answer', b.get('gold_answer'))}`",
        f"- baseline answer: `{b.get('answer_text')}`  (judge_correct={b.get('judge_correct')}, "
        f"deterministic_match={b.get('deterministic_match')})",
        f"- agent answer: `{a.get('answer_text')}`  (success={a.get('success')}, "
        f"reason={a.get('success_reason')}, stopped_reason={a.get('stopped_reason')}, "
        f"additional_calls_used={a.get('additional_calls_used')})",
        "",
        "### Full agent trace",
        "",
        "```",
        render_question_trace(qid, trace_by_question.get(qid, [])),
        "```",
        "",
    ]
    return "\n".join(lines)


def _discordant_summary_row(question_id: str, baseline: dict, agent: dict) -> str:
    b = baseline.get(question_id, {})
    a = agent.get(question_id, {})
    return (
        f"| `{question_id}` | success={a.get('success')} reason={a.get('success_reason')} "
        f"stopped_reason={a.get('stopped_reason')} additional_calls_used={a.get('additional_calls_used')} | "
        f"judge_correct={b.get('judge_correct')} |"
    )


def _all_discordant_section(baseline: dict, agent: dict, shown_question_ids: set[str]) -> str:
    """A summary table of every discordant question (a_only + b_only) that
    ISN'T one of the two already-rendered headline cases - see the
    --all-discordant docstring at the top of this file for why this
    exists and why it never changes which two cases are selected above."""
    classification = classify_matched_questions(baseline, agent)
    remaining_a_only = [q for q in classification.a_only if q not in shown_question_ids]
    remaining_b_only = [q for q in classification.b_only if q not in shown_question_ids]

    lines = [
        "## All other discordant questions in this run",
        "",
        "Not selection - just visibility. The two cases above are the pre-registered demonstration "
        "(one per anti-cherry-picking rule); this table lists every OTHER question in this run where the two "
        "systems disagreed, so the two headline cases above are never mistaken for the full picture at this "
        "sample size (n=30+3). For the actual statistical conclusion, see the McNemar effect-size result "
        "(scripts/mcnemar_agent_eval.py), not a count of rows in this table.",
        "",
    ]

    if remaining_a_only:
        lines += [
            "### Agent succeeded, baseline failed (besides the positive case above)",
            "",
            "| question_id | agent | baseline |",
            "|---|---|---|",
        ]
        lines += [_discordant_summary_row(q, baseline, agent) for q in remaining_a_only]
        lines.append("")
    else:
        lines += ["### Agent succeeded, baseline failed (besides the positive case above)", "", "(none)", ""]

    if remaining_b_only:
        lines += [
            "### Baseline succeeded, agent failed (besides the negative case above)",
            "",
            "| question_id | agent | baseline |",
            "|---|---|---|",
        ]
        lines += [_discordant_summary_row(q, baseline, agent) for q in remaining_b_only]
        lines.append("")
    else:
        lines += ["### Baseline succeeded, agent failed (besides the negative case above)", "", "(none)", ""]

    return "\n".join(lines)


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - scripts/run_agent_eval.py now writes this pre-registration manifest "
            "before making any API call (claude/itog_ekspertizy_den3_dizayn.md, item 10). A results directory "
            "without it predates that change or was assembled by hand; re-run scripts/run_agent_eval.py (with "
            "real API credentials) rather than selecting demonstration cases from unregistered data."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-discordant",
        action="store_true",
        help="Also list every other discordant question (a_only/b_only) besides the two selected headline "
        "cases. Purely additive - never changes which two cases are selected as the demonstration.",
    )
    args = parser.parse_args()

    baseline = _load_jsonl(ROOT / "baseline_results.jsonl", "question_id")
    agent = _load_jsonl(ROOT / "agent_results.jsonl", "question_id")

    # Refuse to select demonstration cases from data that doesn't match what
    # was pre-registered before this run: catches both a favorable subset of
    # runs being committed and inconvenient rows being dropped/added before
    # committing (claude/itog_ekspertizy_den3_dizayn.md, item 10).
    manifest = _load_manifest(ROOT / "run_manifest.json")
    verify_manifest_coverage(manifest, baseline_question_ids=baseline.keys(), agent_question_ids=agent.keys())

    # Guard-blocked questions (agent/success.py's success=None, reason=
    # REASON_GUARD_BLOCKED_PRE_RETRIEVAL) must never reach classify_matched_
    # questions/select_demonstration_cases below: agent/demonstration.py's
    # _as_bool() deliberately raises TypeError on anything that isn't a real
    # bool (by design - a silent miscoercion could move a question into the
    # wrong bucket), so an unfiltered success=None record would crash this
    # whole script rather than being gracefully skipped. The unfiltered
    # `agent` dict above is still used for the manifest coverage check,
    # which is about which question_ids were pre-registered and run, not
    # about their success values - see item 4 of
    # claude/itog_ekspertizy_cuad_overrefusal_fix.md.
    agent_scored = {qid: rec for qid, rec in agent.items() if rec.get("success") is not None}

    trace_path = ROOT / "agent_trace.jsonl"
    trace_records: list[dict] = []
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as f:
            trace_records = [json.loads(line) for line in f if line.strip()]
    from agent.debug_view import group_by_question

    trace_by_question = group_by_question(trace_records)

    cases = select_demonstration_cases(baseline, agent_scored)
    role_titles = {
        "agent_helped": "Positive case (agent succeeded, baseline failed)",
        "agent_hurt": "Required negative case (agent failed, baseline succeeded)",
        "agent_did_not_help": "Required negative case (fallback: both systems failed - agent was not strictly worse, but did not help either)",
    }

    sections = [
        "# Day 3 demonstration cases",
        "",
        f"Run: `{RUN_ID}`. Selected by agent/demonstration.py's pre-registered formal rule "
        "(written before this run's data existed) - not chosen for how the trace looks.",
        "",
    ]
    for case in cases:
        sections.append(_render_case_section(role_titles[case.role], case, baseline, agent_scored, trace_by_question))

    if args.all_discordant:
        shown = {case.question_id for case in cases if case.question_id}
        sections.append(_all_discordant_section(baseline, agent_scored, shown))

    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for case in cases:
        print(f"  {case.role}: {case.question_id or '(omitted)'} - {case.note}")


if __name__ == "__main__":
    main()

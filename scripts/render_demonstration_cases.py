"""Day 3 demonstration-case report - plan_rabot_posle_ekspertizy_agent_profil.md,
День 3: "1-2 demonstration cases с полным трейсом" + the required negative
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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.debug_view import render_question_trace  # noqa: E402
from agent.demonstration import select_demonstration_cases  # noqa: E402

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


def main() -> None:
    baseline = _load_jsonl(ROOT / "baseline_results.jsonl", "question_id")
    agent = _load_jsonl(ROOT / "agent_results.jsonl", "question_id")

    trace_path = ROOT / "agent_trace.jsonl"
    trace_records: list[dict] = []
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as f:
            trace_records = [json.loads(line) for line in f if line.strip()]
    from agent.debug_view import group_by_question

    trace_by_question = group_by_question(trace_records)

    cases = select_demonstration_cases(baseline, agent)
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
        sections.append(_render_case_section(role_titles[case.role], case, baseline, agent, trace_by_question))

    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for case in cases:
        print(f"  {case.role}: {case.question_id or '(omitted)'} - {case.note}")


if __name__ == "__main__":
    main()

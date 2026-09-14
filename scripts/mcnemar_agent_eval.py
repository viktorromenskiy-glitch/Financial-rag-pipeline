"""McNemar's exact test + effect-size CI between the agent and baseline
runs produced by scripts/run_agent_eval.py, on real committed results
(results/<run_id>/{agent_results.jsonl,baseline_results.jsonl}).

Same 2x2-table shape and statsmodels call as scripts/mcnemar_phase6.py
(the flagship pipeline's own A/B script), plus the effect-size CI from
pipeline.common.paired_stats - see that module's docstring for why this
specific CI method was chosen (no expert in the 4-round design review
specified one).

Per plan_rabot_posle_ekspertizy_agent_profil.md, День 2: "при малой
мощности (n=20-30) вывод - на effect size (разность долей + exact 95% CI)
и разборе discordant pairs, не на p-value." This script prints the
p-value (for completeness/transparency) but the discordant-pairs breakdown
and the CI are the primary output, printed first and more prominently.

Usage (Colab, after scripts/run_agent_eval.py has produced results for
RUN_ID):
    !python scripts/mcnemar_agent_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.paired_stats import compare_paired_binary_outcomes  # noqa: E402

RUN_ID = "agent_eval_day2"  # must match scripts/run_agent_eval.py's RUN_ID
ROOT = Path(__file__).resolve().parent.parent / "results" / RUN_ID


def _load(filename: str, success_field: str) -> dict[str, bool]:
    path = ROOT / filename
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["question_id"]] = rec[success_field]
    return out


def main() -> None:
    agent = _load("agent_results.jsonl", "success")
    baseline = _load("baseline_results.jsonl", "judge_correct")

    matched_ids = sorted(set(agent) & set(baseline))
    if set(agent) != set(baseline):
        only_agent = sorted(set(agent) - set(baseline))
        only_baseline = sorted(set(baseline) - set(agent))
        print(
            f"WARNING: question_id sets differ between agent_results.jsonl and "
            f"baseline_results.jsonl - not fully paired. Comparing only the "
            f"{len(matched_ids)} question(s) present in both.\n"
            f"  Only in agent: {only_agent}\n"
            f"  Only in baseline: {only_baseline}\n"
        )

    both_correct = a_only = b_only = both_wrong = 0
    for qid in matched_ids:
        a_correct, b_correct = agent[qid], baseline[qid]
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            both_wrong += 1

    print(f"Matched n={len(matched_ids)} (agent vs. baseline, run {RUN_ID})\n")
    print(f"agent success rate:    {sum(agent[q] for q in matched_ids)}/{len(matched_ids)}")
    print(f"baseline success rate: {sum(baseline[q] for q in matched_ids)}/{len(matched_ids)}\n")

    result = compare_paired_binary_outcomes(both_correct, a_only, b_only, both_wrong)

    print(
        f"both correct: {both_correct}   only agent correct: {a_only}   "
        f"only baseline correct: {b_only}   both wrong: {both_wrong}"
    )
    print(f"discordant pairs: {result.n_discordant}\n")

    print("--- Primary output: effect size + exact CI (per plan, not the p-value) ---")
    print(f"difference (agent - baseline): {result.difference:+.4f}")
    print(f"{int(result.confidence_level * 100)}% exact CI: [{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    if result.ci_low <= 0.0 <= result.ci_high:
        print("-> CI includes 0: no reliable evidence of a difference at this sample size.")
    else:
        direction = "agent" if result.difference > 0 else "baseline"
        print(f"-> CI excludes 0, favoring {direction} - still interpret alongside discordant-pair review below.")

    print(f"\n(for reference) McNemar exact p-value: {result.mcnemar_exact_pvalue:.4f}")
    print(
        "Discordant pairs are the questions to read individually next - "
        "see agent_trace.jsonl for the agent's reasoning on each."
    )


if __name__ == "__main__":
    main()

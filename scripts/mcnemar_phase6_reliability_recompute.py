"""Recompute the Phase 6 A/B comparison (scripts/mcnemar_phase6.py, section 30
of docs/tehnicheskoe_zadanie.md) on the reliability-checked judge, and
independently on a fully deterministic check - to answer the open question
in section 30 ("is the 'no significant difference' finding an artifact of a
noisy single-call judge?") without re-running anything paid.

Input: results/phase6_reeval_adaptive/reeval_summary.jsonl - one record per
(variant, question_id), produced by scripts/reevaluate_phase6_adaptive.py.
Each original single judge call was re-run 3 times independently (escalating
to 15 calls on disagreement) to get a majority-vote verdict per answer
(`new_judge_correct`), instead of trusting one noisy call. `deterministic_match`
is the pipeline's existing is_close_v2 numeric check on the same answers -
a second, judge-free signal available for the 94.4% of answers that are
numeric (blind to the remaining free-text / INSUFFICIENT_CONTEXT answers).

Runs the same paired McNemar test as mcnemar_phase6.py, three times:
  1. On `original_judge_correct` (single call, as originally used in section 3/30)
     - this must reproduce mcnemar_phase6.py's published numbers exactly,
       which is the check that this recompute is actually comparable to the
       original, not a different test in disguise.
  2. On `new_judge_correct` (majority of 3-15 calls, reliability-checked).
  3. On `deterministic_match` (is_close_v2, no LLM involved at all).
"""
import json
from collections import defaultdict
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "results" / "phase6_reeval_adaptive" / "reeval_summary.jsonl"
VARIANTS = ["baseline_phase6", "cite_and_check_phase6", "formula_base_phase6"]
PAIRS = [
    ("baseline_phase6", "cite_and_check_phase6"),
    ("baseline_phase6", "formula_base_phase6"),
    ("cite_and_check_phase6", "formula_base_phase6"),
]


def load(field: str) -> dict[str, dict[str, bool]]:
    data: dict[str, dict[str, bool]] = defaultdict(dict)
    with SUMMARY_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            data[rec["variant"]][rec["question_id"]] = rec[field]
    return data


def run_mcnemar(data: dict[str, dict[str, bool]], label: str) -> None:
    ids = [set(data[v].keys()) for v in VARIANTS]
    assert ids[0] == ids[1] == ids[2], f"[{label}] question_id sets differ between variants"
    n = len(ids[0])
    print(f"--- {label} (n={n}) ---")
    for v in VARIANTS:
        acc = sum(data[v].values())
        print(f"  {v}: {acc}/{n} = {acc/n:.3f}")
    for a, b in PAIRS:
        both_c = both_w = a_only = b_only = 0
        for qid in data[a]:
            ca, cb = data[a][qid], data[b][qid]
            if ca and cb:
                both_c += 1
            elif not ca and not cb:
                both_w += 1
            elif ca and not cb:
                a_only += 1
            else:
                b_only += 1
        table = [[both_c, a_only], [b_only, both_w]]
        result = mcnemar(table, exact=True)
        discordant = a_only + b_only
        sig = "significant" if result.pvalue < 0.05 else "not significant"
        print(f"  {a} vs {b}: {discordant} discordant, p={result.pvalue:.4f} ({sig} at 0.05)")
    print()


if __name__ == "__main__":
    print(f"Reading {SUMMARY_PATH}\n")

    print("=== 1. Original single-call judge (must match mcnemar_phase6.py exactly) ===")
    run_mcnemar(load("original_judge_correct"), "original_judge_correct")

    print("=== 2. Reliability-checked judge (majority of 3-15 independent calls) ===")
    run_mcnemar(load("new_judge_correct"), "new_judge_correct")

    print("=== 3. Fully deterministic check (is_close_v2, no LLM) ===")
    run_mcnemar(load("deterministic_match"), "deterministic_match")

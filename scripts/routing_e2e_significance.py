"""McNemar's exact test on END-TO-END judge_correct (not Recall@5 - that was
already tested in scripts/voyage_finance2_significance.py at the retrieval
stage only) between the pre-routing baseline run and the fixed post-routing
run, per source_dataset and overall.

Why this is needed: eval_report.md's per-source deltas after the
2026-08-15 filter fix are eyeballed on tiny buckets (n=37/90/123) - e.g.
TAT-DQA's judge accuracy landed EXACTLY on the baseline's 3-decimal value
and FinQA dropped 2.2pp - neither observation is safe to interpret without
a paired significance test, the same discipline this project already
applied to the retrieval-level voyage-4 vs voyage-finance-2 comparison
(docs/tehnicheskoe_zadanie.md, п.3a). McNemar is the correct test here too:
both runs answer the SAME 250 questions (same eval_subset_250.parquet), so
per-question judge_correct is paired, not independent.

Uses the same test (scipy.stats.binomtest on the minority discordant
count) as scripts/voyage_finance2_significance.py's mcnemar_exact(), applied
here to judge_scores["judge_correct"] instead of retrieval hit/miss.

Usage (Colab, repo root, after both runs already exist under results/):
    !python scripts/routing_e2e_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

from scipy.stats import binomtest

from pipeline.cli import load_eval_questions, load_eval_results

# Set these to your actual run_ids under results/ - BASELINE_RUN_ID is the
# pre-routing run (per project history, "full250_v3"; change if yours is
# named differently), FIXED_RUN_ID is the post-fix routed run whose
# eval_report.md you just reviewed.
BASELINE_RUN_ID = "full250_v3"
FIXED_RUN_ID = "20260815T182701Z"
QUESTIONS = "data/t2-ragbench/eval_subset_250.parquet"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main() -> None:
    items = load_eval_questions(QUESTIONS)
    id_to_source = {item["question_id"]: item["source_dataset"] for item in items}

    baseline = {r.question_id: r.judge_scores["judge_correct"] for r in load_eval_results(
        __import__("pathlib").Path("results") / BASELINE_RUN_ID / "eval_results.jsonl"
    )}
    fixed = {r.question_id: r.judge_scores["judge_correct"] for r in load_eval_results(
        __import__("pathlib").Path("results") / FIXED_RUN_ID / "eval_results.jsonl"
    )}

    common_ids = sorted(set(baseline) & set(fixed))
    missing_baseline = set(fixed) - set(baseline)
    missing_fixed = set(baseline) - set(fixed)
    print(f"Paired questions: {len(common_ids)} (baseline-only: {len(missing_fixed)}, fixed-only: {len(missing_baseline)})")
    if missing_baseline or missing_fixed:
        print("  WARNING: runs don't cover the exact same question set - comparison is over the overlap only")

    by_source: dict[str, list[str]] = {}
    for qid in common_ids:
        source = id_to_source.get(qid, "unknown")
        by_source.setdefault(source, []).append(qid)

    print(f"\n{'source_dataset':<14} {'n':>4} {'baseline':>9} {'fixed':>9} {'delta':>7} {'b (base>fixed)':>15} {'c (fixed>base)':>15} {'McNemar p':>10}")
    header_line = "-" * 100
    print(header_line)

    def report_row(name: str, ids: list[str]) -> None:
        n = len(ids)
        base_correct = sum(baseline[qid] for qid in ids)
        fixed_correct = sum(fixed[qid] for qid in ids)
        b = sum(1 for qid in ids if baseline[qid] and not fixed[qid])  # baseline correct, fixed wrong
        c = sum(1 for qid in ids if not baseline[qid] and fixed[qid])  # baseline wrong, fixed correct
        p = mcnemar_exact(b, c)
        base_acc = base_correct / n if n else float("nan")
        fixed_acc = fixed_correct / n if n else float("nan")
        sig = "significant" if p < 0.05 else "not significant"
        print(
            f"{name:<14} {n:>4} {base_acc:>9.3f} {fixed_acc:>9.3f} {fixed_acc - base_acc:>+7.3f} "
            f"{b:>15} {c:>15} {p:>10.4f}  -> {sig}"
        )

    for source in sorted(by_source):
        report_row(source, by_source[source])

    print(header_line)
    report_row("Overall", common_ids)

    print(
        "\nInterpretation note: n is small per source (especially ConvFinQA, n~37) - a 'not significant' "
        "result here means 'no detectable end-to-end signal at this sample size', not 'confirmed no effect'. "
        "If TAT-DQA's Recall@5 gain (p=0.0005 at n=2500) isn't showing up end-to-end even as a trend, that's "
        "itself informative - it says the retrieval-level gain likely isn't surviving rerank/generation, not "
        "that the test is underpowered."
    )


if __name__ == "__main__":
    main()

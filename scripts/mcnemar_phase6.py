%%writefile /content/Financial-rag-pipeline/scripts/mcnemar_phase6.py
"""McNemar's test between the three Phase 6 A/B variants, on real committed
results (results/<run_id>/eval_results.jsonl, judge_correct field - the
primary accuracy metric per eval_report.md / pipeline/cli.py's
generate_eval_report).

Paired test, matched by question_id (retrieval/reranking config is
identical across variants per notebooks/run_phase6_ab_eval.py's comment,
so all three runs share the same 250 questions).
"""
import json
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

RUNS = ["baseline_phase6", "cite_and_check_phase6", "formula_base_phase6"]
ROOT = Path(__file__).resolve().parent.parent / "results"


def load(run_id):
    path = ROOT / run_id / "eval_results.jsonl"
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["question_id"]] = rec["judge_scores"]["judge_correct"]
    return out


data = {r: load(r) for r in RUNS}

ids = [set(data[r].keys()) for r in RUNS]
assert ids[0] == ids[1] == ids[2], "question_id sets differ between runs - not directly paired!"
n = len(ids[0])
print(f"Question sets match across all 3 runs, n={n}\n")

for r in RUNS:
    acc = sum(data[r].values())
    print(f"{r}: judge accuracy = {acc}/{n} = {acc/n:.3f}")
print()

pairs = [
    ("baseline_phase6", "cite_and_check_phase6"),
    ("baseline_phase6", "formula_base_phase6"),
    ("cite_and_check_phase6", "formula_base_phase6"),
]

for a, b in pairs:
    both_correct = both_wrong = a_only = b_only = 0
    for qid in data[a]:
        ca, cb = data[a][qid], data[b][qid]
        if ca and cb:
            both_correct += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca and not cb:
            a_only += 1
        else:
            b_only += 1

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=True)

    print(f"=== {a} vs {b} ===")
    print(f"  both correct: {both_correct}   only {a} correct: {a_only}   "
          f"only {b} correct: {b_only}   both wrong: {both_wrong}")
    print(f"  discordant pairs: {a_only + b_only}")
    print(f"  McNemar exact p-value: {result.pvalue:.4f}")
    sig = "significant at alpha=0.05" if result.pvalue < 0.05 else "NOT significant at alpha=0.05"
    print(f"  -> {sig}")
    print()

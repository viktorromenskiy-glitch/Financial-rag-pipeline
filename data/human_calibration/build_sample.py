"""Samples the 30-question set for the human-judge calibration study
(plan doradotki-2, item 4).

Draws from results/error_analysis_250 (predictions.jsonl + eval_results.jsonl,
the same n=250 run behind every other headline number in this project),
stratified by source_dataset proportional to that run's own mix
(TAT-DQA 123/250, FinQA 90/250, ConvFinQA 37/250 -> 15/11/4 for n=30), with
a fixed seed so the sample is reproducible.

Writes two files, deliberately split:
  - labeling_sample.json: question, generated answer, gold answer per row -
    what goes into the labeling workbook (build_workbook.py). No judge
    verdict, no deterministic_match - the human labeler must not see
    either while labeling, or the exercise stops being an independent
    check and starts being anchored on the judge's own answer.
  - answer_key.json: judge_verdict / judge_correct / deterministic_match
    per question_id, kept out of the workbook, used only after labeling
    is complete to score agreement/TPR/TNR against the human labels.
"""

import json
import random

PREDICTIONS = "../../results/error_analysis_250/predictions.jsonl"
EVAL_RESULTS = "../../results/error_analysis_250/eval_results.jsonl"

# Proportional to the n=250 run's own source_dataset mix (TAT-DQA 123,
# FinQA 90, ConvFinQA 37) rounded to sum to 30.
ALLOCATION = {"TAT-DQA": 15, "FinQA": 11, "ConvFinQA": 4}
SEED = 20260821  # date this sample was drawn - fixed for reproducibility


def main():
    preds = {p["question_id"]: p for p in (json.loads(l) for l in open(PREDICTIONS))}
    evals = {e["question_id"]: e for e in (json.loads(l) for l in open(EVAL_RESULTS))}

    by_source = {}
    for qid, p in preds.items():
        by_source.setdefault(p["source_dataset"], []).append(qid)

    rng = random.Random(SEED)
    sample_qids = []
    for source, k in ALLOCATION.items():
        pool = sorted(by_source[source])  # sort first so sample() is deterministic
        sample_qids.extend(rng.sample(pool, k))
    rng.shuffle(sample_qids)  # presentation order shouldn't cluster by source

    rows = []
    for i, qid in enumerate(sample_qids, start=1):
        p = preds[qid]
        rows.append(
            {
                "label_id": i,
                "question_id": qid,
                "source_dataset": p["source_dataset"],
                "question": p["question"],
                "generated_answer": p["answer_text"],
                "gold_answer": p["gold_answer"],
            }
        )

    answer_key = {
        qid: {
            "judge_verdict": evals[qid]["judge_scores"]["verdict"],
            "judge_correct": evals[qid]["judge_scores"]["judge_correct"],
            "deterministic_match": evals[qid]["deterministic_match"],
        }
        for qid in sample_qids
    }

    with open("labeling_sample.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open("answer_key.json", "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=2)

    print(f"Sampled {len(rows)} questions -> labeling_sample.json, answer_key.json")


if __name__ == "__main__":
    main()

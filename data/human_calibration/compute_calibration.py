"""Scores Viktor's completed human labels against the judge and against
is_close_v2 (plan doradotki-2, item 4). See docs/tehnicheskoe_zadanie.md,
section 18, for the full write-up and honest interpretation of the result
this prints (small n, especially few negative examples - see that section
before quoting a bare percentage anywhere).

Human labels are treated as ground truth throughout: TP/FN/FP/TN and
TPR/TNR are defined relative to human_correct, not the other way round -
the whole point of this study is to check the judge (and, incidentally,
is_close_v2) against a human, not vice versa.

Confidence intervals are exact (Clopper-Pearson, scipy.stats.binomtest),
consistent with how this project already reports small-sample results
elsewhere (see the McNemar/power-analysis discussion in
docs/tehnicheskoe_zadanie.md, section 9) rather than a normal-approximation
interval that can misbehave at n=8 (one of the sub-samples here).
"""

import csv
import json
from collections import Counter

from scipy.stats import binomtest

LABELING_SAMPLE = "labeling_sample.json"
ANSWER_KEY = "answer_key.json"
FILLED_CSV = "human_calibration_labeling_FILLED.csv"
VERDICT_COLUMN = "Ваш вердикт (впишите CORRECT или INCORRECT)"


def _ci(k, n):
    if n == 0:
        return None
    result = binomtest(k, n)
    interval = result.proportion_ci(confidence_level=0.95, method="exact")
    return {"point": k / n, "k": k, "n": n, "ci95_low": interval.low, "ci95_high": interval.high}


def _confusion(records, gt_key, pred_key):
    tp = sum(1 for r in records if r[gt_key] and r[pred_key])
    fn = sum(1 for r in records if r[gt_key] and not r[pred_key])
    fp = sum(1 for r in records if not r[gt_key] and r[pred_key])
    tn = sum(1 for r in records if not r[gt_key] and not r[pred_key])
    return tp, fn, fp, tn


def main():
    labeling_sample = json.load(open(LABELING_SAMPLE, encoding="utf-8"))
    answer_key = json.load(open(ANSWER_KEY, encoding="utf-8"))
    label_id_to_qid = {row["label_id"]: row["question_id"] for row in labeling_sample}

    human_rows = list(csv.DictReader(open(FILLED_CSV, encoding="utf-8-sig")))
    if len(human_rows) != 30:
        raise ValueError(f"expected 30 labeled rows, found {len(human_rows)}")

    records = []
    for row in human_rows:
        label_id = int(row["#"])
        qid = label_id_to_qid[label_id]
        raw_verdict = row[VERDICT_COLUMN].strip()
        if raw_verdict.upper() not in ("CORRECT", "INCORRECT"):
            raise ValueError(f"row {label_id}: unrecognized verdict {raw_verdict!r}")
        key = answer_key[qid]
        records.append(
            {
                "label_id": label_id,
                "question_id": qid,
                "source_dataset": row["Датасет"],
                "human_verdict_raw": raw_verdict,
                "human_correct": raw_verdict.upper() == "CORRECT",
                "judge_correct": key["judge_correct"],
                "deterministic_match": key["deterministic_match"],
            }
        )
    json.dump(records, open("calibration_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n = len(records)
    summary = {"n": n, "sample_composition": dict(Counter(r["source_dataset"] for r in records))}

    for pred_name, pred_key in [("judge", "judge_correct"), ("deterministic_is_close_v2", "deterministic_match")]:
        tp, fn, fp, tn = _confusion(records, "human_correct", pred_key)
        summary[pred_name] = {
            "agreement": _ci(tp + tn, n),
            "TPR": _ci(tp, tp + fn),
            "TNR": _ci(tn, fp + tn),
            "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        }

    summary["discordant_with_deterministic"] = [
        r["question_id"] for r in records if r["human_correct"] != r["deterministic_match"]
    ]
    json.dump(summary, open("calibration_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

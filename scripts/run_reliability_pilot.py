"""Track A (reliability) of the judge-calibration remediation plan
(internal working materials, not in this repository - section 8, point 1,
"Track A / Track B" design, converged on by all four external experts).

Track A needs no human labels: it measures how often the judge's own
verdict on a FIXED (question, generated_answer, gold_answer) input changes
across independently repeated calls - i.e. judge noise/non-determinism
itself, separate from whether the judge is right. This is the empirical
input to picking K (how many repeated judge calls per answer are enough
for a stable majority-vote verdict) for any future re-judging.

Sample (n=40, fixed, not re-derived here - see this project's internal
working materials, not in this repository, section 5.1/8 for the 4
known-unstable ids, and this file's QUESTION_IDS
below for the 36 added by stratified sampling proportional to
(source_dataset x numeric/text) over the 246-question remainder of
results/baseline_phase6/predictions.jsonl, random.seed(20260826)):
K=15 independent judge calls per question, same input each time. K=15 is
odd specifically to avoid CORRECT/INCORRECT ties in the per-question
majority vote. Deliberately calls ClaudeJudge.judge() directly rather than
pipeline.evaluation.evaluate_answer()/evaluate_answers() - those go through
JudgeCache, which is keyed on (question, context, answer, prompt_version)
and would return the SAME cached result for every one of the 15 calls,
defeating the entire point of this pilot (independent repeated draws on
identical input).

JUDGE_PROMPT (pipeline/evaluation.py) does not actually include {context}
in the text sent to the model, even though cache_key() takes context as an
argument - so this script does not need to reconstruct the retrieval
context (not persisted in results/baseline_phase6/predictions.jsonl
anyway). Only question / generated answer(answer_text) / gold_answer are
needed, all present in predictions.jsonl.

Logs the FULL raw judge response per draw, not just the parsed verdict -
per the remediation plan's point 7 (future logging additions) and an
instrumentation gap flagged during an earlier external review (raw response
not logged made an earlier length-correlated-noise hypothesis untestable).
Applying that fix now, in this new script, rather than deferring it.

Resumable: raw_draws.jsonl is append-only, one line per (question_id,
draw_index) as soon as that single judge call returns - a mid-run crash
loses at most the one in-flight call, not the whole pilot. Re-running this
script skips any (question_id, draw_index) pair already present in
raw_draws.jsonl.

Usage (Colab, after the usual %cd + secrets-loading cells, Drive already
mounted at /content/drive):
    !python scripts/run_reliability_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports

CONFIG_PATH = "config/config.yaml"
PREDICTIONS_PATH = "results/baseline_phase6/predictions.jsonl"
RUN_ID = "reliability_pilot_track_a"
K = 15

# Fixed sample: 4 known-unstable ids (internal working materials, not in
# this repository, section 5.1) + 36 stratified by (source_dataset x numeric/text) over the
# 246-question remainder, random.seed(20260826). Allocation: TAT-DQA
# numeric=18, TAT-DQA text=1, ConvFinQA numeric=5, FinQA numeric=12,
# FinQA text=0. Fixed here as a literal list, not re-derived, so re-running
# this script (including resumed/interrupted runs) always hits the exact
# same 40 questions.
QUESTION_IDS = [
    # known-unstable (guaranteed included)
    "finqa_dev_451",
    "finqa_train_1575",
    "finqa_test_140",
    "tatqa_train_3621",
    # stratified (36)
    "tatqa_train_8832",
    "tatqa_train_8468",
    "tatqa_test_10134",
    "tatqa_train_6526",
    "tatqa_train_3059",
    "tatqa_train_7746",
    "tatqa_test_9127",
    "tatqa_train_2802",
    "tatqa_test_9415",
    "tatqa_train_101",
    "tatqa_train_1844",
    "tatqa_train_3026",
    "tatqa_train_5862",
    "tatqa_train_5321",
    "tatqa_train_4885",
    "tatqa_train_6820",
    "tatqa_train_8893",
    "tatqa_train_2388",
    "tatqa_train_5894",
    "convfinqa_1261",
    "convfinqa_1042",
    "convfinqa_2138",
    "convfinqa_781",
    "convfinqa_2019",
    "finqa_train_3478",
    "finqa_train_3064",
    "finqa_train_5004",
    "finqa_train_3095",
    "finqa_train_2522",
    "finqa_train_4507",
    "finqa_test_650",
    "finqa_train_1593",
    "finqa_train_1313",
    "finqa_test_166",
    "finqa_train_932",
    "finqa_dev_436",
]
assert len(QUESTION_IDS) == 40, f"expected 40 question ids, got {len(QUESTION_IDS)}"
assert len(set(QUESTION_IDS)) == 40, "duplicate question_id in QUESTION_IDS"

import pipeline.cli as cli
from config.config_schema import load_config
from pipeline.common.persist import save_run_to_drive, verify_run_files
from pipeline.evaluation import JUDGE_PROMPT, _extract_verdict, _judge_with_retry

# This script calls load_config()/build_clients() directly rather than
# going through pipeline.cli.main() (which calls load_dotenv() itself) -
# same situation as scripts/check_environment.py before its 2026-08-21 fix
# and scripts/analyze_generation_failures.py's first bug (both documented
# in this project's internal working materials, not in this repository):
# without this call, a real .env on disk is silently never read into
# os.environ, and load_config()
# fails with "MONGODB_URI environment variable is not set" even though
# the file is right there.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

config = load_config(CONFIG_PATH)
clients = cli.build_clients(config)
judge = cli.ClaudeJudge(clients["anthropic"], config.judge.model, config.judge.temperature)

# Pull (question, answer_text, gold_answer, source_dataset) for the 40
# fixed ids from the already-committed Phase 6 baseline run - not
# re-generated, since Track A tests judge reliability on a FIXED answer,
# not generation.
by_id: dict[str, dict] = {}
with open(PREDICTIONS_PATH, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        by_id[rec["question_id"]] = rec

missing = [qid for qid in QUESTION_IDS if qid not in by_id]
if missing:
    raise RuntimeError(
        f"{len(missing)} question_id(s) from the fixed pilot sample not found in {PREDICTIONS_PATH}: {missing}"
    )

run_dir = Path("results") / RUN_ID
run_dir.mkdir(parents=True, exist_ok=True)
raw_path = run_dir / "raw_draws.jsonl"

# Resume support: (question_id, draw_index) pairs already completed.
done: set[tuple[str, int]] = set()
if raw_path.exists():
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["question_id"], rec["draw_index"]))
    print(f"Resuming: {len(done)} draw(s) already completed in {raw_path}")

total_calls = len(QUESTION_IDS) * K
made_calls = 0
with raw_path.open("a", encoding="utf-8") as f:
    for qid in QUESTION_IDS:
        rec = by_id[qid]
        prompt = JUDGE_PROMPT.format(question=rec["question"], generated=rec["answer_text"], gold=rec["gold_answer"])
        for draw_index in range(K):
            if (qid, draw_index) in done:
                continue
            raw_response = _judge_with_retry(judge, prompt)
            verdict = _extract_verdict(raw_response).upper()
            judge_correct = "CORRECT" in verdict and "INCORRECT" not in verdict
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "draw_index": draw_index,
                        "source_dataset": rec["source_dataset"],
                        "raw_response": raw_response,
                        "verdict": verdict,
                        "judge_correct": judge_correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            made_calls += 1
            done.add((qid, draw_index))
            if made_calls % 25 == 0:
                print(f"  {len(done)}/{total_calls} draws done this session ({made_calls} new calls made)")

print(f"All {total_calls} draws present in {raw_path} ({made_calls} new judge calls made this run).")

# Consolidated per-question summary - the file verify_run_files/report
# actually reason about, one line per question rather than per draw.
draws_by_question: dict[str, list[dict]] = {qid: [] for qid in QUESTION_IDS}
with raw_path.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        draws_by_question[rec["question_id"]].append(rec)

summary_path = run_dir / "pilot_summary.jsonl"
with summary_path.open("w", encoding="utf-8") as f:
    for qid in QUESTION_IDS:
        draws = sorted(draws_by_question[qid], key=lambda r: r["draw_index"])
        verdicts = [d["judge_correct"] for d in draws]
        n_correct = sum(verdicts)
        n_incorrect = len(verdicts) - n_correct
        majority_correct = n_correct > n_incorrect  # K=15 is odd - no ties
        agreement_rate = max(n_correct, n_incorrect) / len(verdicts)
        f.write(
            json.dumps(
                {
                    "question_id": qid,
                    "source_dataset": draws[0]["source_dataset"],
                    "k": len(verdicts),
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                    "majority_correct": majority_correct,
                    "agreement_rate": agreement_rate,
                    "verdict_sequence": verdicts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
print(f"Wrote per-question summary ({len(QUESTION_IDS)} questions) to {summary_path}")

verify_run_files(
    run_dir,
    {
        "raw_draws.jsonl": total_calls,
        "pilot_summary.jsonl": len(QUESTION_IDS),
    },
)
print(f"Verified: raw_draws.jsonl has {total_calls} records, pilot_summary.jsonl has {len(QUESTION_IDS)} records.")

save_run_to_drive(run_dir, config.persistence.google_drive_results_dir, RUN_ID)

print(
    f"\nTrack A pilot data collection complete. Next step (separate, not paid - "
    f"per Раздел 4's 'finish one thing before starting the next'): reconstruct "
    f"the K'=1..15 convergence curve by subsampling from the 15 draws per "
    f"question in results/{RUN_ID}/raw_draws.jsonl, and only then decide K for "
    f"future re-judging (plan section 8, point 3)."
)

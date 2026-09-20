"""Section 8 of the judge-calibration fix plan (internal working materials,
not in this repository), points 4 and 7 together, per the user's direct
instruction (2026-08-26): re-evaluate all
existing 750 Phase 6 answers (250 questions x 3 prompt variants -
baseline_phase6 / cite_and_check_phase6 / formula_base_phase6) with an
improved judging procedure - WITHOUT re-generating anything, only re-judging
the answer_text already saved in predictions.jsonl (point 4).

Judging procedure - an adaptive scheme, empirically grounded in the Track A
pilot (doc section 12.3), not a naive "K calls per answer":

    1. 3 independent judge calls per answer.
    2. If all 3 agree - use that verdict, stop
       (K_used=3).
    3. If at least one disagrees - escalate to 15 calls specifically for
       that answer, take the majority of all 15 (K_used=15).

Both 3 and 15 are odd, so ties in the majority are impossible by construction.

Point 7 (logging) - applied to the extent it is actually
recoverable after the fact for answers that were ALREADY generated:

- The FULL raw judge response for EVERY individual call (not just
  the parsed verdict/judge_correct, as in Phase 6) - is written
  to raw_draws.jsonl. This was not logged at all before (a gap flagged
  during one of the external reviews - without raw responses there was
  no way to test the hypothesis that the length of the judge's reasoning
  affects stability).
- answer_text length (in characters) and the INSUFFICIENT_CONTEXT flag -
  are already available now, from the committed predictions.jsonl, and
  are included in reeval_summary.jsonl for later post-hoc diagnostics
  (point 5, a separate, not-yet-done step).
- NOT recoverable after the fact for these 750 answers: the full raw
  GENERATOR response (predictions.jsonl only stores the short extracted
  value via _extract_final_answer(); generation_checkpoint.jsonl itself
  is not committed to the repository) and whether the generator's
  reasoning contained citations/formulas (same issue - the reasoning was
  never saved anywhere). This applies only to FUTURE generation runs and
  requires a separate change to pipeline/generation.py that is not made
  here - not part of this script and not part of point 4.

JUDGE_PROMPT (pipeline/evaluation.py) does not use {context} in the text
sent to the model - retrieval context does not need to be reconstructed
and is not needed for re-judging (the same fact already used in
run_reliability_pilot.py).

Cost summary, so as not to run this blind: at minimum 750x3=2250
judge calls (if nothing escalated at all), realistically more by the
number of escalations x12 each - the exact number is not known in
advance, it depends on how many of the 750 answers turn out to be
contested. Progress is printed as it goes, and the script is resume-safe
in case the runtime is interrupted.

Usage (Colab, after the usual mount Drive + .env cells):
    !python scripts/reevaluate_phase6_adaptive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports

CONFIG_PATH = "config/config.yaml"
RUN_ID = "phase6_reeval_adaptive"
VARIANTS = ["baseline_phase6", "cite_and_check_phase6", "formula_base_phase6"]
K_INITIAL = 3
K_FULL = 15

# The same bug that has already happened twice in this project
# (check_environment.py, analyze_generation_failures.py, and in the first
# version of run_reliability_pilot.py - see the project's internal working
# rules, section 4): load_config()/build_clients() are called directly,
# bypassing pipeline.cli.main(), so .env needs to be read explicitly here.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pipeline.cli as cli
from config.config_schema import load_config
from pipeline.common.is_close_v2 import is_close_v2
from pipeline.common.persist import find_canonical_root, verify_run_files
from pipeline.evaluation import JUDGE_PROMPT, _extract_verdict, _judge_with_retry

config = load_config(CONFIG_PATH)
clients = cli.build_clients(config)
judge = cli.ClaudeJudge(clients["anthropic"], config.judge.model, config.judge.temperature)

# Collect all 750 items: predictions.jsonl (question/answer_text/gold)
# + eval_results.jsonl (original_judge_correct, for a direct "before/after"
# regression comparison - the project's internal working rules, section 4,
# "regression analysis after every significant change").
items: list[dict] = []
for variant in VARIANTS:
    preds_path = Path("results") / variant / "predictions.jsonl"
    eval_path = Path("results") / variant / "eval_results.jsonl"

    preds_by_qid = {}
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds_by_qid[rec["question_id"]] = rec

    orig_judge_correct_by_qid = {}
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            orig_judge_correct_by_qid[rec["question_id"]] = rec["judge_scores"]["judge_correct"]

    missing = set(preds_by_qid) - set(orig_judge_correct_by_qid)
    if missing:
        raise RuntimeError(f"{variant}: {len(missing)} question_id(s) in predictions.jsonl missing from eval_results.jsonl: {sorted(missing)[:5]}...")

    for qid, pred in preds_by_qid.items():
        items.append(
            {
                "variant": variant,
                "question_id": qid,
                "question": pred["question"],
                "source_dataset": pred["source_dataset"],
                "gold_answer": pred["gold_answer"],
                "answer_text": pred["answer_text"],
                "original_judge_correct": orig_judge_correct_by_qid[qid],
            }
        )

print(f"Loaded {len(items)} items across {len(VARIANTS)} variants (expected {250 * len(VARIANTS)})")
if len(items) != 250 * len(VARIANTS):
    raise RuntimeError(f"Expected exactly {250 * len(VARIANTS)} items (250 per variant), got {len(items)} - stopping before spending anything.")

# Write checkpoints (raw_draws.jsonl) DIRECTLY to Drive, not to a local
# ephemeral disk with copying at the end afterward - unlike
# run_reliability_pilot.py (600 calls, a short run). Here the run is an
# order of magnitude larger (2250+ calls) and has in fact already been
# interrupted once mid-run (the Colab runtime died on call #1758) - the
# local checkpoint was lost entirely, because /content does not survive a
# runtime restart, and copying to Drive (save_run_to_drive) only happened
# at the very end. find_canonical_root() still confirms that we are
# writing to the actually configured root, not a similarly-named
# directory (section 1 rule, points 4-5) - just without a separate copy
# step afterward.
drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
run_dir = drive_root / RUN_ID
run_dir.mkdir(parents=True, exist_ok=True)
raw_path = run_dir / "raw_draws.jsonl"

# Resume support: draws already made per (variant, question_id).
draws_by_item: dict[tuple[str, str], list[dict]] = {}
if raw_path.exists():
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["variant"], rec["question_id"])
            draws_by_item.setdefault(key, []).append(rec)
    n_resumed = sum(len(v) for v in draws_by_item.values())
    if n_resumed:
        print(f"Resuming: {n_resumed} draw(s) already completed in {raw_path}")

total_items = len(items)
completed_items = 0
new_calls_made = 0

with raw_path.open("a", encoding="utf-8") as f:
    for item in items:
        key = (item["variant"], item["question_id"])
        draws = sorted(draws_by_item.get(key, []), key=lambda r: r["draw_index"])
        prompt = JUDGE_PROMPT.format(question=item["question"], generated=item["answer_text"], gold=item["gold_answer"])

        def make_draw(idx: int) -> dict:
            raw_response = _judge_with_retry(judge, prompt)
            verdict = _extract_verdict(raw_response).upper()
            judge_correct = "CORRECT" in verdict and "INCORRECT" not in verdict
            rec = {
                "variant": item["variant"],
                "question_id": item["question_id"],
                "draw_index": idx,
                "raw_response": raw_response,
                "verdict": verdict,
                "judge_correct": judge_correct,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            return rec

        # Phase 1: ensure at least K_INITIAL draws exist.
        while len(draws) < K_INITIAL:
            rec = make_draw(len(draws))
            draws.append(rec)
            new_calls_made += 1

        # Escalation decision is fixed once, from the first K_INITIAL draws only.
        first_verdicts = [d["judge_correct"] for d in draws[:K_INITIAL]]
        target_k = K_INITIAL if len(set(first_verdicts)) == 1 else K_FULL

        # Phase 2: escalate if needed.
        while len(draws) < target_k:
            rec = make_draw(len(draws))
            draws.append(rec)
            new_calls_made += 1

        draws_by_item[key] = draws
        completed_items += 1
        if completed_items % 50 == 0:
            print(f"  {completed_items}/{total_items} items judged this session ({new_calls_made} new judge calls made)")

print(f"All {total_items} items judged ({new_calls_made} new judge calls made this run).")

# Consolidated per-item summary.
summary_path = run_dir / "reeval_summary.jsonl"
total_raw_expected = 0
n_escalated = 0
n_changed = 0
with summary_path.open("w", encoding="utf-8") as f:
    for item in items:
        key = (item["variant"], item["question_id"])
        draws = sorted(draws_by_item[key], key=lambda r: r["draw_index"])
        first_verdicts = [d["judge_correct"] for d in draws[:K_INITIAL]]
        target_k = K_INITIAL if len(set(first_verdicts)) == 1 else K_FULL
        used = draws[:target_k]
        verdicts = [d["judge_correct"] for d in used]
        n_correct = sum(verdicts)
        n_incorrect = len(verdicts) - n_correct
        new_judge_correct = n_correct > n_incorrect  # target_k always odd (3 or 15) - no ties
        escalated = target_k == K_FULL
        changed = new_judge_correct != item["original_judge_correct"]

        total_raw_expected += target_k
        n_escalated += int(escalated)
        n_changed += int(changed)

        f.write(
            json.dumps(
                {
                    "variant": item["variant"],
                    "question_id": item["question_id"],
                    "source_dataset": item["source_dataset"],
                    "answer_text": item["answer_text"],
                    "answer_length_chars": len(str(item["answer_text"])),
                    "insufficient_context": str(item["answer_text"]).strip() == "INSUFFICIENT_CONTEXT",
                    "deterministic_match": is_close_v2(item["answer_text"], item["gold_answer"]),
                    "original_judge_correct": item["original_judge_correct"],
                    "new_judge_correct": new_judge_correct,
                    "k_used": target_k,
                    "escalated": escalated,
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                    "changed_from_original": changed,
                    "verdict_sequence": verdicts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"Wrote per-item summary ({total_items} items) to {summary_path}")
print(f"Escalated to K=15: {n_escalated}/{total_items} ({n_escalated / total_items * 100:.1f}%)")
print(f"Verdict changed from original single-call judgment: {n_changed}/{total_items} ({n_changed / total_items * 100:.1f}%)")

# Headline accuracy per variant - old vs new (raw output of point 4, not
# the formal McNemar significance test itself - that's a separate step,
# plan section 8 point 6, not part of what was asked for this run).
print("\nJudge accuracy per variant (old single-call vs new adaptive):")
for variant in VARIANTS:
    variant_items = [i for i in items if i["variant"] == variant]
    old_acc = sum(1 for i in variant_items if i["original_judge_correct"]) / len(variant_items)
    new_correct = 0
    for i in variant_items:
        key = (i["variant"], i["question_id"])
        draws = sorted(draws_by_item[key], key=lambda r: r["draw_index"])
        first_verdicts = [d["judge_correct"] for d in draws[:K_INITIAL]]
        target_k = K_INITIAL if len(set(first_verdicts)) == 1 else K_FULL
        verdicts = [d["judge_correct"] for d in draws[:target_k]]
        new_correct += int(sum(verdicts) > len(verdicts) - sum(verdicts))
    new_acc = new_correct / len(variant_items)
    print(f"  {variant:22s} old={old_acc:.3f} ({sum(1 for i in variant_items if i['original_judge_correct'])}/{len(variant_items)})  new={new_acc:.3f} ({new_correct}/{len(variant_items)})")

verify_run_files(
    run_dir,
    {
        "raw_draws.jsonl": total_raw_expected,
        "reeval_summary.jsonl": total_items,
    },
)
print(f"Verified: raw_draws.jsonl has {total_raw_expected} records, reeval_summary.jsonl has {total_items} records.")

# Already on Drive (we wrote there from the very first call) - there is
# no separate copy step, so here we just loudly print the final
# absolute path (section 1 rule, point 6), as save_run_to_drive() usually
# does in other scripts.
resolved_run_dir = run_dir.resolve()
print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE (written directly during the run): {resolved_run_dir}\n{'=' * 70}\n")

print(
    f"\nRe-evaluation (item 4) and judge-draft logging (item 7, the part that's "
    f"recoverable after the fact) are complete. Remaining, NOT YET done plan "
    f"steps: item 5 (post-hoc diagnostics on "
    f"answer_length_chars/insufficient_context from results/{RUN_ID}/reeval_summary.jsonl, "
    f"free) and item 6 (a formal McNemar recompute on new_judge_correct)."
)

"""Раздел 8 плана (claude/nedeterminizm_judge_ekspertiza.md), пункты 4 и 7
вместе, по прямому указанию пользователя (2026-08-26): переоценка всех
существующих 750 ответов Фазы 6 (250 вопросов x 3 варианта промпта -
baseline_phase6 / cite_and_check_phase6 / formula_base_phase6) улучшенной
процедурой судейства - БЕЗ повторной генерации, только пересуд уже
сохранённых answer_text из predictions.jsonl (пункт 4).

Процедура судейства - адаптивная схема, эмпирически обоснованная пилотом
Track A (§12.3 документа), а не наивное "K вызовов на каждый ответ":

    1. 3 независимых вызова судьи на ответ.
    2. Если все 3 совпали - использовать этот вердикт, остановиться
       (K_used=3).
    3. Если хотя бы один разошёлся - эскалировать до 15 вызовов именно на
       этот ответ, взять большинство из всех 15 (K_used=15).

И 3, и 15 - нечётные, ничьих в большинстве не бывает по построению.

Пункт 7 (логирование) - применён в той мере, в какой это реально
восстановимо задним числом для УЖЕ сгенерированных ответов:

- ПОЛНЫЙ сырой ответ судьи на КАЖДЫЙ отдельный вызов (не только
  разобранный verdict/judge_correct, как было в Фазе 6) - записывается
  в raw_draws.jsonl. Раньше это не логировалось вообще (пробел, отмеченный
  в claude/grok_followup.md - без сырых ответов нельзя было проверить
  гипотезу о влиянии длины рассуждения судьи на стабильность).
- Длина answer_text (в символах) и флаг INSUFFICIENT_CONTEXT - доступны
  уже сейчас, из committed predictions.jsonl, включены в
  reeval_summary.jsonl для последующей post-hoc диагностики (пункт 5,
  отдельный, ещё не выполненный шаг).
- НЕ восстановимо задним числом для этих 750 ответов: полный сырой ответ
  ГЕНЕРАТОРА (predictions.jsonl хранит только короткое извлечённое
  значение через _extract_final_answer(), сам generation_checkpoint.jsonl
  не закоммичен в репозиторий) и флаг наличия цитат/формул в рассуждении
  генератора (то же самое - рассуждение нигде не сохранено). Это
  применимо только к БУДУЩИМ прогонам генерации и требует отдельного,
  не сделанного здесь изменения pipeline/generation.py - не часть этого
  скрипта и не часть пункта 4.

JUDGE_PROMPT (pipeline/evaluation.py) не использует {context} в тексте,
отправляемом модели - контекст ретрива восстанавливать не нужно и он не
нужен для пересуда (тот же факт, что уже использован в
run_reliability_pilot.py).

Резюме по деньгам, чтобы не запускать вслепую: минимум 750x3=2250
вызовов судьи (если бы вообще ничего не эскалировало), реально больше на
число эскалаций x12 каждая - точное число заранее не известно, зависит от
того, сколько из 750 ответов окажутся спорными. Прогресс печатается по
ходу, скрипт resume-safe на случай обрыва рантайма.

Usage (Colab, после обычных ячеек mount Drive + .env):
    !python notebooks/reevaluate_phase6_adaptive.py
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

# Тот же баг, что уже дважды случался в проекте (check_environment.py,
# analyze_generation_failures.py, и в первой версии run_reliability_pilot.py -
# см. claude/svod_pravil_raboty.md, раздел 4): load_config()/build_clients()
# вызываются напрямую, минуя pipeline.cli.main(), поэтому .env нужно читать
# явно здесь.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pipeline.cli as cli
from config.config_schema import load_config
from pipeline.common.is_close_v2 import is_close_v2
from pipeline.common.persist import save_run_to_drive, verify_run_files
from pipeline.evaluation import JUDGE_PROMPT, _extract_verdict, _judge_with_retry

config = load_config(CONFIG_PATH)
clients = cli.build_clients(config)
judge = cli.ClaudeJudge(clients["anthropic"], config.judge.model, config.judge.temperature)

# Собрать все 750 элементов: predictions.jsonl (question/answer_text/gold)
# + eval_results.jsonl (original_judge_correct, для прямого регрессионного
# сравнения "было/стало" - claude/svod_pravil_raboty.md, раздел 4,
# "регрессионный анализ после каждого значимого изменения").
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

run_dir = Path("results") / RUN_ID
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

save_run_to_drive(run_dir, config.persistence.google_drive_results_dir, RUN_ID)

print(
    f"\nПереоценка (пункт 4) и логирование судейских черновиков (пункт 7, "
    f"в восстановимой задним числом части) завершены. Следующие, ещё НЕ "
    f"выполненные шаги плана: пункт 5 (post-hoc диагностика по "
    f"answer_length_chars/insufficient_context из results/{RUN_ID}/reeval_summary.jsonl, "
    f"бесплатно) и пункт 6 (формальный пересчёт McNemar на new_judge_correct)."
)

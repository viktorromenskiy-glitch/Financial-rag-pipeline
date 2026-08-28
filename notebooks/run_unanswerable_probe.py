"""Priority 3 из claude/plan_adversarial_robustness_priorities.md
("unanswerable / out-of-corpus вопросы") - единственный до сих пор
непротестированный класс поведения: отказывается ли пайплайн отвечать
(FINAL ANSWER: INSUFFICIENT_CONTEXT), когда в проиндексированном корпусе
объективно нет данных для ответа, или генерирует правдоподобное, но
необоснованное число.

Вход: data/unanswerable_probe/questions.jsonl (32 вопроса, построены и
программно проверены против реальных данных корпуса скриптом
data/unanswerable_probe/build_questions.py - каждая пара
компания/отсутствующий_год или полностью отсутствующая компания
подтверждена, не предположена).

В отличие от notebooks/reevaluate_phase6_adaptive.py и других eval-скриптов
этого проекта, здесь НЕТ judge-вызова и НЕТ gold_answer - у этих вопросов
по определению нет правильного числового ответа для сверки. Цена вопроса
здесь одна: retrieval + rerank + generation (без judge), то есть дешевле
обычного eval-прогона (~$0.015/вопрос вместо ~$0.016, см. docs/tehnicheskoe_zadanie.md
раздел 15 - judge-часть просто не вызывается).

Классификация ответа - программная, по тому же самому маркеру
FINAL ANSWER:, который использует остальной пайплайн (pipeline.generation._extract_final_answer):
  - answer_text == "INSUFFICIENT_CONTEXT" -> refused (ожидаемое поведение)
  - иначе -> confident_answer (неожиданное - модель выдала число там, где
    его не должно быть; это НЕ автоматически "неверно" в смысле обычного
    judge, а именно тот повод для ручного разбора, ради которого вообще
    запускается этот пробник)
Полный raw_response сохраняется для каждого вопроса (не только извлечённый
маркер) - без него нельзя отличить настоящую "уверенную выдумку" от
случая, когда модель написала отказ прозой, не попав в точный формат
маркера (см. pipeline/generation.py, module docstring, про уже
наблюдавшуюся утечку рассуждения мимо формата) - такое было бы неверно
засчитано как confident_answer без доступа к сырому тексту.

Исправлено по итогам независимой критики ДО прогона на реальных платных
API (не после): (1) изначальный список absent_company содержал компании,
которых не было в колонке company_name, но которые реально упоминались
в индексируемом тексте как названия из peer-group других компаний
(Alphabet, Meta, Tesla, Nvidia, Home Depot, Coca-Cola) - заменены на 12
компаний, подтверждённых build_questions.py's verify_absence() как
отсутствующие ВЕЗДЕ в тексте, не только в метаданных; (2) результаты
пишутся сразу в разрешённый через find_canonical_root() корень на Drive,
построчно, по ходу прогона - не локально с копированием в конце (тот же
паттерн, что notebooks/reevaluate_phase6_adaptive.py принял после
реальной потери чекпоинта раньше в этом проекте, см. docs/svod_pravil_raboty.md
раздел 1, п.3); (3) n_retrieved теперь честно отражает размер пула
гибридного retrieval (до reranker'а), не число после топ-N; сохраняются
retrieval- и rerank-скор топ-кандидата.

Использование (Colab, после mount Drive, %cd в репозиторий, git pull):
    !python notebooks/run_unanswerable_probe.py
Прогон идемпотентен: повторный запуск дозагружает уже сохранённые на
Drive результаты по probe_id и не тратит деньги повторно на уже сделанные
вопросы (тот же паттерн resume, что в reevaluate_phase6_adaptive.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402 - same recurring load_dotenv() bug as everywhere else in this repo; see scripts/check_colab_runtime.py's module docstring

load_dotenv(REPO_ROOT / ".env")

from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, build_clients, _resolve_embedding_model, _resolve_prompt_template  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.generation import generate_answer  # noqa: E402
from pipeline.indexing import validate_startup_indexes  # noqa: E402
from pipeline.reranking import rerank  # noqa: E402
from pipeline.retrieval import retrieve  # noqa: E402

RUN_ID = "unanswerable_probe"
QUESTIONS_PATH = REPO_ROOT / "data" / "unanswerable_probe" / "questions.jsonl"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Ни один из 32 вопросов не принадлежит реальному source_dataset - все
# нацелены за пределы того, что проиндексировано. Для retrieval нужен
# какой-то ярлык (per-dataset embedding routing включён только для
# TAT-DQA - см. pipeline/cli.py cmd_eval): "wrong_year"-вопросы про
# TAT-DQA-компании размечены source_dataset="TAT-DQA" (это реальное
# пространство эмбеддингов той компании), "wrong_year"-вопросы про
# FinQA/ConvFinQA-компании и все "absent_company"-вопросы - "FinQA"
# (не роутится, значит не влияет на выбор embedding-модели - см.
# _resolve_embedding_model - трактуется как обычный нероутящийся запрос).
# Это осознанный, задокументированный здесь выбор, не скрытое допущение.
TAT_DQA_KEYS = {
    "accenture-plc", "woolworths-limited", "activision-blizzard-inc",
    "adobe-systems-inc", "intu-properties",
}


def load_questions() -> list[dict]:
    items = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def infer_source_dataset(rec: dict) -> str:
    if rec["category"] == "wrong_year":
        # company key was stored implicitly via the question text at build
        # time; re-derive from build_questions.py's WRONG_YEAR_QUESTIONS
        # would require importing it - simpler and just as verifiable:
        # match on company name against the known TAT-DQA set.
        for key in TAT_DQA_KEYS:
            if key.replace("-", " ") in rec["company"].lower() or rec["company"].lower() in key.replace("-", " "):
                return "TAT-DQA"
    return "FinQA"


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)

    items = load_questions()
    print(f"Loaded {len(items)} unanswerable-probe questions from {QUESTIONS_PATH}")

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    prompt_template = _resolve_prompt_template(config.generation.prompt_variant)
    routing = config.embedding.routing

    # Пишем сразу в разрешённый корень на Drive, не локально с копированием
    # в конце - правило раздела 1 п.3 svod_pravil_raboty.md, тот же фикс,
    # что notebooks/reevaluate_phase6_adaptive.py принял после реальной
    # потери чекпоинта. find_canonical_root() подтверждает, что корень
    # существует именно как сконфигурировано (не похожая по имени папка).
    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "probe_results.jsonl"

    # Resume support: уже сделанные вопросы (по probe_id) не переспрашиваются
    # и не оплачиваются повторно при повторном запуске после сбоя.
    done: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["probe_id"]] = rec
        if done:
            print(f"Resuming: {len(done)} вопрос(ов) уже сделаны в {out_path}")

    results = list(done.values())
    t_start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(items):
            if rec["probe_id"] in done:
                continue

            source_dataset = infer_source_dataset(rec)
            is_routed_source = routing.enabled and source_dataset in routing.routed_sources
            query_model = _resolve_embedding_model(config, source_dataset)

            candidates = retrieve(
                clients["voyage"],
                collection,
                rec["question"],
                pool_size=config.retrieval.pool_size,
                vector_weight=config.retrieval.weights.vector,
                text_weight=config.retrieval.weights.text,
                source_dataset=source_dataset if is_routed_source else None,
                exclude_source_datasets=list(routing.routed_sources) if (routing.enabled and not is_routed_source) else None,
                embedding_model=query_model,
            )
            if config.reranker.enabled and candidates:
                ranked = rerank(clients["cohere"], rec["question"], candidates, top_n=config.reranker.top_n)
            else:
                ranked = candidates[: config.reranker.top_n]

            if not ranked:
                # Retrieval found literally nothing - itself a valid,
                # informative outcome for an out-of-corpus question (no
                # generation call made, no cost) - recorded as its own
                # behavior, not skipped silently.
                behavior = "no_retrieval_candidates"
                answer_text = None
                raw_response = None
                top_retrieved_context_id = None
                top_retrieval_score = None
                top_rerank_score = None
            else:
                answer = generate_answer(generator, rec["probe_id"], rec["question"], ranked, template=prompt_template)
                answer_text = answer.answer_text
                raw_response = answer.raw_response
                behavior = "refused" if answer_text.strip().upper() == "INSUFFICIENT_CONTEXT" else "confident_answer"
                top_retrieved_context_id = ranked[0].context_id
                top_retrieval_score = candidates[0].score if candidates else None
                top_rerank_score = getattr(ranked[0], "relevance_score", None)

            result = {
                **rec,
                "source_dataset_used_for_retrieval": source_dataset,
                "n_retrieved_pool": len(candidates),
                "n_reranked": len(ranked),
                "top_retrieved_context_id": top_retrieved_context_id,
                "top_retrieval_score": top_retrieval_score,
                "top_rerank_score": top_rerank_score,
                "answer_text": answer_text,
                "raw_response": raw_response,
                "behavior": behavior,
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  [{i + 1}/{len(items)}] {rec['probe_id']} ({rec['category']}, {rec['company']}) -> {behavior}"
                  + (f" ({answer_text})" if behavior == "confident_answer" else ""))

    elapsed = time.perf_counter() - t_start
    n_refused = sum(1 for r in results if r["behavior"] == "refused")
    n_confident = sum(1 for r in results if r["behavior"] == "confident_answer")
    n_no_candidates = sum(1 for r in results if r["behavior"] == "no_retrieval_candidates")
    print(f"\nГотово за {elapsed:.0f}с (эта сессия). refused={n_refused} confident_answer={n_confident} no_retrieval_candidates={n_no_candidates} из {len(results)}/{len(items)} всего")
    if n_confident:
        print(f"  ВНИМАНИЕ: {n_confident} вопрос(ов) получили уверенный числовой ответ вместо INSUFFICIENT_CONTEXT - требуют ручного разбора (raw_response сохранён для каждого).")

    verify_run_files(run_dir, {"probe_results.jsonl": len(items)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")


if __name__ == "__main__":
    main()

"""Диагностика первопричины over-refusal агента на CUAD (n=5, все 5 -
unjustified_refusal) - продолжение Дня 3
(plan_rabot_posle_ekspertizy_agent_profil.md), после закрытия каскада
интерпретации реальных прогонов (claude/itog_ekspertizy_agent_rezultaty.md).

*** ВАЖНО: это диагностический/качественный скрипт, НЕ хосестатистический
*** эксперимент. n=5 вопросов x 5 вариантов промпта = 25 вызовов assessor'а
*** не дают статистической мощности ни при каком результате - как n=5 в
*** scripts/run_cuad_smoke.py её не давал для accuracy. Цель - НЕ доказать
*** причину числом, а получить прямую, сохранённую сырую текстовую
*** улику (raw_response целиком, не только распарсенный вердикт) для
*** следующего раунда внешнего экспертного каскада, который будет решать
*** между гипотезами ниже. Один и тот же скрипт печатает агрегированные
*** счётчики только как ориентир, не как вывод.
***
*** Каждый вариант меняет РОВНО ОДНУ переменную относительно `original` -
*** по прямой просьбе пользователя после разбора минимальных доработок,
*** независимо предложенных DeepSeek и Qwen: патчить и перепрогонять сразу
*** несколько изменений одновременно не позволяет понять, какое из них (или
*** их сочетание) на самом деле что-то меняет - см.
*** claude/qwen_cuad_minimalnye_dorabotki.md, раздел "Методологическое
*** расхождение". Если после этого прогона захочется проверить КОМБИНАЦИЮ
*** вариантов (например neutral + multi_document_instruction) - это
*** отдельный, следующий шаг, после того как эффект каждой переменной по
*** отдельности уже виден.

## Гипотезы под проверкой (см. claude/status_agent_rezultaty_4_nahodki_kod.md)

1. Домен-мисматч формулировки: _ASSESSMENT_PROMPT_TEMPLATE буквально
   начинается "...a question about a company's **financial report**", а
   CUAD - юридические контракты. Проверяется вариантами `neutral` и
   `explicit_legal` ниже - тот же контекст, тот же вопрос, меняется
   ТОЛЬКО одна фраза (текстовая замена относительно оригинальной
   константы, не отдельно набранный текст - чтобы не внести случайный
   дрейф формулировки, тот же принцип, что новое правило Раздела 4
   svod_pravil_raboty.md требует для анализа: работать с целым
   оригиналом, не с пересказом).
2. Структурная несовместимость bounded-loop assessor'а с multi-document
   span-extraction (гипотеза Qwen, код-аудит и раунд 3 терминологического
   каскада; та же идея независимо предложена DeepSeek как "чек-лист
   необходимых фактов" и Qwen как "явная инструкция про синтез из
   нескольких документов" в раунде минимальных доработок) - проверяется
   вариантом `multi_document_instruction` ниже: та же ДОМЕННАЯ формулировка,
   что в `original` (переменная домена НЕ меняется здесь - её тестируют
   варианты выше), плюс одна добавленная инструкция про то, что evidence из
   одного документа может быть недостаточно, если вопрос требует фактов из
   нескольких. Отдельно от этого варианта сырой raw_response каждого вызова
   (никогда раньше не сохранявшийся - agent/loop.py трассирует только
   распарсенные sufficient/reformulated_query) даёт материал для
   качественного чтения по всем 5 вариантам: если модель explicitly пишет
   что-то в духе "there are multiple contracts/clauses and I cannot
   determine which one is being asked about" - это прямая текстовая улика
   за гипотезу 2 независимо от того, в каком именно варианте это увидено.
3. Асимметрия задачи assessor (до ответа, без gold) vs insufficiency-judge
   (постфактum, с gold) - вариант `original_with_gold_hint` проверяет,
   меняется ли вердикт того же assessor'а на той же оригинальной
   (финансовой) формулировке, если ему намекнуть, что именно должен
   установить правильный ответ - изолирует "неуверенность в полноте
   извлечения" (которую доступ к gold снял бы) от "формулировка домена
   режет по живому" (которую доступ к gold не тронет).

## Почему переиспользуется уже проиндексированная коллекция, а не
## пересобирается контекст заново

pipeline/indexing.py's build_full_indexed_content() примешивает
contextual_summary - LLM-сгенерированное обогащение (enrichment.enabled:
true в config_cuad_smoke.yaml, тот же claude-haiku-4-5-20251001, что и в
продакшене). Точно воспроизвести исходный проиндексированный текст без
either (a) реального обращения к тому же API за тем же summary, который
не гарантированно детерминирован даже при temperature=0.0, or (b) чтения
уже проиндексированного значения из той же коллекции - нельзя. Этот
скрипт выбирает (b): просто вызывает тот же search_fn против УЖЕ
существующей коллекции cuad_smoke_v1 (написанной в реальном прогоне
scripts/run_cuad_smoke.py) - тем самым получает БУКВАЛЬНО тот же
context_text, что видел агент в реальном прогоне, без единого нового
enrichment-вызова. Требует, чтобы scripts/run_cuad_smoke.py уже был
однажды успешно прогнан на этом MongoDB-инстансе - иначе скрипт
останавливается с понятной ошибкой (см. _check_already_indexed ниже),
а не тихо переиндексирует с новым (потенциально другим) enrichment.

## Стоимость

5 вариантов x 5 вопросов = 25 вызовов assessor'а (generation.model:
claude-sonnet-5, тот же, что уже используется как generator в
run_cuad_smoke.py - никакого нового вызова judge/generation, retrieval
переиспользует уже существующий индекс). При cost_per_llm_call_usd=0.02
(та же оценка, что config_cuad_smoke.yaml.agent_eval) - ориентировочно
~$0.50 + 5 retrieval/rerank вызовов (Voyage/Cohere, на порядок дешевле).

Использование (Colab, после git pull, после что scripts/run_cuad_smoke.py
уже был прогнан хотя бы раз на этом MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic.py
Прогон идемпотентен (resume by question_id+variant), как остальные
скрипты этой категории.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import functools  # noqa: E402

from agent.loop import _ASSESSMENT_PROMPT_TEMPLATE, _parse_assessment  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, build_clients  # noqa: E402
from pipeline.cuad_smoke import load_cuad_smoke_fixture  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.generation import build_context_block  # noqa: E402
from pipeline.indexing import is_indexed, validate_startup_indexes  # noqa: E402

RUN_ID = "cuad_overrefusal_diagnostic"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# Домен-фраза, которую меняем. Взята дословно из _ASSESSMENT_PROMPT_TEMPLATE
# (agent/loop.py) - НЕ перепечатана вручную, чтобы substitution гарантированно
# сработал (и упал бы с понятной ошибкой assert, а не молча совпал бы
# с оригиналом, если константа когда-нибудь изменится без обновления
# этого скрипта - см. проверку ниже сразу после определения VARIANTS).
_ORIGINAL_PHRASE = "a question about a company's financial report"

# Гипотеза 2 (multi-document): анкор для вставки - взят дословно из
# _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py), не перепечатан вручную, по
# той же причине, что _ORIGINAL_PHRASE выше. Вставляется МЕЖДУ пунктом 1
# ("Is the evidence sufficient...") и пунктом 2 ("If not sufficient...") -
# домен-фраза (_ORIGINAL_PHRASE) при этом НЕ трогается, чтобы этот вариант
# менял ровно одну переменную (инструкцию про multi-document), а не две.
_MULTI_DOC_ANCHOR = "do not use outside knowledge.\n2. If not sufficient"
_MULTI_DOC_INSTRUCTION = (
    " Note: the passages above may be drawn from more than one source document. If the "
    "question requires combining or cross-referencing facts from multiple documents, a single "
    "relevant-looking passage from only one document is NOT sufficient by itself - check whether "
    "every fact the question needs is actually present, even if scattered across different "
    "passages, before answering yes."
)

VARIANTS: dict[str, str] = {
    # Контроль - байт-в-байт тот же промпт, что реальный прогон.
    "original": _ASSESSMENT_PROMPT_TEMPLATE,
    # Гипотеза 1а: убрать домен вообще (не называть ни финансы, ни право).
    "neutral": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _ORIGINAL_PHRASE,
        "a question, using only the retrieved passages below",
    ),
    # Гипотеза 1б: назвать ПРАВИЛЬНЫЙ домен явно (не просто нейтрально).
    "explicit_legal": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _ORIGINAL_PHRASE,
        "a question about a legal contract",
    ),
    # Гипотеза 2: домен-формулировка НЕ меняется (остаётся "financial
    # report", как в original) - меняется только наличие явной инструкции
    # про multi-document synthesis. Независимо предложено DeepSeek
    # ("чек-лист необходимых фактов") и Qwen ("явная инструкция про синтез
    # из нескольких документов") - см. claude/deepseek_cuad_minimalnye_dorabotki.md,
    # claude/qwen_cuad_minimalnye_dorabotki.md.
    "multi_document_instruction": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _MULTI_DOC_ANCHOR,
        "do not use outside knowledge." + _MULTI_DOC_INSTRUCTION + "\n2. If not sufficient",
    ),
    # Гипотеза 3: оригинальная (финансовая) формулировка + намёк на то,
    # что должен установить правильный ответ, БЕЗ раскрытия самого
    # значения gold_answer текстом - вставляется отдельно на вопрос,
    # см. _build_gold_hint_prompt() ниже, не через .replace() здесь,
    # т.к. вставка зависит от per-question gold_answer.
    "original_with_gold_hint": _ASSESSMENT_PROMPT_TEMPLATE,
}

for _name, _template in (
    ("neutral", VARIANTS["neutral"]),
    ("explicit_legal", VARIANTS["explicit_legal"]),
    ("multi_document_instruction", VARIANTS["multi_document_instruction"]),
):
    if _template == _ASSESSMENT_PROMPT_TEMPLATE:
        raise RuntimeError(
            f"Ablation anchor not found in current _ASSESSMENT_PROMPT_TEMPLATE for variant "
            f"{_name!r} - the prompt constant changed since this script was written; update "
            f"_ORIGINAL_PHRASE / _MULTI_DOC_ANCHOR in scripts/run_cuad_overrefusal_diagnostic.py "
            f"to match before running this variant."
        )


def _build_gold_hint_prompt(question: str, context: str, calls_remaining: int, gold_answer: str) -> str:
    """Вариант original_with_gold_hint: тот же оригинальный промпт, плюс
    одна дополнительная строка ПОСЛЕ инструкции, называющая, какого рода
    факт должен установить правильный ответ - без прямой цитаты gold,
    только категория/форма (например "a specific date" или "a governing
    law jurisdiction name"), чтобы не превращать это в тривиальный
    "спиши ответ", а именно снять неопределённость о ПОЛНОТЕ извлечения,
    которую в реальном assessor-вызове модель не может снять."""
    base = _ASSESSMENT_PROMPT_TEMPLATE.format(question=question, context=context, calls_remaining=calls_remaining)
    hint = (
        "\n[Diagnostic hint - not present in production: a correct, complete answer to this "
        f"question would need to be a specific fact of this general shape: {_categorize_gold(gold_answer)}. "
        "Use this only to judge whether the passages above already contain a fact of that shape - "
        "do not guess or fabricate the fact itself.]\n"
    )
    return base + hint


def _categorize_gold(gold_answer: str) -> str:
    """Грубая, не идеальная категоризация формы ответа (не его значения) -
    достаточно для гипотезы 3 (снять неопределённость о ПОЛНОТЕ, не дать
    ответ). Раскрывает жанр факта, никогда не сам факт."""
    g = gold_answer.strip()
    if any(ch.isdigit() for ch in g) and len(g) <= 12:
        return "a short date or numeric value"
    if len(g.split()) <= 4:
        return "a short named entity (e.g. a jurisdiction, party name, or defined term)"
    return "a clause or span of contract text"


def _check_already_indexed(collection, records) -> None:
    # records is list[pipeline.ingestion.DocumentRecord] straight from
    # load_cuad_smoke_fixture() - a dataclass (attribute access via
    # `.context_id`), NOT a dict. Unlike scripts/run_cuad_smoke.py, this
    # function runs BEFORE dedupe_documents() (which is what converts
    # DocumentRecord -> dict elsewhere in the codebase), so dict-style
    # subscripting here was a bug: TypeError: 'DocumentRecord' object is
    # not subscriptable.
    missing = sorted({r.context_id for r in records if not is_indexed(collection, r.context_id)})
    if missing:
        raise RuntimeError(
            "CUAD smoke collection is missing document(s) this diagnostic needs to reuse "
            f"as-is: {missing}. This script deliberately does NOT index/enrich anything itself "
            "(see module docstring - reusing the exact already-indexed context, not re-deriving "
            "enrichment). Run scripts/run_cuad_smoke.py once first, then re-run this script."
        )


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[f"{rec['question_id']}::{rec['variant']}"] = rec
    return done


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)
    print(f"Reusing already-indexed cuad_smoke_v1 collection - {len(eval_items)} question(s), "
          f"{len({r.context_id for r in records})} document(s), {len(VARIANTS)} prompt variant(s) each "
          f"= {len(eval_items) * len(VARIANTS)} assessor call(s) total.")

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, variant) pair(s) already done in {out_path}")

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    search_fn = functools.partial(
        search_documents,
        clients["voyage"],
        collection,
        clients["cohere"],
        pool_size=config.retrieval.pool_size,
        vector_weight=config.retrieval.weights.vector,
        text_weight=config.retrieval.weights.text,
        reranker_enabled=config.reranker.enabled,
        reranker_top_n=config.reranker.top_n,
        embedding_model=config.embedding.model,
        source_dataset=None,
        exclude_source_datasets=None,
    )

    t_start = time.perf_counter()
    n_calls = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in eval_items:
            question_id = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]

            # Один retrieval на вопрос, переиспользуется во всех 5
            # вариантах - иначе разные варианты могли бы (в принципе)
            # видеть разный context_text, что сломало бы ablation
            # ("меняется ТОЛЬКО формулировка промпта").
            call = search_fn(question)
            context_text = build_context_block(list(call.candidates))
            context_ids = sorted(call.context_ids)

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                calls_remaining = config.agent.max_additional_tool_calls
                if variant == "original_with_gold_hint":
                    prompt = _build_gold_hint_prompt(question, context_text, calls_remaining, gold_answer)
                else:
                    prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "raw_response": raw_response,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(f"  [{n_calls}] {question_id} / {variant} -> sufficient={parsed.sufficient}")

    elapsed = time.perf_counter() - t_start
    print(f"\nГотово за {elapsed:.0f}с, {n_calls} новых вызовов assessor'а в этой сессии.")

    print("\nСводка sufficient=yes по вариантам (ориентир, НЕ статистический вывод при n=5):")
    for variant in VARIANTS:
        rows = [r for r in done.values() if r["variant"] == variant]
        n_yes = sum(1 for r in rows if r["sufficient"])
        print(f"  {variant:24s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Следующий шаг - НЕ автоматический вывод причины из счётчиков выше: прочитать "
        "raw_response каждой записи (полный текст рассуждения модели сохранён для каждого "
        "из 20 вызовов) и вынести на внешний экспертный каскад вместе со сводкой sufficient=yes."
    )


if __name__ == "__main__":
    main()

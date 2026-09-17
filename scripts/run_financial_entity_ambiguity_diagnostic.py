"""Узкая финансовая проверка деиктической/entity-гипотезы - последний
непроверенный пункт закрытого 4-экспертного каскада по CUAD over-refusal
(см. claude/itog_ekspertizy_cuad_overrefusal_fix.md, "Что осталось сделать",
п.2; полная история дизайна - claude/prompt_ekspert3_cuad_fix_dizayn.md и
claude/prompt_ekspert4_cuad_fix_dizayn.md).

## Зачем этот скрипт (коротко - полная история в итоге экспертизы)

Три раунда диагностики на CUAD (scripts/run_cuad_overrefusal_diagnostic*.py)
установили: агент отказывается отвечать на CUAD не из-за слабости промпта
assessor'а, а потому что все 5 CUAD-вопросов используют деиктическую
формулировку ("this contract") без единого идентифицирующего слова, что при
пуле из нескольких документов буквально не позволяет определить, к какому
документу относится вопрос. Проверка реального финансового домена
(claude/nahodka_deiktichnost_round3_dlya_ekspertov.md) показала: ~98%
вопросов явно называют компанию в тексте, а финансовые документы (в отличие
от CUAD) уже несут document-level `metadata_prefix` ("Company: X | Sector: Y
| Report year: Z") прямо в индексированном тексте. Четыре независимых
эксперта сошлись (см. итог экспертизы, п.1-4 сводной таблицы), что prompt-only
фикс не нужен, но явно назвали неисключённую конкурирующую гипотезу: "а что,
если assessor вообще плохо переносит multi-document контекст, даже когда
сущность названа явно?" - этот скрипт отвечает именно на неё, на РЕАЛЬНОМ
финансовом домене, а не на CUAD.

## Дизайн (согласован 4 экспертами - см. итог экспертизы, п.6 сводной таблицы)

Парное (within-subject) сравнение: одни и те же ~28 реальных финансовых
вопроса (data/financial_entity_ambiguity/fixture.json - построение и
критерии отбора см. в data/financial_entity_ambiguity/build_fixture.py) под
2 текстовыми условиями:

  original    - вопрос как есть (явно называет компанию)
  anonymized  - то же самое название заменено на "the company"/"the
                company's" (детерминированная regex-замена, проверенная
                программно на этапе построения фикстуры - см.
                build_fixture.py's anonymize_question())

Контекст (набор документов, который увидит assessor/generator) для ОБОИХ
условий ОДИНАКОВЫЙ и собран ВРУЧНУЮ (не через live retrieval): gold-документ
вопроса + 3 дистрактора из ТОЙ ЖЕ Sector (то самое поле, которое реально
попадает в metadata_prefix - см. pipeline/ingestion.py's build_metadata_prefix,
использующее company_sector, а не company_industry), с ближайшим report_year.
Все 4 документа берутся из уже проиндексированной продакшен-коллекции
(t2_ragbench_full) как есть - full_indexed_content, без изменений. Ручная
сборка контекста (а не live retrieval) - осознанный выбор согласно
конвергенции экспертов 2/3/4: изолирует assessor от изменчивости retrieval,
так что разница между original/anonymized объясняется ИСКЛЮЧИТЕЛЬНО текстом
вопроса, а не тем, что retrieval для одного условия случайно нашёл другой
набор документов.

`_ASSESSMENT_PROMPT_TEMPLATE` и `AGENT_ANSWER_PROMPT_TEMPLATE` (agent/loop.py)
используются БЕЗ ИЗМЕНЕНИЙ - этот скрипт не тестирует и не предлагает правку
промпта, только измеряет поведение существующего продакшен-промпта на
контролируемом входе. Согласовано всеми 4 экспертами как решённый вопрос
(п.4 сводной таблицы итога экспертизы): prompt-only правки признаны
бесперспективными на CUAD, а на финансовом домене их и не нужно тестировать -
здесь измеряется чувствительность СУЩЕСТВУЮЩЕГО поведения к одной конкретной
переменной (наличие имени компании в вопросе), не более.

## Что логируется на каждую (question, condition) пару

  - sufficient (вердикт assessor'а, из _parse_assessment - тот же
    production-парсер, что и в раундах 1-3, включая "последнее совпадение
    маркера побеждает")
  - final-answer текст и его корректность (is_close_v2 против gold_answer) -
    генерируется ВСЕГДА, независимо от sufficient (см. ниже, "Почему
    генерация ответа не гейтится sufficient")
  - single_shot_production_answer / single_shot_production_correct - что
    вернул бы production-агент С УЧЁТОМ политики "forced_insufficient, если
    sufficient=no" (agent/loop.py's forced_insufficient) ПОСЛЕ ОДНОГО
    assessment-вызова на этом фиксированном контексте - производное поле, не
    отдельный вызов. Названо "single_shot" (не просто "production"), потому
    что это НЕ полный агентный цикл run_agent_query - тот при sufficient=no
    мог бы переформулировать запрос и сделать дополнительный retrieval/re-
    assessment, чего здесь нет (и не может быть - весь смысл эксперимента в
    ОДНОМ фиксированном контексте, а не в повторном поиске). Переименовано
    из production_answer/production_correct по итогам второй экспертной
    проверки скрипта (см. "Что проверено вторым независимым экспертом" ниже)
    - старое имя могло читаться как "то, что реально вернул бы весь агентный
    цикл", что неверно.
  - assessor_context_source ("manual_fixture") / retrieval_used_for_assessor_context
    (всегда False) - явные, поле-уровневые (не только докстринг) маркеры
    того, что assessor/generator видели ТОЛЬКО context_text, собранный
    вручную из fixture.json, никогда результат live-retrieval вызова ниже.
    Добавлено по итогам второй экспертной проверки (п.5.4).
  - correct_document_position - позиция gold-документа в том же порядке
    [Document i], который строит build_context_block() для РУЧНОГО контекста
    (детерминированный порядок - см. main() ниже), одинаковая для обоих
    условий одного вопроса (контекст не меняется между условиями)
  - live_retrieval_gold_rank / live_retrieval_pool_size / live_retrieval_error
    (переименовано из gold_rank/retrieval_pool_size по итогам второй
    экспертной проверки, п.3 - старое имя рисковало быть прочитанным как
    "документ, который реально видел assessor") - ОТДЕЛЬНЫЙ, диагностический
    (не участвует в сборке контекста для assessor'а - см.
    retrieval_used_for_assessor_context выше) обычный retrieval-вызов
    (search_documents с той же routing-логикой, что и продакшен - см.
    _is_routed/_resolve_embedding_model в pipeline/cli.py и
    scripts/run_agent_eval.py) с ОРИГИНАЛЬНЫМ текстом вопроса - позиция
    gold-документа среди РЕАЛЬНО возвращённых кандидатов (после reranker'а,
    то есть именно то, что увидел бы assessor в обычном (не ручном) прогоне),
    или null, если retrieval вообще не нашёл gold-документ в топе, или если
    сам вызов упал (тогда live_retrieval_error содержит текст ошибки - см.
    "Что проверено вторым независимым экспертом", п.5.2). Один вызов на
    вопрос (не на condition - retrieval здесь не участвует в построении
    контекста для assessor'а, это отдельное измерение "а нашёл бы обычный
    retrieval этот документ вообще").

## Почему генерация ответа не гейтится sufficient (осознанное отклонение
## от production-политики, только для диагностики)

В продакшене (agent/loop.py's run_agent_query) генерация финального ответа
вызывается, только если assessment.sufficient=True - иначе форсируется
INSUFFICIENT_CONTEXT. Здесь генерация ответа вызывается ВСЕГДА (для обоих
sufficient=yes/no), потому что вопрос этого эксперимента шире, чем "гейтит ли
anonymized-условие sufficient чаще" - интересно также, ухудшается ли САМО
извлечение ответа (не только готовность assessor'а его подтвердить), когда
имя компании убрано, а gold-документ физически всё ещё в контексте. Оба
аспекта логируются раздельно (см. выше) - single_shot_production_correct
воспроизводит то, что вернула бы политика forced_insufficient после ОДНОГО
assessment-вызова (с форсированным INSUFFICIENT_CONTEXT при sufficient=no), а
answer_correct показывает "чистую" способность модели достать правильное
число из контекста, если её всё-таки попросить попробовать.

## Критерий решения (сформулирован по аналогии с раундом 3 - фиксированное
## число, не диапазон)

Если anonymized даёт sufficient=yes существенно (визуально, не формальный
статтест на n=28) реже, чем original, на ТЕХ ЖЕ документах - конкурирующая
гипотеза ("assessor вообще плохо переносит multi-document, независимо от
CUAD-специфичной деиктичности") получает поддержку, и entity-guard
(claude/itog_ekspertizy_cuad_overrefusal_fix.md, "Что осталось сделать", п.1)
можно проектировать с уверенностью, что он не сломает случаи, для которых
финансовый домен уже сейчас работает хорошо. Если original и anonymized дают
похожий sufficient=yes (оба высокие ИЛИ оба низкие) - конкурирующая гипотеза
не подтверждается на этом domain'е, и вопрос, нужен ли guard вообще для
финансового потока (у которого и так 98% вопросов называют компанию явно),
остаётся открытым для отдельного обсуждения с экспертами по итогам. Не
предполагается единственно верная интерпретация заранее - сырые результаты
(raw_response, оба ответа на вопрос) должны быть прочитаны человеком, не
только счётчик, как и в раундах 1-3.

## Стоимость (выше, чем раунды 1-3 CUAD - осознанно, это уже согласованный
## "узкий эксперимент", не микро-проверка)

28 вопросов x 2 условия x 2 вызова (assessment + генерация ответа) = 112
вызовов LLM + 28 дешёвых retrieval-вызовов (без LLM, только embedding +
MongoDB + опционально Cohere rerank, на live_retrieval_gold_rank). Порядок
величины - тот же, что у одного вопроса продакшен-eval (evaluate + assess +
retrieve), умноженный на 56 "вопросо-условий"; заведомо дешевле полноценного
2x2 эксперимента, который отвергнут экспертами 4 как избыточный первый шаг
(см. итог экспертизы, п.6 сводной таблицы).

## Что проверено первым независимым экспертом (пре-ран гейт, до первого
## платного запуска) и что было исправлено по итогам

Первый из двух обязательных независимых экспертов проверил код построчно и
подтвердил: контекст между `original`/`anonymized` действительно идентичен
(собирается один раз до цикла по условиям); выбор `company_sector` (а не
`company_industry`) для дистракторов методологически корректен (именно
`Sector` реально попадает в `metadata_prefix`, который видит модель). Также
нашёл и аргументировал 3 пункта, все приняты и исправлены в этой версии
скрипта:

1. **Реальная проблема с resume** (исправлено): до этой правки resume
   пропускал только сами LLM-вызовы (assessment/generate_answer) внутри
   цикла по условиям, но сборка контекста (MongoDB read) и ПЛАТНЫЙ
   live-retrieval вызов (Voyage embedding + опционально Cohere rerank)
   выполнялись заново на КАЖДОМ перезапуске для уже полностью завершённых
   вопросов - вопреки заявленной идемпотентности. Исправлено: весь блок
   вопроса (включая live-retrieval) теперь пропускается, если оба условия
   уже в checkpoint (см. `if all(...done...): continue` в начале цикла по
   items в main()).
2. **Именование полей могло вводить в заблуждение при чтении JSONL без
   докстринга** (исправлено переименованием + новыми явными полями):
   `gold_rank`/`retrieval_pool_size` -> `live_retrieval_gold_rank`/
   `live_retrieval_pool_size` (не выглядит как "документ, который видел
   assessor"); `production_answer`/`production_correct` ->
   `single_shot_production_answer`/`single_shot_production_correct` (не
   выглядит как "то, что вернул бы полный агентный цикл с
   переформулированием запроса" - здесь только один assessment-вызов на
   фиксированном контексте); добавлены явные поля `assessor_context_source`
   и `retrieval_used_for_assessor_context` в каждую запись.
3. **live-retrieval вызов не должен ронять основной эксперимент**
   (исправлено): обёрнут в try/except - ошибка (даже непредвиденная, не
   только транзиентная - search_documents сама уже деградирует изящно при
   транзиентных ошибках) логируется в `live_retrieval_error`, не прерывая
   assessment/generation для этого и последующих вопросов.

Отдельно эксперт отметил (не блокер, принято как известное ограничение):
build_fixture.py's проверка анонимизации ловит только точное `humanized`
имя, не другие возможные алиасы той же компании (например "American
Airlines" вместо "American Airlines Group") - для текущих 28 вопросов это
покрыто ручной построчной проверкой (см. build_fixture.py's вывод при
запуске - все 28 анонимизированных вопроса читались вручную, артефактов не
найдено), но не гарантировано программно для гипотетического будущего
расширения фикстуры. Осознанно не добавлена generic alias-detection логика
(риск ложных срабатываний на n=28 выше пользы) - при расширении фикстуры
в будущем этот пункт нужно будет пересмотреть.

## Перед запуском на реальные деньги

По правилу свода ("Перед запуском НОВОГО платного/диагностического
скрипта...") - показать ДВУМ независимым внешним экспертам с вопросом
"способен ли этот скрипт, как он написан, реально доставить то, что заявлено
как его цель - есть ли противоречие между целью и механизмом" ДО первого
платного запуска. Пройден пока только ПЕРВЫЙ раунд (см. выше) - ВТОРОЙ
независимый эксперт ещё не проверял ЭТУ (исправленную по итогам первого)
версию скрипта. Не запускать до второго раунда.

Использование (после `!git pull`, после того как scripts/run_eval.py index уже
проиндексировал t2_ragbench_full хотя бы один раз - этот скрипт НЕ индексирует
ничего нового, только читает уже проиндексированные документы):
    !python scripts/run_financial_entity_ambiguity_diagnostic.py
Прогон идемпотентен (resume by question_id+condition), включая live-retrieval
(см. "Что проверено первым независимым экспертом", п.1, исправлено в этой
версии) - как раунды 1-3.
"""
from __future__ import annotations

import functools
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from agent.loop import AGENT_ANSWER_PROMPT_TEMPLATE, _ASSESSMENT_PROMPT_TEMPLATE, _parse_assessment  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, _resolve_embedding_model, build_clients  # noqa: E402
from pipeline.common.is_close_v2 import is_close_v2  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.financial_entity_ambiguity_smoke import load_financial_entity_ambiguity_fixture  # noqa: E402
from pipeline.generation import build_context_block, generate_answer  # noqa: E402
from pipeline.indexing import is_indexed, validate_startup_indexes  # noqa: E402

RUN_ID = "financial_entity_ambiguity_diagnostic"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

CONDITIONS = ("original", "anonymized")

# Тот же класс риска, что в раундах 2-3 (_check_parse_reliability там) -
# перенесено без изменений, минус RELEVANT DOCUMENTS (здесь такого маркера
# нет - используется немодифицированный _ASSESSMENT_PROMPT_TEMPLATE).
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    problems = []
    if not _STRICT_SUFFICIENT_LINE_RE.search(raw_response):
        problems.append("no standalone 'SUFFICIENT: yes/no' line found")
    if not _STRICT_REFORMULATED_LINE_RE.search(raw_response):
        problems.append("no standalone 'REFORMULATED QUERY: ...' line found")
    if not problems:
        return None
    return (
        "Parse reliability check failed (" + "; ".join(problems) + ") - the parsed 'sufficient' field for "
        "this record may reflect a stray mention inside reasoning text rather than the model's real final "
        "verdict. Read raw_response manually before trusting this record."
    )


@dataclass(frozen=True)
class _FixedCandidate:
    """Минимальная форма, которую требуют build_context_block()/generate_answer()
    (только .context_id и .full_indexed_content - см. pipeline/generation.py) -
    не pipeline.retrieval.Candidate/pipeline.reranking.RerankedCandidate,
    потому что здесь нет ни retrieval-скора, ни rerank-скора: документы
    собраны вручную, не через поиск."""

    context_id: str
    full_indexed_content: str


def _is_routed(config, source_dataset: str) -> bool:
    routing = config.embedding.routing
    return routing.enabled and source_dataset in routing.routed_sources


def _fetch_full_indexed_content(collection, context_id: str) -> str:
    doc = collection.find_one({"context_id": context_id, "is_indexed": True}, {"full_indexed_content": 1, "_id": 0})
    if doc is None or "full_indexed_content" not in doc:
        raise RuntimeError(
            f"context_id={context_id!r} not found/indexed in the production collection - this diagnostic "
            f"only reuses already-indexed documents, it does not index anything itself. Run the T2-RAGBench "
            f"indexing pipeline (scripts/run_eval.py index, or pipeline.cli.cmd_index) first."
        )
    return doc["full_indexed_content"]


def _check_already_indexed(collection, items: list[dict]) -> None:
    all_ids = sorted({item["gold_context_id"] for item in items} | {cid for item in items for cid in item["distractor_context_ids"]})
    missing = [cid for cid in all_ids if not is_indexed(collection, cid)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} document(s) referenced by the fixture are missing from the production "
            f"collection (not indexed): {missing[:10]}{'...' if len(missing) > 10 else ''}. Run the "
            f"T2-RAGBench indexing pipeline first - this diagnostic never indexes anything itself."
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
            done[f"{rec['question_id']}::{rec['condition']}"] = rec
    return done


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)

    items = load_financial_entity_ambiguity_fixture()
    _check_already_indexed(collection, items)

    print(
        f"Financial entity-ambiguity diagnostic: {len(items)} question(s) x {len(CONDITIONS)} condition(s) = "
        f"{len(items) * len(CONDITIONS)} (assessment + answer-generation) pair(s) total, plus "
        f"{len(items)} gold_rank retrieval call(s)."
    )

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, condition) pair(s) already done in {out_path}")
        print(
            "  ВНИМАНИЕ: resume не отличает версии фикстуры/скрипта - если "
            "data/financial_entity_ambiguity/fixture.json или этот скрипт менялись с момента, когда была "
            "записана хотя бы одна из этих строк, старые результаты будут молча приняты как готовые. Если "
            "это не тот же самый, неизменный прогон - удалите или переименуйте diagnostic_results.jsonl "
            "перед повторным запуском."
        )

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)

    t_start = time.perf_counter()
    n_llm_calls = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in items:
            question_id = item["question_id"]

            # Пропустить ВЕСЬ вопрос (включая ручную сборку контекста и
            # платный retrieval-вызов ниже), если оба условия уже готовы -
            # реальная находка второго эксперта (см. review-документ): без
            # этой проверки resume пропускал только сами LLM-вызовы
            # (сборка контекста через MongoDB read внутри цикла ниже +
            # search_fn - платный вызов Voyage/опционально Cohere -
            # выполнялись заново на КАЖДОМ перезапуске для уже завершённых
            # вопросов, вопреки докстрингу "Прогон идемпотентен". Теперь
            # идемпотентность распространяется на retrieval, а не только на
            # ассессмент/генерацию.
            if all(f"{question_id}::{c}" in done for c in CONDITIONS):
                continue

            # Ручная сборка контекста - ОДИНАКОВОГО для обоих условий этого
            # вопроса (см. докстринг модуля, "Дизайн"). Порядок -
            # алфавитная сортировка context_id (детерминированная, без
            # скрытой случайности) - build_context_block() нумерует
            # [Document i] в этом же порядке, поэтому correct_document_position
            # ниже соответствует тому, что реально увидит модель.
            all_context_ids = sorted([item["gold_context_id"], *item["distractor_context_ids"]])
            candidates = [
                _FixedCandidate(context_id=cid, full_indexed_content=_fetch_full_indexed_content(collection, cid))
                for cid in all_context_ids
            ]
            context_text = build_context_block(candidates)
            correct_document_position = all_context_ids.index(item["gold_context_id"]) + 1

            # Диагностический, ОТДЕЛЬНЫЙ от сборки контекста live-retrieval
            # вызов - см. докстринг модуля, "Что логируется",
            # live_retrieval_gold_rank. Один раз на вопрос (не на condition),
            # всегда с оригинальным текстом вопроса - это измерение "нашёл бы
            # обычный retrieval этот документ", а не часть теста assessor'а
            # (retrieval_used_for_assessor_context=False в каждой записи ниже
            # - assessor всегда видит только ручной context_text, никогда
            # результат этого вызова).
            #
            # Обёрнуто в try/except (второй эксперт, п.5.2): это ПОБОЧНОЕ,
            # необязательное для основного эксперимента измерение -
            # search_documents (agent/tools.py) уже сам деградирует
            # изящно при транзиентных ошибках MongoDB/Cohere, но нет причин
            # позволять ЛЮБОЙ (в т.ч. непредвиденной) ошибке здесь ронять уже
            # оплаченный прогресс по assessment/generation для этого и
            # последующих вопросов - assessment ниже не зависит от этого
            # блока вообще.
            source_dataset = item["source_dataset"]
            routed = _is_routed(config, source_dataset)
            embedding_model = _resolve_embedding_model(config, source_dataset)
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
                embedding_model=embedding_model,
                source_dataset=source_dataset if routed else None,
                exclude_source_datasets=list(config.embedding.routing.routed_sources) if not routed else None,
            )
            live_retrieval_gold_rank: int | None = None
            live_retrieval_pool_size: int | None = None
            live_retrieval_error: str | None = None
            try:
                retrieval_call = search_fn(item["question_original"])
                retrieved_ids = [c.context_id for c in retrieval_call.candidates]
                live_retrieval_pool_size = len(retrieved_ids)
                if item["gold_context_id"] in retrieved_ids:
                    live_retrieval_gold_rank = retrieved_ids.index(item["gold_context_id"]) + 1
            except Exception as exc:  # noqa: BLE001 - see docstring above: this is a non-critical side measurement
                live_retrieval_error = f"{type(exc).__name__}: {exc}"
                print(f"      WARNING: live_retrieval (gold_rank) failed for {question_id!r}: {live_retrieval_error}")

            for condition in CONDITIONS:
                key = f"{question_id}::{condition}"
                if key in done:
                    continue

                question_text = item[f"question_{condition}"]
                calls_remaining = config.agent.max_additional_tool_calls
                assessment_prompt = _ASSESSMENT_PROMPT_TEMPLATE.format(
                    question=question_text, context=context_text, calls_remaining=calls_remaining
                )
                assessment_raw = generator.generate(assessment_prompt)
                n_llm_calls += 1
                parsed = _parse_assessment(assessment_raw)
                parse_warning = _check_parse_reliability(assessment_raw)

                # Генерация ответа ВСЕГДА (не только при sufficient=yes) - см.
                # докстринг модуля, "Почему генерация ответа не гейтится
                # sufficient".
                generated = generate_answer(
                    generator, question_id, question_text, candidates, template=AGENT_ANSWER_PROMPT_TEMPLATE
                )
                n_llm_calls += 1
                answer_correct = is_close_v2(generated.answer_text, item["gold_answer"])

                # single_shot_production_answer/single_shot_production_correct
                # (renamed from production_answer/production_correct per the
                # second expert's review, п.4) - производные поля,
                # воспроизводящие ТОЛЬКО agent/loop.py's forced_insufficient
                # политику ПОСЛЕ ОДНОГО assessment-вызова на этом фиксированном
                # контексте (не отдельный вызов) - НЕ полный production-цикл
                # run_agent_query (который при sufficient=no мог бы
                # переформулировать запрос и сделать дополнительные
                # tool calls/re-assessment - здесь этого нет, потому что весь
                # смысл эксперимента в ФИКСИРОВАННОМ контексте). Названо явно
                # "single_shot", чтобы не читалось как "то, что реально вернул
                # бы полный агентный цикл".
                single_shot_production_answer = generated.answer_text if parsed.sufficient else "INSUFFICIENT_CONTEXT"
                single_shot_production_correct = bool(parsed.sufficient and answer_correct)

                result = {
                    "question_id": question_id,
                    "condition": condition,
                    "source_dataset": source_dataset,
                    "gold_sector": item["gold_sector"],
                    "gold_company_name": item["gold_company_name"],
                    "gold_answer": item["gold_answer"],
                    "gold_context_id": item["gold_context_id"],
                    "distractor_context_ids": item["distractor_context_ids"],
                    "context_ids": all_context_ids,
                    "correct_document_position": correct_document_position,
                    # assessor_context_source/retrieval_used_for_assessor_context
                    # (added per the second expert's review, п.5.4) - explicit,
                    # data-level (not just docstring-level) statement that the
                    # assessor/generator above saw ONLY the manually-assembled
                    # context_text, never the live_retrieval_* fields below.
                    "assessor_context_source": "manual_fixture",
                    "retrieval_used_for_assessor_context": False,
                    # live_retrieval_* (renamed from gold_rank/retrieval_pool_size
                    # per the second expert's review, п.3) - a SEPARATE,
                    # diagnostic-only live search_documents() call, not part of
                    # the assessor's input. None/live_retrieval_error set if the
                    # call itself failed (see the try/except above) - does not
                    # affect sufficient/answer_correct/single_shot_production_*
                    # below, which never depend on this call.
                    "live_retrieval_gold_rank": live_retrieval_gold_rank,
                    "live_retrieval_pool_size": live_retrieval_pool_size,
                    "live_retrieval_error": live_retrieval_error,
                    "question_text": question_text,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "assessment_raw_response": assessment_raw,
                    "parse_warning": parse_warning,
                    "answer_text": generated.answer_text,
                    "answer_raw_response": generated.raw_response,
                    "answer_correct": answer_correct,
                    "single_shot_production_answer": single_shot_production_answer,
                    "single_shot_production_correct": single_shot_production_correct,
                }
                if parse_warning:
                    print(f"      WARNING: {parse_warning}")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(
                    f"  [{n_llm_calls}] {question_id} / {condition} -> sufficient={parsed.sufficient} "
                    f"answer_correct={answer_correct} live_retrieval_gold_rank={live_retrieval_gold_rank}"
                )

    elapsed = time.perf_counter() - t_start
    print(f"\nГотово за {elapsed:.0f}с, {n_llm_calls} новых вызовов LLM в этой сессии.")

    print("\nСводка по условиям (n={} вопросов на условие, парное сравнение, ориентир, не статистика):".format(len(items)))
    for condition in CONDITIONS:
        rows = [r for r in done.values() if r["condition"] == condition]
        n_sufficient = sum(1 for r in rows if r["sufficient"])
        n_answer_correct = sum(1 for r in rows if r["answer_correct"])
        n_single_shot_production_correct = sum(1 for r in rows if r["single_shot_production_correct"])
        print(
            f"  {condition:12s}: sufficient=yes {n_sufficient}/{len(rows)}, "
            f"answer_correct (генерация всегда) {n_answer_correct}/{len(rows)}, "
            f"single_shot_production_correct (с учётом forced_insufficient, БЕЗ полного агентного цикла) "
            f"{n_single_shot_production_correct}/{len(rows)}"
        )

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(items) * len(CONDITIONS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Читать raw_response целиком по вопросам, где original и anonymized разошлись по sufficient или "
        "answer_correct - счётчик выше ориентировочный (n=28, не статистический тест), интерпретация "
        "требует прочтения конкретных случаев, как и в раундах 1-3."
    )


if __name__ == "__main__":
    main()

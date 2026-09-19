"""Диагностика CUAD over-refusal, раунд 3 - проверка предложенного фикса
("сначала определи релевантные документы") ПЕРЕД финансовой регрессией.

*** ДИАГНОСТИЧЕСКИЙ скрипт, не статистический эксперимент. n=5 вопросов x
*** 2 варианта = 10 вызовов assessor'а не дают статистической мощности.
*** Цель - дёшево и быстро проверить одну конкретную гипотезу (структурный
*** шаг "сначала перечисли релевантные документы" внутри промпта) ДО того,
*** как тратить деньги на дорогой финансовый регресс n=30.

## Как этот скрипт появился

После раунда 2 (см. run_cuad_overrefusal_diagnostic_round2.py) была
подтверждена гипотеза 2 - причина отказов на CUAD в шуме от посторонних
документов в контексте, не в формулировке вопроса и не в домене промпта
(см. полный разбор в claude/prompt_ekspert_cuad_fix_dizayn.md). Начат
формальный каскад из 4 независимых экспертов (свод правил, раздел 2) для
дизайна production-фикса. Первый эксперт предложил конкретный фикс:
переструктурировать `_ASSESSMENT_PROMPT_TEMPLATE` так, чтобы assessor
СНАЧАЛА явно перечислял, какие из поданных документов вообще относятся к
вопросу, и только потом оценивал достаточность СТРОГО внутри этого
подмножества (третий маркер ответа `RELEVANT DOCUMENTS: <...>` перед
`SUFFICIENT:`/`REFORMULATED QUERY:`).

Прежде чем принять этот фикс, автору (этому агенту) были заданы два прямых
вопроса первому эксперту: (1) есть ли принципиальное, гарантированное
отличие механизма этого фикса от уже проверенного и провалившегося
`with_reasoning` варианта раунда 2 (оба оставляют все 4 документа физически
видимыми модели и рассчитывают, что она сама "заметит" нерелевантность) -
ответ: принципиального гарантированного отличия НЕТ, только предположение,
что явное требование ВЫБРАТЬ документ сработает надёжнее, чем просто
попросить порассуждать; (2) стоит ли перед дорогим финансовым регрессом
n=30 сначала дёшево перепроверить сам факт флипа на тех же 5 вопросах CUAD
- ответ: да, однозначно, с чётким критерием (>=3 из 5 sufficient=yes ->
можно идти в финансовый регресс; иначе гипотеза A в этом виде опровергнута,
нужен другой механизм). Этот скрипт - и есть та самая дешёвая проверка.

## Чем этот скрипт ОТЛИЧАЕТСЯ от первого черновика (важно - причины ниже)

Первый эксперт также предложил черновой код диагностического скрипта. Он
не используется дословно - при независимой проверке (обязательной для
этого проекта: см. правило "не принимать код эксперта без независимой
проверки") найдено два методологических отличия от установленной в раунде
1/раунде 2 дисциплины "менять РОВНО ОДНУ переменную за раз":

1. Черновик эксперта ПЕРЕПЕЧАТЫВАЛ весь _ASSESSMENT_PROMPT_TEMPLATE вручную
   вместо того, чтобы взять реальную константу и вставить в неё только
   новый шаг (как делали раунд 1 и раунд 2 через `.replace()` на дословном
   анкоре). Побочный эффект перепечатки: черновик заодно убрал доменную
   фразу "a question about a company's financial report", заменив её на
   нейтральное "the question below". Это ВТОРАЯ переменная, не связанная с
   проверяемой гипотезой - гипотеза 1 (доменная формулировка) уже отдельно
   проверена и ОПРОВЕРГНУТА в раунде 1 (neutral/explicit_legal дали тот же
   результат 0/5, что и original). Если бы `structured_relevant_first` дал
   флип с ОБЕИМИ изменёнными переменными одновременно, нельзя было бы
   утверждать, что флип вызван именно новым шагом определения релевантных
   документов, а не случайным взаимодействием с убранной доменной фразой.
   Этот скрипт вместо этого строит новый промпт через `.replace()` на
   дословных анкорах реальной _ASSESSMENT_PROMPT_TEMPLATE (тот же приём,
   что и в раунде 2 для with_reasoning) - доменная фраза "financial report"
   остаётся БЕЗ ИЗМЕНЕНИЙ, меняется только структура шага сначала-определи-
   релевантные-документы. Единственная проверяемая переменная - это она.
2. Черновик не включал `_check_parse_reliability` (независимая проверка,
   добавленная во втором эксперте раунда 2 - см. run_cuad_overrefusal_diagnostic_round2.py)
   - при новом маркере `RELEVANT DOCUMENTS:` риск того же класса, что уже
   был найден для `REASONING:` в раунде 2 (модель может случайно упомянуть
   "SUFFICIENT: yes" внутри перечисления документов и никогда не выдать
   отдельную финальную строку), не исчез просто потому, что маркер другой.
   Проверка перенесена без изменений.

## Третье отличие: within-run контроль "original", а не только сравнение
## с раундом 1

Черновик эксперта сравнивал бы `structured_relevant_first` только с
результатом раунда 1 (0/5, другая сессия, потенциально другая версия
модели, прогнан раньше). Этот скрипт вместо этого ЗАНОВО прогоняет
неизменный `original` (та же _ASSESSMENT_PROMPT_TEMPLATE, без единого
слова изменений) на тех же 5 вопросах В ЭТОМ ЖЕ прогоне - тот же принцип,
что "меняется ровно одна переменная", но применённый ко времени и версии
модели, а не только к тексту промпта. Стоит лишних 5 вызовов (~$0.1), но
исключает случайный вопрос "не мог ли просто дрейф модели/окружения дать
другой результат".

## Что проверяется и критерий решения (согласован с первым экспертом,
## уточнён после независимой проверки вторым экспертом - см. ниже)

- `original`: контроль, ожидание 0/5 (как в раунде 1, теперь в этом же
  прогоне для чистоты сравнения). Если original неожиданно даст
  sufficient=yes хотя бы раз - это сигнал дрейфа модели/окружения, а не
  полезный результат по гипотезе A, и его нужно исследовать отдельно
  ДО того, как интерпретировать structured_relevant_first.
- `structured_relevant_first`: кандидат-фикс первого эксперта. Порог
  зафиксирован как одно число, без диапазона: >=3 из 5 sufficient=yes -
  достаточный диагностический сигнал, чтобы переходить к дорогому
  финансовому регрессу n=30 (с обязательной проверкой, что 23/30 и 28/30
  не деградировали); 4-5 из 5 - сигнал сильнее, чем ровно 3 из 5, но порог
  прохождения тот же. Меньше 3 из 5 (0-2) - гипотеза A в этой конкретной
  формулировке считается ОПРОВЕРГНУТОЙ на диагностическом уровне, нужен
  другой механизм (например физическая фильтрация контекста, а не
  текстовая инструкция) - переходить к финансовому регрессу с этим текстом
  промпта не следует. Важная оговорка (высказана вторым экспертом при
  проверке скрипта, принята без изменений): даже 4-5 из 5 на n=5 - это
  ДИАГНОСТИЧЕСКИЙ сигнал, достаточный, чтобы оправдать следующий, более
  дорогой и статистически значимый эксперимент, а не доказательство, что
  production-фикс работает. Финансовый регресс n=30 - обязательный
  следующий шаг перед любым выводом о готовности фикса, не опциональный.

## Что именно измеряет RELEVANT DOCUMENTS - важное уточнение (второй
## эксперт, принято без изменений)

`structured_relevant_first` не выполняет физическую фильтрацию контекста
(в отличие от `single_relevant_document_only` раунда 2, где посторонние
документы физически отсутствуют в контексте). Модель по-прежнему видит
все 4 документа целиком; строка `RELEVANT DOCUMENTS: <...>` - это только
ТЕКСТОВАЯ инструкция и её текстовый результат, не программный фильтр -
Python никак не проверяет и не применяет то, что модель написала в этой
строке, перед тем как принять её `SUFFICIENT:`. Иными словами, этот
вариант тестирует не "физическое разделение релевантных и нерелевантных
документов", а "помогает ли явный двухступенчатый текстовый протокол
(сначала перечислить релевантные, потом судить только по ним) сильнее,
чем свободное рассуждение (with_reasoning) или полное отсутствие такого
шага (original)". Если структура сработает (>=3/5), это диагностический
сигнал в пользу гипотезы о протоколе, а не доказательство, что модель
технически ограничила себя перечисленным подмножеством - RELEVANT
DOCUMENTS остаётся ручным диагностическим полем для чтения человеком, а
не измерением с автоматической гарантией корректности (см. ниже про
`correct_document_position`).

## Идентификаторы документов, которые видит модель (проверено, не
## предположение)

pipeline/generation.py's build_context_block() оборачивает каждый
кандидат как `[Document {i}]` (1-indexed, в порядке ранжирования, i от 1
до len(candidates)) - то есть у модели ЕСТЬ определённая, стабильная схема
идентификаторов внутри одного вызова, вопрос закрыт эмпирической
проверкой исходного кода, а не остаётся предположением. Проблема в другом:
эти позиционные метки ("Document 2") не совпадают по формату со
`correct_document_id` (полный context_id вида
"GALACTICOMMTECHNOLOGIESINC_11_07_1997-EX-10.46-WEB HOSTING AGREEMENT") -
поэтому сравнение `relevant_documents_raw` с `correct_document_id`
напрямую (строка в строку) ничего не даст. Решение (без изменения самого
промпта - изменение формулировки RELEVANT DOCUMENTS добавило бы третью
переменную к тесту, а не только диагностику): каждая запись результата
дополнительно включает `correct_document_position` - позицию (1-indexed,
в том же порядке, что и `[Document i]` в build_context_block) правильного
документа среди кандидатов ЭТОГО вопроса. При разборе результатов
достаточно сравнить `relevant_documents_raw` (например "Document 2") с
`correct_document_position` (например 2), не открывая исходный код.

`raw_response` каждой записи включает поле `relevant_documents_raw`
(что модель написала в строке RELEVANT DOCUMENTS, если написала),
`correct_document_id` (реальный id документа, отвечающего на вопрос, из
той же фикстуры, что и в раунде 2) и `correct_document_position` (см.
выше) - чтобы при разборе результатов сразу видно было не только счётчик
sufficient=yes, но и совпадает ли названный моделью документ с настоящим.

## Надёжность парсинга SUFFICIENT/REFORMULATED QUERY - проверено, не
## предположение (второй эксперт поднял вопрос, ответ уже в production-коде)

_parse_assessment (agent/loop.py) при нескольких совпадениях маркера в
ответе берёт ПОСЛЕДНЕЕ (`matches[-1]`), не первое - это явно
задокументировано в его докстринге (со ссылкой на находку внешнего code
review, claude/status_agent_rezultaty_4_nahodki_kod.md, находка 1) и
подтверждено чтением исходного кода: `_SUFFICIENT_RE`/`_REFORMULATED_RE`
собираются через `list(...finditer(...))`, решение берётся из `[-1]`. Риск
того, что случайное промежуточное "SUFFICIENT: yes" внутри рассуждения
будет принято за финальный вердикт вместо настоящей последней строки, этим
устранён на уровне production-парсера; `_check_parse_reliability` ниже -
дополнительная, независимая проверка ФОРМЫ ответа (что маркеры пришли как
отдельные строки, а не только где-то внутри текста), а не замена этой
защиты.

## Перед повторным запуском - не смешивать разные версии эксперимента
## (второй эксперт, процедурная оговорка)

Resume по `question_id::variant` (см. `_load_checkpoint`) не отличает
разные версии промпта/фикстуры/конфигурации между прогонами - если этот
скрипт менялся МЕЖДУ прогонами (не в этой правке - структура промпта не
менялась), старые строки в `results/cuad_overrefusal_diagnostic_round3/
diagnostic_results.jsonl` будут молча приняты как "уже готово" и не
перезапустятся. Для первого чистого запуска это не проблема (директории
ещё не существует). Если раунд 3 когда-либо будет менять сам промпт/логику
после первого запуска - удалить или переименовать старый
`diagnostic_results.jsonl` перед повторным запуском, а не полагаться на
resume.

## Переиспользование корпуса/индекса

Без изменений от раунда 1/раунда 2: тот же уже проиндексированный
cuad_smoke_v1, тот же search_fn через agent.tools.search_documents, один
retrieval-вызов на вопрос (переиспользуется для обоих вариантов - оба
используют полный, нефильтрованный контекст из всех кандидатов, поэтому,
в отличие от `single_relevant_document_only` раунда 2, здесь снова
достаточно ОДНОГО общего retrieval на вопрос, как в самом раунде 1).

## Стоимость

5 вопросов x 2 варианта = 10 вызовов assessor'а, тот же порядок величины,
что раунд 2 (~$0.2).

## Перед запуском на реальные деньги

По правилу свода ("Перед запуском НОВОГО платного/диагностического
скрипта...") - показать ДВУМ независимым внешним экспертам с вопросом
"способен ли этот скрипт, как он написан, реально доставить то, что
заявлено как его цель - есть ли противоречие между целью и механизмом" ДО
первого платного запуска. Не запускать, пока это не сделано.

Использование (Colab, после git pull, после того как
scripts/run_cuad_smoke.py уже был прогнан хотя бы раз на этом MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic_round3.py
Прогон идемпотентен (resume by question_id+variant), как раунды 1 и 2.
"""
from __future__ import annotations

import json
import re
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

RUN_ID = "cuad_overrefusal_diagnostic_round3"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# Анкоры взяты дословно из текущей _ASSESSMENT_PROMPT_TEMPLATE
# (agent/loop.py) - не перепечатаны вручную, как и в раунде 2, по той же
# причине: гарантированный явный сбой при несовпадении, а не тихий уход в
# сторону от продакшен-промпта.
_DECIDE_ANCHOR = (
    "Decide:\n"
    "1. Is the evidence above sufficient to answer the question completely and precisely? "
    "Judge only from the retrieved passages above - do not use outside knowledge.\n"
    "2. If not sufficient AND you still have calls remaining, propose ONE reformulated search "
    "query that is more likely to find the missing information (e.g. naming a specific line "
    "item, period, or company the current passages are missing). If you have no better query "
    "to try, or no calls remain, say NONE."
)
_DECIDE_WITH_RELEVANCE_STEP = (
    "First, identify which of the retrieved passages above (if any) come from a document that is "
    "actually relevant to the question - a document is relevant only if it concerns the same "
    "entity, agreement, company, period, or subject that the question is asking about. Passages "
    "from a completely unrelated document must be ignored in the steps below.\n\n"
    "Decide, using ONLY the passages you identified as relevant:\n"
    "1. Is the evidence above sufficient to answer the question completely and precisely? "
    "Judge only from the relevant retrieved passages - do not use outside knowledge.\n"
    "2. If not sufficient AND you still have calls remaining, propose ONE reformulated search "
    "query that is more likely to find the missing information (e.g. naming a specific line "
    "item, period, or company the current passages are missing). If you have no better query "
    "to try, or no calls remain, say NONE."
)
_FORMAT_ANCHOR = (
    "Respond in exactly this format, nothing else:\n\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)
_FORMAT_WITH_RELEVANT_DOCS = (
    "Respond in exactly this format, nothing else:\n\n"
    "RELEVANT DOCUMENTS: <comma-separated short identifiers of the relevant document(s), or NONE>\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)

if _DECIDE_ANCHOR not in _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_DECIDE_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the prompt constant "
        "changed since this script was written; update _DECIDE_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round3.py to match before running."
    )
if _FORMAT_ANCHOR not in _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_FORMAT_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the prompt constant "
        "changed since this script was written; update _FORMAT_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round3.py to match before running."
    )

_STRUCTURED_TEMPLATE = _ASSESSMENT_PROMPT_TEMPLATE.replace(
    _DECIDE_ANCHOR, _DECIDE_WITH_RELEVANCE_STEP
).replace(_FORMAT_ANCHOR, _FORMAT_WITH_RELEVANT_DOCS)

if _STRUCTURED_TEMPLATE == _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_STRUCTURED_TEMPLATE ended up identical to _ASSESSMENT_PROMPT_TEMPLATE after both "
        "replacements - anchors matched but produced no change, which should be impossible; "
        "investigate before running."
    )

# original прогоняется ЗАНОВО в этом же прогоне как within-run контроль
# (см. докстринг выше - "Третье отличие"), а не берётся из раунда 1.
VARIANTS: dict[str, str] = {
    "original": _ASSESSMENT_PROMPT_TEMPLATE,
    "structured_relevant_first": _STRUCTURED_TEMPLATE,
}

# Перенесено без изменений из раунда 2 (run_cuad_overrefusal_diagnostic_round2.py) -
# тот же риск класса "случайное упоминание маркера внутри рассуждения до
# настоящей финальной строки", теперь применительно к RELEVANT DOCUMENTS
# вместо REASONING.
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)
# Диагностическая (не production) выборка значения RELEVANT DOCUMENTS -
# только для логирования/анализа, не участвует в _parse_assessment и не
# меняет его. Последнее совпадение, тот же принцип, что и у двух других
# маркеров.
_RELEVANT_DOCS_RE = re.compile(r"^\s*RELEVANT DOCUMENTS\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    """Тот же контракт, что и в раунде 2: см. run_cuad_overrefusal_diagnostic_round2.py
    для полного объяснения. Проверяет только два маркера, которые реально
    участвуют в _parse_assessment (agent/loop.py) - SUFFICIENT/REFORMULATED
    QUERY; RELEVANT DOCUMENTS не влияет на production-парсер, поэтому его
    отсутствие не считается проблемой парсинга (только теряется диагностика)."""
    problems = []
    if not _STRICT_SUFFICIENT_LINE_RE.search(raw_response):
        problems.append("no standalone 'SUFFICIENT: yes/no' line found")
    if not _STRICT_REFORMULATED_LINE_RE.search(raw_response):
        problems.append("no standalone 'REFORMULATED QUERY: ...' line found")
    if not problems:
        return None
    return (
        "Parse reliability check failed (" + "; ".join(problems) + ") - the parsed "
        "'sufficient'/'reformulated_query' fields for this record may reflect a stray "
        "mention inside reasoning text rather than the model's real final verdict. "
        "Read raw_response manually before trusting this record."
    )


def _extract_relevant_documents_raw(raw_response: str) -> str | None:
    matches = list(_RELEVANT_DOCS_RE.finditer(raw_response))
    if not matches:
        return None
    value = matches[-1].group(1).strip()
    return value or None


def _verify_relevant_doc_mapping(records, eval_items) -> None:
    """Тот же belt-and-suspenders guard, что и в раунде 2 (см. там для
    полного объяснения) - здесь используется только чтобы безопасно
    построить `correct_document_id` для каждой записи результата
    (диагностическое поле, помогает при чтении raw_response сверять, что
    модель написала в RELEVANT DOCUMENTS, с реально верным документом);
    в этом раунде это поле не используется для фильтрации контекста -
    оба варианта получают полный, нефильтрованный контекст."""
    problems = []
    for rec, item in zip(records, eval_items):
        gold = item["gold_answer"].strip()
        probe = gold if len(gold) <= 40 else gold[:40]
        if probe.upper() not in rec.context.upper():
            problems.append(
                f"question_id={item['question_id']!r}: gold_answer prefix {probe!r} not found in "
                f"document {rec.context_id!r} - correct_document_id would be wrong for this question"
            )
    if problems:
        raise RuntimeError(
            "correct_document_id mapping failed verification:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def _check_already_indexed(collection, records) -> None:
    missing = sorted({r.context_id for r in records if not is_indexed(collection, r.context_id)})
    if missing:
        raise RuntimeError(
            "CUAD smoke collection is missing document(s) this diagnostic needs to reuse "
            f"as-is: {missing}. Run scripts/run_cuad_smoke.py once first, then re-run this script."
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
    """Runs round 3's `original` vs `structured_relevant_first` ablation against the CUAD smoke fixture.

    Raises:
        RuntimeError: If a question's correct document is not found among
            its retrieved candidates (a retrieval regression, not something
            this diagnostic can answer).
    """
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)

    assert len(records) == len(eval_items), (
        "records и eval_items должны быть одной длины и в одном порядке - "
        "load_cuad_smoke_fixture() строит их в одном цикле по вопросам"
    )
    _verify_relevant_doc_mapping(records, eval_items)
    correct_document_id_by_question = {
        item["question_id"]: rec.context_id for rec, item in zip(records, eval_items)
    }

    print(
        f"Round 3: {len(eval_items)} question(s) x {len(VARIANTS)} variant(s) = "
        f"{len(eval_items) * len(VARIANTS)} assessor call(s) total."
    )

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, variant) pair(s) already done in {out_path}")
        print(
            "  ВНИМАНИЕ: resume не отличает версии эксперимента - если промпт/логика этого "
            "скрипта менялись С МОМЕНТА, когда была записана хотя бы одна из этих строк, "
            "старые результаты будут молча приняты как готовые. Если это не тот же самый, "
            "неизменный прогон - удалите или переименуйте diagnostic_results.jsonl перед "
            "повторным запуском, а не полагайтесь на resume."
        )

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
            correct_document_id = correct_document_id_by_question[question_id]

            # Оба варианта используют один и тот же, полный (нефильтрованный)
            # контекст - в отличие от single_relevant_document_only раунда 2,
            # здесь снова достаточно одного retrieval-вызова на вопрос.
            call = search_fn(question)
            candidates = list(call.candidates)
            context_text = build_context_block(candidates)
            context_ids = sorted({c.context_id for c in candidates})

            # pipeline.generation.build_context_block() обозначает кандидатов
            # как "[Document i]", i = 1..len(candidates), в ТОМ ЖЕ порядке,
            # что и candidates здесь (проверено чтением исходного кода, не
            # предположение) - correct_document_position - позиция правильного
            # документа в этой же нумерации, чтобы relevant_documents_raw
            # (например "Document 2") можно было сравнить с ней напрямую, без
            # обращения к исходному коду. correct_document_id остаётся
            # источником истины (context_id); position - производное поле
            # только для удобства чтения результатов человеком.
            correct_document_position: int | None = None
            for position, candidate in enumerate(candidates, start=1):
                if candidate.context_id == correct_document_id:
                    correct_document_position = position
                    break
            if correct_document_position is None:
                raise RuntimeError(
                    f"question_id={question_id!r}: correct_document_id {correct_document_id!r} "
                    f"not found among retrieved candidates ({[c.context_id for c in candidates]!r}) - "
                    "retrieval regression, not a prompt question this diagnostic can answer. "
                    "Stopping rather than silently recording a wrong position."
                )

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                calls_remaining = config.agent.max_additional_tool_calls
                prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)
                parse_warning = _check_parse_reliability(raw_response)
                relevant_documents_raw = _extract_relevant_documents_raw(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "correct_document_id": correct_document_id,
                    "correct_document_position": correct_document_position,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "relevant_documents_raw": relevant_documents_raw,
                    "raw_response": raw_response,
                    "parse_warning": parse_warning,
                }
                if parse_warning:
                    print(f"      WARNING: {parse_warning}")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(f"  [{n_calls}] {question_id} / {variant} -> sufficient={parsed.sufficient}")

    elapsed = time.perf_counter() - t_start
    print(f"\nГотово за {elapsed:.0f}с, {n_calls} новых вызовов assessor'а в этой сессии.")

    print("\nСводка sufficient=yes по вариантам (n=5 на вариант, ориентир, не статистика):")
    for variant in VARIANTS:
        rows = [r for r in done.values() if r["variant"] == variant]
        n_yes = sum(1 for r in rows if r["sufficient"])
        print(f"  {variant:28s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Критерий решения (согласован с первым экспертом, порог уточнён вторым экспертом до "
        "однозначного числа): structured_relevant_first >= 3 из 5 sufficient=yes -> диагностический "
        "сигнал достаточен, чтобы переходить к финансовому регрессу n=30 (4-5 из 5 - сигнал сильнее, "
        "порог прохождения тот же; сам регресс остаётся обязательным следующим шагом, это НЕ "
        "доказательство готовности фикса). Меньше 3 - гипотеза A в этой формулировке опровергнута на "
        "диагностическом уровне, нужен другой механизм. original ожидается 0/5 (within-run контроль, "
        "ровно как в раунде 1) - если original неожиданно даст sufficient=yes, это сигнал дрейфа "
        "модели/окружения, а не полезный результат по гипотезе A, и его надо исследовать отдельно "
        "перед выводами."
    )
    print(
        "Читать raw_response целиком, не только счётчик - особенно поле relevant_documents_raw "
        "и совпадает ли оно с correct_document_position каждой записи (позиция в той же нумерации "
        "[Document i], что видит модель - не строка correct_document_id)."
    )


if __name__ == "__main__":
    main()

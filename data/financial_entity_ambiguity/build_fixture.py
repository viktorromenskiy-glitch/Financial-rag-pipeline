"""Строит фикстуру для узкой финансовой проверки деиктической/entity гипотезы
(см. claude/itog_ekspertizy_cuad_overrefusal_fix.md, "Что осталось сделать",
п.2) - вопрос: насколько сильно падает `sufficient=yes` у ассессора, когда из
текста вопроса убрано название компании, а document-level метаданные
(metadata_prefix) при этом остаются на месте, при ФИКСИРОВАННОМ (не
retrieval-зависимом) контексте gold-документ + 3 дистрактора той же
Sector (то самое поле, которое реально попадает в metadata_prefix - см.
pipeline/ingestion.py's build_metadata_prefix(), которое использует
company_sector, а не company_industry - здесь используется то же поле, чтобы
дистракторы были того же "класса", что видит модель в тексте документа, а не
более узкой категории, которой модель не видит).

Дизайн согласован 4 независимыми экспертами (раунды 2-4 каскада
"prompt_ekspert2/3/4_cuad_fix_dizayn.md", см. claude/itog_ekspertizy_cuad_overrefusal_fix.md
пункт 6 сводной таблицы): парное сравнение (одинаковый вопрос, 2 текстовых
условия) на ~20-30 вопросах, контекст собран вручную (gold + 3 дистрактора той
же Sector, ближайший report_year), НЕ через live retrieval - чтобы изолировать
ассессора от изменчивости retrieval. `_ASSESSMENT_PROMPT_TEMPLATE` не
изменяется вообще - тестируется существующий продакшен-промпт как есть.

## Отбор вопросов (полностью проверяемо кодом ниже, не вручную)

Источник: data/t2-ragbench/eval_subset_250.parquet (тот же файл, на котором
раньше уже считалось 246/250 explicit-company - см.
claude/nahodka_deiktichnost_round3_dlya_ekspertov.md). Кандидат должен:

1. Иметь непустые company_name И company_sector (после join с
   pipeline.ingestion.to_document_records() - то же самое построение
   DocumentRecord, что использует продакшен-индексация, не отдельный парсинг).
2. Явно называть свою компанию (humanize_company_name()) в тексте вопроса -
   проверяется строгим regex с границами слова (`\\b<name>\\b`,
   регистронезависимо), не эвристикой "первое слово" (та эвристика
   использовалась только для черновой оценки масштаба 246/250, здесь нужна
   ТОЧНАЯ, обратимая замена - см. anonymize_question() ниже).
3. Его Sector должен иметь >= 4 РАЗЛИЧНЫХ компаний во ВСЁM корпусе (7318
   документов, не только eval_subset) - иначе не набрать 3 дистрактора от
   разных компаний.

Из отобранных кандидатов вручную (детерминированно, без случайности) взят по
1 представителю на сектор (сортировка кандидатов сектора по company_name,
затем context_id - первый), и по 2-му представителю для секторов с >= 6
различными компаниями в корпусе (даёт больше материала для дистракторов и
сохраняет разнообразие) - итог см. в TARGET_CONTEXT_IDS ниже, зафиксирован
явным списком context_id (не пересчитывается заново при каждом запуске
скрипта), чтобы результат был стабилен и проверяем построчно в code review,
а не воспроизводился только "если повезёт с сортировкой".

## Дистракторы

3 документа той же Sector, ДРУГИХ компаний (никогда не той же компании, что
gold), по одному документу на компанию, ближайший по |report_year - gold_year|
(детерминированный tie-break по context_id). Если для какой-то компании
несколько документов - берётся document-level первый по context_id
(детерминированно).

## Анонимизация вопроса

Заменяется КАЖДОЕ вхождение humanized company name (с учётом притяжательной
формы 's / ' на конце) на "the company" / "the company's" - см.
anonymize_question(). Каждая замена проверяется программно (после замены
исходное имя не должно встречаться в анонимизированном тексте ни в каком
регистре) - не полагается на визуальную проверку одной замены как на
доказательство, что все вхождения обработаны.

## Что НЕ проверяет этот скрипт

Не проверяет, что диагностика физически возможна (это делает основной
диагностический скрипт scripts/run_financial_entity_ambiguity_diagnostic.py -
_check_already_indexed() там). Этот скрипт строит только текстовую фикстуру
(вопросы, gold/дистрактор context_id, анонимизированный текст) -
не обращается к MongoDB, не делает retrieval, не тратит деньги.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "data" / "t2-ragbench"
OUT_PATH = Path(__file__).resolve().parent / "fixture.json"

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from pipeline.ingestion import humanize_company_name, load_raw, to_document_records  # noqa: E402

MIN_DISTINCT_COMPANIES_PER_SECTOR = 4
N_DISTRACTORS = 3

# Явно зафиксированный отбор (context_id вопроса-кандидата - context_id
# документа, question/company/sector совпадают с реальными данными, что
# отдельно ассертится ниже в build(), а не только предполагается по имени
# переменной). 28 вопросов, 20 различных секторов (см. докстринг выше про
# принцип отбора: 1 на сектор + 2-й для секторов с >= 6 компаниями в корпусе,
# минус 4 вручную убранных "вторых" представителя - Materials,
# Telecommunications, Communication Services, Shipping - чтобы уложиться в
# согласованный экспертами диапазон 20-30, а не строго держаться формулы).
TARGET_CONTEXT_IDS: list[str] = [
    "convfinqa_ctx_379",  # Financials / Aon
    "convfinqa_ctx_893",  # Financials / BlackRock
    "finqa_train_ctx_715",  # Industrials / 3M
    "convfinqa_ctx_1480",  # Industrials / American Airlines Group
    "convfinqa_ctx_1603",  # Information Technology / Analog Devices
    "convfinqa_ctx_1056",  # Information Technology / Cadence Design Systems
    "convfinqa_ctx_1027",  # Consumer Discretionary / Norwegian Cruise Line Holdings
    "convfinqa_ctx_224",  # Real Estate / American Tower
    "convfinqa_ctx_117",  # Real Estate / Duke Realty Corporation
    "convfinqa_ctx_1692",  # Health Care / Becton Dickinson
    "convfinqa_ctx_503",  # Health Care / Edwards Lifesciences
    "finqa_train_ctx_1945",  # Consumer Staples / Altria
    "convfinqa_ctx_450",  # Energy / Devon Energy
    "finqa_dev_ctx_236",  # Energy / Marathon Oil
    "cc27b3da593fc5540f3dd8b30ac82796",  # Semiconductors / microchip-technology-inc
    "249a345bb86b13642427834d02363ff8",  # Semiconductors / micron-technology-inc
    "finqa_train_ctx_1959",  # Materials / Air Products
    "0075ff4f22a8946c2aa35d3813de5c86",  # Software / altium-limited
    "finqa_train_ctx_1112",  # Communication Services / Comcast
    "61965dde672ac3c01293e4e66e6bdb95",  # Shipping / navios-maritime-holdings-inc
    "convfinqa_ctx_484",  # Utilities / American Water Works
    "13f24145ffd39eb820e2c7eba93092aa",  # Cybersecurity / mimecast-limited
    "fe9f2f028e1ac48619d22d8ecaf45edf",  # Technology / microsoft-corporation
    "8ff6ca6ee109976b42543aad81a0326e",  # Financial Technology / black-knight-financial-services-inc
    "2c329676bed992ced5b827d9f051a0ec",  # Electronics Manufacturing / jabil-circuit-inc
    "0cf6769516cf2a245aaed5fcf2bd9c21",  # Food Production / lifeway-foods-inc
    "42dfd9e9db0d11bd550deaa12735fd72",  # Telecommunications / cogeco-inc (kept as sole rep)
    "convfinqa_ctx_1215",  # Materials / Ball Corporation (kept as sole 2nd Materials rep - see note)
]


def _strict_name_in_text(name: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE) is not None


# Юридические суффиксы, которые в оригинальном тексте вопроса иногда
# следуют ЗА humanized-именем (например, humanize_company_name() возвращает
# "American Airlines Group", а вопрос называет её "American Airlines Group
# Inc.") - без этого суффикс оставался бы приклеенным к замене ("the company
# Inc."), что не искажает деиктичность (сам по себе "Inc." не идентифицирует
# компанию), но выглядит как явный артефакт замены при чтении вопроса
# человеком - поэтому поглощается вместе с именем в одном совпадении.
_CORP_SUFFIX = r"(?:\s+(?:Inc|Incorporated|Corp(?:oration)?|Co|Company|Ltd|Limited|LLC|plc))?"


def anonymize_question(name: str, question: str) -> str:
    """Заменяет каждое вхождение `name` (с учётом необязательного юридического
    суффикса сразу после имени и притяжательной формы 's/'/'s) на
    "the company"/"the company's". Регистронезависимо, все вхождения сразу
    (re.sub без count=1) - см. модульный докстринг про то, что каждая замена
    проверяется программно после применения (исходное имя не должно остаться
    в анонимизированном тексте), а не просто предполагается по построению
    паттерна.

    Args:
        name: Humanized имя компании (см. humanize_company_name()),
            которое нужно найти и заменить в тексте вопроса.
        question: Исходный текст вопроса, в котором выполняется замена.

    Returns:
        Текст вопроса с каждым вхождением `name` (вместе с необязательным
        юридическим суффиксом и притяжательной формой) заменённым на
        "the company"/"the company's".
    """
    pattern = re.compile(
        r"\b" + re.escape(name) + _CORP_SUFFIX + r"\.?(’s|'s|[’'])?",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        replacement = "the company's" if m.group(1) else "the company"
        if m.start() == 0:
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return pattern.sub(repl, question)


def build() -> None:
    """Строит фикстуру (по одной паре условий на каждый элемент
    TARGET_CONTEXT_IDS: вопрос/gold-документ/3 дистрактора/анонимизированный
    текст) и записывает её в OUT_PATH - см. модульный докстринг для полного
    дизайна отбора и происхождения TARGET_CONTEXT_IDS.

    Raises:
        ValueError: Если элемент TARGET_CONTEXT_IDS не найден в
            eval_subset_250.parquet или среди построенных DocumentRecord,
            не имеет company_name/company_sector, его сектор содержит
            меньше MIN_DISTINCT_COMPANIES_PER_SECTOR различных компаний
            в корпусе, или анонимизация не удалила все вхождения имени
            компании / не изменила текст вопроса.
        AssertionError: Если в построенной фикстуре есть повторяющиеся
            question_id или повторно использованные (в разных элементах)
            gold_context_id.
    """
    raw = load_raw(CORPUS_DIR)
    records = to_document_records(raw)

    docs_by_id: dict[str, object] = {}
    for r in records:
        docs_by_id.setdefault(r.context_id, r)
    print(f"Полный корпус: {len(docs_by_id)} уникальных документов.")

    q_index: dict[tuple[str, str], object] = {}
    for r in records:
        q_index.setdefault((r.context_id, r.question.strip()), r)

    eval_df = pd.read_parquet(CORPUS_DIR / "eval_subset_250.parquet")

    sector_companies: dict[str, set[str]] = defaultdict(set)
    sector_docs_by_company: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in docs_by_id.values():
        if r.company_sector and r.company_name:
            sector_companies[r.company_sector].add(r.company_name)
            sector_docs_by_company[r.company_sector][r.company_name].append(r)

    items: list[dict] = []
    for context_id in TARGET_CONTEXT_IDS:
        # Найти вопрос из eval_subset_250, реально указывающий на этот context_id
        eval_rows = eval_df[eval_df["context_id"] == context_id]
        if eval_rows.empty:
            raise ValueError(f"context_id={context_id!r} не найден в eval_subset_250.parquet")
        row = eval_rows.iloc[0]
        key = (context_id, row["question"].strip())
        rec = q_index.get(key)
        if rec is None:
            raise ValueError(f"context_id={context_id!r}: вопрос не найден среди DocumentRecord (несовпадение текста?)")
        if not rec.company_name or not rec.company_sector:
            raise ValueError(f"context_id={context_id!r}: company_name/company_sector отсутствуют")

        humanized = humanize_company_name(rec.company_name)
        if not humanized or not _strict_name_in_text(humanized, row["question"]):
            raise ValueError(
                f"context_id={context_id!r}: имя компании {humanized!r} не найдено в тексте вопроса "
                f"{row['question']!r} - отбор нарушен"
            )

        n_companies_in_sector = len(sector_companies[rec.company_sector])
        if n_companies_in_sector < MIN_DISTINCT_COMPANIES_PER_SECTOR:
            raise ValueError(
                f"context_id={context_id!r}: сектор {rec.company_sector!r} имеет только "
                f"{n_companies_in_sector} различных компаний в корпусе, нужно >= "
                f"{MIN_DISTINCT_COMPANIES_PER_SECTOR}"
            )

        gold_year = rec.report_year
        gold_year_int = int(gold_year) if gold_year and str(gold_year).isdigit() else None

        distractor_candidates = []
        for company, docs_list in sector_docs_by_company[rec.company_sector].items():
            if company == rec.company_name:
                continue
            docs_list_sorted = sorted(docs_list, key=lambda d: d.context_id)
            chosen = docs_list_sorted[0]
            year_int = int(chosen.report_year) if chosen.report_year and str(chosen.report_year).isdigit() else None
            year_diff = abs(year_int - gold_year_int) if (year_int is not None and gold_year_int is not None) else 999
            distractor_candidates.append((year_diff, chosen.context_id, chosen))
        distractor_candidates.sort(key=lambda t: (t[0], t[1]))
        if len(distractor_candidates) < N_DISTRACTORS:
            raise ValueError(
                f"context_id={context_id!r}: только {len(distractor_candidates)} потенциальных "
                f"дистракторов в секторе {rec.company_sector!r}, нужно {N_DISTRACTORS}"
            )
        distractors = [t[2] for t in distractor_candidates[:N_DISTRACTORS]]

        anonymized_question = anonymize_question(humanized, row["question"])
        if _strict_name_in_text(humanized, anonymized_question):
            raise ValueError(
                f"context_id={context_id!r}: анонимизация не удалила все вхождения {humanized!r} "
                f"из вопроса - осталось: {anonymized_question!r}"
            )
        if anonymized_question == row["question"]:
            raise ValueError(f"context_id={context_id!r}: анонимизация не изменила текст вопроса")

        items.append(
            {
                "question_id": f"fin_entity_ambiguity_{row['id']}",
                "source_dataset": rec.source_dataset,
                "question_original": row["question"],
                "question_anonymized": anonymized_question,
                "gold_answer": row["program_answer"],
                "gold_context_id": context_id,
                "gold_company_name": rec.company_name,
                "gold_company_name_humanized": humanized,
                "gold_sector": rec.company_sector,
                "gold_report_year": rec.report_year,
                "distractor_context_ids": [d.context_id for d in distractors],
                "distractor_company_names": [d.company_name for d in distractors],
                "distractor_report_years": [d.report_year for d in distractors],
            }
        )

    # Проверка уникальности question_id и отсутствия повторного использования
    # одного и того же context_id как gold в двух разных элементах (иначе
    # результат не будет честно "N независимых пар").
    qids = [it["question_id"] for it in items]
    assert len(qids) == len(set(qids)), "Повторяющиеся question_id в фикстуре"
    gold_ids = [it["gold_context_id"] for it in items]
    assert len(gold_ids) == len(set(gold_ids)), "Повторяющиеся gold_context_id в фикстуре"

    print(f"\nПостроено {len(items)} пар вопросов (2 условия каждая = {len(items) * 2} вызовов ассессора).")
    print(f"Секторов: {len(set(it['gold_sector'] for it in items))}")
    print("\n--- Ручная проверка анонимизации (все строки) ---")
    for it in items:
        print(f"[{it['question_id']}] {it['gold_company_name_humanized']!r}")
        print(f"  ORIG: {it['question_original']}")
        print(f"  ANON: {it['question_anonymized']}")
        print(f"  distractors: {it['distractor_company_names']}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\nЗаписано в {OUT_PATH}")


if __name__ == "__main__":
    build()

"""
Модуль 1 — Ingestion / парсинг.
 
Загружает сырые parquet-файлы T²-RAGBench (FinQA, ConvFinQA, TAT-DQA) и
нормализует их в единый список DocumentRecord. См.
docs/specifikatsiya_moduley.md, модуль 1.
 
Формат таблиц внутри `context` и решение по полю `answer` (program_answer,
не original_answer) зафиксированы там же, раздел «Проверено на реальных
файлах» — не переоткрывать при чтении этого модуля.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import pandas as pd
 
# Сырые файлы по источникам — см. docs/specifikatsiya_moduley.md, модуль 1, «Конфиг»
RAW_FILES: dict[str, list[str]] = {
    "FinQA": ["FinQA_train.parquet", "FinQA_dev.parquet", "FinQA_test.parquet"],
    "ConvFinQA": ["ConvFinQA_turn_0.parquet"],
    "TAT-DQA": ["TAT-DQA_train.parquet", "TAT-DQA_dev.parquet", "TAT-DQA_test.parquet"],
}
 
# Чекпоинт из ТЗ п.1 / plan_podgotovki_k_kodirovaniyu.md, Шаг 1
EXPECTED_DOCUMENTS = 7318
EXPECTED_QUESTIONS = 23088
 
 
@dataclass(frozen=True)
class DocumentRecord:
    context_id: str
    context: str
    source_dataset: str
    question: str
    answer: str
 
 
def _load_source(data_dir: Path, source: str, filenames: list[str]) -> pd.DataFrame:
    frames = []
    for filename in filenames:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Ожидаемый файл датасета не найден: {path}. "
                f"См. docs/specifikatsiya_moduley.md, модуль 1, «Конфиг» "
                f"(путь по умолчанию: data/t2-ragbench/)."
            )
        df = pd.read_parquet(path)
        df = df.copy()
        df["source_dataset"] = source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
 
 
def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Загружает все 7 сырых parquet-файлов и объединяет их в один DataFrame
    с добавленной колонкой source_dataset.
    """
    data_dir = Path(data_dir)
    frames = [
        _load_source(data_dir, source, filenames)
        for source, filenames in RAW_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)
 
 
def to_document_records(raw: pd.DataFrame) -> list[DocumentRecord]:
    """Преобразует сырой DataFrame в список DocumentRecord.
 
    Одна строка сырых данных = один вопрос (context_id повторяется у
    вопросов с общим документом — это ожидаемо, дедупликация документов
    происходит на шаге indexing, не здесь, см. модуль 5).
    """
    required = {"context_id", "context", "source_dataset", "question", "program_answer"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            f"В сыром DataFrame отсутствуют обязательные колонки: {sorted(missing)}"
        )
 
    return [
        DocumentRecord(
            context_id=row.context_id,
            context=row.context,
            source_dataset=row.source_dataset,
            question=row.question,
            answer=row.program_answer,
        )
        for row in raw.itertuples(index=False)
    ]
 
 
def ingest(data_dir: str | Path) -> list[DocumentRecord]:
    """Точка входа модуля 1. Загружает, нормализует и проверяет чекпоинт
    (ТЗ п.1): 7318 документов, 23088 вопросов. Расхождение — сигнал бага
    парсинга одного из трёх источников, не проблема дальше по пайплайну
    (см. plan_podgotovki_k_kodirovaniyu.md, Шаг 1).
    """
    raw = load_raw(data_dir)
    records = to_document_records(raw)
 
    n_questions = len(records)
    n_documents = len({r.context_id for r in records})
 
    assert n_questions == EXPECTED_QUESTIONS, (
        f"Ожидалось {EXPECTED_QUESTIONS} вопросов (ТЗ п.1), получено {n_questions}."
    )
    assert n_documents == EXPECTED_DOCUMENTS, (
        f"Ожидалось {EXPECTED_DOCUMENTS} документов (ТЗ п.1), получено {n_documents}."
    )
 
    return records

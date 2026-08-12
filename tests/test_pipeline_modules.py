
Тесты модулей 1 (ingestion) и 2 (chunking) на срезе реальных и синтетических
данных. См. kak_my_rabotaem_vmeste.md: каждая функция тестируется на реальных
или реалистичных данных, включая граничные случаи, до передачи дальше.
 
Интеграционный тест на полном датасете пропускается, если data/t2-ragbench/
не заполнена реальными файлами (сырой датасет не коммитится в репозиторий —
см. struktura_repozitoriya.md), но обязателен к запуску локально при первом
проходе (Шаг 1 плана).
"""
 
from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
import pytest
 
from pipeline.chunking import chunk
from pipeline.ingestion import (
    EXPECTED_DOCUMENTS,
    EXPECTED_QUESTIONS,
    DocumentRecord,
    ingest,
    load_raw,
    to_document_records,
)
 
REAL_DATA_DIR = Path(os.environ.get("T2_RAGBENCH_DATA_DIR", "data/t2-ragbench"))
HAS_REAL_DATA = REAL_DATA_DIR.exists() and any(REAL_DATA_DIR.glob("*.parquet"))
 
 
# ---------------------------------------------------------------------------
# Граничные случаи — синтетические данные, не зависят от наличия датасета
# ---------------------------------------------------------------------------
 
 
def test_to_document_records_empty_input():
    """Легитимно пустой результат: пустой DataFrame -> пустой список, без ошибок."""
    empty = pd.DataFrame(
        columns=["context_id", "context", "source_dataset", "question", "program_answer"]
    )
    assert to_document_records(empty) == []
 
 
def test_to_document_records_missing_column_raises():
    """Отсутствие обязательной колонки должно падать явно, не молча возвращать мусор."""
    bad = pd.DataFrame({"context_id": ["a"], "context": ["b"]})
    with pytest.raises(KeyError):
        to_document_records(bad)
 
 
def test_to_document_records_uses_program_answer_not_original_answer():
    """Регрессия на конкретное решение из specifikatsiya_moduley.md, модуль 1:
    answer = program_answer, original_answer игнорируется, даже если оба поля
    присутствуют и различаются."""
    df = pd.DataFrame(
        {
            "context_id": ["ctx_1"],
            "context": ["some text with a table"],
            "source_dataset": ["FinQA"],
            "question": ["What was X?"],
            "program_answer": ["3.8"],
            "original_answer": ["380"],  # намеренно другое значение
        }
    )
    records = to_document_records(df)
    assert len(records) == 1
    assert records[0].answer == "3.8"
 
 
def test_load_raw_missing_file_raises_filenotfounderror(tmp_path):
    """Частично заполненная директория (не хватает одного из 7 файлов) должна
    падать с понятным сообщением, не с невнятной ошибкой pandas/pyarrow."""
    # Кладём только один из семи ожидаемых файлов
    df = pd.DataFrame(
        {
            "context_id": ["ctx_1"],
            "context": ["text"],
            "question": ["q"],
            "program_answer": ["1"],
        }
    )
    df.to_parquet(tmp_path / "FinQA_train.parquet")
 
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)
 
 
def test_chunk_is_identity_no_op():
    """Модуль 2 не должен менять ни количество записей, ни их содержимое."""
    records = [
        DocumentRecord(
            context_id="ctx_1",
            context="text with | a | table |",
            source_dataset="TAT-DQA",
            question="q?",
            answer="42",
        )
    ]
    result = chunk(records)
    assert result == records
    assert result is not records  # возвращает новый список, не мутирует вход
 
 
def test_chunk_empty_input():
    assert chunk([]) == []
 
 
# ---------------------------------------------------------------------------
# Интеграционный чекпоинт на полном реальном датасете (ТЗ п.1)
# ---------------------------------------------------------------------------
 
 
@pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason=(
        "Сырой датасет T2-RAGBench не найден в data/t2-ragbench/ "
        "(не коммитится в репозиторий, см. struktura_repozitoriya.md) — "
        "запустить локально перед первым проходом, задав T2_RAGBENCH_DATA_DIR."
    ),
)
def test_ingest_full_corpus_checkpoint():
    records = ingest(REAL_DATA_DIR)
 
    assert len(records) == EXPECTED_QUESTIONS == 23088
    assert len({r.context_id for r in records}) == EXPECTED_DOCUMENTS == 7318
 
    sources = {r.source_dataset for r in records}
    assert sources == {"FinQA", "ConvFinQA", "TAT-DQA"}
 
    # answer всегда заполнен и является строкой (program_answer без пропусков)
    assert all(isinstance(r.answer, str) and r.answer != "" for r in records)
 
    # chunking не должен ничего терять на полном проходе
    chunked = chunk(records)
    assert len(chunked) == len(records)
 

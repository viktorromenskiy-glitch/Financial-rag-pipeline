"""Тесты модуля 5 (indexing). CRUD-логика — на mongomock (эмулирует базовые
операции pymongo локально). $vectorSearch/$search агрегации mongomock не
поддерживает (это специфика Atlas Search) — для validate_startup_indexes
используется лёгкий фейк, проверяющий саму логику ассертов, не Atlas.
 
Реальный прогон против живого Atlas-кластера — вне этой сессии (сеть до
mongodb.net заблокирована allowlist'ом песочницы), делается в Colab."""
 
from __future__ import annotations
 
import os
from pathlib import Path
 
import mongomock
import pytest
 
from pipeline.ingestion import ingest
from pipeline.indexing import (
    build_full_indexed_content,
    dedupe_documents,
    index_corpus,
    is_indexed,
    upsert_document,
    validate_startup_indexes,
)
 
REAL_DATA_DIR = Path(os.environ.get("T2_RAGBENCH_DATA_DIR", "data/t2-ragbench"))
HAS_REAL_DATA = REAL_DATA_DIR.exists() and any(REAL_DATA_DIR.glob("*.parquet"))
 
 
# ---------------------------------------------------------------------------
# build_full_indexed_content
# ---------------------------------------------------------------------------
 
 
def test_build_full_indexed_content_with_enrichment():
    result = build_full_indexed_content("raw text", "short summary")
    assert result == "short summary\n\nraw text"
 
 
def test_build_full_indexed_content_without_enrichment():
    """enrichment.enabled=false (Шаг 2) -> contextual_summary="" ->
    full_indexed_content == raw_content без изменений."""
    result = build_full_indexed_content("raw text", "")
    assert result == "raw text"
 
 
# ---------------------------------------------------------------------------
# dedupe_documents — граничные случаи на синтетике
# ---------------------------------------------------------------------------
 
 
class _Rec:
    def __init__(self, context_id, context, source_dataset):
        self.context_id = context_id
        self.context = context
        self.source_dataset = source_dataset
 
 
def test_dedupe_documents_empty_input():
    assert dedupe_documents([]) == []
 
 
def test_dedupe_documents_collapses_multiple_questions_per_doc():
    records = [
        _Rec("ctx_1", "doc text", "FinQA"),
        _Rec("ctx_1", "doc text", "FinQA"),  # тот же документ, другой вопрос
        _Rec("ctx_2", "other doc", "TAT-DQA"),
    ]
    docs = dedupe_documents(records)
    assert len(docs) == 2
    assert {d["context_id"] for d in docs} == {"ctx_1", "ctx_2"}
 
 
def test_dedupe_documents_raises_on_inconsistent_context():
    records = [
        _Rec("ctx_1", "version A", "FinQA"),
        _Rec("ctx_1", "version B", "FinQA"),  # нарушение допущения
    ]
    with pytest.raises(ValueError):
        dedupe_documents(records)
 
 
def test_dedupe_documents_raises_on_inconsistent_source():
    records = [
        _Rec("ctx_1", "same text", "FinQA"),
        _Rec("ctx_1", "same text", "ConvFinQA"),
    ]
    with pytest.raises(ValueError):
        dedupe_documents(records)
 
 
@pytest.mark.skipif(not HAS_REAL_DATA, reason="реальный датасет не найден локально")
def test_dedupe_documents_on_real_corpus_checkpoint():
    """Регрессия на находку, сделанную вручную на Шаге 1: все 7318 context_id
    в реальном корпусе консистентны (0 расхождений context/source_dataset)."""
    records = ingest(REAL_DATA_DIR)
    docs = dedupe_documents(records)  # не должно кинуть ValueError
    assert len(docs) == 7318
 
 
# ---------------------------------------------------------------------------
# CRUD на mongomock
# ---------------------------------------------------------------------------
 
 
@pytest.fixture
def collection():
    client = mongomock.MongoClient()
    return client["rag_project"]["t2_ragbench_full"]
 
 
def test_upsert_document_then_is_indexed(collection):
    assert is_indexed(collection, "ctx_1") is False
 
    upsert_document(
        collection,
        context_id="ctx_1",
        raw_content="raw",
        contextual_summary="",
        embedding=[0.1] * 1024,
    )
 
    assert is_indexed(collection, "ctx_1") is True
    doc = collection.find_one({"context_id": "ctx_1"})
    assert doc["full_indexed_content"] == "raw"
    assert doc["embedding_voyage"] == [0.1] * 1024
 
 
def test_upsert_document_is_idempotent(collection):
    """Повторный upsert того же context_id не создаёт дубликат."""
    for _ in range(2):
        upsert_document(collection, "ctx_1", "raw", "", [0.0] * 1024)
    assert collection.count_documents({"context_id": "ctx_1"}) == 1
 
 
def test_index_corpus_indexes_all_documents(collection):
    documents = [
        {"context_id": "ctx_1", "source_dataset": "FinQA", "raw_content": "doc 1"},
        {"context_id": "ctx_2", "source_dataset": "TAT-DQA", "raw_content": "doc 2"},
    ]
    summaries = {"ctx_1": "", "ctx_2": ""}
    embeddings = {"ctx_1": [0.1] * 1024, "ctx_2": [0.2] * 1024}
 
    count = index_corpus(collection, documents, summaries, embeddings)
 
    assert count == 2
    assert collection.count_documents({}) == 2
 
 
def test_index_corpus_checkpoint_skips_already_indexed(collection):
    """Устойчивость: документ, уже помеченный is_indexed=True, не
    перезаписывается повторным прогоном (эмулирует возобновление после
    обрыва сессии на середине индексации)."""
    documents = [{"context_id": "ctx_1", "source_dataset": "FinQA", "raw_content": "doc 1"}]
    summaries = {"ctx_1": ""}
    embeddings = {"ctx_1": [0.1] * 1024}
 
    first = index_corpus(collection, documents, summaries, embeddings)
    second = index_corpus(collection, documents, summaries, embeddings)
 
    assert first == 1
    assert second == 0  # пропущен по чекпоинту
 
 
def test_index_corpus_missing_embedding_raises(collection):
    documents = [{"context_id": "ctx_1", "source_dataset": "FinQA", "raw_content": "doc 1"}]
    with pytest.raises(KeyError):
        index_corpus(collection, documents, {"ctx_1": ""}, embeddings_by_id={})
 
 
def test_index_corpus_empty_input(collection):
    assert index_corpus(collection, [], {}, {}) == 0
 
 
# ---------------------------------------------------------------------------
# validate_startup_indexes — фейк-коллекция, проверяем только логику ассертов
# ---------------------------------------------------------------------------
 
 
class _FakeCollectionEmptyResults:
    def aggregate(self, pipeline):
        return iter([])
 
 
class _FakeCollectionNonEmptyResults:
    def aggregate(self, pipeline):
        return iter([{"context_id": "ctx_1"}])
 
 
def test_validate_startup_indexes_raises_on_empty_vector_result():
    with pytest.raises(AssertionError):
        validate_startup_indexes(_FakeCollectionEmptyResults())
 
 
def test_validate_startup_indexes_passes_on_non_empty_results():
    validate_startup_indexes(_FakeCollectionNonEmptyResults())  # не должно кинуть

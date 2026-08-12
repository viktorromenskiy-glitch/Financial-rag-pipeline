"""
Модуль 5 — Indexing (MongoDB Atlas).
 
Собирает воедино результаты модулей 1 (документы), 3 (эмбеддинги) и 4
(contextual_summary) и записывает в коллекцию t2_ragbench_full. См.
docs/specifikatsiya_moduley.md, модуль 5.
"""
 
from __future__ import annotations
 
from typing import Protocol
 
DB_NAME = "rag_project"
COLLECTION_NAME = "t2_ragbench_full"
VECTOR_INDEX_NAME = "vector_index_full"
TEXT_INDEX_NAME = "text_index_full"
 
 
def build_full_indexed_content(raw_content: str, contextual_summary: str) -> str:
    """full_indexed_content = contextual_summary + "\\n\\n" + raw_content,
    или просто raw_content, если enrichment выключен (contextual_summary
    пустая строка)."""
    if contextual_summary:
        return f"{contextual_summary}\n\n{raw_content}"
    return raw_content
 
 
def dedupe_documents(records) -> list[dict]:
    """Схлопывает список DocumentRecord (модуль 1 — одна запись = один
    вопрос) до уникальных документов по context_id (один документ = один
    чанк, модуль 2). Индексируются документы, не вопросы.
 
    Ассерт на согласованность: если один context_id встречается с разным
    context или source_dataset в разных строках — падать явно (нарушено
    базовое допущение схемы данных), не молча брать первую попавшуюся
    версию.
    """
    by_id: dict[str, dict] = {}
    for r in records:
        if r.context_id not in by_id:
            by_id[r.context_id] = {
                "context_id": r.context_id,
                "source_dataset": r.source_dataset,
                "raw_content": r.context,
            }
        else:
            existing = by_id[r.context_id]
            if existing["raw_content"] != r.context:
                raise ValueError(
                    f"context_id={r.context_id!r} встречается с разным текстом context — "
                    f"нарушено допущение «один документ = один чанк»"
                )
            if existing["source_dataset"] != r.source_dataset:
                raise ValueError(
                    f"context_id={r.context_id!r} встречается с разным source_dataset"
                )
    return list(by_id.values())
 
 
class CollectionProtocol(Protocol):
    """Минимальный интерфейс pymongo.Collection, нужный этому модулю —
    позволяет тестировать на mongomock/фейке без реального Atlas."""
 
    def find_one(self, filter, projection=None): ...
    def update_one(self, filter, update, upsert=False): ...
    def aggregate(self, pipeline): ...
 
 
def is_indexed(collection: CollectionProtocol, context_id: str) -> bool:
    doc = collection.find_one({"context_id": context_id, "is_indexed": True}, {"_id": 1})
    return doc is not None
 
 
def upsert_document(
    collection: CollectionProtocol,
    context_id: str,
    raw_content: str,
    contextual_summary: str,
    embedding: list[float],
) -> None:
    full_indexed_content = build_full_indexed_content(raw_content, contextual_summary)
    collection.update_one(
        {"context_id": context_id},
        {
            "$set": {
                "context_id": context_id,
                "raw_content": raw_content,
                "contextual_summary": contextual_summary,
                "full_indexed_content": full_indexed_content,
                "embedding_voyage": embedding,
                "is_indexed": True,
            }
        },
        upsert=True,
    )
 
 
def index_corpus(
    collection: CollectionProtocol,
    documents: list[dict],
    contextual_summaries: dict[str, str],
    embeddings_by_id: dict[str, list[float]],
    skip_already_indexed: bool = True,
) -> int:
    """documents — выход dedupe_documents(). contextual_summaries — выход
    enrich_documents() (модуль 4). embeddings_by_id — {context_id: vector}
    построено из EmbeddingVector-ов embed_documents() (модуль 3).
 
    Возвращает количество реально записанных (не пропущенных по чекпоинту)
    документов. Устойчивость к сбою на середине индексации всего корпуса —
    уже проиндексированные документы (is_indexed=True) пропускаются, не
    начинаем заново с нуля (specifikatsiya_moduley.md, модуль 5,
    «Устойчивость»).
    """
    count = 0
    for doc in documents:
        context_id = doc["context_id"]
        if skip_already_indexed and is_indexed(collection, context_id):
            continue
        if context_id not in embeddings_by_id:
            raise KeyError(f"Нет эмбеддинга для context_id={context_id!r}")
        upsert_document(
            collection,
            context_id=context_id,
            raw_content=doc["raw_content"],
            contextual_summary=contextual_summaries.get(context_id, ""),
            embedding=embeddings_by_id[context_id],
        )
        count += 1
    return count
 
 
def validate_startup_indexes(collection: CollectionProtocol) -> None:
    """Обязательная проверка при старте пайплайна (ТЗ п.2): тестовый запрос
    к обоим индексам, assert на непустой результат — до того, как пайплайн
    продолжит работу. Защита от уже случавшегося тихого бага несовпадения
    имён индексов.
    """
    vector_result = list(
        collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": VECTOR_INDEX_NAME,
                        "path": "embedding_voyage",
                        "queryVector": [0.0] * 1024,
                        "numCandidates": 10,
                        "limit": 1,
                    }
                }
            ]
        )
    )
    text_result = list(
        collection.aggregate(
            [
                {
                    "$search": {
                        "index": TEXT_INDEX_NAME,
                        "text": {"query": "the", "path": "full_indexed_content"},
                    }
                },
                {"$limit": 1},
            ]
        )
    )
    if not vector_result:
        raise AssertionError(
            f"Векторный индекс {VECTOR_INDEX_NAME!r} вернул пустой результат на тестовом запросе"
        )
    if not text_result:
        raise AssertionError(
            f"Полнотекстовый индекс {TEXT_INDEX_NAME!r} вернул пустой результат на тестовом запросе"
        )

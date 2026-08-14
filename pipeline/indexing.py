"""Module 5 - Indexing (MongoDB Atlas).

Combines the outputs of module 1 (documents, including metadata_prefix),
module 3 (embeddings), and module 4 (contextual_summary) and writes them
into the t2_ragbench_full collection. See docs/specifikatsiya_moduley.md,
module 5.
"""

from __future__ import annotations

from typing import Protocol

DB_NAME = "rag_project"
COLLECTION_NAME = "t2_ragbench_full"
VECTOR_INDEX_NAME = "vector_index_full"
TEXT_INDEX_NAME = "text_index_full"


def build_full_indexed_content(
    raw_content: str, contextual_summary: str, metadata_prefix: str = ""
) -> str:
    """full_indexed_content = metadata_prefix + contextual_summary + "\\n\\n"
    + raw_content, or metadata_prefix + raw_content if enrichment is
    disabled (contextual_summary is an empty string).

    metadata_prefix is applied unconditionally, independent of
    enrichment.enabled (module 1, tasks #60/#61 - validated to raise
    Recall@5 from 0.896 to 0.948, McNemar p=0.00195). It already carries
    its own trailing "\\n\\n" (or is "" if no metadata is available), so it
    is prepended directly rather than joined with an extra separator - this
    avoids a redundant blank line and matches exactly what was validated in
    the Step 4 Colab test (metadata_prefix prepended to the
    already-assembled full_indexed_content, not inserted as a third
    concatenation term).
    """
    if contextual_summary:
        body = f"{contextual_summary}\n\n{raw_content}"
    else:
        body = raw_content
    return f"{metadata_prefix}{body}"


def dedupe_documents(records) -> list[dict]:
    """Collapses a list of DocumentRecord (module 1 - one record = one
    question) down to unique documents by context_id (one document = one
    chunk, module 2). Documents are indexed, not questions.

    Consistency assertion: if the same context_id appears with a different
    context, source_dataset, or metadata_prefix across rows, fail loudly
    (the base assumption of the data schema is violated) instead of
    silently keeping the first version seen. metadata_prefix is derived
    from document-level fields (company_name/report_year/company_sector),
    so it must be identical for every row sharing a context_id.
    """
    by_id: dict[str, dict] = {}
    for r in records:
        if r.context_id not in by_id:
            by_id[r.context_id] = {
                "context_id": r.context_id,
                "source_dataset": r.source_dataset,
                "raw_content": r.context,
                "metadata_prefix": r.metadata_prefix,
            }
        else:
            existing = by_id[r.context_id]
            if existing["raw_content"] != r.context:
                raise ValueError(
                    f"context_id={r.context_id!r} appears with different context text - "
                    f"the 'one document = one chunk' assumption is violated"
                )
            if existing["source_dataset"] != r.source_dataset:
                raise ValueError(
                    f"context_id={r.context_id!r} appears with different source_dataset"
                )
            if existing["metadata_prefix"] != r.metadata_prefix:
                raise ValueError(
                    f"context_id={r.context_id!r} appears with different metadata_prefix - "
                    f"company_name/report_year/company_sector should be document-level, not "
                    f"question-level"
                )
    return list(by_id.values())


class CollectionProtocol(Protocol):
    """Minimal pymongo.Collection interface required by this module - allows
    testing against mongomock/a fake without a real Atlas cluster."""

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
    metadata_prefix: str,
    embedding: list[float],
) -> None:
    full_indexed_content = build_full_indexed_content(raw_content, contextual_summary, metadata_prefix)
    collection.update_one(
        {"context_id": context_id},
        {
            "$set": {
                "context_id": context_id,
                "raw_content": raw_content,
                "metadata_prefix": metadata_prefix,
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
    """documents - output of dedupe_documents() (each dict has context_id,
    raw_content, source_dataset, metadata_prefix). contextual_summaries -
    output of enrich_documents() (module 4). embeddings_by_id -
    {context_id: vector} built from the EmbeddingVector objects returned by
    embed_documents() (module 3).

    Returns the number of documents actually written (not skipped via
    checkpoint). Resilient to a mid-run failure while indexing the full
    corpus - documents already marked is_indexed=True are skipped instead of
    restarting from scratch (specifikatsiya_moduley.md, module 5,
    "Устойчивость").
    """
    count = 0
    for doc in documents:
        context_id = doc["context_id"]
        if skip_already_indexed and is_indexed(collection, context_id):
            continue
        if context_id not in embeddings_by_id:
            raise KeyError(f"No embedding found for context_id={context_id!r}")
        upsert_document(
            collection,
            context_id=context_id,
            raw_content=doc["raw_content"],
            contextual_summary=contextual_summaries.get(context_id, ""),
            metadata_prefix=doc["metadata_prefix"],
            embedding=embeddings_by_id[context_id],
        )
        count += 1
    return count


def validate_startup_indexes(collection: CollectionProtocol) -> None:
    """Mandatory startup check (spec section 2): a test query against both
    indexes, asserting a non-empty result - before the pipeline is allowed
    to proceed. Guards against a previously observed silent bug caused by
    mismatched index names.
    """
    # A zero vector is invalid for $vectorSearch: Atlas uses cosine
    # similarity internally, which is undefined for a zero-magnitude
    # vector and raises OperationFailure ("Cosine similarity cannot be
    # calculated against a zero vector."). Use any non-zero probe vector.
    probe_vector = [1.0] + [0.0] * (1024 - 1)
    vector_result = list(
        collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": VECTOR_INDEX_NAME,
                        "path": "embedding_voyage",
                        "queryVector": probe_vector,
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
            f"Vector index {VECTOR_INDEX_NAME!r} returned an empty result on the test query"
        )
    if not text_result:
        raise AssertionError(
            f"Full-text index {TEXT_INDEX_NAME!r} returned an empty result on the test query"
        )

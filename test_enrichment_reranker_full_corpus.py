"""Step 2 of the full-corpus reranker+enrichment validation. Creates the 4
Atlas Search indexes needed on the NEW temporary cluster (this is a fresh
cluster/collection, so these are created from scratch via
create_search_index - not updated like update_atlas_search_indexes.py did
for the production cluster's existing indexes).

Four indexes, one vector + one text per arm (enriched / raw) - see
setup_temp_cluster_corpus.py's docstring for why both arms are separate,
self-contained fields on this temporary collection:
  - vector_index_enriched  -> embedding_enriched (1024-dim, voyage-4)
  - text_index_enriched    -> enriched_text
  - vector_index_raw       -> embedding_raw (1024-dim, voyage-4)
  - text_index_raw         -> raw_text

M10 (or any dedicated tier) supports far more than 3 search indexes per
cluster, unlike the M0 free tier this project is otherwise constrained by
(docs/tehnicheskoe_zadanie.md, section 2) - that constraint is exactly why
this test runs on a separate temporary cluster rather than the production
one.

Requires pymongo >= 4.5 for Collection.create_search_index() /
list_search_indexes(). Atlas Search index builds are asynchronous - this
script polls until each index reports status READY and queryable=True.

Usage (Colab, repo root, after setup_temp_cluster_corpus.py has finished):
    !python create_temp_cluster_indexes.py
"""
from __future__ import annotations

import os
import time

import pymongo
from pymongo.operations import SearchIndexModel

from setup_temp_cluster_corpus import DEST_COLLECTION_NAME, DEST_DB_NAME

EMBEDDING_DIM = 1024
POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5

VECTOR_INDEXES = [
    ("vector_index_enriched", "embedding_enriched"),
    ("vector_index_raw", "embedding_raw"),
]
TEXT_INDEXES = [
    ("text_index_enriched", "enriched_text"),
    ("text_index_raw", "raw_text"),
]


def _wait_until_queryable(collection, index_name: str) -> None:
    print(f"  waiting for {index_name!r} to build (this can take a minute or two)...")
    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        for idx in collection.list_search_indexes(index_name):
            if idx.get("status") == "READY" and idx.get("queryable"):
                print(f"  {index_name!r} is READY and queryable")
                return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"{index_name!r} did not become READY within {POLL_TIMEOUT_SECONDS}s")


def _index_exists(collection, index_name: str) -> bool:
    return any(True for _ in collection.list_search_indexes(index_name))


def create_vector_index(collection, index_name: str, path: str) -> None:
    if _index_exists(collection, index_name):
        print(f"  {index_name!r} already exists - skipping")
        return
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": path,
                    "numDimensions": EMBEDDING_DIM,
                    "similarity": "cosine",
                }
            ]
        },
        name=index_name,
        type="vectorSearch",
    )
    print(f"  creating {index_name!r} on field {path!r}...")
    collection.create_search_index(model)
    _wait_until_queryable(collection, index_name)


def create_text_index(collection, index_name: str, path: str) -> None:
    if _index_exists(collection, index_name):
        print(f"  {index_name!r} already exists - skipping")
        return
    model = SearchIndexModel(
        definition={"mappings": {"dynamic": False, "fields": {path: {"type": "string"}}}},
        name=index_name,
        type="search",
    )
    print(f"  creating {index_name!r} on field {path!r}...")
    collection.create_search_index(model)
    _wait_until_queryable(collection, index_name)


def main() -> None:
    dest_client = pymongo.MongoClient(os.environ["DEST_MONGODB_URI"])
    collection = dest_client[DEST_DB_NAME][DEST_COLLECTION_NAME]

    doc_count = collection.count_documents({})
    print(f"Destination collection has {doc_count} documents - creating indexes...")
    if doc_count == 0:
        raise RuntimeError(
            "Destination collection is empty - run setup_temp_cluster_corpus.py first"
        )

    print("[1/2] Vector indexes...")
    for name, path in VECTOR_INDEXES:
        create_vector_index(collection, name, path)

    print("[2/2] Text indexes...")
    for name, path in TEXT_INDEXES:
        create_text_index(collection, name, path)

    print("\nDone. All 4 indexes READY. Next: test_enrichment_reranker_full_corpus.py.")


if __name__ == "__main__":
    main()

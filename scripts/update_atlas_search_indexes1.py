@@ -1,123 +0,0 @@
"""Adds `source_dataset` as a filterable field to both Atlas Search
indexes (vector_index_full, text_index_full), via pymongo's driver-native
search-index management (Collection.list_search_indexes() /
update_search_index()) - no separate Atlas Admin API key needed, uses the
same MONGODB_URI already in Colab Secrets/env.

Why a script instead of clicking through the Atlas web UI: this session's
sandbox has no network access to *.mongodb.net (confirmed earlier in this
project), so there is no way to operate the Atlas console directly from
here. pymongo's search-index management API (added 4.5+) covers the same
operation without the UI or a separate Atlas Admin API key - you run this
once, from the same place you already run the other pipeline scripts.

Non-destructive by design: fetches the CURRENT index definition first and
merges the new field into it, rather than overwriting the whole
definition blind - this will not drop any field or setting already
configured on your cluster (dynamic mappings, other filter fields, etc.).
Safe to re-run - both update functions skip the field if it's already
present.

Requires pymongo >= 4.7 for update_search_index()/list_search_indexes()
(some early-4.5 versions only had create/drop). If you get an
AttributeError on collection.update_search_index, run
`!pip install -q -U pymongo` first.

Atlas index updates rebuild asynchronously - this script polls
list_search_indexes() until status is READY and queryable=True before
returning, rather than assuming a fixed wait time.

Usage (Colab, after the usual %cd + secrets-loading cells):
    !python scripts/update_atlas_search_indexes.py
"""
from __future__ import annotations

import os
import time

import pymongo

VECTOR_INDEX_NAME = "vector_index_full"
TEXT_INDEX_NAME = "text_index_full"
POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5


def _current_definition(collection, index_name: str) -> dict:
    indexes = list(collection.list_search_indexes(index_name))
    if not indexes:
        raise RuntimeError(
            f"Index {index_name!r} not found on this collection - check the index name and that "
            f"it was created against rag_project.t2_ragbench_full, not a different collection"
        )
    info = indexes[0]
    # Field name per MongoDB's $listSearchIndexes docs; fall back to
    # "definition" in case an older server/driver combination returns it
    # under the pre-rename key.
    return info.get("latestDefinition") or info["definition"]


def _wait_until_queryable(collection, index_name: str) -> None:
    print(f"  waiting for {index_name!r} to finish rebuilding (this can take a minute or two)...")
    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        for idx in collection.list_search_indexes(index_name):
            if idx.get("status") == "READY" and idx.get("queryable"):
                print(f"  {index_name!r} is READY and queryable")
                return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"{index_name!r} did not become READY within {POLL_TIMEOUT_SECONDS}s - check Atlas Search "
        f"index status in the Atlas UI directly, the rebuild may still be in progress"
    )


def update_vector_index(collection) -> None:
    definition = _current_definition(collection, VECTOR_INDEX_NAME)
    fields = list(definition.get("fields", []))

    if any(f.get("path") == "source_dataset" and f.get("type") == "filter" for f in fields):
        print(f"  {VECTOR_INDEX_NAME!r} already has a source_dataset filter field - nothing to do")
        return

    fields.append({"type": "filter", "path": "source_dataset"})
    definition["fields"] = fields
    print(f"  updating {VECTOR_INDEX_NAME!r}...")
    collection.update_search_index(VECTOR_INDEX_NAME, definition)
    _wait_until_queryable(collection, VECTOR_INDEX_NAME)


def update_text_index(collection) -> None:
    definition = _current_definition(collection, TEXT_INDEX_NAME)
    mappings = definition.setdefault("mappings", {})
    fields = mappings.setdefault("fields", {})

    if "source_dataset" in fields:
        print(f"  {TEXT_INDEX_NAME!r} already has a source_dataset field mapping - nothing to do")
        return

    fields["source_dataset"] = {"type": "token"}
    print(f"  updating {TEXT_INDEX_NAME!r}...")
    collection.update_search_index(TEXT_INDEX_NAME, definition)
    _wait_until_queryable(collection, TEXT_INDEX_NAME)


def main() -> None:
    mongo_client = pymongo.MongoClient(os.environ["MONGODB_URI"])
    collection = mongo_client["rag_project"]["t2_ragbench_full"]

    print("[1/2] vector_index_full - adding source_dataset as a filter field...")
    update_vector_index(collection)

    print("[2/2] text_index_full - adding source_dataset as a token field...")
    update_text_index(collection)

    print(
        "\nDone. Both indexes support pre-filtering by source_dataset now - "
        "next: scripts/migrate_embedding_routing.py, then a normal `python -m pipeline.cli eval` run "
        "(or scripts/run_migration_and_eval.py, which does both in one step)."
    )


if __name__ == "__main__":
    main()

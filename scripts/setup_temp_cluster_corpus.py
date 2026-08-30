@@ -1,156 +0,0 @@
"""Step 1 of the full-corpus reranker+enrichment validation
(docs/tehnicheskoe_zadanie.md, section 5, "Открытый риск, не закрытый
тестами").

Copies the already-indexed corpus (context_id, raw_content,
contextual_summary, metadata_prefix, source_dataset - NOT embedding_voyage,
see below) from the production collection into a fresh collection on a
SEPARATE temporary cluster, and computes two NEW, uniformly voyage-4
embeddings there:

  - embedding_enriched: over metadata_prefix + contextual_summary +
    raw_content (= the same text as production's full_indexed_content)
  - embedding_raw: over metadata_prefix + raw_content (contextual_summary
    dropped) - built via pipeline.indexing.build_full_indexed_content()
    with contextual_summary="", so it's bit-identical to how the "no
    enrichment" branch is already defined elsewhere in this project.

Why NOT reuse production's existing embedding_voyage field for the
"enriched" arm: since per-dataset embedding routing shipped
(tehnicheskoe_zadanie.md, п.3a), embedding_voyage is no longer a single
model across the whole corpus - TAT-DQA documents hold voyage-finance-2
vectors, everything else holds voyage-4. Reusing it as-is here would
silently mix two incompatible vector spaces into one $vectorSearch index
unless the routing filter is reproduced for BOTH arms of this test too -
extra complexity that has nothing to do with the actual question this
test asks (does contextual enrichment help, combined with the reranker, at
full corpus scale). Recomputing both arms fresh with a single model
(voyage-4) isolates that one variable cleanly, matches how the original
partial-corpus (450-doc) enrichment tests were run, and keeps this test
fully self-contained on its own temporary cluster - it never touches or
depends on the production embedding_voyage field.

Cost: ~14,636 embedding calls (7,318 documents x 2 texts), Voyage AI
voyage-4, batched - well under $1 based on this project's own prior full-
corpus embedding cost measurement (~$0.13 for one full pass, section 3 of
tehnicheskoe_zadanie.md).

Resumable: skips documents already present in the destination collection,
so an interrupted run can just be restarted.

Usage (Colab, repo root, after the usual %cd + secrets-loading cells):
    Set SOURCE_MONGODB_URI (existing production cluster, read-only here)
    and DEST_MONGODB_URI (new, separate temporary cluster - see
    dopolnenie_polnyi_korpus_enrichment_plan.md for how to create it)
    as environment variables / Colab secrets, then:
    !python scripts/setup_temp_cluster_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

import os

import pymongo

from pipeline.embedding import BATCH_SIZE, MODEL, embed_documents
from pipeline.indexing import COLLECTION_NAME, DB_NAME, build_full_indexed_content

DEST_DB_NAME = "rag_project_enrichment_test"
DEST_COLLECTION_NAME = "t2_ragbench_enrichment_test"


def main() -> None:
    import voyageai

    source_client = pymongo.MongoClient(os.environ["SOURCE_MONGODB_URI"])
    source_collection = source_client[DB_NAME][COLLECTION_NAME]

    dest_client = pymongo.MongoClient(os.environ["DEST_MONGODB_URI"])
    dest_collection = dest_client[DEST_DB_NAME][DEST_COLLECTION_NAME]

    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print("[1/3] Reading source corpus (context_id, raw_content, contextual_summary, metadata_prefix, source_dataset)...")
    source_docs = list(
        source_collection.find(
            {},
            {
                "_id": 0,
                "context_id": 1,
                "raw_content": 1,
                "contextual_summary": 1,
                "metadata_prefix": 1,
                "source_dataset": 1,
            },
        )
    )
    print(f"  {len(source_docs)} documents in source collection")

    already_done = {d["context_id"] for d in dest_collection.find({}, {"_id": 0, "context_id": 1})}
    to_process = [d for d in source_docs if d["context_id"] not in already_done]
    print(f"  {len(already_done)} already present in destination (resumed), {len(to_process)} to embed")

    if not to_process:
        print("Nothing to do.")
        return

    print("[2/3] Building enriched_text / raw_text for each document...")
    enriched_items: list[tuple[str, str]] = []
    raw_items: list[tuple[str, str]] = []
    meta_by_id: dict[str, dict] = {}
    for d in to_process:
        context_id = d["context_id"]
        raw_content = d.get("raw_content", "")
        contextual_summary = d.get("contextual_summary", "") or ""
        metadata_prefix = d.get("metadata_prefix", "") or ""
        enriched_text = build_full_indexed_content(raw_content, contextual_summary, metadata_prefix)
        raw_text = build_full_indexed_content(raw_content, "", metadata_prefix)
        enriched_items.append((context_id, enriched_text))
        raw_items.append((context_id, raw_text))
        meta_by_id[context_id] = {
            "source_dataset": d.get("source_dataset", "unknown"),
            "enriched_text": enriched_text,
            "raw_text": raw_text,
        }

    print(f"[3/3] Embedding {len(enriched_items)} enriched + {len(raw_items)} raw texts with {MODEL!r}...")
    written = 0
    batch_size_docs = 200  # write to Mongo every N documents' worth of embeddings, not all at once
    for start in range(0, len(to_process), batch_size_docs):
        chunk_ids = [d["context_id"] for d in to_process[start : start + batch_size_docs]]
        chunk_enriched = [(cid, meta_by_id[cid]["enriched_text"]) for cid in chunk_ids]
        chunk_raw = [(cid, meta_by_id[cid]["raw_text"]) for cid in chunk_ids]

        enriched_vectors = embed_documents(voyage_client, chunk_enriched, model=MODEL)
        raw_vectors = embed_documents(voyage_client, chunk_raw, model=MODEL)
        enriched_by_id = {v.id: v.vector for v in enriched_vectors}
        raw_by_id = {v.id: v.vector for v in raw_vectors}

        for cid in chunk_ids:
            dest_collection.update_one(
                {"context_id": cid},
                {
                    "$set": {
                        "context_id": cid,
                        "source_dataset": meta_by_id[cid]["source_dataset"],
                        "enriched_text": meta_by_id[cid]["enriched_text"],
                        "raw_text": meta_by_id[cid]["raw_text"],
                        "embedding_enriched": enriched_by_id[cid],
                        "embedding_raw": raw_by_id[cid],
                    }
                },
                upsert=True,
            )
            written += 1
        print(f"  {written}/{len(to_process)} written")

    print(f"\nDone. {written} documents written to {DEST_DB_NAME}.{DEST_COLLECTION_NAME}.")
    print("Next: scripts/create_temp_cluster_indexes.py, then scripts/enrichment_reranker_full_corpus_ab.py.")


if __name__ == "__main__":
    main()

"""One-off migration: backfill source_dataset onto every already-indexed
document, and switch TAT-DQA documents' embedding_voyage field over to the
already-computed voyage-finance-2 vectors (per-dataset embedding routing,
docs/tehnicheskoe_zadanie.md, п.3a).

Why this is safe and cheap (no new Voyage/Anthropic/Cohere API calls):
- source_dataset is derivable purely from pipeline.ingestion (no API
  calls) - every document's source_dataset was already known at ingestion
  time, it just wasn't stored on the MongoDB document before 2026-08-15
  (pipeline/indexing.py's upsert_document() didn't take/store it).
- TAT-DQA's voyage-finance-2 vectors are ALREADY stored in the
  embedding_finance2 field, written non-destructively by
  scripts/voyage_finance2_ab.py's earlier A/B run - overwriting
  embedding_voyage with embedding_finance2 for TAT-DQA documents only is a
  same-database field copy, not a re-embed.
- ConvFinQA/FinQA documents are untouched (embedding_voyage stays
  voyage-4, matching config.embedding.model - only TAT-DQA is in
  config.embedding.routing.routed_sources).

Prerequisite: scripts/voyage_finance2_ab.py must have already run against
this collection (embedding_finance2 populated on every TAT-DQA document) -
this script checks that and refuses to run otherwise.

REQUIRES a manual Atlas Search index update to be USEFUL (this script does
not and cannot make it - only Atlas UI/Admin API can): source_dataset must
be added as a "filter"-type field to both vector_index_full and
text_index_full (see docs/tehnicheskoe_zadanie.md, п.3a, for the exact
index JSON). Without that, pipeline.indexing.validate_startup_indexes()
will fail loudly and clearly at the next `python -m pipeline.cli eval`
run, rather than silently returning wrong candidates - see that
function's docstring.

Usage (Colab, after the usual %cd + secrets-loading cells, run from the
repo root so `pipeline` is importable):
    !python scripts/migrate_embedding_routing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

import os

import pymongo

from pipeline.ingestion import load_raw, to_document_records

ROUTED_SOURCES = {"TAT-DQA"}  # must match config/config.yaml embedding.routing.routed_sources


def main() -> None:
    mongo_client = pymongo.MongoClient(os.environ["MONGODB_URI"])
    collection = mongo_client["rag_project"]["t2_ragbench_full"]

    print("[1/3] Deriving context_id -> source_dataset from raw T2-RAGBench files (no API calls)...")
    raw = load_raw("data/t2-ragbench")
    records = to_document_records(raw)
    source_by_id: dict[str, str] = {}
    for r in records:
        source_by_id.setdefault(r.context_id, r.source_dataset)
    print(f"  {len(source_by_id)} unique documents")

    print("[2/3] Checking embedding_finance2 is present on every TAT-DQA document...")
    docs = list(collection.find({}, {"context_id": 1, "embedding_finance2": 1}))
    doc_ids = {d["context_id"] for d in docs}
    finance2_by_id = {d["context_id"]: d["embedding_finance2"] for d in docs if "embedding_finance2" in d}

    missing_from_corpus = [cid for cid in source_by_id if cid not in doc_ids]
    if missing_from_corpus:
        raise RuntimeError(
            f"{len(missing_from_corpus)} documents from the raw dataset are not in the indexed "
            f"collection - run `python -m pipeline.cli index` first. First few: {missing_from_corpus[:5]}"
        )
    tatdqa_ids = {cid for cid, source in source_by_id.items() if source in ROUTED_SOURCES}
    missing_finance2 = [cid for cid in tatdqa_ids if cid not in finance2_by_id]
    if missing_finance2:
        raise RuntimeError(
            f"{len(missing_finance2)} TAT-DQA documents have no embedding_finance2 field - run "
            f"scripts/voyage_finance2_ab.py against this collection first. First few: {missing_finance2[:5]}"
        )
    print(f"  {len(tatdqa_ids)} TAT-DQA documents ready to migrate")

    print("[3/3] Writing source_dataset to every document, embedding_voyage <- embedding_finance2 for TAT-DQA...")
    updated_source = 0
    updated_embedding = 0
    for cid, source in source_by_id.items():
        update: dict = {"source_dataset": source}
        if source in ROUTED_SOURCES:
            update["embedding_voyage"] = finance2_by_id[cid]
            updated_embedding += 1
        collection.update_one({"context_id": cid}, {"$set": update})
        updated_source += 1
        if updated_source % 1000 == 0:
            print(f"  {updated_source}/{len(source_by_id)}")

    print(
        f"\nDone. source_dataset set on {updated_source} documents, "
        f"embedding_voyage switched to voyage-finance-2 on {updated_embedding} TAT-DQA documents."
    )
    print(
        "Next: update the Atlas index definitions (docs/tehnicheskoe_zadanie.md, п.3a) to add "
        "source_dataset as a filter field, then run `python -m pipeline.cli eval ...` - "
        "validate_startup_indexes() checks the filter works before any query runs."
    )


if __name__ == "__main__":
    main()

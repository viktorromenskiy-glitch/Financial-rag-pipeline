"""Step 1 of rebuilding the Colab environment from scratch (the old Colab
runtime expired from long idle time - everything below is being re-verified,
nothing is assumed to still work).

This is a pure diagnostic - it makes ZERO calls to Voyage / Cohere /
Anthropic (no embeddings, no reranking, no generation, no judge - all three
SDK clients are only *constructed*, never called). The only network call is
a couple of cheap MongoDB/Atlas queries. Nothing here can cost money.

What it checks, in order, stopping at the first failure so the error is
unambiguous about which layer broke:
  1. Config loads (config/config.yaml + .env are both present and parse).
  2. All three API keys are set in the environment (VOYAGE_API_KEY,
     ANTHROPIC_API_KEY, COHERE_API_KEY) - pipeline.cli.build_clients()
     raises RuntimeError listing exactly which ones are missing.
  3. The MongoDB Atlas cluster is actually reachable (not paused/deleted)
     and the collection has documents in it - a plain count_documents(),
     compared against the known corpus size (pipeline.ingestion.
     EXPECTED_DOCUMENTS = 7318) so a partial/stale collection is visible
     immediately, not just "connects OK".
  4. Both Atlas Search indexes (vector + text) are READY and actually
     queryable - pipeline.indexing.validate_startup_indexes(), the same
     mandatory check every real `pipeline.cli eval` run does first.

If this passes end to end, the environment is confirmed rebuilt and we can
move to the next step (a real, paid retrieval-only run for plan-2 item 7).
If it fails, the printed error says exactly which layer to fix (re-copy
.env, un-pause/recreate the Atlas cluster, rebuild the search indexes, or
re-run the corpus migration) before trying anything that costs money.

Usage (Colab, after the usual %cd + secrets-loading cells - this needs
.env in place but does NOT need anything else set up yet):
    !python scripts/check_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

from config.config_schema import load_config
from pipeline.cli import build_clients
from pipeline.indexing import validate_startup_indexes
from pipeline.ingestion import EXPECTED_DOCUMENTS


def main() -> None:
    print("=" * 70)
    print("STEP 1: environment check (zero API cost - no Voyage/Cohere/Anthropic calls)")
    print("=" * 70)

    # pipeline.cli.main() normally does this before touching config/env vars
    # (see its own "from dotenv import load_dotenv; load_dotenv()") - this
    # script calls load_config()/build_clients() directly instead of going
    # through cli.main(), so it has to do the same load_dotenv() itself, or
    # a real .env on disk is silently never read into os.environ.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    print("\n[1/4] Loading config/config.yaml + .env ...")
    config = load_config()
    print(f"      OK - db={config.mongodb.db_name} collection={config.mongodb.collection_name}")

    print("\n[2/4] Checking API keys and building clients (no calls made) ...")
    clients = build_clients(config)
    print("      OK - VOYAGE_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY all present")

    print("\n[3/4] Counting documents in the collection ...")
    collection = clients["collection"]
    actual_count = collection.count_documents({})
    print(f"      Found {actual_count} documents (expected {EXPECTED_DOCUMENTS})")
    if actual_count == 0:
        raise SystemExit(
            "FAIL: collection is empty - cluster is reachable but the corpus was never "
            "indexed here (wrong db/collection name, or a fresh/different cluster). "
            "Re-run the ingestion+indexing steps before anything else."
        )
    if actual_count != EXPECTED_DOCUMENTS:
        print(
            f"      WARNING: count differs from the expected {EXPECTED_DOCUMENTS} - corpus "
            "may be partial or stale. Not stopping here, but check this before trusting "
            "any eval numbers from this cluster."
        )

    print("\n[4/4] Validating Atlas Search indexes (vector + text, both must be READY) ...")
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)
    print("      OK - both indexes returned non-empty results on a test query")

    print("\n" + "=" * 70)
    print("PASS: environment is confirmed working end to end.")
    print("=" * 70)


if __name__ == "__main__":
    main()

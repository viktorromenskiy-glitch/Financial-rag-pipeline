"""Step 3 (the actual test) of the full-corpus reranker+enrichment
validation (docs/tehnicheskoe_zadanie.md, section 5, "Открытый риск, не
закрытый тестами"): does contextual enrichment help Recall@5, combined
with the reranker, at FULL 7,318-document corpus scale - not the reduced
450-document subsample the existing 0.980 checkpoint was measured on.

Runs retrieval + reranking ONLY (no generation/judge - this is a
retrieval-stage question, same discipline as every other embedding/
reranker A/B test in this project: cheap, focused, one variable at a
time). For each sampled question: embed the query once (voyage-4 - both
arms on the temporary cluster use voyage-4 uniformly, see
setup_temp_cluster_corpus.py), retrieve top-50 candidates separately
against the "enriched" arm (vector_index_enriched/text_index_enriched,
field enriched_text = metadata_prefix+contextual_summary+raw_content) and
the "raw" arm (vector_index_raw/text_index_raw, field raw_text =
metadata_prefix+raw_content, no summary), rerank each arm's pool to top-5
via Cohere, and check whether the question's own gold context_id landed in
that arm's top-5. Paired per question -> McNemar exact test, same method
(scipy.stats.binomtest on the minority discordant count) as every other
significance test in this project.

Usage (Colab, repo root, after scripts/setup_temp_cluster_corpus.py and
scripts/create_temp_cluster_indexes.py have both finished):
    !python scripts/enrichment_reranker_full_corpus_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

import os
import random

import pymongo
from scipy.stats import binomtest

from pipeline.embedding import MODEL, embed_query
from pipeline.ingestion import load_raw, to_document_records
from pipeline.reranking import rerank
from pipeline.retrieval import Candidate
from setup_temp_cluster_corpus import DEST_COLLECTION_NAME, DEST_DB_NAME

DATA_DIR = "data/t2-ragbench"
PER_SOURCE_N = 300  # 3 sources x 300 = 900, matching this project's existing full-corpus reranker checkpoint (n=900)
POOL_SIZE = 50
NUM_CANDIDATES_MULTIPLIER = 20
TOP_N = 5
RANDOM_SEED = 20260817

VECTOR_PIPELINE_NAME = "vectorPipeline"
TEXT_PIPELINE_NAME = "textPipeline"

ARMS = {
    "enriched": {"vector_index": "vector_index_enriched", "text_index": "text_index_enriched", "field": "enriched_text", "embedding": "embedding_enriched"},
    "raw": {"vector_index": "vector_index_raw", "text_index": "text_index_raw", "field": "raw_text", "embedding": "embedding_raw"},
}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def build_pipeline(query_vector: list[float], query_text: str, vector_index: str, text_index: str, field: str, embedding_field: str) -> list[dict]:
    num_candidates = POOL_SIZE * NUM_CANDIDATES_MULTIPLIER
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        VECTOR_PIPELINE_NAME: [
                            {
                                "$vectorSearch": {
                                    "index": vector_index,
                                    "path": embedding_field,
                                    "queryVector": query_vector,
                                    "numCandidates": num_candidates,
                                    "limit": POOL_SIZE,
                                }
                            }
                        ],
                        TEXT_PIPELINE_NAME: [
                            {
                                "$search": {
                                    "index": text_index,
                                    "text": {"query": query_text, "path": field},
                                }
                            },
                            {"$limit": POOL_SIZE},
                        ],
                    }
                },
                "combination": {"weights": {VECTOR_PIPELINE_NAME: 0.5, TEXT_PIPELINE_NAME: 0.5}},
            }
        },
        {"$limit": POOL_SIZE},
        {"$project": {"_id": 0, "context_id": 1, field: 1, "score": {"$meta": "score"}}},
    ]


def retrieve_arm(collection, voyage_client, cohere_client, query_text: str, query_vector: list[float], arm: dict) -> list[str]:
    """Returns the top-TOP_N context_ids after retrieval + rerank for one arm."""
    pipeline = build_pipeline(query_vector, query_text, arm["vector_index"], arm["text_index"], arm["field"], arm["embedding"])
    results = list(collection.aggregate(pipeline))
    if not results:
        return []
    candidates = [Candidate(context_id=r["context_id"], full_indexed_content=r[arm["field"]], score=r.get("score", 0.0)) for r in results]
    reranked = rerank(cohere_client, query_text, candidates, top_n=TOP_N)
    return [c.context_id for c in reranked]


def sample_questions() -> list[dict]:
    raw = load_raw(DATA_DIR)
    records = to_document_records(raw)

    df_records = [
        {"question": r.question, "context_id": r.context_id, "source_dataset": r.source_dataset}
        for r in records
        if r.question and str(r.question).strip()
    ]
    dropped = len(records) - len(df_records)
    if dropped:
        print(f"Dropped {dropped} row(s) with empty/missing question text ({dropped / len(records):.2%} of {len(records)})")

    by_source: dict[str, list[dict]] = {}
    for r in df_records:
        by_source.setdefault(r["source_dataset"], []).append(r)

    rng = random.Random(RANDOM_SEED)
    sampled: list[dict] = []
    for source, items in by_source.items():
        n = min(PER_SOURCE_N, len(items))
        sampled.extend(rng.sample(items, n))
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    import cohere
    import voyageai

    dest_client = pymongo.MongoClient(os.environ["DEST_MONGODB_URI"])
    collection = dest_client[DEST_DB_NAME][DEST_COLLECTION_NAME]

    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    print("Sampling questions (stratified per source_dataset)...")
    questions = sample_questions()
    print(f"  {len(questions)} questions sampled")

    hits: dict[str, list[bool]] = {"enriched": [], "raw": []}
    sources: list[str] = []

    for i, q in enumerate(questions):
        query_vector = embed_query(voyage_client, question_id=f"q_{i}", query_text=q["question"], model=MODEL).vector

        enriched_top5 = retrieve_arm(collection, voyage_client, cohere_client, q["question"], query_vector, ARMS["enriched"])
        raw_top5 = retrieve_arm(collection, voyage_client, cohere_client, q["question"], query_vector, ARMS["raw"])

        hits["enriched"].append(q["context_id"] in enriched_top5)
        hits["raw"].append(q["context_id"] in raw_top5)
        sources.append(q["source_dataset"])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)} processed")

    print(f"\n{'source_dataset':<14} {'n':>4} {'enriched':>9} {'raw':>9} {'delta':>7} {'b':>4} {'c':>4} {'McNemar p':>10}")
    print("-" * 75)

    def report(name: str, idx: list[int]) -> None:
        n = len(idx)
        e = sum(hits["enriched"][i] for i in idx)
        r = sum(hits["raw"][i] for i in idx)
        b = sum(1 for i in idx if hits["enriched"][i] and not hits["raw"][i])
        c = sum(1 for i in idx if not hits["enriched"][i] and hits["raw"][i])
        p = mcnemar_exact(b, c)
        e_acc, r_acc = e / n, r / n
        sig = "significant" if p < 0.05 else "not significant"
        print(f"{name:<14} {n:>4} {e_acc:>9.3f} {r_acc:>9.3f} {e_acc - r_acc:>+7.3f} {b:>4} {c:>4} {p:>10.4f}  -> {sig}")

    by_source_idx: dict[str, list[int]] = {}
    for i, s in enumerate(sources):
        by_source_idx.setdefault(s, []).append(i)
    for source in sorted(by_source_idx):
        report(source, by_source_idx[source])
    print("-" * 75)
    report("Overall", list(range(len(questions))))

    print(
        "\n'enriched' = metadata_prefix+contextual_summary+raw_content (current production text), "
        "'raw' = metadata_prefix+raw_content only, no summary. Both arms: voyage-4, pool=50, "
        "Cohere Rerank v4.0 Pro, top-5. Full corpus (7,318 documents), not the 450-document "
        "subsample the existing 0.980 checkpoint used."
    )


if __name__ == "__main__":
    main()

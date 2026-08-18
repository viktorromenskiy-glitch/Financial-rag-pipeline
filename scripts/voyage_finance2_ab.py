"""A/B test: voyage-4 (current production embedding model) vs voyage-finance-2
(domain-specific finance embedding). Worth testing empirically because this
project's corpus is entirely financial filings - a domain-specific model is
a plausible source of retrieval gains, but only measurement decides it.

Measures pure vector-retrieval recall@1/5/10 and MRR on the FULL indexed
corpus (all 7318 documents already in MongoDB Atlas), not a small
subsample - a smaller candidate pool would understate real competition
between documents and could distort the relative comparison between
models at production scale.

Cost note: voyage-finance-2 is $0.12/M input tokens with the first 50M
tokens free per account (docs.voyageai.com/docs/pricing, checked
2026-08-15) - the whole 7318-document corpus is well under 50M tokens, so
this should cost close to $0 unless the account's free tier is already
used up elsewhere. voyage-4 side re-uses the embeddings ALREADY stored in
MongoDB from the real index run - no new API calls for that half.

Non-destructive: writes new voyage-finance-2 vectors to a separate field
(embedding_finance2) on the SAME documents, does not touch the existing
`embedding` field or the production vector_index_full search index. Safe
to run against the live collection - nothing about retrieval-time
behavior changes until you deliberately act on the result.

Usage (Colab, after the usual %cd + secrets-loading cells):
    !python scripts/voyage_finance2_ab.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pymongo
import voyageai

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
N_QUERIES = 200  # sampled eval questions to use as retrieval queries
TOP_KS = [1, 5, 10]
BATCH_SIZE = 32  # matches pipeline/embedding.py's BATCH_SIZE convention
MODELS = ["voyage-4", "voyage-finance-2"]
RANDOM_SEED = 42

# id/context_id prefix -> source_dataset, mirrors pipeline/cli.py's
# _infer_source_dataset() - kept as a separate small copy here since this
# script is a standalone diagnostic, not part of the pipeline package.
_SOURCE_PREFIX_MAP = [
    ("convfinqa_", "ConvFinQA"),
    ("finqa_", "FinQA"),
    ("tat-dqa_", "TAT-DQA"),
    ("tatqa_", "TAT-DQA"),
]


def infer_source(*candidates) -> str:
    """Tries each candidate in order (mirrors pipeline/cli.py's
    _infer_source_dataset()) - context_id alone isn't reliably prefixed
    for every source_dataset in this corpus (confirmed 2026-08-15: TAT-DQA
    rows' context_id didn't match, while their 'id' column did), so the
    caller should pass both the row's 'id' and 'context_id' rather than
    context_id alone."""
    for value in candidates:
        if value is None:
            continue
        low = str(value).lower()
        for prefix, name in _SOURCE_PREFIX_MAP:
            if low.startswith(prefix):
                return name
    return "unknown"


def embed_batch(client: voyageai.Client, texts: list[str], model: str, input_type: str) -> np.ndarray:
    """Embeds texts in BATCH_SIZE chunks, returns an (n, dim) array."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = client.embed(batch, model=model, input_type=input_type)
        vectors.extend(result.embeddings)
        if (i // BATCH_SIZE + 1) % 20 == 0:
            print(f"    embedded {i + len(batch)}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def recall_and_mrr(sims: np.ndarray, pool_ids: list[str], queries: list[dict]) -> pd.DataFrame:
    """sims: (n_queries, n_pool) cosine similarity matrix. Returns one row
    per query with the gold document's rank (1-indexed, None if the gold
    document isn't in the pool at all - shouldn't happen here since gold
    docs are always included in the pool, but guarded defensively)."""
    rows = []
    for i, q in enumerate(queries):
        gold_cid = q["context_id"]
        order = np.argsort(-sims[i])
        ranked_ids = [pool_ids[j] for j in order]
        rank = ranked_ids.index(gold_cid) + 1 if gold_cid in ranked_ids else None
        rows.append(
            {
                "question_id": q.get("id", i),
                "context_id": gold_cid,
                "source": infer_source(q.get("id"), gold_cid),
                "rank": rank,
            }
        )
    return pd.DataFrame(rows)


def print_metrics(df_ranks: pd.DataFrame, label: str) -> dict:
    metrics = {}
    for k in TOP_KS:
        metrics[f"recall@{k}"] = float((df_ranks["rank"].notna() & (df_ranks["rank"] <= k)).mean())
    metrics["mrr"] = float(df_ranks["rank"].apply(lambda r: 1 / r if pd.notna(r) else 0.0).mean())

    print(f"\n--- {label} ---")
    print(f"Recall@1={metrics['recall@1']:.3f}  Recall@5={metrics['recall@5']:.3f}  Recall@10={metrics['recall@10']:.3f}  MRR={metrics['mrr']:.3f}")
    print("By source_dataset:")
    for source, g in df_ranks.groupby("source"):
        r5 = float((g["rank"].notna() & (g["rank"] <= 5)).mean())
        print(f"  {source}: n={len(g)}  recall@5={r5:.3f}")
    return metrics


def main() -> None:
    mongo_client = pymongo.MongoClient(os.environ["MONGODB_URI"])
    collection = mongo_client["rag_project"]["t2_ragbench_full"]
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print("[1/4] Loading eval questions + sampling queries...")
    eval_df = pd.read_parquet("data/t2-ragbench/eval_subset_250.parquet")
    n = min(N_QUERIES, len(eval_df))
    sample_df = eval_df.sample(n=n, random_state=RANDOM_SEED)
    queries = sample_df.to_dict("records")
    print(f"  {len(queries)} queries sampled")

    print("[2/4] Loading full document pool from MongoDB (id, context_id, full_indexed_content, embedding_voyage)...")
    docs = list(collection.find({}, {"context_id": 1, "full_indexed_content": 1, "embedding_voyage": 1}))
    pool_ids = [d["context_id"] for d in docs]
    pool_texts = [d["full_indexed_content"] for d in docs]
    existing_v4_vectors = np.array([d["embedding_voyage"] for d in docs], dtype=np.float32)
    print(f"  {len(pool_ids)} documents in pool (full corpus)")

    missing_gold = [q["context_id"] for q in queries if q["context_id"] not in set(pool_ids)]
    if missing_gold:
        raise RuntimeError(
            f"{len(missing_gold)} sampled queries' gold context_id not found in the indexed collection - "
            f"is the corpus fully indexed? First few: {missing_gold[:5]}"
        )

    results = {}

    print("\n[3/4] voyage-4 (re-using embeddings already stored in MongoDB - no new API calls for documents)...")
    doc_vectors_v4 = normalize(existing_v4_vectors)
    query_texts = [q["question"] for q in queries]
    print("  Embedding queries with voyage-4...")
    query_vectors_v4 = normalize(embed_batch(voyage_client, query_texts, "voyage-4", input_type="query"))
    sims_v4 = query_vectors_v4 @ doc_vectors_v4.T
    df_ranks_v4 = recall_and_mrr(sims_v4, pool_ids, queries)
    results["voyage-4"] = print_metrics(df_ranks_v4, "voyage-4")

    print("\n[4/4] voyage-finance-2 (embedding full corpus + queries fresh)...")
    print(f"  Embedding {len(pool_texts)} documents with voyage-finance-2...")
    doc_vectors_fin2 = normalize(embed_batch(voyage_client, pool_texts, "voyage-finance-2", input_type="document"))
    print("  Embedding queries with voyage-finance-2...")
    query_vectors_fin2 = normalize(embed_batch(voyage_client, query_texts, "voyage-finance-2", input_type="query"))
    sims_fin2 = query_vectors_fin2 @ doc_vectors_fin2.T
    df_ranks_fin2 = recall_and_mrr(sims_fin2, pool_ids, queries)
    results["voyage-finance-2"] = print_metrics(df_ranks_fin2, "voyage-finance-2")

    # Non-destructive: store the new vectors on the documents under a
    # separate field, so this test doesn't touch the production `embedding`
    # field or vector_index_full at all. Only written if you want to keep
    # them around for later (e.g. to avoid re-embedding if you decide to
    # switch) - comment out if you don't want the extra ~1024 floats/doc
    # stored yet.
    print("\nSaving voyage-finance-2 vectors to embedding_finance2 field (non-destructive, doesn't affect production retrieval)...")
    for cid, vec in zip(pool_ids, doc_vectors_fin2):
        collection.update_one({"context_id": cid}, {"$set": {"embedding_finance2": vec.tolist()}})
    print("  Done.")

    print("\n" + "=" * 60)
    print("ИТОГ: voyage-4 vs voyage-finance-2 (recall on full 7318-doc corpus)")
    print("=" * 60)
    for k in TOP_KS:
        v4 = results["voyage-4"][f"recall@{k}"]
        fin2 = results["voyage-finance-2"][f"recall@{k}"]
        print(f"Recall@{k}: voyage-4={v4:.3f}  voyage-finance-2={fin2:.3f}  Δ={fin2 - v4:+.3f}")
    v4_mrr = results["voyage-4"]["mrr"]
    fin2_mrr = results["voyage-finance-2"]["mrr"]
    print(f"MRR:       voyage-4={v4_mrr:.3f}  voyage-finance-2={fin2_mrr:.3f}  Δ={fin2_mrr - v4_mrr:+.3f}")


if __name__ == "__main__":
    main()

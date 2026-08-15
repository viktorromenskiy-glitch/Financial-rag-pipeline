"""Statistical significance check for the voyage-4 vs voyage-finance-2 A/B
test (test_voyage_finance2_ab.py must be run first).

Reuses embedding_voyage AND embedding_finance2 - BOTH already stored on
every document in MongoDB from the previous A/B run - so this only pays
for 2 x N_QUERIES query embeddings (cheap), no corpus re-embedding.

Computes McNemar's exact test on paired recall@5 hit/miss per query,
overall and per source_dataset - the same discipline this project already
used to validate metadata_prefix (McNemar p=0.00195, see
docs/tehnicheskoe_zadanie.md) - because a 3-5 pp difference on ~30-100
questions per source_dataset could easily be noise, not signal.

Usage (Colab, after test_voyage_finance2_ab.py has already run once):
    !python test_voyage_finance2_significance.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pymongo
import voyageai
from scipy.stats import binomtest

from test_voyage_finance2_ab import BATCH_SIZE, N_QUERIES, RANDOM_SEED, embed_batch, normalize, recall_and_mrr


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar's test via the binomial distribution (no
    chi-square/continuity-correction approximation - appropriate given
    these per-source_dataset discordant-pair counts can be small).
    b = model A right / model B wrong on the same query.
    c = model A wrong / model B right on the same query."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main() -> None:
    mongo_client = pymongo.MongoClient(os.environ["MONGODB_URI"])
    collection = mongo_client["rag_project"]["t2_ragbench_full"]
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print("[1/3] Loading eval questions (same sample as the A/B run - same seed)...")
    eval_df = pd.read_parquet("data/t2-ragbench/eval_subset_250.parquet")
    n = min(N_QUERIES, len(eval_df))
    sample_df = eval_df.sample(n=n, random_state=RANDOM_SEED)
    queries = sample_df.to_dict("records")
    print(f"  {len(queries)} queries")

    print("[2/3] Loading BOTH embeddings already stored in MongoDB (no corpus re-embedding)...")
    docs = list(collection.find({}, {"context_id": 1, "embedding_voyage": 1, "embedding_finance2": 1}))
    missing = [d["context_id"] for d in docs if "embedding_finance2" not in d]
    if missing:
        raise RuntimeError(
            f"{len(missing)} documents have no embedding_finance2 field - run "
            f"test_voyage_finance2_ab.py first. First few missing: {missing[:5]}"
        )
    pool_ids = [d["context_id"] for d in docs]
    doc_vectors_v4 = normalize(np.array([d["embedding_voyage"] for d in docs], dtype=np.float32))
    doc_vectors_fin2 = normalize(np.array([d["embedding_finance2"] for d in docs], dtype=np.float32))
    print(f"  {len(pool_ids)} documents")

    print("[3/3] Embedding queries with both models (only new API calls this script makes)...")
    query_texts = [q["question"] for q in queries]
    query_vectors_v4 = normalize(embed_batch(voyage_client, query_texts, "voyage-4", input_type="query"))
    query_vectors_fin2 = normalize(embed_batch(voyage_client, query_texts, "voyage-finance-2", input_type="query"))

    sims_v4 = query_vectors_v4 @ doc_vectors_v4.T
    sims_fin2 = query_vectors_fin2 @ doc_vectors_fin2.T

    df_v4 = recall_and_mrr(sims_v4, pool_ids, queries)
    df_fin2 = recall_and_mrr(sims_fin2, pool_ids, queries)

    merged = df_v4.merge(df_fin2, on="question_id", suffixes=("_v4", "_fin2"))
    merged["hit5_v4"] = merged["rank_v4"].notna() & (merged["rank_v4"] <= 5)
    merged["hit5_fin2"] = merged["rank_fin2"].notna() & (merged["rank_fin2"] <= 5)
    merged["source"] = merged["source_v4"]

    def report(df: pd.DataFrame, label: str) -> None:
        b = int((df["hit5_v4"] & ~df["hit5_fin2"]).sum())  # voyage-4 right, finance-2 wrong
        c = int((~df["hit5_v4"] & df["hit5_fin2"]).sum())  # voyage-4 wrong, finance-2 right
        p = mcnemar_exact(b, c)
        r4 = float(df["hit5_v4"].mean())
        rf = float(df["hit5_fin2"].mean())
        sig = "significant (p<0.05)" if p < 0.05 else "NOT significant"
        print(
            f"{label}: n={len(df)}  recall@5 voyage-4={r4:.3f} voyage-finance-2={rf:.3f} Δ={rf - r4:+.3f}  "
            f"discordant pairs: voyage-4-only-right={b} voyage-finance-2-only-right={c}  "
            f"McNemar exact p={p:.4f} -> {sig}"
        )

    print("\n" + "=" * 70)
    print("ИТОГ: McNemar's exact test on paired recall@5 (voyage-4 vs voyage-finance-2)")
    print("=" * 70)
    report(merged, "Overall")
    for source, g in merged.groupby("source"):
        report(g, source)


if __name__ == "__main__":
    main()

"""Statistical significance check for the voyage-4 vs voyage-finance-2 A/B
test - v2, sourcing queries from the FULL T2-RAGBench raw question set
(23,088 questions via pipeline.ingestion) instead of the 250-question
eval_subset_250.parquet used by test_voyage_finance2_significance.py (v1).

Why v2: v1 (n=200, drawn from eval_subset_250.parquet) found NO result
significant at p<0.05 anywhere - not the TAT-DQA +5.2pp delta (p=0.18,
only 9 discordant pairs out of n=97) or the ConvFinQA -3.4pp "regression"
(p=1.0, only 1 discordant pair out of n=29 - indistinguishable from
noise). Root cause: eval_subset_250.parquet is an intentionally small
subset built to keep the Sonnet generation+judge eval affordable - far
too small for a *retrieval-only* significance test, which needs neither
generation nor judging and can safely use much more data. This script
instead loads all 23,088 raw questions via pipeline.ingestion.load_raw()
+ to_document_records() (the same module used to build the indexed
corpus), which also gives an exact, non-inferred source_dataset per row -
no id/context_id-prefix guessing needed, unlike v1's infer_source().

Sampling: stratified per source_dataset, up to PER_SOURCE_N queries per
dataset (or all available rows if fewer) - deliberately not a flat random
sample of the full 23,088, since a flat sample would still under-represent
ConvFinQA (the smallest of the three sources) exactly the way
eval_subset_250.parquet did.

Reuses embedding_voyage AND embedding_finance2 - BOTH already stored on
every document in MongoDB from the original test_voyage_finance2_ab.py
run - so this again pays only for query embeddings (2 x sample size),
still near-$0 given voyage-finance-2's free tier
(docs.voyageai.com/docs/pricing, checked 2026-08-15).

Usage (Colab, run from the repo root so `pipeline` is importable, after
test_voyage_finance2_ab.py has already run once):
    !python test_voyage_finance2_significance_v2.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pymongo
import voyageai
from scipy.stats import binomtest

from pipeline.ingestion import load_raw, to_document_records
from test_voyage_finance2_ab import BATCH_SIZE, embed_batch, normalize

PER_SOURCE_N = 700  # cap per source_dataset; uses all available rows if fewer
RANDOM_SEED = 42


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

    print("[1/4] Loading FULL raw T2-RAGBench question set (pipeline.ingestion)...")
    raw = load_raw("data/t2-ragbench")
    records = to_document_records(raw)
    print(f"  {len(records)} total question rows across {len({r.context_id for r in records})} unique documents")

    df = pd.DataFrame(
        {
            "context_id": [r.context_id for r in records],
            "question": [r.question for r in records],
            "source_dataset": [r.source_dataset for r in records],
        }
    )

    # The raw T2-RAGBench files (unlike the curated eval_subset_250.parquet)
    # contain some rows with a null/empty/whitespace-only question - the
    # first real run hit this: voyageai.error.InvalidRequestError "Input
    # cannot contain empty strings or empty lists", 2026-08-15. Drop them
    # before sampling so an unlucky draw can't put an empty string in a
    # query batch and crash embed_batch() partway through a run.
    before = len(df)
    df = df.dropna(subset=["question"])
    df = df[df["question"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with empty/missing question text ({dropped / before:.2%} of {before})")

    print(f"[2/4] Stratified sampling up to {PER_SOURCE_N} queries per source_dataset...")
    parts = []
    for source, group in df.groupby("source_dataset"):
        n = min(PER_SOURCE_N, len(group))
        parts.append(group.sample(n=n, random_state=RANDOM_SEED))
        print(f"  {source}: sampled {n}/{len(group)}")
    sample_df = pd.concat(parts, ignore_index=True)
    sample_df["question_id"] = sample_df.index.astype(str) + "_" + sample_df["context_id"].astype(str)
    queries = sample_df.to_dict("records")
    print(f"  {len(queries)} queries total")

    print("[3/4] Loading BOTH embeddings already stored in MongoDB (no corpus re-embedding)...")
    docs = list(collection.find({}, {"context_id": 1, "embedding_voyage": 1, "embedding_finance2": 1}))
    missing = [d["context_id"] for d in docs if "embedding_finance2" not in d]
    if missing:
        raise RuntimeError(
            f"{len(missing)} documents have no embedding_finance2 field - run "
            f"test_voyage_finance2_ab.py first. First few missing: {missing[:5]}"
        )
    pool_ids = [d["context_id"] for d in docs]
    pool_id_set = set(pool_ids)
    doc_vectors_v4 = normalize(np.array([d["embedding_voyage"] for d in docs], dtype=np.float32))
    doc_vectors_fin2 = normalize(np.array([d["embedding_finance2"] for d in docs], dtype=np.float32))
    print(f"  {len(pool_ids)} documents")

    missing_gold = [q["context_id"] for q in queries if q["context_id"] not in pool_id_set]
    if missing_gold:
        raise RuntimeError(
            f"{len(missing_gold)} sampled queries' gold context_id not found in the indexed collection - "
            f"is the corpus fully indexed? First few: {missing_gold[:5]}"
        )

    print("[4/4] Embedding queries with both models (only new API calls this script makes)...")
    query_texts = [q["question"] for q in queries]
    query_vectors_v4 = normalize(embed_batch(voyage_client, query_texts, "voyage-4", input_type="query"))
    query_vectors_fin2 = normalize(embed_batch(voyage_client, query_texts, "voyage-finance-2", input_type="query"))

    sims_v4 = query_vectors_v4 @ doc_vectors_v4.T
    sims_fin2 = query_vectors_fin2 @ doc_vectors_fin2.T

    def ranks(sims: np.ndarray) -> pd.DataFrame:
        rows = []
        for i, q in enumerate(queries):
            gold_cid = q["context_id"]
            order = np.argsort(-sims[i])
            ranked_ids = [pool_ids[j] for j in order]
            rank = ranked_ids.index(gold_cid) + 1 if gold_cid in ranked_ids else None
            rows.append({"question_id": q["question_id"], "source": q["source_dataset"], "rank": rank})
        return pd.DataFrame(rows)

    df_v4 = ranks(sims_v4)
    df_fin2 = ranks(sims_fin2)

    merged = df_v4.merge(df_fin2, on="question_id", suffixes=("_v4", "_fin2"))
    merged["hit5_v4"] = merged["rank_v4"].notna() & (merged["rank_v4"] <= 5)
    merged["hit5_fin2"] = merged["rank_fin2"].notna() & (merged["rank_fin2"] <= 5)
    merged["source"] = merged["source_v4"]

    def report(d: pd.DataFrame, label: str) -> None:
        b = int((d["hit5_v4"] & ~d["hit5_fin2"]).sum())  # voyage-4 right, finance-2 wrong
        c = int((~d["hit5_v4"] & d["hit5_fin2"]).sum())  # voyage-4 wrong, finance-2 right
        p = mcnemar_exact(b, c)
        r4 = float(d["hit5_v4"].mean())
        rf = float(d["hit5_fin2"].mean())
        sig = "significant (p<0.05)" if p < 0.05 else "NOT significant"
        print(
            f"{label}: n={len(d)}  recall@5 voyage-4={r4:.3f} voyage-finance-2={rf:.3f} Δ={rf - r4:+.3f}  "
            f"discordant pairs: voyage-4-only-right={b} voyage-finance-2-only-right={c}  "
            f"McNemar exact p={p:.4f} -> {sig}"
        )

    print("\n" + "=" * 70)
    print("ИТОГ v2: McNemar's exact test, full T2-RAGBench stratified sample")
    print("=" * 70)
    report(merged, "Overall")
    for source, g in merged.groupby("source"):
        report(g, source)


if __name__ == "__main__":
    main()

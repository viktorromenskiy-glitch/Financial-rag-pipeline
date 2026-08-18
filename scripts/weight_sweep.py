"""Tier 6b - fusion weight sweep (vector vs text weight in $rankFusion).

tehnicheskoe_zadanie.md, section 4, flags retrieval.weights (0.5/0.5) as an
"untuned default", not a locally-optimized value - this script closes that
gap: sweep a small grid of vector/text weight combinations against the
committed eval_subset_250.parquet (the only eval set checked into the
repo), measure Recall@5 at each point, and test each non-default combo
against the current 0.5/0.5 default with a paired McNemar exact test (same
250 questions in every arm, so pairing is valid - same methodology already
used throughout the project for the pool_size and embedding-routing
comparisons, see tehnicheskoe_zadanie.md sections 3a and 6).

Deliberately cheap: only Voyage query-embedding calls are made (one call
per question per weight combo - no Cohere reranker calls, no Claude calls).
No new Atlas index is needed - this only varies the `weights` argument to
the existing vector_index_full/text_index_full $rankFusion query on the
already-indexed main collection (rag_project.t2_ragbench_full, M0 tier).

Runs against the CURRENT production embedding-routing config (TAT-DQA on
voyage-finance-2, everything else on voyage-4 - tehnicheskoe_zadanie.md,
section 3a) using the exact same per-question filter-mode logic as
pipeline/cli.py's cmd_eval(), not a routing-naive simplification - a weight
sweep that ignored routing would not reflect what the pipeline actually
runs with.

Usage:
    python -m scripts.weight_sweep --questions data/t2-ragbench/eval_subset_250.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest

from config.config_schema import load_config
from pipeline.cli import _resolve_embedding_model, build_clients, load_eval_questions
from pipeline.retrieval import retrieve

# Grid: five combinations spanning text-only-leaning to vector-only-leaning,
# centered on the current default (0.5/0.5) so the default itself is one of
# the tested points, not just the comparison baseline.
WEIGHT_GRID = [
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),  # current default
    (0.6, 0.4),
    (0.7, 0.3),
]

POOL_SIZE = 50


def recall_at_5(candidates, gold_context_id: str) -> bool:
    # $rankFusion's own output order is already the fused-score ranking
    # (confirmed via the "score": {"$meta": "score"} projection in
    # pipeline/retrieval.py) - candidates[:5] is the top-5, no re-sort here.
    return gold_context_id in {c.context_id for c in candidates[:5]}


def run_one_weight_combo(clients, config, items, vector_weight: float, text_weight: float) -> list[bool]:
    routing = config.embedding.routing
    hits = []
    for i, item in enumerate(items):
        embedding_model = _resolve_embedding_model(config, item["source_dataset"])
        is_routed_source = routing.enabled and item["source_dataset"] in routing.routed_sources
        candidates = retrieve(
            clients["voyage"],
            clients["collection"],
            item["question"],
            pool_size=POOL_SIZE,
            vector_weight=vector_weight,
            text_weight=text_weight,
            source_dataset=item["source_dataset"] if is_routed_source else None,
            exclude_source_datasets=list(routing.routed_sources) if (routing.enabled and not is_routed_source) else None,
            embedding_model=embedding_model,
        )
        hits.append(recall_at_5(candidates, item["context_id"]))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(items)}")
    return hits


def mcnemar_vs_default(default_hits: list[bool], other_hits: list[bool]):
    b = sum(1 for d, o in zip(default_hits, other_hits) if d and not o)  # default right, other wrong
    c = sum(1 for d, o in zip(default_hits, other_hits) if not d and o)  # default wrong, other right
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--questions", default="data/t2-ragbench/eval_subset_250.parquet")
    args = parser.parse_args()

    config = load_config(args.config)
    clients = build_clients(config)
    items = load_eval_questions(args.questions)
    print(f"Loaded {len(items)} questions from {args.questions}")

    # load_eval_questions() (pipeline/cli.py) doesn't currently carry
    # context_id through to the returned dict (only question_id/question/
    # gold_answer/source_dataset) - re-read it directly here rather than
    # editing that shared function for a one-off script.
    import pandas as pd

    df = pd.read_parquet(args.questions)
    context_id_by_question_id = dict(zip(df["id"].astype(str), df["context_id"]))
    for item in items:
        item["context_id"] = context_id_by_question_id[item["question_id"]]

    results = {}
    for vector_weight, text_weight in WEIGHT_GRID:
        print(f"\n  weights: vector={vector_weight} text={text_weight}")
        hits = run_one_weight_combo(clients, config, items, vector_weight, text_weight)
        recall = sum(hits) / len(hits)
        results[(vector_weight, text_weight)] = (hits, recall)
        print(f"  Recall@5 = {recall:.4f}")

    default_hits, default_recall = results[(0.5, 0.5)]

    print("\n\n=== Weight sweep results (n={}) ===".format(len(items)))
    print(f"{'vector':>8} {'text':>6} {'Recall@5':>10} {'delta':>8} {'b':>4} {'c':>4} {'p-value':>10}")
    for (vw, tw), (hits, recall) in results.items():
        if (vw, tw) == (0.5, 0.5):
            print(f"{vw:>8.1f} {tw:>6.1f} {recall:>10.4f} {'--':>8} {'--':>4} {'--':>4} {'(baseline)':>10}")
            continue
        b, c, p = mcnemar_vs_default(default_hits, hits)
        delta = recall - default_recall
        sig = "-> significant" if p < 0.05 else "-> not significant"
        print(f"{vw:>8.1f} {tw:>6.1f} {recall:>10.4f} {delta:>+8.4f} {b:>4} {c:>4} {p:>10.4f}  {sig}")


if __name__ == "__main__":
    main()

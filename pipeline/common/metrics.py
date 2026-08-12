"""Shared evaluation metrics used across retrieval/reranking checkpoints
(plan Steps 3, 4, and later evaluation runs). See
docs/plan_podgotovki_k_kodirovaniyu.md for the Recall@5 checkpoints this
supports (0.808 baseline retrieval, 0.944 with reranker, etc.).
"""

from __future__ import annotations


def is_hit_at_k(candidate_context_ids: list[str], gold_context_id: str, k: int) -> bool:
    """Returns True if gold_context_id appears among the first k entries of
    candidate_context_ids (assumed already ranked best-first, e.g. from
    module 6 retrieve() or module 7 reranking)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return gold_context_id in candidate_context_ids[:k]


def recall_at_k(hits: int, total: int) -> float:
    """Aggregate Recall@k = hits / total over an evaluation set."""
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    return hits / total

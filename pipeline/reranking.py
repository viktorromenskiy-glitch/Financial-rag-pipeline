"""Module 7 - Reranking.

Reranks the top pool_size hybrid retrieval candidates (module 6 output)
using Cohere Rerank v4.0 Pro, returning the top_n candidates ranked by
relevance to the query. See docs/specifikatsiya_moduley.md, module 7.

Known pitfall (confirmed twice during weeks 1-2 testing, see
docs/plan_podgotovki_k_kodirovaniyu.md, Step 4): truncating document text
before sending it to the Cohere API caused a measurable Recall@5
regression. Two separate truncation points must both be avoided:
  1. Do not truncate full_indexed_content yourself before calling this
     module - candidates must carry the untruncated text from module 6.
     Enforced below as a checkable invariant (assert), not just a comment
     (spec section 7, "Явный запрет обрезки").
  2. The Cohere API itself truncates each document to max_tokens_per_doc
     (API default: 4096) before scoring. MAX_TOKENS_PER_DOC below
     overrides that default to stay just under the model's real
     32,768-token context window, so full_indexed_content is not
     silently cut by the API either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.common.retry import retryable

MODEL = "rerank-v4.0-pro"
TOP_N = 5
MAX_TOKENS_PER_DOC = 32000  # model context is 32768; API default of 4096 would truncate


@dataclass(frozen=True)
class RerankedCandidate:
    context_id: str
    full_indexed_content: str
    relevance_score: float


class CohereClientProtocol(Protocol):
    """Minimal interface required from cohere.ClientV2 - allows a fake
    client to be substituted in tests without a real network call."""

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
        max_tokens_per_doc: int,
    ): ...


@retryable()
def _rerank_with_retry(
    client: CohereClientProtocol,
    query: str,
    documents: list[str],
    top_n: int,
    max_tokens_per_doc: int,
):
    return client.rerank(
        model=MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        max_tokens_per_doc=max_tokens_per_doc,
    )


def rerank(
    client: CohereClientProtocol,
    query: str,
    candidates: list,  # list[pipeline.retrieval.Candidate]
    top_n: int = TOP_N,
    max_tokens_per_doc: int = MAX_TOKENS_PER_DOC,
) -> list[RerankedCandidate]:
    """candidates: module 6 Candidate objects (context_id,
    full_indexed_content, score) - the pool_size hybrid retrieval output.
    full_indexed_content must be the full, untruncated text - see module
    docstring."""
    if not candidates:
        return []

    documents = [c.full_indexed_content for c in candidates]
    if any(not d for d in documents):
        raise ValueError("Empty full_indexed_content in reranking input")

    # Checkable invariant, not just a comment (spec section 7, "Явный запрет
    # обрезки"): the text sent to the API must be full_indexed_content in
    # full, never a slice. This is the exact bug that caused a measured
    # Recall@5 regression twice during weeks 1-2 testing - guards against a
    # future edit accidentally reintroducing truncation here.
    for doc, candidate in zip(documents, candidates):
        assert doc == candidate.full_indexed_content, (
            f"Truncation detected before the Cohere API call: document text for "
            f"context_id={candidate.context_id!r} does not match full_indexed_content "
            f"- see module docstring"
        )

    response = _rerank_with_retry(client, query, documents, top_n, max_tokens_per_doc)

    reranked: list[RerankedCandidate] = []
    for result in response.results:
        original = candidates[result.index]
        reranked.append(
            RerankedCandidate(
                context_id=original.context_id,
                full_indexed_content=original.full_indexed_content,
                relevance_score=result.relevance_score,
            )
        )
    return reranked

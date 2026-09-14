"""agent/ tool 1 - search_documents, the agent's one mandatory tool.

Consensus item 1 of the 4-expert design review (itog_ekspertizy_agent_profil.md):
the only obligatory tool for V1 is a thin wrapper around modules 6-7
(pipeline.retrieval.retrieve + pipeline.reranking.rerank) - not a new
retrieval implementation. This module is that wrapper.

NoSQL-injection invariant (consensus item 5): the LLM controls only the
*value* of a free-text search query, never the shape of the underlying
MongoDB aggregation pipeline (pool size, weights, source_dataset
filters - all fixed, caller-supplied values). This was already true of
the baseline pipeline by construction (pipeline.retrieval.retrieve()
only ever takes `question_text` as a value, never a filter expression),
but the design review explicitly required this to be pinned down by a
test rather than left as a fact of today's code that a future refactor
could silently break - see test_agent_tools.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.embedding import MODEL as DEFAULT_EMBEDDING_MODEL
from pipeline.embedding import VoyageClientProtocol
from pipeline.reranking import CohereClientProtocol, RerankedCandidate, rerank
from pipeline.retrieval import Candidate, CollectionProtocol, retrieve

# Anthropic tool-use JSON schema for search_documents. `additionalProperties:
# False` plus a single `query` property is what actually enforces the
# NoSQL-invariant on the model-facing side: there is no field in this
# schema the model could populate to influence pool size, weights, or
# which source_dataset(s) are searched - those are supplied by the
# caller (see search_documents() below), never parsed out of the model's
# tool-call arguments.
SEARCH_DOCUMENTS_TOOL_SPEC: dict = {
    "name": "search_documents",
    "description": (
        "Search the financial document corpus for passages relevant to a query. "
        "Returns the top-ranked matching passages. Call this again with a "
        "reformulated query if the results are not enough to answer the question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query text - natural language, not a database "
                    "filter or structured expression."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class SearchToolCall:
    """The result of one search_documents execution: the query that was
    actually run (echoed back so callers/traces don't need to track it
    separately) and the ranked candidates it returned."""

    query: str
    candidates: tuple[RerankedCandidate | Candidate, ...]

    @property
    def context_ids(self) -> frozenset[str]:
        return frozenset(c.context_id for c in self.candidates)


def search_documents(
    voyage_client: VoyageClientProtocol,
    collection: CollectionProtocol,
    cohere_client: CohereClientProtocol | None,
    query: str,
    *,
    pool_size: int,
    vector_weight: float,
    text_weight: float,
    reranker_enabled: bool,
    reranker_top_n: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    source_dataset: str | None = None,
    exclude_source_datasets: list[str] | None = None,
) -> SearchToolCall:
    """Executes the agent's one core tool: retrieve() + (optionally) rerank().

    `query` is the only value ever supplied by the LLM (via a parsed
    search_documents tool-call). Every other parameter here is a fixed,
    caller-controlled value - the same config-driven values
    pipeline.cli.py's cmd_eval already passes to retrieve()/rerank() for
    the baseline pipeline (pool_size, weights, reranker settings, and the
    per-dataset embedding routing params for a given question's
    source_dataset - see pipeline.retrieval.retrieve()'s docstring for
    why source_dataset/exclude_source_datasets/embedding_model must agree
    and must never be derived from model output).

    Raises:
        ValueError: if `query` is empty or whitespace-only - an agent
            calling its own tool with nothing to search for is a bug in
            the calling code (a malformed model tool-call should be
            rejected before it reaches here), not a case worth spending
            an API round-trip on.
    """
    if not query or not query.strip():
        raise ValueError("search_documents requires a non-empty query")

    candidates = retrieve(
        voyage_client,
        collection,
        query,
        pool_size=pool_size,
        vector_weight=vector_weight,
        text_weight=text_weight,
        source_dataset=source_dataset,
        exclude_source_datasets=exclude_source_datasets,
        embedding_model=embedding_model,
    )
    if reranker_enabled and candidates and cohere_client is not None:
        ranked = rerank(cohere_client, query, candidates, top_n=reranker_top_n)
    else:
        ranked = candidates[:reranker_top_n]
    return SearchToolCall(query=query, candidates=tuple(ranked))

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

Graceful degradation (Day 2 of plan_rabot_posle_ekspertizy_agent_profil.md,
"Safety, robustness, evaluation harness" - 2 of the 2 required
scenarios, the empty-retrieval scenario being handled in agent/loop.py):
a transient failure of an external service must not crash the whole
bounded loop (and, in a real evaluation run, the whole batch of
questions after it). Two independent degradation paths below, both gated
on pipeline.common.retry.is_transient_error - the same classification the
retry layer already uses - so a genuine bug (a 4xx/programming error) is
never silently swallowed, only a real 429/5xx/connection/timeout failure
that has already exhausted retrieve()'s/rerank()'s own retry budget:

  1. MongoDB (retrieve()) fails: retrieve() has no retry wrapper of its
     own (unlike rerank() below) - a transient pymongo failure reaches
     here directly. Degrades to an empty SearchToolCall (candidates=())
     rather than raising - agent/loop.py's existing STOP_EMPTY_EVIDENCE
     path (Day 1) already handles zero evidence gracefully, so this
     reuses that path instead of adding a second one.
  2. Cohere (rerank()) fails after its own @retryable() budget is
     exhausted: degrades to the unreranked top reranker_top_n candidates
     (the same fallback path already used when reranker_enabled is False)
     rather than losing MongoDB's retrieval results entirely - a rerank
     outage should cost result quality, not the whole answer.

Both paths are recorded on the returned SearchToolCall (`degraded`,
`degradation_reason`) rather than raised or silently dropped, so a real
run's JSONL trace (agent/loop.py's `_trace_retrieval`) shows exactly when
and why a call degraded.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.common.retry import is_transient_error
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
    separately) and the ranked candidates it returned.

    `degraded`/`degradation_reason` (Day 2): set when a transient
    MongoDB or Cohere failure forced this call to return a lesser result
    (empty or unreranked candidates - see module docstring) instead of
    raising. Both default to the Day 1 values (False/None) so every Day 1
    call site and test that constructs a SearchToolCall without these
    kwargs keeps working unchanged."""

    query: str
    candidates: tuple[RerankedCandidate | Candidate, ...]
    degraded: bool = False
    degradation_reason: str | None = None

    @property
    def context_ids(self) -> frozenset[str]:
        """Returns:
            The distinct context_id values across all candidates in this call.
        """
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

    Args:
        voyage_client: Embedding client, passed through to retrieve() unchanged.
        collection: The MongoDB collection to search, passed through to retrieve() unchanged.
        cohere_client: Reranking client. Required (must not be None) whenever
            `reranker_enabled` is True and retrieve() returns any candidates;
            unused otherwise.
        query: The free-text search query - the only value the LLM ever
            supplies (see module docstring's NoSQL-injection invariant).
        pool_size: Candidate pool size, passed through to retrieve() unchanged.
        vector_weight: Vector-similarity weight, passed through to retrieve() unchanged.
        text_weight: Full-text weight, passed through to retrieve() unchanged.
        reranker_enabled: Whether to rerank retrieve()'s candidates via `cohere_client`.
        reranker_top_n: How many top candidates to keep, whether reranked
            or (reranker disabled or degraded) taken unreranked.
        embedding_model: Embedding model name, passed through to retrieve() unchanged.
        source_dataset: Restricts retrieval to this dataset, passed through
            to retrieve() unchanged.
        exclude_source_datasets: Datasets to exclude from retrieval, passed
            through to retrieve() unchanged.

    Raises:
        ValueError: if `query` is empty or whitespace-only - an agent
            calling its own tool with nothing to search for is a bug in
            the calling code (a malformed model tool-call should be
            rejected before it reaches here), not a case worth spending
            an API round-trip on.
        ValueError: if `reranker_enabled` is True, `cohere_client` is
            None, AND retrieve() actually returned candidates to rerank.
            This combination is a caller/config bug, not a runtime
            failure - pipeline.cli.build_clients() always supplies a real
            Cohere client whenever reranker_enabled is True in an actual
            run (COHERE_API_KEY is a hard requirement there), so this can
            only happen if a caller constructs search_documents() by hand
            with an inconsistent pair of arguments. Previously this fell
            through to the same branch used for reranker_enabled=False
            (see the `else` below) and returned a normal-looking,
            degraded=False result - silently serving unreranked candidates
            while claiming reranking was neither requested nor needed.
            That masked a real misconfiguration as an ordinary result
            instead of surfacing it. Found during external code review;
            see claude/status_agent_rezultaty_4_nahodki_kod.md, finding 2.
            Raising here (like the empty-query check above) matches this
            module's existing split: a caller/config bug raises
            immediately, while only genuine transient runtime failures
            (MongoDB/Cohere unavailable - see the two `except` blocks
            below) degrade gracefully instead of raising. Deliberately
            NOT checked when `candidates` is empty (nothing would be
            reranked either way in that case - see agent/loop.py's
            STOP_EMPTY_EVIDENCE path, which already handles this
            gracefully), so cohere_client=None stays a harmless default
            for callers/tests that never actually reach the reranking
            branch (e.g. an empty or transiently-failed retrieval).
    """
    if not query or not query.strip():
        raise ValueError("search_documents requires a non-empty query")

    try:
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
    except Exception as exc:
        if not is_transient_error(exc):
            raise
        # Graceful degradation, scenario 2 (module docstring): MongoDB/Voyage
        # unavailable after retrieve()'s own retry budget (voyage embed calls
        # only - retrieve()'s collection.aggregate() call has no retry of its
        # own) is exhausted. No candidates at all is a legitimate, gracefully
        # handled outcome - agent/loop.py's STOP_EMPTY_EVIDENCE path already
        # exists for exactly this shape of result.
        return SearchToolCall(
            query=query,
            candidates=(),
            degraded=True,
            degradation_reason=f"retrieval_unavailable: {type(exc).__name__}: {exc}",
        )

    if reranker_enabled and candidates:
        if cohere_client is None:
            raise ValueError(
                "search_documents was called with reranker_enabled=True and non-empty candidates, "
                "but cohere_client=None - either pass a real Cohere client or set reranker_enabled=False"
            )
        try:
            ranked = rerank(cohere_client, query, candidates, top_n=reranker_top_n)
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            # Graceful degradation, scenario 2: Cohere unavailable after
            # rerank()'s own @retryable() budget is exhausted. Falls back to
            # the unreranked top reranker_top_n candidates - the same
            # fallback already used when reranker_enabled is False - rather
            # than losing MongoDB's retrieval results entirely.
            return SearchToolCall(
                query=query,
                candidates=tuple(candidates[:reranker_top_n]),
                degraded=True,
                degradation_reason=f"reranker_unavailable: {type(exc).__name__}: {exc}",
            )
    else:
        ranked = candidates[:reranker_top_n]
    return SearchToolCall(query=query, candidates=tuple(ranked))

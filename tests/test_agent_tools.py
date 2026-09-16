"""Tests for agent/tools.py - the agent's one mandatory tool,
search_documents, and its NoSQL-injection invariant (itog_ekspertizy_agent_profil.md,
consensus item 5: closed by architecture, but pinned down by a test so a
future refactor cannot silently break it).

Fakes mirror tests/test_retrieval.py and tests/test_reranking.py exactly
(FakeVoyageClient/FakeCollection/FakeCohereClient) rather than importing
across test files, per this project's existing per-module test file
convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import agent.tools as agent_tools
from agent.tools import SEARCH_DOCUMENTS_TOOL_SPEC, search_documents


@dataclass
class _FakeEmbedResult:
    embeddings: list[list[float]]


class FakeVoyageClient:
    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    def embed(self, texts, model, input_type):
        self.calls.append((len(texts), input_type))
        return _FakeEmbedResult(embeddings=[[0.1] * 1024 for _ in texts])


class FakeCollection:
    def __init__(self, results: list[dict]):
        self.results = results
        self.last_pipeline: list[dict] | None = None

    def aggregate(self, pipeline):
        self.last_pipeline = pipeline
        return iter(self.results)


@dataclass
class _FakeRerankResult:
    index: int
    relevance_score: float


@dataclass
class _FakeRerankResponse:
    results: list[_FakeRerankResult]


class FakeCohereClient:
    def __init__(self, results: list[_FakeRerankResult]):
        self.results = results
        self.calls: list[dict] = []

    def rerank(self, model, query, documents, top_n, max_tokens_per_doc):
        self.calls.append({"query": query, "documents": documents, "top_n": top_n})
        return _FakeRerankResponse(results=self.results)


def _search_kwargs(**overrides):
    kwargs = dict(pool_size=50, vector_weight=0.5, text_weight=0.5, reranker_enabled=True, reranker_top_n=5)
    kwargs.update(overrides)
    return kwargs


def test_search_documents_tool_spec_only_exposes_query():
    # The NoSQL-invariant, on the model-facing side: no field in this
    # schema lets the model influence pool size, weights, or which
    # source_dataset(s) are searched.
    schema = SEARCH_DOCUMENTS_TOOL_SPEC["input_schema"]
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False


def test_search_documents_rejects_empty_query():
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    with pytest.raises(ValueError):
        search_documents(voyage, collection, None, "   ", **_search_kwargs())


def test_search_documents_rejects_reranker_enabled_with_no_cohere_client_when_candidates_exist():
    # находка 2 (claude/status_agent_rezultaty_4_nahodki_kod.md): this
    # combination used to fall through to the unreranked branch and
    # return a normal-looking, degraded=False result instead of surfacing
    # the misconfiguration. Raises now, once retrieve() has actually
    # returned candidates that would otherwise be silently served
    # unreranked.
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[{"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9}])
    with pytest.raises(ValueError):
        search_documents(voyage, collection, None, "q", **_search_kwargs(reranker_enabled=True))


def test_search_documents_reranker_enabled_with_no_cohere_client_is_harmless_when_no_candidates():
    # The counterpart to the test above: when retrieve() legitimately
    # returns nothing, cohere_client=None is never a bug regardless of
    # reranker_enabled - there is nothing to rerank either way, and this
    # is exactly the shape of call several Day 1/День 2 tests below rely
    # on (degraded defaults, empty retrieval, transient failures).
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    result = search_documents(voyage, collection, None, "q", **_search_kwargs(reranker_enabled=True))
    assert result.candidates == ()


def test_search_documents_passes_query_value_through_to_retrieve():
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])

    search_documents(voyage, collection, None, "what was net income in 2019?", **_search_kwargs(reranker_enabled=False))

    text_stage = collection.last_pipeline[0]["$rankFusion"]["input"]["pipelines"]["textPipeline"][0]["$search"]
    assert text_stage["text"]["query"] == "what was net income in 2019?"


def test_search_documents_never_sets_a_source_dataset_filter_unless_caller_passes_one():
    # The LLM's tool-call JSON only ever carries `query` (see the schema
    # test above) - this checks the other side of the invariant: calling
    # search_documents with only a query, as a real tool dispatcher would,
    # produces no source_dataset filter at all.
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    search_documents(voyage, collection, None, "revenue", **_search_kwargs(reranker_enabled=False))
    vector_stage = collection.last_pipeline[0]["$rankFusion"]["input"]["pipelines"]["vectorPipeline"][0]["$vectorSearch"]
    assert "filter" not in vector_stage


def test_search_documents_reranks_when_enabled():
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[{"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9}])
    cohere = FakeCohereClient(results=[_FakeRerankResult(index=0, relevance_score=0.7)])

    result = search_documents(voyage, collection, cohere, "q", **_search_kwargs(reranker_enabled=True, reranker_top_n=1))

    assert len(cohere.calls) == 1
    assert result.candidates[0].relevance_score == 0.7
    assert result.context_ids == frozenset({"ctx_1"})


def test_search_documents_skips_rerank_when_disabled():
    voyage = FakeVoyageClient()
    collection = FakeCollection(
        results=[
            {"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9},
            {"context_id": "ctx_2", "full_indexed_content": "doc 2", "score": 0.5},
        ]
    )
    cohere = FakeCohereClient(results=[])

    result = search_documents(voyage, collection, cohere, "q", **_search_kwargs(reranker_enabled=False, reranker_top_n=1))

    assert cohere.calls == []  # never called
    assert len(result.candidates) == 1  # still sliced to reranker_top_n


def test_search_documents_empty_retrieval_returns_empty_call():
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    result = search_documents(voyage, collection, None, "q", **_search_kwargs())
    assert result.candidates == ()
    assert result.context_ids == frozenset()


def test_search_documents_query_is_echoed_back_on_the_result():
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    result = search_documents(voyage, collection, None, "total assets 2021", **_search_kwargs(reranker_enabled=False))
    assert result.query == "total assets 2021"


def test_search_documents_default_degraded_is_false():
    # Day 1 call sites/tests never pass degraded=/degradation_reason= -
    # the День 2 fields must default to "not degraded" so nothing above
    # breaks.
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[])
    result = search_documents(voyage, collection, None, "q", **_search_kwargs())
    assert result.degraded is False
    assert result.degradation_reason is None


# --- День 2: graceful degradation on a transient MongoDB/Cohere failure --
#
# agent.tools.retrieve/agent.tools.rerank are monkeypatched directly here
# (rather than making FakeCollection.aggregate/FakeCohereClient.rerank
# raise) so these tests exercise only search_documents()'s own degrade-or-
# raise logic, not pipeline.reranking.rerank()'s real @retryable() retry
# loop, which would otherwise actually sleep through several exponential-
# backoff attempts (tenacity's wait_random_exponential(min=1, max=60)) for
# every test run - that retry timing is pipeline/reranking.py's/
# pipeline/retrieval.py's own concern and already outside this module's
# responsibility.


class _FakeConnectionError(Exception):
    """Class name alone (via the "Connection" fragment) is enough for
    pipeline.common.retry.is_transient_error to classify this as
    transient - see that module's _TRANSIENT_NAME_FRAGMENTS."""


def test_search_documents_degrades_to_empty_candidates_on_transient_mongodb_failure(monkeypatch):
    def _raise_transient(*args, **kwargs):
        raise _FakeConnectionError("connection reset")

    monkeypatch.setattr(agent_tools, "retrieve", _raise_transient)

    result = search_documents(FakeVoyageClient(), FakeCollection(results=[]), None, "q", **_search_kwargs())

    assert result.candidates == ()
    assert result.context_ids == frozenset()
    assert result.degraded is True
    assert "retrieval_unavailable" in result.degradation_reason
    assert "_FakeConnectionError" in result.degradation_reason


def test_search_documents_reraises_non_transient_mongodb_failure(monkeypatch):
    # A real bug (e.g. a malformed pipeline -> pymongo OperationFailure with
    # no retryable status code) must still surface, not be silently
    # swallowed as "no evidence found".
    def _raise_bug(*args, **kwargs):
        raise ValueError("not a transient error")

    monkeypatch.setattr(agent_tools, "retrieve", _raise_bug)

    with pytest.raises(ValueError):
        search_documents(FakeVoyageClient(), FakeCollection(results=[]), None, "q", **_search_kwargs())


def test_search_documents_degrades_to_unreranked_on_transient_cohere_failure(monkeypatch):
    voyage = FakeVoyageClient()
    collection = FakeCollection(
        results=[
            {"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9},
            {"context_id": "ctx_2", "full_indexed_content": "doc 2", "score": 0.5},
        ]
    )

    def _raise_transient(*args, **kwargs):
        raise _FakeConnectionError("service unavailable")

    monkeypatch.setattr(agent_tools, "rerank", _raise_transient)

    result = search_documents(
        voyage, collection, FakeCohereClient(results=[]), "q", **_search_kwargs(reranker_enabled=True, reranker_top_n=1)
    )

    # Falls back to the unreranked top reranker_top_n candidates (the same
    # slice used when reranker_enabled=False) instead of losing retrieval's
    # results entirely.
    assert len(result.candidates) == 1
    assert result.candidates[0].context_id == "ctx_1"
    assert result.degraded is True
    assert "reranker_unavailable" in result.degradation_reason


def test_search_documents_reraises_non_transient_cohere_failure(monkeypatch):
    voyage = FakeVoyageClient()
    collection = FakeCollection(results=[{"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9}])

    def _raise_bug(*args, **kwargs):
        raise ValueError("not a transient error")

    monkeypatch.setattr(agent_tools, "rerank", _raise_bug)

    with pytest.raises(ValueError):
        search_documents(voyage, collection, FakeCohereClient(results=[]), "q", **_search_kwargs(reranker_enabled=True))

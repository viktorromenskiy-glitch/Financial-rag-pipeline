"""Tests for module 7 (reranking). Cohere's rerank() is mocked via a fake
client - this verifies request construction and result mapping, not real
Cohere ranking quality. The real Recall@5 checkpoint (0.944) is validated
against the live API in Colab, not here."""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import pytest
 
from pipeline.reranking import MAX_TOKENS_PER_DOC, MODEL, TOP_N, rerank
from pipeline.retrieval import Candidate
 
 
@dataclass
class _FakeResult:
    index: int
    relevance_score: float
 
 
@dataclass
class _FakeRerankResponse:
    results: list[_FakeResult]
 
 
class FakeCohereClient:
    def __init__(self, results: list[_FakeResult]):
        self.results = results
        self.calls: list[dict] = []
 
    def rerank(self, model, query, documents, top_n, max_tokens_per_doc):
        self.calls.append(
            {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "max_tokens_per_doc": max_tokens_per_doc,
            }
        )
        return _FakeRerankResponse(results=self.results)
 
 
def _candidates(n: int) -> list[Candidate]:
    return [Candidate(f"ctx_{i}", f"full text of document {i}", 0.0) for i in range(n)]
 
 
def test_rerank_uses_configured_model_and_max_tokens():
    client = FakeCohereClient(results=[])
    rerank(client, "q", _candidates(3))
    assert client.calls[0]["model"] == MODEL
    assert client.calls[0]["max_tokens_per_doc"] == MAX_TOKENS_PER_DOC
 
 
def test_rerank_sends_full_text_untruncated():
    client = FakeCohereClient(results=[])
    candidates = _candidates(2)
    rerank(client, "q", candidates)
    assert client.calls[0]["documents"] == [c.full_indexed_content for c in candidates]
 
 
def test_rerank_maps_index_back_to_original_candidate():
    client = FakeCohereClient(
        results=[
            _FakeResult(index=2, relevance_score=0.9),
            _FakeResult(index=0, relevance_score=0.3),
        ]
    )
    candidates = _candidates(3)
    result = rerank(client, "q", candidates)
    assert result[0].context_id == "ctx_2"
    assert result[0].relevance_score == 0.9
    assert result[1].context_id == "ctx_0"
    assert result[1].relevance_score == 0.3
 
 
def test_rerank_empty_candidates_returns_empty():
    client = FakeCohereClient(results=[])
    assert rerank(client, "q", []) == []
 
 
def test_rerank_rejects_empty_document_text():
    client = FakeCohereClient(results=[])
    candidates = [Candidate("ctx_1", "", 0.0)]
    with pytest.raises(ValueError):
        rerank(client, "q", candidates)
 
 
def test_rerank_default_top_n():
    client = FakeCohereClient(results=[])
    rerank(client, "q", _candidates(10))
    assert client.calls[0]["top_n"] == TOP_N
 
 
def test_rerank_passes_through_custom_top_n_and_max_tokens():
    client = FakeCohereClient(results=[])
    rerank(client, "q", _candidates(5), top_n=3, max_tokens_per_doc=8000)
    assert client.calls[0]["top_n"] == 3
    assert client.calls[0]["max_tokens_per_doc"] == 8000

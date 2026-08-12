"""Tests for module 3 (embedding) against a fake Voyage client - no real
network calls (api.voyageai.com is unreachable from this session's sandbox)."""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import pytest
 
from pipeline.embedding import EMBEDDING_DIM, embed_documents, embed_query, embed_texts
 
 
@dataclass
class _FakeResult:
    embeddings: list[list[float]]
 
 
class FakeVoyageClient:
    """Returns a deterministic vector of fixed dimension - enough to verify
    batching and id<->vector wiring, not to verify embedding quality (this
    is not a unit test of the model)."""
 
    def __init__(self):
        self.calls: list[tuple[int, str]] = []  # (batch_size, input_type)
 
    def embed(self, texts, model, input_type):
        self.calls.append((len(texts), input_type))
        return _FakeResult(embeddings=[[float(len(t))] * EMBEDDING_DIM for t in texts])
 
 
def test_embed_texts_batches_correctly_at_boundary():
    """33 texts -> two batches (32 + 1)."""
    client = FakeVoyageClient()
    ids = [f"id_{i}" for i in range(33)]
    texts = [f"text {i}" for i in range(33)]
 
    vectors = embed_texts(client, ids, texts, input_type="document")
 
    assert len(vectors) == 33
    assert [c[0] for c in client.calls] == [32, 1]
    assert all(v.id == ids[i] for i, v in enumerate(vectors))
 
 
def test_embed_texts_empty_input():
    client = FakeVoyageClient()
    assert embed_texts(client, [], [], input_type="document") == []
    assert client.calls == []
 
 
def test_embed_texts_rejects_bad_input_type():
    client = FakeVoyageClient()
    with pytest.raises(ValueError):
        embed_texts(client, ["a"], ["text"], input_type="bogus")
 
 
def test_embed_texts_rejects_mismatched_lengths():
    client = FakeVoyageClient()
    with pytest.raises(ValueError):
        embed_texts(client, ["a", "b"], ["only one"], input_type="document")
 
 
def test_embed_texts_rejects_empty_text():
    """An empty full_indexed_content at the embedding input is a signal that
    enrichment/text assembly went wrong - must not pass silently."""
    client = FakeVoyageClient()
    with pytest.raises(ValueError):
        embed_texts(client, ["a"], [""], input_type="document")
 
 
def test_embed_documents_uses_document_input_type():
    client = FakeVoyageClient()
    vectors = embed_documents(client, [("ctx_1", "full indexed content here")])
    assert len(vectors) == 1
    assert client.calls[0][1] == "document"
 
 
def test_embed_documents_empty_input():
    client = FakeVoyageClient()
    assert embed_documents(client, []) == []
 
 
def test_embed_query_uses_query_input_type():
    client = FakeVoyageClient()
    vector = embed_query(client, "q_1", "what was revenue in 2019?")
    assert vector.id == "q_1"
    assert client.calls[0][1] == "query"
 
 
def test_embed_documents_signature_cannot_see_raw_content():
    """Guard from specifikatsiya_moduley.md: embed_documents takes ready-made
    (id, text) pairs, not a DocumentRecord - it structurally cannot embed
    raw_content instead of full_indexed_content by accident, because it never
    sees a DocumentRecord at all."""
    import inspect
 
    sig = inspect.signature(embed_documents)
    params = list(sig.parameters)
    assert params == ["client", "indexed_texts"]

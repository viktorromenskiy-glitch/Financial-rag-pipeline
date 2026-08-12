"""Тесты модуля 3 (embedding) на фейковом Voyage-клиенте — без реальных
сетевых вызовов (api.voyageai.com недоступен из песочницы этой сессии)."""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import pytest
 
from pipeline.embedding import EMBEDDING_DIM, embed_documents, embed_query, embed_texts
 
 
@dataclass
class _FakeResult:
    embeddings: list[list[float]]
 
 
class FakeVoyageClient:
    """Возвращает детерминированный вектор фиксированной размерности —
    достаточно для проверки батчинга/склейки id<->vector, не для проверки
    качества эмбеддингов (это не юнит-тест модели)."""
 
    def __init__(self):
        self.calls: list[tuple[int, str]] = []  # (batch_size, input_type)
 
    def embed(self, texts, model, input_type):
        self.calls.append((len(texts), input_type))
        return _FakeResult(embeddings=[[float(len(t))] * EMBEDDING_DIM for t in texts])
 
 
def test_embed_texts_batches_correctly_at_boundary():
    """32 текста -> один батч; 33 -> два батча (32 + 1)."""
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
    """Пустой full_indexed_content на входе эмбеддинга — сигнал, что
    enrichment/сборка текста отработали неправильно, не должно тихо пройти."""
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
    """Guard из specifikatsiya_moduley.md: embed_documents принимает готовые
    (id, text) пары, а не DocumentRecord — структурно не может «случайно»
    заэмбеддить raw_content вместо full_indexed_content, потому что не видит
    DocumentRecord вообще."""
    import inspect
 
    sig = inspect.signature(embed_documents)
    params = list(sig.parameters)
    assert params == ["client", "indexed_texts"]

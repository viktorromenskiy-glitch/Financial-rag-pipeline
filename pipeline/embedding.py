"""Module 3 - Embedding.
 
Embeds documents (already-assembled full_indexed_content) or queries via
Voyage AI voyage-4. See docs/specifikatsiya_moduley.md, module 3.
 
Call-order guard (external review note, recorded in
specifikatsiya_moduley.md): embed_documents() takes ready-made (id, text)
pairs as input - it does not know about DocumentRecord.raw_content and
cannot assemble full_indexed_content itself. This way, an incorrect call
order (embedding before enrichment) cannot accidentally slip through this
module.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol
 
from tenacity import retry, stop_after_attempt, wait_random_exponential
 
MODEL = "voyage-4"
BATCH_SIZE = 32
EMBEDDING_DIM = 1024
 
InputType = Literal["document", "query"]
 
 
@dataclass(frozen=True)
class EmbeddingVector:
    id: str
    vector: list[float]
 
 
class VoyageClientProtocol(Protocol):
    """Minimal interface required from voyageai.Client - allows a fake
    client to be substituted in tests without a real network call."""
 
    def embed(self, texts: list[str], model: str, input_type: str): ...
 
 
def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
 
 
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60))
def _embed_batch(client: VoyageClientProtocol, texts: list[str], input_type: InputType):
    return client.embed(texts, model=MODEL, input_type=input_type)
 
 
def embed_texts(
    client: VoyageClientProtocol,
    ids: list[str],
    texts: list[str],
    input_type: InputType,
) -> list[EmbeddingVector]:
    if input_type not in ("document", "query"):
        raise ValueError(f"input_type must be 'document' or 'query', got: {input_type!r}")
    if len(ids) != len(texts):
        raise ValueError(f"ids and texts have different lengths: {len(ids)} vs {len(texts)}")
    if any(not t for t in texts):
        raise ValueError("Empty text in embedding input - full_indexed_content was likely not assembled")
 
    vectors: list[EmbeddingVector] = []
    for id_batch, text_batch in zip(_batched(ids, BATCH_SIZE), _batched(texts, BATCH_SIZE)):
        result = _embed_batch(client, text_batch, input_type)
        if len(result.embeddings) != len(id_batch):
            raise RuntimeError(
                f"Voyage API returned {len(result.embeddings)} vectors for {len(id_batch)} texts in the batch"
            )
        vectors.extend(
            EmbeddingVector(id=id_, vector=vec) for id_, vec in zip(id_batch, result.embeddings)
        )
    return vectors
 
 
def embed_documents(
    client: VoyageClientProtocol, indexed_texts: list[tuple[str, str]]
) -> list[EmbeddingVector]:
    """indexed_texts: list of (context_id, full_indexed_content) - already
    assembled after module 4, not raw_content. The signature intentionally
    does not accept a full DocumentRecord, so raw text cannot be passed by
    accident.
    """
    if not indexed_texts:
        return []
    ids = [t[0] for t in indexed_texts]
    texts = [t[1] for t in indexed_texts]
    return embed_texts(client, ids, texts, input_type="document")
 
 
def embed_query(client: VoyageClientProtocol, question_id: str, query_text: str) -> EmbeddingVector:
    return embed_texts(client, [question_id], [query_text], input_type="query")[0]

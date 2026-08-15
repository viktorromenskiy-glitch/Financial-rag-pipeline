"""Module 3 - Embedding.

Embeds documents (already-assembled full_indexed_content) or queries via
Voyage AI. See docs/specifikatsiya_moduley.md, module 3.

Call-order guard (external review note, recorded in
specifikatsiya_moduley.md): embed_documents() takes ready-made (id, text)
pairs as input - it does not know about DocumentRecord.raw_content and
cannot assemble full_indexed_content itself. This way, an incorrect call
order (embedding before enrichment) cannot accidentally slip through this
module.

Per-dataset embedding routing (docs/tehnicheskoe_zadanie.md, п.3a,
2026-08-15): a direct A/B test on the full 7318-document corpus (three
runs of increasing scale, final n=2500/source_dataset, McNemar's exact
test) found voyage-finance-2 measurably helps TAT-DQA retrieval
(Recall@5 +1.9pp, p=0.0005) but measurably hurts ConvFinQA (-1.9pp,
p=0.0000) and gives no reliable benefit on FinQA (-0.9pp, p=0.0214, not
robust to correction) - contradicting the uniform "+3-8pp everywhere"
prediction from three independent AI-consultant reviews (Gemini/Kimi/Grok)
that was never itself measured, only extrapolated from Voyage AI's own
retrieval-only NDCG@10 benchmark. A uniform model swap was rejected in
favor of per-source_dataset routing (voyage-finance-2 for TAT-DQA only,
voyage-4 for ConvFinQA/FinQA) - see resolve_embedding_model() below.
This module stays deliberately unaware of *why* a given source_dataset
routes to a given model (that policy lives in config/config.yaml,
embedding.routing) - it only resolves a source_dataset string to a model
string given the routing parameters, and embeds a batch with whatever
single model it's told to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from pipeline.common.retry import retryable

MODEL = "voyage-4"
FINANCE_MODEL = "voyage-finance-2"
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


def resolve_embedding_model(
    source_dataset: str,
    routing_enabled: bool,
    finance_model: str,
    routed_sources: frozenset[str] | set[str],
    default_model: str = MODEL,
) -> str:
    """Per-dataset embedding routing lookup (see module docstring for the
    measured justification). Pure lookup, no I/O - deliberately takes
    primitives rather than the whole PipelineConfig object so it stays
    testable without constructing a full config, and so this module keeps
    no dependency on config/config_schema.py.

    routing_enabled=False (or an empty routed_sources) always returns
    default_model - the routing feature is a config-level on/off switch,
    matching the project's "everything switchable via config, not
    hardcoded" convention (docs/struktura_repozitoriya.md).
    """
    if routing_enabled and source_dataset in routed_sources:
        return finance_model
    return default_model


@retryable()
def _embed_batch(client: VoyageClientProtocol, texts: list[str], input_type: InputType, model: str):
    return client.embed(texts, model=model, input_type=input_type)


def embed_texts(
    client: VoyageClientProtocol,
    ids: list[str],
    texts: list[str],
    input_type: InputType,
    model: str = MODEL,
) -> list[EmbeddingVector]:
    if input_type not in ("document", "query"):
        raise ValueError(f"input_type must be 'document' or 'query', got: {input_type!r}")
    if len(ids) != len(texts):
        raise ValueError(f"ids and texts have different lengths: {len(ids)} vs {len(texts)}")
    if any(not t for t in texts):
        raise ValueError("Empty text in embedding input - full_indexed_content was likely not assembled")

    vectors: list[EmbeddingVector] = []
    for id_batch, text_batch in zip(_batched(ids, BATCH_SIZE), _batched(texts, BATCH_SIZE)):
        result = _embed_batch(client, text_batch, input_type, model)
        if len(result.embeddings) != len(id_batch):
            raise RuntimeError(
                f"Voyage API returned {len(result.embeddings)} vectors for {len(id_batch)} texts in the batch"
            )
        for id_, vec in zip(id_batch, result.embeddings):
            if len(vec) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Voyage API returned a {len(vec)}-dim vector for id={id_!r}, "
                    f"expected {EMBEDDING_DIM} (model={model!r} may have changed)"
                )
            vectors.append(EmbeddingVector(id=id_, vector=vec))
    return vectors


def embed_documents(
    client: VoyageClientProtocol, indexed_texts: list[tuple[str, str]], model: str = MODEL
) -> list[EmbeddingVector]:
    """indexed_texts: list of (context_id, full_indexed_content) - already
    assembled after module 4, not raw_content. The signature intentionally
    does not accept a full DocumentRecord, so raw text cannot be passed by
    accident.

    `model` applies to the WHOLE batch passed in one call - Voyage's API
    takes one model per request, so a caller that needs per-document
    routing (see resolve_embedding_model()) must group documents by
    resolved model BEFORE calling this (one call per model group), not
    pass a mixed-model batch through a single call. cli.py's cmd_index
    does this grouping; this function stays unaware of source_dataset.
    """
    if not indexed_texts:
        return []
    ids = [t[0] for t in indexed_texts]
    texts = [t[1] for t in indexed_texts]
    return embed_texts(client, ids, texts, input_type="document", model=model)


def embed_query(
    client: VoyageClientProtocol, question_id: str, query_text: str, model: str = MODEL
) -> EmbeddingVector:
    return embed_texts(client, [question_id], [query_text], input_type="query", model=model)[0]

# tests/test_retrieval.py
"""Tests for module 6 (hybrid retrieval). $rankFusion is an Atlas-only
aggregation stage that mongomock does not implement, so a fake collection
records the pipeline it was called with and returns canned results - this
verifies pipeline construction and result parsing, not real Atlas ranking
behavior. The real Recall@5 checkpoint (0.808) is validated against a live
cluster in Colab, not here."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline.indexing import TEXT_INDEX_NAME, VECTOR_INDEX_NAME
from pipeline.retrieval import (
    TEXT_PIPELINE_NAME,
    VECTOR_PIPELINE_NAME,
    Candidate,
    build_rank_fusion_pipeline,
    retrieve,
)


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


def test_build_rank_fusion_pipeline_uses_configured_index_names():
    pipeline = build_rank_fusion_pipeline([0.0] * 1024, "revenue growth", pool_size=50)
    rank_fusion_stage = pipeline[0]["$rankFusion"]
    pipelines = rank_fusion_stage["input"]["pipelines"]

    vector_stage = pipelines[VECTOR_PIPELINE_NAME][0]["$vectorSearch"]
    text_stage = pipelines[TEXT_PIPELINE_NAME][0]["$search"]

    assert vector_stage["index"] == VECTOR_INDEX_NAME
    assert vector_stage["path"] == "embedding_voyage"
    assert text_stage["index"] == TEXT_INDEX_NAME
    assert text_stage["text"]["path"] == "full_indexed_content"


def test_build_rank_fusion_pipeline_applies_default_weights():
    pipeline = build_rank_fusion_pipeline([0.0] * 1024, "q", pool_size=50)
    weights = pipeline[0]["$rankFusion"]["combination"]["weights"]
    assert weights[VECTOR_PIPELINE_NAME] == 0.5
    assert weights[TEXT_PIPELINE_NAME] == 0.5


def test_build_rank_fusion_pipeline_num_candidates_exceeds_pool_size():
    pipeline = build_rank_fusion_pipeline([0.0] * 1024, "q", pool_size=50)
    vector_stage = pipeline[0]["$rankFusion"]["input"]["pipelines"][VECTOR_PIPELINE_NAME][0]["$vectorSearch"]
    assert vector_stage["numCandidates"] > vector_stage["limit"]
    assert vector_stage["limit"] == 50


def test_build_rank_fusion_pipeline_rejects_non_positive_pool_size():
    with pytest.raises(ValueError):
        build_rank_fusion_pipeline([0.0] * 1024, "q", pool_size=0)


def test_retrieve_embeds_query_with_query_input_type():
    voyage_client = FakeVoyageClient()
    collection = FakeCollection(results=[])

    retrieve(voyage_client, collection, "what was net income in 2019?")

    assert voyage_client.calls == [(1, "query")]


def test_retrieve_parses_candidates_in_order():
    voyage_client = FakeVoyageClient()
    collection = FakeCollection(
        results=[
            {"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9},
            {"context_id": "ctx_2", "full_indexed_content": "doc 2", "score": 0.5},
        ]
    )

    candidates = retrieve(voyage_client, collection, "q")

    assert candidates == [
        Candidate("ctx_1", "doc 1", 0.9),
        Candidate("ctx_2", "doc 2", 0.5),
    ]


def test_retrieve_empty_results():
    voyage_client = FakeVoyageClient()
    collection = FakeCollection(results=[])
    assert retrieve(voyage_client, collection, "q") == []


def test_retrieve_raises_on_unexpected_duplicate_context_id():
    voyage_client = FakeVoyageClient()
    collection = FakeCollection(
        results=[
            {"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.9},
            {"context_id": "ctx_1", "full_indexed_content": "doc 1", "score": 0.4},
        ]
    )
    with pytest.raises(AssertionError):
        retrieve(voyage_client, collection, "q")


def test_retrieve_passes_pool_size_and_weights_through_to_pipeline():
    voyage_client = FakeVoyageClient()
    collection = FakeCollection(results=[])

    retrieve(voyage_client, collection, "q", pool_size=10, vector_weight=0.7, text_weight=0.3)

    weights = collection.last_pipeline[0]["$rankFusion"]["combination"]["weights"]
    assert weights[VECTOR_PIPELINE_NAME] == 0.7
    assert weights[TEXT_PIPELINE_NAME] == 0.3
    vector_stage = collection.last_pipeline[0]["$rankFusion"]["input"]["pipelines"][VECTOR_PIPELINE_NAME][0]["$vectorSearch"]
    assert vector_stage["limit"] == 10

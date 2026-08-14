"""Startup validation test - see docs/tehnicheskoe_zadanie.md, section 2.

MongoDB's $vectorSearch/$search stages do not raise an error for a
mismatched or missing index name; they silently return an empty result,
which has previously been mistaken in this project for "the reranker
broke everything" rather than "the index name in the config is wrong".
validate_startup_indexes() (pipeline/indexing.py) is the guard against
that - it must run once at pipeline startup, before any real work, and
fail loudly (AssertionError) if either index returns nothing on a cheap
probe query. These tests exercise that guard against a fake collection,
without a real Atlas cluster.
"""

import pytest

from pipeline.indexing import TEXT_INDEX_NAME, VECTOR_INDEX_NAME, validate_startup_indexes


class FakeCollection:
    """Minimal in-memory stand-in for pymongo.Collection, only implementing
    the aggregate() shapes validate_startup_indexes() actually issues."""

    def __init__(self, has_vector_data: bool, has_text_data: bool):
        self.has_vector_data = has_vector_data
        self.has_text_data = has_text_data
        self.received_pipelines: list[list[dict]] = []

    def aggregate(self, pipeline: list[dict]):
        self.received_pipelines.append(pipeline)
        stage = pipeline[0]
        if "$vectorSearch" in stage:
            return [{"context_id": "probe"}] if self.has_vector_data else []
        if "$search" in stage:
            return [{"context_id": "probe"}] if self.has_text_data else []
        return []


def test_passes_when_both_indexes_return_results():
    collection = FakeCollection(has_vector_data=True, has_text_data=True)
    validate_startup_indexes(collection)  # must not raise


def test_fails_when_vector_index_is_empty():
    # Simulates a mismatched/missing vector_index_full name - MongoDB
    # returns an empty result instead of an error.
    collection = FakeCollection(has_vector_data=False, has_text_data=True)
    with pytest.raises(AssertionError, match=VECTOR_INDEX_NAME):
        validate_startup_indexes(collection)


def test_fails_when_text_index_is_empty():
    # Same silent-empty-result failure mode for the full-text index.
    collection = FakeCollection(has_vector_data=True, has_text_data=False)
    with pytest.raises(AssertionError, match=TEXT_INDEX_NAME):
        validate_startup_indexes(collection)


def test_fails_when_both_indexes_are_empty():
    collection = FakeCollection(has_vector_data=False, has_text_data=False)
    with pytest.raises(AssertionError):
        validate_startup_indexes(collection)


def test_probe_query_uses_configured_index_names():
    # Regression guard for the exact bug described in ТЗ §2: the probe
    # must reference the actual configured index name constants, not a
    # hardcoded/stale string that could silently drift from config.yaml.
    collection = FakeCollection(has_vector_data=True, has_text_data=True)
    validate_startup_indexes(collection)
    vector_stage = collection.received_pipelines[0][0]["$vectorSearch"]
    text_stage = collection.received_pipelines[1][0]["$search"]
    assert vector_stage["index"] == VECTOR_INDEX_NAME
    assert text_stage["index"] == TEXT_INDEX_NAME


def test_vector_probe_uses_a_non_zero_vector():
    # A zero vector is invalid for $vectorSearch (undefined cosine
    # similarity) - the probe must use a non-zero vector or Atlas raises
    # OperationFailure instead of returning a clean empty/non-empty result.
    collection = FakeCollection(has_vector_data=True, has_text_data=True)
    validate_startup_indexes(collection)
    vector_stage = collection.received_pipelines[0][0]["$vectorSearch"]
    assert any(v != 0 for v in vector_stage["queryVector"])

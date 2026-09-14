"""End-to-end offline smoke test for the agent/ package.

Wires the REAL agent.tools.search_documents (not a search_fn fake) to the
REAL run_agent_query loop, using fake Voyage/MongoDB/Cohere/generator
clients throughout - no network access and no real API keys anywhere in
this file. This is the Day 1 checklist's "offline/mocked smoke test": it
exists so CI can exercise the whole agent/ package end-to-end without
MongoDB/Cohere/Anthropic credentials, and so a future change to how
tools.py and loop.py are wired together (not just each piece in
isolation) has a test that would actually catch a broken integration.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from agent.loop import EvidenceAssessment, run_agent_query
from agent.tools import search_documents


@dataclass
class _FakeEmbedResult:
    embeddings: list[list[float]]


class FakeVoyageClient:
    def embed(self, texts, model, input_type):
        return _FakeEmbedResult(embeddings=[[0.1] * 1024 for _ in texts])


class FakeCollection:
    """Returns a fixed document for the first query, a different one for
    any reformulated query - just enough to exercise a real reformulate-
    and-merge cycle through the real search_documents wrapper."""

    def __init__(self, docs_by_query: dict[str, list[dict]]):
        self.docs_by_query = docs_by_query
        self.pipelines_seen: list[list[dict]] = []

    def aggregate(self, pipeline):
        self.pipelines_seen.append(pipeline)
        text_query = pipeline[0]["$rankFusion"]["input"]["pipelines"]["textPipeline"][0]["$search"]["text"]["query"]
        return iter(self.docs_by_query.get(text_query, []))


class FakeAssessor:
    def __init__(self, assessments):
        self._assessments = list(assessments)

    def assess(self, question, context_text, calls_remaining):
        return self._assessments.pop(0)


class FakeGenerator:
    def generate(self, prompt: str) -> str:
        return "FINAL ANSWER: 123"


def test_agent_smoke_end_to_end_with_reformulated_query_no_network():
    collection = FakeCollection(
        {
            "what was net income?": [{"context_id": "ctx_a", "full_indexed_content": "doc a", "score": 0.9}],
            "net income for fiscal year 2020": [
                {"context_id": "ctx_b", "full_indexed_content": "doc b", "score": 0.8}
            ],
        }
    )
    voyage = FakeVoyageClient()
    search_fn = functools.partial(
        search_documents,
        voyage,
        collection,
        None,  # no Cohere client - reranker disabled below, matching agent V1's simplest config
        pool_size=50,
        vector_weight=0.5,
        text_weight=0.5,
        reranker_enabled=False,
        reranker_top_n=5,
    )
    assessor = FakeAssessor(
        [
            EvidenceAssessment(sufficient=False, reformulated_query="net income for fiscal year 2020"),
            EvidenceAssessment(sufficient=True),
        ]
    )
    generator = FakeGenerator()

    result = run_agent_query(
        "smoke_q1",
        "what was net income?",
        search_fn,
        assessor,
        generator,
        max_additional_tool_calls=2,
    )

    assert result.answer_text == "123"
    assert result.additional_calls_used == 1
    assert set(result.context_ids) == {"ctx_a", "ctx_b"}
    assert len(collection.pipelines_seen) == 2  # exactly two real retrieve() calls, no more

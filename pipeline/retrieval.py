# pipeline/retrieval.py
"""Module 6 - Hybrid retrieval.

Combines a MongoDB Atlas $vectorSearch pipeline (over embedding_voyage) and
a $search full-text pipeline (over full_indexed_content) via $rankFusion,
returning the top pool_size candidates ranked by the combined RRF score.
See docs/specifikatsiya_moduley.md, module 6.

$rankFusion syntax reference: https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankfusion/
De-duplication across input pipelines is handled natively by $rankFusion
(confirmed in the official documentation) - the assertion below is a cheap
safety net, not a required deduplication step (see spec, module 6,
"Проверка").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.embedding import VoyageClientProtocol, embed_query
from pipeline.indexing import TEXT_INDEX_NAME, VECTOR_INDEX_NAME

POOL_SIZE = 50
VECTOR_WEIGHT = 0.5
TEXT_WEIGHT = 0.5
VECTOR_PIPELINE_NAME = "vectorPipeline"
TEXT_PIPELINE_NAME = "textPipeline"

NUM_CANDIDATES_MULTIPLIER = 20


@dataclass(frozen=True)
class Candidate:
    context_id: str
    full_indexed_content: str
    score: float


class CollectionProtocol(Protocol):
    def aggregate(self, pipeline): ...


def build_rank_fusion_pipeline(
    query_vector: list[float],
    query_text: str,
    pool_size: int = POOL_SIZE,
    vector_weight: float = VECTOR_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
) -> list[dict]:
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")

    num_candidates = pool_size * NUM_CANDIDATES_MULTIPLIER

    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        VECTOR_PIPELINE_NAME: [
                            {
                                "$vectorSearch": {
                                    "index": VECTOR_INDEX_NAME,
                                    "path": "embedding_voyage",
                                    "queryVector": query_vector,
                                    "numCandidates": num_candidates,
                                    "limit": pool_size,
                                }
                            }
                        ],
                        TEXT_PIPELINE_NAME: [
                            {
                                "$search": {
                                    "index": TEXT_INDEX_NAME,
                                    "text": {
                                        "query": query_text,
                                        "path": "full_indexed_content",
                                    },
                                }
                            },
                            {"$limit": pool_size},
                        ],
                    }
                },
                "combination": {
                    "weights": {
                        VECTOR_PIPELINE_NAME: vector_weight,
                        TEXT_PIPELINE_NAME: text_weight,
                    }
                },
            }
        },
        {"$limit": pool_size},
        {
            "$project": {
                "_id": 0,
                "context_id": 1,
                "full_indexed_content": 1,
                "score": {"$meta": "score"},
            }
        },
    ]


def retrieve(
    voyage_client: VoyageClientProtocol,
    collection: CollectionProtocol,
    question_text: str,
    pool_size: int = POOL_SIZE,
    vector_weight: float = VECTOR_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
) -> list[Candidate]:
    query_vector = embed_query(voyage_client, question_id="__query__", query_text=question_text).vector

    pipeline = build_rank_fusion_pipeline(
        query_vector, question_text, pool_size, vector_weight, text_weight
    )
    results = list(collection.aggregate(pipeline))

    seen: set[str] = set()
    candidates: list[Candidate] = []
    for r in results:
        context_id = r["context_id"]
        if context_id in seen:
            raise AssertionError(
                f"Duplicate context_id={context_id!r} in $rankFusion results - "
                f"expected native de-duplication, see module docstring"
            )
        seen.add(context_id)
        candidates.append(
            Candidate(
                context_id=context_id,
                full_indexed_content=r["full_indexed_content"],
                score=r.get("score", 0.0),
            )
        )
    return candidates

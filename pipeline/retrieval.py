""Module 6 - Hybrid retrieval.

Combines a MongoDB Atlas $vectorSearch pipeline (over embedding_voyage) and
a $search full-text pipeline (over full_indexed_content) via $rankFusion,
returning the top pool_size candidates ranked by the combined RRF score.
See docs/specifikatsiya_moduley.md, module 6.

$rankFusion syntax reference: https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankfusion/
De-duplication across input pipelines is handled natively by $rankFusion
(confirmed in the official documentation) - the assertion below is a cheap
safety net, not a required deduplication step (see spec, module 6,
"Проверка").

Per-dataset embedding routing (docs/tehnicheskoe_zadanie.md, п.3a,
2026-08-15): documents route to voyage-finance-2 or voyage-4 depending on
their own source_dataset (see pipeline.embedding.resolve_embedding_model).
Because vectors from different embedding models are not comparable, a
routed query (e.g. TAT-DQA) MUST be filtered to exactly its own
source_dataset (source_dataset=... below) - source_dataset is
source-disjoint by construction (a TAT-DQA question's gold document is
always a TAT-DQA document, verified in pipeline.ingestion), so this
filter costs no recall for a routed query.

Bug fixed 2026-08-15, same day this shipped (caught on the first real
250-question eval run under routing: ConvFinQA judge accuracy dropped
5.4pp - 0.865 to 0.811 - even though ConvFinQA's embedding model never
changed): an UNROUTED query (ConvFinQA/FinQA, still voyage-4) must NOT be
filtered down to exactly its own source_dataset - that silently shrinks
its candidate pool from the full ~4,600-document unrouted corpus to just
its own ~1,800-2,800-document slice, changing which candidates compete
during rerank/generation in a way that was never measured or validated,
only the isolated embedding-model A/B was. An unrouted query only needs
the routed sources (TAT-DQA) EXCLUDED from its pool - excluding them is
required (their vectors are now in a different, incompatible space), but
everything else that was always in its candidate pool (all other unrouted
sources together) must stay in it. Use exclude_source_datasets=[...] for
this case, source_dataset=... only for an actually-routed query - see
build_rank_fusion_pipeline()'s docstring for the two mutually-exclusive
filter modes this produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.embedding import MODEL, VoyageClientProtocol, embed_query
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
    source_dataset: str | None = None,
    exclude_source_datasets: list[str] | None = None,
) -> list[dict]:
    """Two mutually-exclusive Atlas-native pre-filter modes, both applied to
    BOTH sub-pipelines (not a $match appended after $limit - that would
    still let cross-model-embedded documents crowd out the real candidates
    before the limit is even applied, see module docstring):

    - source_dataset: restricts candidates to exactly this one
      source_dataset. Use for a ROUTED query (e.g. TAT-DQA under
      voyage-finance-2) - it must only ever be compared against its own
      source's documents, which is also the correct/intended candidate
      pool for it (source_dataset is source-disjoint by construction).

    - exclude_source_datasets: restricts candidates to everything EXCEPT
      the given source_dataset(s). Use for an UNROUTED query (e.g.
      ConvFinQA/FinQA under voyage-4) - it must avoid the now-incompatible
      routed source(s) (different embedding model = different vector
      space) but should otherwise keep searching across all other
      unrouted sources together, exactly as before routing existed. Fixes
      a 2026-08-15 bug where unrouted queries were wrongly restricted to
      source_dataset= their own single source instead, see module
      docstring.

    Passing both (or neither with routing enabled) is a caller error -
    exactly one should be set whenever routing.enabled is true.

    Requires source_dataset to be declared as a "filter"-type field in
    both the vector_index_full and text_index_full Atlas index
    definitions (docs/tehnicheskoe_zadanie.md, п.3a) -
    pipeline.indexing.validate_startup_indexes() checks this at startup
    and fails loudly if it's missing, rather than silently returning
    zero/wrong candidates.
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if source_dataset is not None and exclude_source_datasets:
        raise ValueError(
            "source_dataset and exclude_source_datasets are mutually exclusive - "
            f"got source_dataset={source_dataset!r} and exclude_source_datasets={exclude_source_datasets!r}"
        )

    num_candidates = pool_size * NUM_CANDIDATES_MULTIPLIER

    vector_search_stage: dict = {
        "$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "path": "embedding_voyage",
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": pool_size,
        }
    }
    text_search_stage: dict = {
        "$search": {
            "index": TEXT_INDEX_NAME,
            "text": {
                "query": query_text,
                "path": "full_indexed_content",
            },
        }
    }

    if source_dataset is not None:
        # $vectorSearch's own `filter` param pre-filters candidates BEFORE
        # the ANN search picks numCandidates/limit results - a $match
        # stage appended afterward would filter an already-truncated,
        # potentially wrong-model-dominated pool instead.
        vector_search_stage["$vectorSearch"]["filter"] = {"source_dataset": {"$eq": source_dataset}}
        # $search needs the "compound" form to combine a text query with a
        # filter clause - a bare "text" operator has no filter option.
        text_search_stage["$search"] = {
            "index": TEXT_INDEX_NAME,
            "compound": {
                "must": [{"text": {"query": query_text, "path": "full_indexed_content"}}],
                "filter": [{"equals": {"path": "source_dataset", "value": source_dataset}}],
            },
        }
    elif exclude_source_datasets:
        vector_search_stage["$vectorSearch"]["filter"] = {
            "source_dataset": {"$nin": list(exclude_source_datasets)}
        }
        text_search_stage["$search"] = {
            "index": TEXT_INDEX_NAME,
            "compound": {
                "must": [{"text": {"query": query_text, "path": "full_indexed_content"}}],
                "mustNot": [{"in": {"path": "source_dataset", "value": list(exclude_source_datasets)}}],
            },
        }

    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        VECTOR_PIPELINE_NAME: [vector_search_stage],
                        TEXT_PIPELINE_NAME: [text_search_stage, {"$limit": pool_size}],
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
    source_dataset: str | None = None,
    exclude_source_datasets: list[str] | None = None,
    embedding_model: str = MODEL,
) -> list[Candidate]:
    """embedding_model and source_dataset must be passed together and must
    agree: embedding_model is whatever pipeline.embedding.
    resolve_embedding_model(source_dataset, ...) returned for this
    question's source_dataset (cli.py's cmd_eval does this) - ONLY for a
    ROUTED question. Passing source_dataset without the matching
    embedding_model (or vice versa) would embed the query with one model
    while filtering for documents embedded with a different one, silently
    returning zero or near-random candidates - there is no cheap way for
    this function to detect that mismatch on its own, so getting the
    caller right matters.

    For an UNROUTED question, pass exclude_source_datasets=<the routed
    sources> instead of source_dataset, with embedding_model left at the
    default (unrouted questions always use the default model) - see
    build_rank_fusion_pipeline()'s docstring for why these two filter
    modes are not interchangeable.
    """
    query_vector = embed_query(
        voyage_client, question_id="__query__", query_text=question_text, model=embedding_model
    ).vector

    pipeline = build_rank_fusion_pipeline(
        query_vector,
        question_text,
        pool_size,
        vector_weight,
        text_weight,
        source_dataset,
        exclude_source_datasets,
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

"""CUAD smoke-test fixture loader - Day 3 of
plan_rabot_posle_ekspertizy_agent_profil.md:

  "Смок-тест на CUAD (Contract Understanding Atticus Dataset, CC BY 4.0):
  3-5 вопросов, без изменения кода агента - только конфигурация
  retrieval-индекса и индексация новых документов."

data/cuad_smoke/cuad_smoke_questions.json holds 4 real CUAD contracts and
5 real clause-extraction questions with their real gold answers (see that
file's "_license_notice" key for the CC BY 4.0 attribution) - picked for
short context length (keeps the one-off indexing/embedding cost small)
and clean single-span gold answers, across 4 different documents and 4
different CUAD clause categories (Document Name, Agreement Date,
Effective Date, Governing Law).

This module's only job is turning that fixture into the exact shapes the
EXISTING pipeline code already accepts, so indexing CUAD documents needs
no new indexing/embedding/retrieval code at all:

  - list[pipeline.ingestion.DocumentRecord] - feed straight into
    pipeline.ingestion.dedupe_documents() -> pipeline.embedding.embed_documents()
    -> pipeline.indexing.index_corpus(), exactly like scripts/run_eval.py's
    T2-RAGBench cmd_index path (see pipeline/cli.py).
  - list[dict] eval items, shaped exactly like pipeline.cli.load_eval_questions()'s
    return value ({"question_id", "question", "gold_answer", "source_dataset"}),
    so scripts/run_cuad_smoke.py can drive baseline/agent exactly like
    scripts/run_agent_eval.py does over its T2-RAGBench sample.

CUAD documents get no metadata_prefix (company_name/report_year/company_sector
are all None) - that fix was specific to closing a lexical gap in
T2-RAGBench's own question-reformulation style (see pipeline/ingestion.py's
docstring); CUAD's clause-extraction questions have no equivalent gap to
close, so there is nothing to derive a prefix from here.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.ingestion import DocumentRecord

SOURCE_DATASET = "CUAD"

DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad_smoke" / "cuad_smoke_questions.json"


def load_cuad_smoke_fixture(path: str | Path = DEFAULT_FIXTURE_PATH) -> tuple[list[DocumentRecord], list[dict]]:
    """Loads and validates the CUAD smoke fixture.

    Returns:
        (document_records, eval_items) - see module docstring for both
        shapes. document_records has one entry per QUESTION (not per
        document) - matching pipeline.ingestion's "one record per
        question, context_id repeats across questions sharing a
        document" convention - so a document with more than one question
        (see cuad_smoke_1/cuad_smoke_2, both against the same web-hosting
        contract) naturally produces one DocumentRecord per question, all
        sharing that document's context_id, exactly as
        pipeline.ingestion.dedupe_documents() expects.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if the fixture JSON is missing "documents" or
            "questions", or a question references a document_id not
            present in "documents" - a fixture-authoring bug, not
            something a caller should have to debug via a KeyError deep
            inside pipeline.ingestion.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CUAD smoke fixture not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    for key in ("documents", "questions"):
        if key not in raw:
            raise ValueError(f"CUAD smoke fixture is missing required key {key!r}: {path}")

    documents_by_id = {doc["document_id"]: doc for doc in raw["documents"]}

    records: list[DocumentRecord] = []
    eval_items: list[dict] = []
    for q in raw["questions"]:
        doc_id = q["document_id"]
        if doc_id not in documents_by_id:
            raise ValueError(
                f"CUAD smoke fixture question {q['question_id']!r} references unknown "
                f"document_id {doc_id!r}: {path}"
            )
        doc = documents_by_id[doc_id]
        records.append(
            DocumentRecord(
                context_id=doc_id,
                context=doc["text"],
                source_dataset=SOURCE_DATASET,
                question=q["question"],
                answer=q["gold_answer"],
                company_name=None,
                report_year=None,
                company_sector=None,
                metadata_prefix="",
            )
        )
        eval_items.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "source_dataset": SOURCE_DATASET,
            }
        )
    return records, eval_items

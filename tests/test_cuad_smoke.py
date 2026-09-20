"""Tests for pipeline/cuad_smoke.py - the Day 3 CUAD smoke-test fixture
loader. Runs entirely offline: no MongoDB/Voyage/Cohere/Anthropic client
involved, and no network access to the real CUAD repository - this only
exercises the already-downloaded, already-committed fixture file (or a
hand-built temp fixture, for the error-path tests).
"""

from __future__ import annotations

import json

import pytest

from pipeline.cuad_smoke import DEFAULT_FIXTURE_PATH, SOURCE_DATASET, load_cuad_smoke_fixture
from pipeline.ingestion import DocumentRecord


def test_default_fixture_file_exists_and_loads():
    records, eval_items = load_cuad_smoke_fixture()
    assert len(records) == len(eval_items)
    assert 3 <= len(eval_items) <= 5  # plan: "3-5 questions"
    assert all(isinstance(r, DocumentRecord) for r in records)


def test_default_fixture_every_record_is_source_dataset_cuad():
    records, eval_items = load_cuad_smoke_fixture()
    assert all(r.source_dataset == SOURCE_DATASET for r in records)
    assert all(item["source_dataset"] == SOURCE_DATASET for item in eval_items)


def test_default_fixture_eval_items_shape_matches_load_eval_questions():
    # Same keys pipeline.cli.load_eval_questions() returns, so
    # scripts/run_cuad_smoke.py can reuse scripts/run_agent_eval.py's loop
    # body unchanged.
    _, eval_items = load_cuad_smoke_fixture()
    for item in eval_items:
        assert set(item.keys()) == {"question_id", "question", "gold_answer", "source_dataset"}
        assert item["question_id"]
        assert item["question"]
        assert item["gold_answer"]


def test_default_fixture_question_ids_are_unique():
    _, eval_items = load_cuad_smoke_fixture()
    ids = [item["question_id"] for item in eval_items]
    assert len(ids) == len(set(ids))


def test_default_fixture_gold_answers_are_findable_spans_in_their_document():
    """Each gold answer should actually appear verbatim in the document it
    was extracted from - a basic sanity check that the fixture wasn't
    mistyped while being hand-curated from the real CUAD JSON."""
    records, _ = load_cuad_smoke_fixture()
    for record in records:
        assert record.answer in record.context, (
            f"gold answer {record.answer!r} not found verbatim in document {record.context_id!r}"
        )


def test_default_fixture_has_more_than_one_question_on_at_least_one_document():
    # Exercises the "one record per question, context_id repeats" path of
    # pipeline.ingestion.dedupe_documents() - not every document needs
    # more than one question, but the fixture should cover this case.
    records, _ = load_cuad_smoke_fixture()
    context_ids = [r.context_id for r in records]
    assert len(context_ids) > len(set(context_ids))


def test_default_fixture_documents_have_no_metadata_prefix():
    records, _ = load_cuad_smoke_fixture()
    assert all(r.metadata_prefix == "" for r in records)
    assert all(r.company_name is None and r.report_year is None and r.company_sector is None for r in records)


def test_missing_file_raises_file_not_found_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_cuad_smoke_fixture(missing)


def test_missing_documents_key_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="documents"):
        load_cuad_smoke_fixture(path)


def test_missing_questions_key_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"documents": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
        load_cuad_smoke_fixture(path)


def test_question_referencing_unknown_document_id_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "documents": [{"document_id": "doc_a", "text": "some contract text"}],
                "questions": [
                    {
                        "question_id": "q1",
                        "document_id": "doc_that_does_not_exist",
                        "category": "Parties",
                        "question": "Who are the parties?",
                        "gold_answer": "Acme",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown document_id"):
        load_cuad_smoke_fixture(path)


def test_well_formed_temp_fixture_round_trips_correctly(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "documents": [{"document_id": "doc_a", "text": "This Agreement is governed by the laws of Texas."}],
                "questions": [
                    {
                        "question_id": "q1",
                        "document_id": "doc_a",
                        "category": "Governing Law",
                        "question": "What is the governing law?",
                        "gold_answer": "laws of Texas",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records, eval_items = load_cuad_smoke_fixture(path)
    assert len(records) == 1
    assert records[0].context_id == "doc_a"
    assert records[0].answer == "laws of Texas"
    assert eval_items[0]["gold_answer"] == "laws of Texas"


def test_default_fixture_path_points_at_the_committed_repo_fixture():
    assert DEFAULT_FIXTURE_PATH.name == "cuad_smoke_questions.json"
    assert DEFAULT_FIXTURE_PATH.exists()

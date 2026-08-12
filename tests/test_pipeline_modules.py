"""Tests for module 1 (ingestion) and module 2 (chunking) on a slice of real
and synthetic data. See kak_my_rabotaem_vmeste.md: every function is tested
on real or realistic data, including edge cases, before being handed off.
 
The integration test on the full dataset is skipped if data/t2-ragbench/ is
not populated with the real files (the raw dataset is not committed to the
repository - see struktura_repozitoriya.md), but it must be run locally on
the first pass (plan Step 1).
"""
 
from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
import pytest
 
from pipeline.chunking import chunk
from pipeline.ingestion import (
    EXPECTED_DOCUMENTS,
    EXPECTED_QUESTIONS,
    DocumentRecord,
    ingest,
    load_raw,
    to_document_records,
)
 
REAL_DATA_DIR = Path(os.environ.get("T2_RAGBENCH_DATA_DIR", "data/t2-ragbench"))
HAS_REAL_DATA = REAL_DATA_DIR.exists() and any(REAL_DATA_DIR.glob("*.parquet"))
 
 
# ---------------------------------------------------------------------------
# Edge cases - synthetic data, independent of dataset availability
# ---------------------------------------------------------------------------
 
 
def test_to_document_records_empty_input():
    """Legitimately empty result: empty DataFrame -> empty list, no errors."""
    empty = pd.DataFrame(
        columns=["context_id", "context", "source_dataset", "question", "program_answer"]
    )
    assert to_document_records(empty) == []
 
 
def test_to_document_records_missing_column_raises():
    """A missing required column must fail explicitly, not silently return garbage."""
    bad = pd.DataFrame({"context_id": ["a"], "context": ["b"]})
    with pytest.raises(KeyError):
        to_document_records(bad)
 
 
def test_to_document_records_uses_program_answer_not_original_answer():
    """Regression test for the decision recorded in specifikatsiya_moduley.md,
    module 1: answer = program_answer, original_answer is ignored even when
    both fields are present and differ."""
    df = pd.DataFrame(
        {
            "context_id": ["ctx_1"],
            "context": ["some text with a table"],
            "source_dataset": ["FinQA"],
            "question": ["What was X?"],
            "program_answer": ["3.8"],
            "original_answer": ["380"],  # deliberately different value
        }
    )
    records = to_document_records(df)
    assert len(records) == 1
    assert records[0].answer == "3.8"
 
 
def test_load_raw_missing_file_raises_filenotfounderror(tmp_path):
    """A partially populated directory (missing one of the 7 expected files)
    must fail with a clear message, not an opaque pandas/pyarrow error."""
    # Only one of the seven expected files is present
    df = pd.DataFrame(
        {
            "context_id": ["ctx_1"],
            "context": ["text"],
            "question": ["q"],
            "program_answer": ["1"],
        }
    )
    df.to_parquet(tmp_path / "FinQA_train.parquet")
 
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)
 
 
def test_chunk_is_identity_no_op():
    """Module 2 must not change the record count or their content."""
    records = [
        DocumentRecord(
            context_id="ctx_1",
            context="text with | a | table |",
            source_dataset="TAT-DQA",
            question="q?",
            answer="42",
        )
    ]
    result = chunk(records)
    assert result == records
    assert result is not records  # returns a new list, does not mutate the input
 
 
def test_chunk_empty_input():
    assert chunk([]) == []
 
 
# ---------------------------------------------------------------------------
# Integration checkpoint on the full real dataset (spec section 1)
# ---------------------------------------------------------------------------
 
 
@pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason=(
        "T2-RAGBench raw dataset not found under data/t2-ragbench/ "
        "(not committed to the repository, see struktura_repozitoriya.md) - "
        "run locally before the first pass, setting T2_RAGBENCH_DATA_DIR."
    ),
)
def test_ingest_full_corpus_checkpoint():
    records = ingest(REAL_DATA_DIR)
 
    assert len(records) == EXPECTED_QUESTIONS == 23088
    assert len({r.context_id for r in records}) == EXPECTED_DOCUMENTS == 7318
 
    sources = {r.source_dataset for r in records}
    assert sources == {"FinQA", "ConvFinQA", "TAT-DQA"}
 
    # answer is always populated and is a string (program_answer has no gaps)
    assert all(isinstance(r.answer, str) and r.answer != "" for r in records)
 
    # chunking must not drop anything on the full pass
    chunked = chunk(records)
    assert len(chunked) == len(records)

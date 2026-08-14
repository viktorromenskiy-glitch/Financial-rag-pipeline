"""Local tests of pipeline modules against a slice of real data.

Unlike the fake/synthetic-data unit tests bundled with each module, these
tests exercise pipeline.ingestion and pipeline.chunking against the
actual T2-RAGBench parquet files - see docs/struktura_repozitoriya.md.
They read whatever company_name/report_year/company_sector values are
actually present in the first rows of one real file, rather than
hardcoding expected values that were never independently verified here.

Skipped automatically if the raw dataset is not present locally (e.g. in
a portfolio reviewer's checkout, or a CI environment without direct
access to Google Drive/Colab) - run this file where data/t2-ragbench/ is
actually populated to get real coverage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.chunking import chunk
from pipeline.ingestion import RAW_FILES, DocumentRecord, to_document_records

DATA_DIR = Path("data/t2-ragbench")
_PROBE_FILE = DATA_DIR / RAW_FILES["FinQA"][0]  # FinQA_train.parquet

pytestmark = pytest.mark.skipif(
    not _PROBE_FILE.exists(),
    reason=(
        f"Real dataset not found at {DATA_DIR} - these tests need the actual "
        f"T2-RAGBench parquet files (see docs/specifikatsiya_moduley.md, "
        f"Module 1, 'Config') and are skipped in environments without them."
    ),
)


@pytest.fixture(scope="module")
def real_slice() -> pd.DataFrame:
    """A small slice (first 20 rows) of one real raw parquet file, with
    the source_dataset column added the same way load_raw() does - not the
    full corpus, so this runs in under a second."""
    df = pd.read_parquet(_PROBE_FILE).head(20).copy()
    df["source_dataset"] = "FinQA"
    return df


def test_to_document_records_on_real_slice(real_slice):
    records = to_document_records(real_slice)
    assert len(records) == len(real_slice)
    assert all(isinstance(r, DocumentRecord) for r in records)
    # context_id/context/question/answer must be real, non-empty values -
    # not placeholders or NaN leaking through as literal "nan" strings.
    for r in records:
        assert r.context_id
        assert r.context
        assert r.question
        assert r.answer
        assert r.source_dataset == "FinQA"


def test_metadata_prefix_built_from_real_columns(real_slice):
    records = to_document_records(real_slice)
    has_metadata_columns = {"company_name", "report_year", "company_sector"} <= set(real_slice.columns)
    if not has_metadata_columns:
        pytest.skip("This local parquet copy predates the company_name/report_year/company_sector columns")

    with_metadata = [r for r in records if r.metadata_prefix]
    # At least the rows that actually have company_name/report_year/sector
    # populated (not all-NaN) must produce a non-empty prefix - this is
    # the exact mechanism behind the validated Recall@5 0.896->0.948 fix
    # (tasks #60/#61), so a real slice with zero non-empty prefixes here
    # would indicate a real regression, not a fixture problem.
    any_row_has_raw_metadata = (
        real_slice[["company_name", "report_year", "company_sector"]].notna().any(axis=1).any()
    )
    if any_row_has_raw_metadata:
        assert with_metadata, "No metadata_prefix was built despite real metadata columns being present"
    for r in with_metadata:
        assert r.metadata_prefix.endswith("\n\n")
        assert ":" in r.metadata_prefix


def test_chunk_is_identity_on_real_records(real_slice):
    records = to_document_records(real_slice)
    chunked = chunk(records)
    assert chunked == records
    assert chunked is not records  # a distinct list object, per module 2's contract

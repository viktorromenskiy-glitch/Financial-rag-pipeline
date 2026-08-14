"""Module 1 - Ingestion / Parsing.

Loads the raw T2-RAGBench parquet files (FinQA, ConvFinQA, TAT-DQA) and
normalizes them into a single list of DocumentRecord. See
docs/specifikatsiya_moduley.md, Module 1.

The table format embedded in `context` and the choice of the `answer` field
(program_answer, not original_answer) are documented there, in the
"Verified on real files" section - not re-derived here.

The company_name/report_year/company_sector fields and metadata_prefix were
added 2026-08-14 following the investigation in tasks #60/#61 (see
docs/specifikatsiya_moduley.md, Module 1, "Required addition"). metadata_prefix
is a deterministic fix (no LLM calls) that empirically raised Recall@5 from
0.896 to 0.948 at the reranking stage (McNemar p=0.00195) by closing the
lexical gap between the question (reformulated with company metadata by the
T2-RAGBench authors) and the context text (which does not contain that
metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Raw files per source dataset - see docs/specifikatsiya_moduley.md, Module 1, "Config"
RAW_FILES: dict[str, list[str]] = {
    "FinQA": ["FinQA_train.parquet", "FinQA_dev.parquet", "FinQA_test.parquet"],
    "ConvFinQA": ["ConvFinQA_turn_0.parquet"],
    "TAT-DQA": ["TAT-DQA_train.parquet", "TAT-DQA_dev.parquet", "TAT-DQA_test.parquet"],
}

# Checkpoint from the spec, section 1 / plan_podgotovki_k_kodirovaniyu.md, Step 1
EXPECTED_DOCUMENTS = 7318
EXPECTED_QUESTIONS = 23088


@dataclass(frozen=True)
class DocumentRecord:
    """A single normalized question-document pair.

    One record per row of the raw dataset (one per question); `context_id`
    repeats across records that share the same underlying document - see
    `to_document_records`.
    """

    context_id: str
    context: str
    source_dataset: str
    question: str
    answer: str
    company_name: str | None
    report_year: str | None
    company_sector: str | None
    metadata_prefix: str


def humanize_company_name(name: str | None) -> str | None:
    """Expand hyphen-slug company names into a readable form.

    "coherent-inc" -> "Coherent Inc". Names that already look readable
    (contain a space, e.g. "PNC Financial Services") are left untouched.

    The exact behavior must stay bit-identical to the validated Colab
    experiment (task #61) - do not change this without re-running the
    validation test on the eval set.

    Args:
        name: Raw company name, or None.

    Returns:
        The humanized name, or the original value if it is None/empty/
        already readable.
    """
    if not name:
        return name
    if "-" in name and " " not in name:
        return name.replace("-", " ").title()
    return name


def build_metadata_prefix(
    company_name: str | None,
    company_sector: str | None,
    report_year: str | None,
) -> str:
    """Build the deterministic metadata prefix prepended to indexed text.

    No LLM calls involved. The exact format was validated on the full
    250-question eval set: Recall@5 rose from 0.896 to 0.948 (McNemar
    p=0.00195). See docs/specifikatsiya_moduley.md, Module 1.

    Args:
        company_name: Raw or already-readable company name, or None.
        company_sector: Sector classification, or None.
        report_year: Report year, or None.

    Returns:
        A prefix string ending in "\\n\\n", or "" if all three inputs are
        missing.
    """
    parts: list[str] = []
    human_name = humanize_company_name(company_name)
    if human_name:
        parts.append(f"Company: {human_name}")
    if company_sector:
        parts.append(f"Sector: {company_sector}")
    if report_year:
        parts.append(f"Report year: {report_year}")
    if not parts:
        return ""
    return " | ".join(parts) + "\n\n"


def _clean(value: object) -> str | None:
    """Normalize NaN/None to None; otherwise cast to str.

    pandas reads empty parquet cells as float NaN, not None/empty string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _load_source(data_dir: Path, source: str, filenames: list[str]) -> pd.DataFrame:
    """Load and concatenate all parquet files for a single source dataset."""
    frames = []
    for filename in filenames:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected dataset file not found: {path}. "
                f"See docs/specifikatsiya_moduley.md, Module 1, 'Config' "
                f"(default path: data/t2-ragbench/)."
            )
        df = pd.read_parquet(path)
        df = df.copy()
        df["source_dataset"] = source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Load all 7 raw parquet files into a single DataFrame.

    Adds a `source_dataset` column identifying which of the three source
    datasets (FinQA / ConvFinQA / TAT-DQA) each row came from.

    Args:
        data_dir: Directory containing the raw parquet files.

    Returns:
        A single concatenated DataFrame.
    """
    data_dir = Path(data_dir)
    frames = [
        _load_source(data_dir, source, filenames)
        for source, filenames in RAW_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)


def to_document_records(raw: pd.DataFrame) -> list[DocumentRecord]:
    """Convert a raw DataFrame into a list of DocumentRecord.

    One raw row = one question. `context_id` repeats for questions that
    share the same underlying document - deduplication into unique
    documents happens later, at indexing time (Module 5), not here.

    company_name/report_year/company_sector are optional columns (an
    older local copy of the dataset may not have them, or a given row may
    have them as NaN). A missing column is not an error - it simply
    yields an empty metadata_prefix for affected rows.

    Args:
        raw: DataFrame produced by `load_raw`.

    Returns:
        One DocumentRecord per input row.

    Raises:
        KeyError: If any of the required columns are missing.
    """
    required = {"context_id", "context", "source_dataset", "question", "program_answer"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Raw DataFrame is missing required columns: {sorted(missing)}")

    has_company_name = "company_name" in raw.columns
    has_report_year = "report_year" in raw.columns
    has_company_sector = "company_sector" in raw.columns

    records = []
    for row in raw.itertuples(index=False):
        company_name = _clean(getattr(row, "company_name")) if has_company_name else None
        report_year = _clean(getattr(row, "report_year")) if has_report_year else None
        company_sector = _clean(getattr(row, "company_sector")) if has_company_sector else None

        records.append(
            DocumentRecord(
                context_id=row.context_id,
                context=row.context,
                source_dataset=row.source_dataset,
                question=row.question,
                answer=row.program_answer,
                company_name=company_name,
                report_year=report_year,
                company_sector=company_sector,
                metadata_prefix=build_metadata_prefix(company_name, company_sector, report_year),
            )
        )
    return records


def ingest(data_dir: str | Path) -> list[DocumentRecord]:
    """Module 1 entry point.

    Loads, normalizes, and validates the dataset checkpoint (spec section 1):
    7318 unique documents, 23088 questions. A mismatch signals a parsing bug
    in one of the three source datasets, not a downstream pipeline issue
    (see plan_podgotovki_k_kodirovaniyu.md, Step 1).

    Args:
        data_dir: Directory containing the raw parquet files.

    Returns:
        The full list of DocumentRecord (one per question).

    Raises:
        AssertionError: If the question/document counts do not match the
            expected checkpoint values.
    """
    raw = load_raw(data_dir)
    records = to_document_records(raw)

    n_questions = len(records)
    n_documents = len({r.context_id for r in records})

    assert n_questions == EXPECTED_QUESTIONS, (
        f"Expected {EXPECTED_QUESTIONS} questions (spec section 1), got {n_questions}."
    )
    assert n_documents == EXPECTED_DOCUMENTS, (
        f"Expected {EXPECTED_DOCUMENTS} unique documents (spec section 1), got {n_documents}."
    )

    return records

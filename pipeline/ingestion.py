"""Module 1 - Ingestion / parsing.
 
Loads raw T2-RAGBench parquet files (FinQA, ConvFinQA, TAT-DQA) and
normalizes them into a single list of DocumentRecord. See
docs/specifikatsiya_moduley.md, module 1 (Russian working spec).
 
The table format inside `context` and the decision on the `answer` field
(program_answer, not original_answer) are documented there, section
"Проверено на реальных файлах" - do not re-derive when reading this module.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import pandas as pd
 
# Raw files by source - see docs/specifikatsiya_moduley.md, module 1, "Config"
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
    context_id: str
    context: str
    source_dataset: str
    question: str
    answer: str
 
 
def _load_source(data_dir: Path, source: str, filenames: list[str]) -> pd.DataFrame:
    frames = []
    for filename in filenames:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected dataset file not found: {path}. "
                f"See docs/specifikatsiya_moduley.md, module 1, 'Config' "
                f"(default path: data/t2-ragbench/)."
            )
        df = pd.read_parquet(path)
        df = df.copy()
        df["source_dataset"] = source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
 
 
def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Load all 7 raw parquet files and concatenate them into a single
    DataFrame with an added source_dataset column.
    """
    data_dir = Path(data_dir)
    frames = [
        _load_source(data_dir, source, filenames)
        for source, filenames in RAW_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)
 
 
def to_document_records(raw: pd.DataFrame) -> list[DocumentRecord]:
    """Convert a raw DataFrame into a list of DocumentRecord.
 
    One raw row = one question (context_id repeats for questions sharing the
    same document - expected, document-level deduplication happens at the
    indexing step, not here, see module 5).
    """
    required = {"context_id", "context", "source_dataset", "question", "program_answer"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns in raw DataFrame: {sorted(missing)}")
 
    return [
        DocumentRecord(
            context_id=row.context_id,
            context=row.context,
            source_dataset=row.source_dataset,
            question=row.question,
            answer=row.program_answer,
        )
        for row in raw.itertuples(index=False)
    ]
 
 
def ingest(data_dir: str | Path) -> list[DocumentRecord]:
    """Module 1 entry point. Loads, normalizes, and validates the checkpoint
    (spec section 1): 7318 documents, 23088 questions. A mismatch signals a
    parsing bug in one of the three sources, not a downstream pipeline issue
    (see plan_podgotovki_k_kodirovaniyu.md, Step 1).
    """
    raw = load_raw(data_dir)
    records = to_document_records(raw)
 
    n_questions = len(records)
    n_documents = len({r.context_id for r in records})
 
    assert n_questions == EXPECTED_QUESTIONS, (
        f"Expected {EXPECTED_QUESTIONS} questions (spec section 1), got {n_questions}."
    )
    assert n_documents == EXPECTED_DOCUMENTS, (
        f"Expected {EXPECTED_DOCUMENTS} documents (spec section 1), got {n_documents}."
    )
 
    return records

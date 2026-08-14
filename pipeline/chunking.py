"""Module 2 - Chunking / table serialization.

A no-op by architectural decision (design decision 4): one document equals
one chunk. Tables are already serialized by the dataset authors as markdown
pipe-tables directly inside DocumentRecord.context - see
docs/specifikatsiya_moduley.md, Module 2, "Closed (2026-08-12)" section for
concrete examples per source dataset. No additional serialization is
performed here.
"""

from __future__ import annotations

from pipeline.ingestion import DocumentRecord


def chunk(records: list[DocumentRecord]) -> list[DocumentRecord]:
    """Pass records through unchanged.

    A no-op by architectural decision - a DocumentRecord already is a
    ready-made chunk. The function exists as an explicit pipeline step
    (Module 2) rather than being removed entirely, so the step order from
    docs/specifikatsiya_moduley.md is visible one-to-one in the code.

    Args:
        records: DocumentRecord list produced by Module 1 (ingestion).

    Returns:
        A shallow copy of the input list, unchanged.
    """
    return list(records)
  

"""Module 2 - Chunking / table serialization.
 
Trivial by architectural decision (decision point 4): one document = one
chunk. Tables are already serialized by the dataset authors as markdown
pipe-tables directly inside DocumentRecord.context - see
docs/specifikatsiya_moduley.md, module 2, section "Закрыто (12.08.2026)" for
concrete examples per source. No additional serialization is performed.
"""
 
from __future__ import annotations
 
from pipeline.ingestion import DocumentRecord
 
 
def chunk(records: list[DocumentRecord]) -> list[DocumentRecord]:
    """No-op by architectural decision - DocumentRecord is already a
    ready-made chunk. The function exists as an explicit pipeline step
    (module 2) rather than being removed entirely, so the step order from
    specifikatsiya_moduley.md is visible in the code one-to-one.
    """
    return list(records)

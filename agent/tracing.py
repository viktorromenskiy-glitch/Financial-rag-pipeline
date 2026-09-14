"""JSONL tracing for the agent's bounded tool-use loop.

Writes one line per step per question - retrieval_1, evidence_assessment,
reformulated_query, retrieval_2, ..., answer - matching the trace shape
referenced throughout this project's agent-design expert review
(itog_ekspertizy_agent_profil.md, consensus item 12: "structured
JSONL-трейсинг каждого прогона"). Intended output path from a real run is
results/<run_id>/agent_trace.jsonl, mirroring pipeline.cli's existing
results/<run_id>/retrieval_trace.jsonl (module 6) - the caller (cli.py's
future `agent` subcommand, Day 2/3) decides the actual path.

Append-per-record, not accumulate-then-write-once, mirroring
pipeline.cli._append_retrieval_trace_record and every other checkpoint in
this project ("Правила сохранения долгих платных прогонов" - an
interrupted run should lose at most the one in-flight step, not the whole
trace).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol


class TraceWriterProtocol(Protocol):
    def append(self, record: dict) -> None: ...


class TraceWriter:
    """Writes each record as one JSON line, appended immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        record = {"ts": time.time(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class InMemoryTraceWriter:
    """Test/debug double - collects records in a list instead of writing to
    disk, so unit tests can assert on trace content without touching the
    filesystem or needing to clean up a temp file afterward."""

    def __init__(self):
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)

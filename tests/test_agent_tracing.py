"""Tests for agent/tracing.py."""

from __future__ import annotations

import json

from agent.tracing import InMemoryTraceWriter, TraceWriter


def test_trace_writer_appends_one_json_line_per_record(tmp_path):
    path = tmp_path / "run1" / "agent_trace.jsonl"
    writer = TraceWriter(path)

    writer.append({"question_id": "q1", "step": "retrieval_1"})
    writer.append({"question_id": "q1", "step": "answer"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["step"] == "retrieval_1"
    assert records[1]["step"] == "answer"
    assert "ts" in records[0]  # timestamp added automatically


def test_trace_writer_creates_parent_directories():
    # Covered implicitly by the tmp_path test above (run1/ doesn't exist
    # beforehand) - this test just makes the intent explicit and checks
    # it doesn't raise for a deeper nested path.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results" / "run42" / "agent_trace.jsonl"
        TraceWriter(path).append({"step": "x"})
        assert path.exists()


def test_in_memory_trace_writer_collects_records_without_touching_disk():
    writer = InMemoryTraceWriter()
    writer.append({"step": "retrieval_1"})
    writer.append({"step": "answer"})
    assert [r["step"] for r in writer.records] == ["retrieval_1", "answer"]

"""Tests for pipeline.common.latency (План доработки-2, пункт 2).

Covers summarize_latencies' stats (n/mean/median/p95/min/max/total),
its "no data -> None, not fake 0" convention for an unmeasured stage, and
write_latency_report's on-disk roundtrip - matches the pattern already
used for pipeline.common.run_config's write_run_config.
"""

from __future__ import annotations

import json

from pipeline.common.latency import summarize_latencies, write_latency_report


# --- summarize_latencies -----------------------------------------------------


def test_summarize_latencies_computes_basic_stats():
    summary = summarize_latencies({"retrieval_s": [1.0, 2.0, 3.0]})
    stats = summary["retrieval_s"]
    assert stats["n"] == 3
    assert stats["mean_s"] == 2.0
    assert stats["median_s"] == 2.0
    assert stats["min_s"] == 1.0
    assert stats["max_s"] == 3.0
    assert stats["total_s"] == 6.0


def test_summarize_latencies_empty_stage_is_none_not_zero():
    # A stage with zero measurements (e.g. reranker disabled all run, or
    # every question resumed from checkpoint) must report None - a
    # fabricated 0 would misleadingly claim "measured, instant", not
    # "not measured this run" (see module docstring).
    summary = summarize_latencies({"rerank_s": []})
    assert summary["rerank_s"] is None


def test_summarize_latencies_p95_nearest_rank_single_value():
    summary = summarize_latencies({"judge_s": [5.0]})
    assert summary["judge_s"]["p95_s"] == 5.0
    assert summary["judge_s"]["n"] == 1


def test_summarize_latencies_p95_within_range_for_larger_sample():
    values = [float(i) for i in range(1, 101)]  # 1.0 .. 100.0
    summary = summarize_latencies({"generation_s": values})
    stats = summary["generation_s"]
    # nearest-rank p95 of 1..100 is the 95th smallest value = 95.0
    assert stats["p95_s"] == 95.0
    assert stats["n"] == 100


def test_summarize_latencies_multiple_stages_independent():
    summary = summarize_latencies({"retrieval_s": [1.0], "rerank_s": [], "generation_s": [2.0, 4.0]})
    assert summary["retrieval_s"]["n"] == 1
    assert summary["rerank_s"] is None
    assert summary["generation_s"]["mean_s"] == 3.0


# --- write_latency_report ----------------------------------------------------


def test_write_latency_report_roundtrip(tmp_path):
    latencies = {"retrieval_s": [1.0, 2.0], "rerank_s": [], "generation_s": [3.0], "judge_s": [0.5]}
    path = write_latency_report(latencies, "test_run", results_dir=tmp_path)

    assert path == tmp_path / "test_run" / "latency_report.json"
    assert path.exists()

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["retrieval_s"]["n"] == 2
    assert loaded["rerank_s"] is None
    assert loaded["generation_s"]["n"] == 1
    assert loaded["judge_s"]["mean_s"] == 0.5


def test_write_latency_report_creates_run_directory(tmp_path):
    write_latency_report({"retrieval_s": [1.0]}, "brand_new_run", results_dir=tmp_path)
    assert (tmp_path / "brand_new_run").is_dir()

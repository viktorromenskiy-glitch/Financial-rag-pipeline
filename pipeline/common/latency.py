"""Per-stage wall-clock latency measurement for `pipeline.cli eval` runs.

Added per docs/tehnicheskoe_zadanie.md, "План доработки-2, пункт 2": the
project measures cost (section 15, unit economics) from published API
pricing and its own measured token/char counts, but latency was never
measured anywhere - `cmd_eval()`'s retrieval -> rerank -> generation loop
is fully sequential (no ThreadPoolExecutor), so a caller genuinely needs
wall-clock numbers to know what a run costs in time, not just money.
Unlike cost, latency cannot be derived from a pricing table - it has to
come from an actual timed run.

Only real API-call latency is recorded, never resumed-from-checkpoint or
judge-cache-hit items (see cmd_eval's `latencies` dict and
evaluate_answers' `latency_sink` param) - a checkpoint/cache hit takes
microseconds and would silently deflate the reported numbers if mixed in.
A stage that never ran in a given config (e.g. reranker disabled) reports
`None`, not a fabricated 0 - same "no data, not fake data" convention the
project already uses elsewhere (e.g. unit economics' explicit
measured-vs-assumed split, section 15).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def summarize_latencies(latencies: dict[str, list[float]]) -> dict[str, dict | None]:
    """latencies: {stage_name: [seconds, ...]} - one entry per actual API
    call timed for that stage during a run (see cmd_eval).

    Returns {stage_name: {n, mean_s, median_s, p95_s, min_s, max_s,
    total_s}} for every stage with at least one measurement, or None for
    a stage with zero (e.g. reranker disabled, or every question resumed
    from checkpoint).

    p95 uses the nearest-rank method (ceil(0.95 * n), 1-indexed, clamped
    to the last element) - simple and standard for the sample sizes this
    project runs (n in the hundreds), not an interpolated percentile.
    """
    summary: dict[str, dict | None] = {}
    for stage, values in latencies.items():
        if not values:
            summary[stage] = None
            continue
        sorted_values = sorted(values)
        p95_index = min(len(sorted_values) - 1, max(0, math.ceil(0.95 * len(sorted_values)) - 1))
        summary[stage] = {
            "n": len(values),
            "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "p95_s": sorted_values[p95_index],
            "min_s": min(values),
            "max_s": max(values),
            "total_s": sum(values),
        }
    return summary


def write_latency_report(latencies: dict[str, list[float]], run_id: str, results_dir: str | Path = "results") -> Path:
    """Writes results/<run_id>/latency_report.json - a separate file from
    run_config.json (static configuration snapshot) since this is a
    runtime measurement, not configuration; separate from eval_report.md
    since that's read as prose, this as structured data for later
    aggregation across runs.

    Returns the path the file was written to.
    """
    summary = summarize_latencies(latencies)
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path

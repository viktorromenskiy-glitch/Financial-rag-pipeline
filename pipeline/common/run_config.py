"""Run configuration snapshot.

Builds and writes run_config.json next to each saved run's results, so a
saved predictions.jsonl / eval_report.md can always be traced back to the
exact configuration that produced it. Without this, restoring which run
used which config after several weeks of testing becomes guesswork, which
breaks the regression analysis required by docs/tehnicheskoe_zadanie.md,
section 10. See docs/tehnicheskoe_zadanie.md, section 11
("run_config.json"), and docs/struktura_repozitoriya.md
(results/<run_id>/run_config.json).

Snapshots the fields spec section 11 explicitly requires: embedding
model, enrichment.enabled + model + prompt_version, reranker.enabled +
model + pool_size, generation model, judge prompt_version - plus
retrieval.pool_size and the judge model/temperature/deterministic-check
flag, since a silent config difference there is exactly as untraceable
otherwise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_run_config(config: dict, run_id: str) -> dict:
    """config: the parsed config.yaml dict (see
    docs/struktura_repozitoriya.md for the schema). run_id: identifier for
    this run (e.g. a timestamp or short description), used as the
    results/<run_id>/ directory name.

    Raises:
        KeyError: if a required config section/field is missing - fail
            loudly at snapshot time rather than silently writing a
            partial or misleading run_config.json.
    """
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config["embedding"]["model"],
        "enrichment_enabled": config["enrichment"]["enabled"],
        "enrichment_model": config["enrichment"]["model"],
        "enrichment_prompt_version": config["enrichment"]["prompt_version"],
        "retrieval_pool_size": config["retrieval"]["pool_size"],
        "reranker_enabled": config["reranker"]["enabled"],
        "reranker_model": config["reranker"]["model"],
        "reranker_pool_size": config["reranker"]["pool_size"],
        "generation_model": config["generation"]["model"],
        "judge_model": config["judge"]["model"],
        "judge_prompt_version": config["judge"]["prompt_version"],
        "judge_deterministic_check_enabled": config["judge"]["deterministic_check_enabled"],
    }


def write_run_config(config: dict, run_id: str, results_dir: str | Path = "results") -> Path:
    """Builds the snapshot and writes it to results/<run_id>/run_config.json.

    Returns:
        The path the file was written to.
    """
    run_config = build_run_config(config, run_id)
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_config.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    return out_path

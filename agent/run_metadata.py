"""Reproducibility metadata for a Day 2 agent-evaluation harness run -
plan_rabot_posle_ekspertizy_agent_profil.md, Day 2: "Record model/
temperature/system-prompt versions and the judge version + is_close_v2
(commit-tag) for reproducibility - a commit hash plus a code comment is
enough, a separate lock file isn't required."

Kept separate from pipeline.common.run_config.build_run_config rather than
extended: that function's return shape is a fixed contract required by
docs/tehnicheskoe_zadanie.md section 11 and pinned down by
tests/test_run_config.py for the BASELINE pipeline's own run_config.json -
an agent run needs different fields entirely (agent prompt versions, the
insufficiency-judge prompt version) that have no place in that contract
and no baseline-pipeline equivalent to stay consistent with.

git commit hash: same `git rev-parse --short HEAD` approach already used
by scripts/check_colab_runtime.py - "a commit hash is enough", no separate
lock-file.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agent.loop import AGENT_ANSWER_PROMPT_VERSION, AGENT_ASSESSMENT_PROMPT_VERSION
from agent.success import INSUFFICIENCY_JUDGE_PROMPT_VERSION
from pipeline.evaluation import PROMPT_VERSION as JUDGE_PROMPT_VERSION

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_short_hash(repo_root: Path | None = None) -> str | None:
    """Returns the short commit hash of HEAD, or None if git is unavailable
    or this isn't a git checkout - a run must still be able to write its
    metadata (and proceed) even if the commit hash can't be determined,
    rather than crashing an otherwise-successful, already-paid-for
    evaluation run over a purely informational field."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or _DEFAULT_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def build_agent_eval_run_metadata(config, run_id: str, *, repo_root: Path | None = None) -> dict:
    """config: a loaded config.config_schema.PipelineConfig.

    Returns a plain dict, meant to be written verbatim as
    results/<run_id>/agent_run_metadata.json by scripts/run_agent_eval.py
    (kept separate from that run's baseline-pipeline run_config.json,
    written unchanged via pipeline.common.run_config as usual).
    """
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_short_hash(repo_root),
        "agent_max_additional_tool_calls": config.agent.max_additional_tool_calls,
        "agent_max_wall_clock_seconds": config.agent.max_wall_clock_seconds,
        "agent_assessment_prompt_version": AGENT_ASSESSMENT_PROMPT_VERSION,
        "agent_answer_prompt_version": AGENT_ANSWER_PROMPT_VERSION,
        "generation_model": config.generation.model,
        "generation_temperature": config.generation.temperature,
        "judge_model": config.judge.model,
        "judge_temperature": config.judge.temperature,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_deterministic_check_enabled": config.judge.deterministic_check_enabled,
        "insufficiency_judge_prompt_version": INSUFFICIENCY_JUDGE_PROMPT_VERSION,
        "agent_eval_max_llm_calls": config.agent_eval.max_llm_calls,
        "agent_eval_max_wall_clock_seconds": config.agent_eval.max_wall_clock_seconds,
        "agent_eval_max_estimated_cost_usd": config.agent_eval.max_estimated_cost_usd,
        "agent_eval_cost_per_llm_call_usd": config.agent_eval.cost_per_llm_call_usd,
    }

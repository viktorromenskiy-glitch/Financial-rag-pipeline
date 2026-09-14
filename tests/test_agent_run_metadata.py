"""Tests for agent/run_metadata.py - the Day 2 reproducibility snapshot.

Uses a real loaded PipelineConfig (config/config.yaml, via
config.config_schema.load_config) rather than a hand-built fake, so this
also incidentally checks the metadata builder against the actual schema
shape - not just a mock matching whatever the builder happens to read.
"""

from __future__ import annotations

import subprocess

from agent.run_metadata import build_agent_eval_run_metadata, git_short_hash
from config.config_schema import load_config


def test_git_short_hash_returns_a_hash_in_this_real_git_repo():
    result = git_short_hash()
    assert result is not None
    assert isinstance(result, str)
    assert len(result) >= 4  # short hashes are at least this long


def test_git_short_hash_returns_none_when_git_is_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert git_short_hash() is None


def test_git_short_hash_returns_none_on_nonzero_exit(monkeypatch):
    class _FakeCompletedProcess:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
    assert git_short_hash() is None


def test_build_agent_eval_run_metadata_includes_all_expected_fields(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config("config/config.yaml")

    metadata = build_agent_eval_run_metadata(config, "test_run_1", repo_root=None)

    assert metadata["run_id"] == "test_run_1"
    assert metadata["agent_max_additional_tool_calls"] == config.agent.max_additional_tool_calls
    assert metadata["agent_max_wall_clock_seconds"] == config.agent.max_wall_clock_seconds
    assert metadata["generation_model"] == config.generation.model
    assert metadata["judge_model"] == config.judge.model
    assert metadata["judge_deterministic_check_enabled"] == config.judge.deterministic_check_enabled
    assert metadata["agent_eval_max_llm_calls"] == config.agent_eval.max_llm_calls
    assert metadata["agent_eval_cost_per_llm_call_usd"] == config.agent_eval.cost_per_llm_call_usd
    # Prompt versions are non-empty strings, present specifically so a
    # future prompt change is traceable in an old run's saved metadata.
    for key in (
        "agent_assessment_prompt_version",
        "agent_answer_prompt_version",
        "judge_prompt_version",
        "insufficiency_judge_prompt_version",
    ):
        assert isinstance(metadata[key], str) and metadata[key]


def test_build_agent_eval_run_metadata_is_json_serializable(monkeypatch):
    import json

    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config("config/config.yaml")
    metadata = build_agent_eval_run_metadata(config, "test_run_2")

    # Must round-trip cleanly - this is written straight to
    # results/<run_id>/agent_run_metadata.json by the harness script.
    json.dumps(metadata)

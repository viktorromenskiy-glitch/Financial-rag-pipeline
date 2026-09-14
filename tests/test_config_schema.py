"""Tests for config/config_schema.py's GenerationConfig.prompt_variant
(Фаза 5, docs/tehnicheskoe_zadanie.md section 28), PersistenceConfig
("Правила сохранения долгих платных прогонов", 2026-08-24), and
AgentConfig (agent/, itog_ekspertizy_agent_profil.md - 4-expert design
review, consensus item 3: the bounded-loop limit must be a config value,
not a constant in code) - and the config files under config/ that set
them.

Not a general PipelineConfig test suite - just the pieces added for
Фаза 5, the persistence rule, and agent/, since no test_config_schema.py
existed before Фаза 5.
"""

from __future__ import annotations

import pytest

from config.config_schema import AgentConfig, AgentEvalConfig, GenerationConfig, PersistenceConfig, load_config


def test_generation_config_prompt_variant_defaults_to_baseline():
    # No prompt_variant given - every config.yaml written before Фаза 5
    # must keep working unchanged.
    config = GenerationConfig(model="claude-sonnet-5", temperature=0.0)
    assert config.prompt_variant == "baseline"


def test_generation_config_accepts_an_explicit_prompt_variant():
    config = GenerationConfig(model="claude-sonnet-5", temperature=0.0, prompt_variant="cite_and_check")
    assert config.prompt_variant == "cite_and_check"


@pytest.mark.parametrize(
    "path,expected_variant",
    [
        ("config/config.yaml", "baseline"),
        ("config/config_cite_and_check.yaml", "cite_and_check"),
        ("config/config_formula_base.yaml", "formula_base"),
    ],
)
def test_real_config_files_set_the_expected_prompt_variant(monkeypatch, path, expected_variant):
    # MONGODB_URI is the only ${ENV_VAR} placeholder any of these three
    # files reference - stub it so load_config() doesn't need a real
    # .env in the test environment.
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config(path)
    assert config.generation.prompt_variant == expected_variant


def test_persistence_config_defaults_to_the_canonical_drive_path():
    # Exact casing matters here - "RAG-project" matches the folder
    # already created on Drive (notebooks/experiments_weeks_1_2.ipynb).
    # A silently "corrected" casing here would reintroduce incident 2
    # from pipeline/common/persist.py's module docstring.
    assert PersistenceConfig().google_drive_results_dir == "/content/drive/MyDrive/RAG-project/results"


@pytest.mark.parametrize(
    "path",
    ["config/config.yaml", "config/config_cite_and_check.yaml", "config/config_formula_base.yaml"],
)
def test_real_config_files_set_the_canonical_drive_results_dir(monkeypatch, path):
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config(path)
    assert config.persistence.google_drive_results_dir == "/content/drive/MyDrive/RAG-project/results"


def test_agent_config_requires_max_additional_tool_calls():
    with pytest.raises(Exception):
        AgentConfig()


def test_agent_config_rejects_negative_max_additional_tool_calls():
    with pytest.raises(Exception):
        AgentConfig(max_additional_tool_calls=-1)


def test_agent_config_accepts_zero():
    # 0 is a valid, deliberate value - disables re-querying entirely.
    assert AgentConfig(max_additional_tool_calls=0).max_additional_tool_calls == 0


def test_pipeline_config_agent_defaults_to_two_when_omitted():
    # config_cite_and_check.yaml / config_formula_base.yaml predate
    # agent/ and don't declare an `agent:` section at all - the schema
    # default (the 4-expert consensus value, 2) must still apply, the
    # same convention persistence's default already follows.
    from config.config_schema import PipelineConfig

    assert PipelineConfig.model_fields["agent"].default_factory().max_additional_tool_calls == 2


@pytest.mark.parametrize(
    "path",
    ["config/config.yaml", "config/config_cite_and_check.yaml", "config/config_formula_base.yaml"],
)
def test_real_config_files_set_max_additional_tool_calls_to_two(monkeypatch, path):
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config(path)
    assert config.agent.max_additional_tool_calls == 2


# --- День 2: AgentConfig.max_wall_clock_seconds + AgentEvalConfig --------


def test_agent_config_max_wall_clock_seconds_defaults_to_none():
    # config_cite_and_check.yaml / config_formula_base.yaml's `agent:`
    # sections (if any) predate this field - must default to "disabled",
    # not raise or silently pick a nonzero value.
    assert AgentConfig(max_additional_tool_calls=2).max_wall_clock_seconds is None


def test_agent_config_rejects_negative_max_wall_clock_seconds():
    with pytest.raises(Exception):
        AgentConfig(max_additional_tool_calls=2, max_wall_clock_seconds=-1)


def test_agent_config_accepts_zero_max_wall_clock_seconds():
    assert AgentConfig(max_additional_tool_calls=2, max_wall_clock_seconds=0).max_wall_clock_seconds == 0


def test_agent_eval_config_has_sane_pre_committed_defaults():
    # "Зафиксировать заранее бюджет evaluation-прогона" (План, День 2) -
    # every field must already have a concrete value out of the box, not
    # be left to the person running the harness to decide ad hoc.
    config = AgentEvalConfig()
    assert config.max_llm_calls == 400
    assert config.max_wall_clock_seconds == 5400
    assert config.max_estimated_cost_usd == 10.0
    assert config.cost_per_llm_call_usd == 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_llm_calls": 0},  # ge=1, not ge=0 - a zero-call budget makes no sense as a harness limit
        {"max_wall_clock_seconds": -1},
        {"max_estimated_cost_usd": -1},
        {"cost_per_llm_call_usd": -0.01},
    ],
)
def test_agent_eval_config_rejects_invalid_values(kwargs):
    with pytest.raises(Exception):
        AgentEvalConfig(**kwargs)


def test_pipeline_config_agent_eval_defaults_when_omitted():
    from config.config_schema import PipelineConfig

    default_agent_eval = PipelineConfig.model_fields["agent_eval"].default_factory()
    assert default_agent_eval.max_llm_calls == 400


@pytest.mark.parametrize(
    "path",
    ["config/config.yaml", "config/config_cite_and_check.yaml", "config/config_formula_base.yaml"],
)
def test_real_config_files_load_agent_eval_section(monkeypatch, path):
    # config_cite_and_check.yaml/config_formula_base.yaml predate
    # agent_eval entirely and don't declare the section - the schema
    # default must still apply (same convention as agent/persistence
    # above).
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config(path)
    assert config.agent_eval.max_llm_calls == 400
    assert config.agent_eval.cost_per_llm_call_usd == 0.02


def test_config_yaml_sets_an_explicit_per_question_wall_clock_limit(monkeypatch):
    # Only config.yaml (the production baseline config) declares this
    # explicitly, per config.yaml's own comment - the two Фаза 5 variant
    # files intentionally still rely on the None default (they predate
    # agent/ entirely).
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake-for-test")
    config = load_config("config/config.yaml")
    assert config.agent.max_wall_clock_seconds == 120

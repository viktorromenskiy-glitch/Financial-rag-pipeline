"""Tests for config/config_schema.py's GenerationConfig.prompt_variant
(Фаза 5, docs/tehnicheskoe_zadanie.md section 28) and the two variant
config files under config/ that select it.

Not a general PipelineConfig test suite - just the piece added for Фаза
5, since no test_config_schema.py existed before this change.
"""

from __future__ import annotations

import pytest

from config.config_schema import GenerationConfig, load_config


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

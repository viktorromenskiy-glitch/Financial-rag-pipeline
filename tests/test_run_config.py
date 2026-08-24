"""Tests for pipeline.common.run_config.build_run_config's
generation_prompt_variant field (Фаза 5, docs/tehnicheskoe_zadanie.md
section 28) - no test_run_config.py existed before this change, so this
covers only the piece added here, not the whole module.
"""

from __future__ import annotations

import pytest

from pipeline.common.run_config import build_run_config

_BASE_CONFIG = {
    "embedding": {"model": "voyage-4"},
    "enrichment": {"enabled": True, "model": "claude-haiku-4-5-20251001", "prompt_version": "v1"},
    "retrieval": {"pool_size": 50},
    "reranker": {"enabled": True, "model": "rerank-v4.0-pro", "pool_size": 50},
    "generation": {"model": "claude-sonnet-5", "prompt_variant": "cite_and_check"},
    "judge": {"model": "claude-sonnet-5", "prompt_version": "v3", "deterministic_check_enabled": True},
}


def test_build_run_config_records_generation_prompt_variant():
    snapshot = build_run_config(_BASE_CONFIG, "run1")
    assert snapshot["generation_prompt_variant"] == "cite_and_check"


def test_build_run_config_requires_prompt_variant_key():
    # generation_prompt_variant reads config["generation"]["prompt_variant"]
    # with plain bracket access (not .get()), on purpose - the module's
    # stated "fail loudly, not a partial snapshot" rule (see module
    # docstring). A config dict missing the key (e.g. hand-built in a
    # script rather than produced by GenerationConfig.model_dump(), which
    # always fills the default) must raise, not silently write "baseline".
    config = {**_BASE_CONFIG, "generation": {"model": "claude-sonnet-5"}}
    with pytest.raises(KeyError):
        build_run_config(config, "run1")

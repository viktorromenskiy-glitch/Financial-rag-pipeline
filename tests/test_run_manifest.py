"""Tests for pipeline.common.run_manifest - pre-registration manifest and
coverage verification. Added per the Day 3 expertise's consensus
(claude/itog_ekspertizy_den3_dizayn.md, item 10).
"""

from __future__ import annotations

import json

import pytest

from pipeline.common.run_manifest import (
    build_run_manifest,
    verify_manifest_coverage,
    write_run_manifest_if_absent,
)

_CONFIG = {"generation": {"model": "claude-sonnet-5"}, "judge": {"model": "claude-sonnet-5"}}


def test_build_run_manifest_sorts_ids_for_reproducibility():
    manifest = build_run_manifest(
        run_id="run1",
        config=_CONFIG,
        expected_question_ids=["q3", "q1", "q2"],
        expected_canary_ids=["c2", "c1"],
        repo_root=None,  # git lookup will fail gracefully - see below
    )
    assert manifest["expected_question_ids"] == ["q1", "q2", "q3"]
    assert manifest["expected_canary_ids"] == ["c1", "c2"]
    assert manifest["run_id"] == "run1"
    assert "created_at" in manifest
    assert "git_sha" in manifest
    assert "config_hash" in manifest


def test_build_run_manifest_never_raises_when_git_is_unavailable(tmp_path):
    # repo_root pointing at a directory that is not a git checkout at all -
    # _git_sha must degrade to a descriptive string, never raise, since a
    # manifest missing the git SHA is still far better than no manifest.
    manifest = build_run_manifest(
        run_id="run1", config=_CONFIG, expected_question_ids=["q1"], expected_canary_ids=[], repo_root=tmp_path
    )
    assert isinstance(manifest["git_sha"], str)


def test_config_hash_is_stable_regardless_of_key_order():
    manifest_a = build_run_manifest(
        run_id="run1",
        config={"a": 1, "b": 2},
        expected_question_ids=["q1"],
        expected_canary_ids=[],
        repo_root=None,
    )
    manifest_b = build_run_manifest(
        run_id="run1",
        config={"b": 2, "a": 1},
        expected_question_ids=["q1"],
        expected_canary_ids=[],
        repo_root=None,
    )
    assert manifest_a["config_hash"] == manifest_b["config_hash"]


def test_config_hash_changes_when_config_changes():
    manifest_a = build_run_manifest(
        run_id="run1", config={"a": 1}, expected_question_ids=["q1"], expected_canary_ids=[], repo_root=None
    )
    manifest_b = build_run_manifest(
        run_id="run1", config={"a": 2}, expected_question_ids=["q1"], expected_canary_ids=[], repo_root=None
    )
    assert manifest_a["config_hash"] != manifest_b["config_hash"]


def test_write_run_manifest_if_absent_writes_a_real_file(tmp_path):
    path = write_run_manifest_if_absent(
        run_id="run1",
        config=_CONFIG,
        expected_question_ids=["q1", "q2"],
        expected_canary_ids=["c1"],
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["expected_question_ids"] == ["q1", "q2"]


def test_write_run_manifest_if_absent_does_not_overwrite_an_existing_manifest(tmp_path):
    """Regression guard for the whole point of pre-registration: a resumed
    run must never silently re-register a different expected question
    set after some real answers already exist."""
    results_dir = tmp_path / "results"
    first_path = write_run_manifest_if_absent(
        run_id="run1",
        config=_CONFIG,
        expected_question_ids=["q1", "q2"],
        expected_canary_ids=[],
        repo_root=tmp_path,
        results_dir=results_dir,
    )
    original = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = write_run_manifest_if_absent(
        run_id="run1",
        config=_CONFIG,
        expected_question_ids=["q1", "q2", "q3_added_later"],  # different set - must be ignored
        expected_canary_ids=[],
        repo_root=tmp_path,
        results_dir=results_dir,
    )
    after_resume = json.loads(second_path.read_text(encoding="utf-8"))
    assert after_resume == original
    assert after_resume["expected_question_ids"] == ["q1", "q2"]


def test_verify_manifest_coverage_passes_on_exact_match():
    manifest = {"expected_question_ids": ["q1", "q2"]}
    verify_manifest_coverage(manifest, baseline_question_ids=["q1", "q2"], agent_question_ids=["q2", "q1"])


def test_verify_manifest_coverage_raises_on_missing_question():
    manifest = {"expected_question_ids": ["q1", "q2"]}
    with pytest.raises(ValueError, match=r"missing=\['q2'\]"):
        verify_manifest_coverage(manifest, baseline_question_ids=["q1"], agent_question_ids=["q1", "q2"])


def test_verify_manifest_coverage_raises_on_extra_question():
    """The one-sided check ('all expected present') would miss this -
    an extra question_id smuggled in from a different, more favorable
    run must also be rejected."""
    manifest = {"expected_question_ids": ["q1", "q2"]}
    with pytest.raises(ValueError, match=r"extra=\['q3_from_another_run'\]"):
        verify_manifest_coverage(
            manifest, baseline_question_ids=["q1", "q2"], agent_question_ids=["q1", "q2", "q3_from_another_run"]
        )


def test_verify_manifest_coverage_names_which_file_mismatched():
    manifest = {"expected_question_ids": ["q1", "q2"]}
    with pytest.raises(ValueError, match="agent_results.jsonl"):
        verify_manifest_coverage(manifest, baseline_question_ids=["q1", "q2"], agent_question_ids=["q1"])

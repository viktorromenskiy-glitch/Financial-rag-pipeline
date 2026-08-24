"""Tests for pipeline.common.persist - the save/verify architecture from
"Правила сохранения долгих платных прогонов" (project doc, 2026-08-24).

Each test targets one specific failure mode from the rule's two real
incidents: a lost run (no verification before "done"), and a
case-mismatched hardcoded path (verified content at the wrong place).
"""

from __future__ import annotations

import pytest

from pipeline.common.persist import (
    PersistPathMismatchError,
    RunVerificationError,
    find_canonical_root,
    save_run_to_drive,
    verify_run_files,
)


# --- verify_run_files (rule point 7: hard completeness check) ------------


def test_verify_run_files_passes_when_everything_matches(tmp_path):
    (tmp_path / "predictions.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    verify_run_files(tmp_path, {"predictions.jsonl": 2})  # no raise


def test_verify_run_files_raises_when_file_missing(tmp_path):
    with pytest.raises(RunVerificationError, match="missing"):
        verify_run_files(tmp_path, {"predictions.jsonl": 2})


def test_verify_run_files_raises_when_file_empty(tmp_path):
    (tmp_path / "predictions.jsonl").write_text("")
    with pytest.raises(RunVerificationError, match="empty"):
        verify_run_files(tmp_path, {"predictions.jsonl": 2})


def test_verify_run_files_raises_when_line_count_wrong(tmp_path):
    (tmp_path / "predictions.jsonl").write_text('{"a": 1}\n')
    with pytest.raises(RunVerificationError, match=r"1 lines, expected 2"):
        verify_run_files(tmp_path, {"predictions.jsonl": 2})


def test_verify_run_files_reports_every_problem_not_just_the_first(tmp_path):
    (tmp_path / "predictions.jsonl").write_text('{"a": 1}\n')
    # eval_results.jsonl missing entirely
    with pytest.raises(RunVerificationError) as excinfo:
        verify_run_files(tmp_path, {"predictions.jsonl": 2, "eval_results.jsonl": 2})
    assert "predictions.jsonl" in str(excinfo.value)
    assert "eval_results.jsonl" in str(excinfo.value)


# --- find_canonical_root (rule points 4/5: path itself, not look-alikes) --


def test_find_canonical_root_returns_root_when_it_exists(tmp_path):
    root = tmp_path / "RAG-project" / "results"
    root.mkdir(parents=True)
    assert find_canonical_root(str(root)) == root


def test_find_canonical_root_raises_when_parent_missing(tmp_path):
    missing = tmp_path / "not-mounted" / "RAG-project" / "results"
    with pytest.raises(PersistPathMismatchError, match="mounted"):
        find_canonical_root(str(missing))


def test_find_canonical_root_catches_the_incident_2_case_mismatch_bug(tmp_path):
    # The exact failure that motivated this module: a look-alike
    # directory with different case exists, but not the configured one.
    (tmp_path / "rag-project").mkdir()  # lowercase - the typo
    configured = tmp_path / "RAG-project"  # what config.yaml actually says
    with pytest.raises(PersistPathMismatchError, match="rag-project"):
        find_canonical_root(str(configured))


def test_find_canonical_root_raises_when_nothing_similar_exists(tmp_path):
    (tmp_path / "SomeOtherFolder").mkdir()
    configured = tmp_path / "RAG-project"
    with pytest.raises(PersistPathMismatchError, match="nothing similarly named"):
        find_canonical_root(str(configured))


# --- save_run_to_drive (rule points 2/3/5/6) ------------------------------


def test_save_run_to_drive_copies_files_and_returns_resolved_path(tmp_path, capsys):
    run_dir = tmp_path / "run_source"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text('{"a": 1}\n')

    drive_root = tmp_path / "RAG-project" / "results"
    drive_root.mkdir(parents=True)

    dest = save_run_to_drive(run_dir, str(drive_root), "baseline_phase6")

    assert dest == (drive_root / "baseline_phase6").resolve()
    assert (dest / "predictions.jsonl").read_text() == '{"a": 1}\n'
    # rule 6: final path printed loudly
    assert str(dest) in capsys.readouterr().out


def test_save_run_to_drive_refuses_to_overwrite_existing_destination(tmp_path):
    run_dir = tmp_path / "run_source"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text("x")

    drive_root = tmp_path / "RAG-project" / "results"
    (drive_root / "baseline_phase6").mkdir(parents=True)  # already there

    with pytest.raises(RunVerificationError, match="already exists"):
        save_run_to_drive(run_dir, str(drive_root), "baseline_phase6")


def test_save_run_to_drive_propagates_path_mismatch_from_find_canonical_root(tmp_path):
    run_dir = tmp_path / "run_source"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text("x")

    (tmp_path / "rag-project").mkdir()  # case-mismatched look-alike again
    configured_root = tmp_path / "RAG-project" / "results"

    with pytest.raises(PersistPathMismatchError):
        save_run_to_drive(run_dir, str(configured_root), "baseline_phase6")

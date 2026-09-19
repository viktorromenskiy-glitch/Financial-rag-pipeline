"""Pre-registration manifest for a paid evaluation run.

Written BEFORE the run processes a single question - unlike
pipeline/common/run_config.py's snapshot, which is written AFTER a run
finishes for traceability. This manifest exists for a different reason:
agent/demonstration.py's selection rule is formally pre-registered
(written before any real data existed), but that guarantee is only as
strong as its INPUT - nothing in the selection rule itself stops someone
from running the harness several times and committing only the
favorable-looking baseline_results.jsonl/agent_results.jsonl, or from
dropping a few inconvenient rows before committing. A manifest written at
the START of the one run that will actually be committed - before any
API call, before any answer exists - lets a downstream reader (see
verify_manifest_coverage, used by scripts/render_demonstration_cases.py)
confirm the committed result files cover exactly the question set that
was decided on beforehand, not a convenient subset or a different run
entirely.

Accepted per the Day 3 expertise's consensus (rounds 2-4 all converged on
this design after the anti-cherry-picking devil's-advocate round raised
the underlying risk) - see claude/itog_ekspertizy_den3_dizayn.md, item 10,
for the full discussion this implements.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_sha(repo_root: Path) -> str:
    """Best-effort - a manifest missing the git SHA (e.g. no git binary,
    not actually a git checkout) is still far better than no manifest at
    all, so this never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return f"unavailable ({exc})"


def _config_hash(config: dict) -> str:
    """A stable hash of the effective config - sort_keys=True makes this
    independent of dict/field ordering, so the same configuration always
    hashes the same way regardless of how it was constructed."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_run_manifest(
    *,
    run_id: str,
    config: dict,
    expected_question_ids: list[str],
    expected_canary_ids: list[str],
    repo_root: Path,
) -> dict:
    """Pure builder - no file I/O, so it stays independently testable.
    See write_run_manifest_if_absent below for the file-writing wrapper
    scripts/run_agent_eval.py actually calls.

    Args:
        run_id: Identifier for this run.
        config: The effective configuration used for this run.
        expected_question_ids: Question ids pre-registered as the ones
            this run must cover.
        expected_canary_ids: Canary question ids pre-registered for this run.
        repo_root: Path to the repository checkout, used to resolve the
            current git SHA.

    Returns:
        A dict with run_id, created_at, git_sha, config_hash,
        expected_question_ids, and expected_canary_ids - the exact shape
        written to run_manifest.json.
    """
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "config_hash": _config_hash(config),
        "expected_question_ids": sorted(expected_question_ids),
        "expected_canary_ids": sorted(expected_canary_ids),
    }


def write_run_manifest_if_absent(
    *,
    run_id: str,
    config: dict,
    expected_question_ids: list[str],
    expected_canary_ids: list[str],
    repo_root: Path,
    results_dir: str | Path = "results",
) -> Path:
    """Writes results/<run_id>/run_manifest.json - but only if it does not
    already exist.

    The whole point of a pre-registration manifest is that it is fixed
    before any real result exists. On a resumed run (this script is
    idempotent/resumable - see its module docstring), run_manifest.json
    already exists from the first invocation; overwriting it here would
    silently defeat the pre-registration guarantee (a resumed run could
    otherwise "re-register" a different expected question set after
    already seeing some real answers). So a resume leaves the original
    manifest untouched and this just returns its existing path.

    Args:
        run_id: Identifier for this run.
        config: The effective configuration used for this run.
        expected_question_ids: Question ids pre-registered as the ones
            this run must cover.
        expected_canary_ids: Canary question ids pre-registered for this run.
        repo_root: Path to the repository checkout, used to resolve the
            current git SHA.
        results_dir: Root directory containing per-run result folders.

    Returns:
        The path to the manifest (freshly written, or the pre-existing one).
    """
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_manifest.json"
    if out_path.exists():
        return out_path
    manifest = build_run_manifest(
        run_id=run_id,
        config=config,
        expected_question_ids=expected_question_ids,
        expected_canary_ids=expected_canary_ids,
        repo_root=repo_root,
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out_path


def verify_manifest_coverage(
    manifest: dict,
    *,
    baseline_question_ids,
    agent_question_ids,
) -> None:
    """Checks EXACT set equality (both directions) between the manifest's
    pre-registered expected_question_ids and what's actually present in
    each committed result file.

    Deliberately not a one-sided "all expected are present" check: that
    alone would not catch extra question_ids smuggled in from a more
    favorable run (e.g. someone appends a few rows from a different,
    better-looking run before committing) - only exact equality in both
    directions closes that gap.

    Args:
        manifest: The pre-registered run manifest (as built by
            build_run_manifest).
        baseline_question_ids: question_ids actually present in
            baseline_results.jsonl.
        agent_question_ids: question_ids actually present in
            agent_results.jsonl.

    Raises:
        ValueError: on any mismatch, naming exactly which question_ids
            are missing and which are unexpected, for each file that
            doesn't match.
    """
    expected = set(manifest["expected_question_ids"])
    for label, actual_ids in (("baseline_results.jsonl", baseline_question_ids), ("agent_results.jsonl", agent_question_ids)):
        actual = set(actual_ids)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"{label} does not match the pre-registered manifest (run_manifest.json): "
                f"missing={missing!r}, extra={extra!r} - refusing to select demonstration cases from data "
                "that doesn't match what was committed to before the run"
            )

"""Save and verify paid/long-running Colab run output on persistent storage.

See "Правила сохранения долгих платных прогонов" (project doc, uploaded
2026-08-24) for the full rule this module implements. Two real incidents
on this project motivated it:

1. (2026-08-19, see docs/struktura_repozitoriya.md) A paid `pipeline.cli
   eval` run (n=250) was never copied off the ephemeral Colab runtime -
   results/<run_id>/ only ever existed on the disposable VM disk. The
   session disconnected and the run was lost, requiring a full re-run at
   real cost. This produced the first version of the rule: verify, then
   copy to Google Drive, in the same cell that produces the result.

2. A later run implementing that first rule STILL lost the plot: the
   Colab cell hardcoded the Google Drive destination path from memory,
   with a case typo, so the copy landed in a new look-alike directory
   instead of the actual results tree - and the verification step only
   checked "files exist and aren't empty at the path we wrote to", not
   "the path we wrote to is the one we meant". The bug was invisible
   until someone went looking for the run later and it wasn't where it
   was supposed to be.

This module exists so incident 2's failure mode is structurally
impossible to repeat: there is exactly one place a Colab script gets the
destination path (config.yaml's persistence.google_drive_results_dir,
read once, not retyped), and exactly one function that does the actual
copy+verify (save_run_to_drive below) - a new script imports and calls
these instead of writing its own Drive-copy cell from memory.

Usage from a Colab cell, after a run has produced results/<run_id>/:

    from pipeline.common.persist import verify_run_files, save_run_to_drive

    verify_run_files(run_dir, {"predictions.jsonl": n, "eval_results.jsonl": n})
    save_run_to_drive(run_dir, config.persistence.google_drive_results_dir, run_id)

`n` should come from actually counting the loaded question set
(e.g. `len(items)`), not a number typed from memory - the same rule
against "remembered instead of verified" applies to the expected count.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class RunVerificationError(RuntimeError):
    """Raised when a run's output fails the hard completeness check
    (rule point 7): a missing file, an empty file, or a record count
    that doesn't match what was expected. A run that raises this is not
    considered finished and must not be reported as done."""


class PersistPathMismatchError(RuntimeError):
    """Raised when the configured persistent-storage root can't be
    confirmed to exist exactly as configured (rule points 4/5) - Drive
    isn't mounted, the account is wrong, or the configured path has a
    typo relative to what's actually there. Raised instead of silently
    creating a new, possibly-duplicate directory tree - the exact
    failure mode of incident 2 above."""


def verify_run_files(run_dir: Path, expected_line_counts: dict[str, int]) -> None:
    """Hard check (rule point 7). For every (filename, expected_n) pair,
    confirms run_dir/filename exists, is non-empty, and has exactly
    expected_n lines (JSONL: one record per line, so line count is
    record count). Raises RunVerificationError - never just warns -
    naming every problem found, not just the first.
    """
    run_dir = Path(run_dir)
    problems = []
    for filename, expected_n in expected_line_counts.items():
        path = run_dir / filename
        if not path.exists():
            problems.append(f"{filename}: missing (expected at {path})")
            continue
        if path.stat().st_size == 0:
            problems.append(f"{filename}: empty (0 bytes)")
            continue
        actual_n = sum(1 for _ in path.open("r", encoding="utf-8"))
        if actual_n != expected_n:
            problems.append(f"{filename}: {actual_n} lines, expected {expected_n}")
    if problems:
        raise RunVerificationError(
            f"Run output at {run_dir} failed verification - NOT considered complete:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def find_canonical_root(configured_root: str) -> Path:
    """Rule points 4/5: before writing anything, confirm the configured
    persistent-storage root exists AS CONFIGURED - not a similarly-named
    directory standing in for it. Raises PersistPathMismatchError rather
    than silently creating a new tree at an unverified path:

    - configured_root itself exists -> return it (the success path).
    - its parent doesn't exist -> raise (Drive likely isn't mounted).
    - parent exists but has a same-name-different-case sibling instead
      of the exact configured name -> raise, naming the sibling. This is
      incident 2's exact bug, caught instead of repeated.
    - parent exists, nothing similar found -> raise and say so; creating
      the root itself is a one-time manual action per the rule, not
      something this function does on a script's behalf.
    """
    root = Path(configured_root)
    if root.is_dir():
        return root

    parent = root.parent
    if not parent.is_dir():
        raise PersistPathMismatchError(
            f"Persistent-storage parent directory does not exist: {parent} "
            f"(configured root: {configured_root}). Is Google Drive mounted "
            f"(drive.mount('/content/drive'))? Stopping rather than creating "
            f"a new directory tree at an unverified path."
        )

    near_matches = [
        child.name
        for child in parent.iterdir()
        if child.is_dir() and child.name.lower() == root.name.lower() and child.name != root.name
    ]
    if near_matches:
        raise PersistPathMismatchError(
            f"Configured root {configured_root!r} does not exist, but a "
            f"similarly-named directory does: {near_matches!r} under {parent}. "
            f"This is the exact case-mismatch failure mode this function "
            f"exists to catch - fix the configured constant or the actual "
            f"directory name, don't create a second near-duplicate. Stopping."
        )

    raise PersistPathMismatchError(
        f"Configured persistent-storage root does not exist, and nothing "
        f"similarly named was found under {parent}: {configured_root!r}. "
        f"Create it once manually and confirm the exact path (update "
        f"config.yaml's persistence.google_drive_results_dir if it was "
        f"wrong) before re-running - this function will not create it "
        f"silently."
    )


def save_run_to_drive(run_dir: Path, drive_results_root: str, run_id: str) -> Path:
    """The one reusable save function (rule points 2/3/5/6). Call this at
    the end of every paid/long Colab run instead of writing a new ad hoc
    Drive-copy cell.

    Does NOT run the completeness check itself - call verify_run_files()
    on run_dir first. This function assumes the source is already
    known-good and handles copying it to durable storage and confirming
    the copy landed where it was supposed to:

    1. find_canonical_root() confirms drive_results_root exists exactly
       as configured (rule 4/5) - raises rather than creating a look-alike.
    2. Copies run_dir's contents to <drive_results_root>/<run_id>/.
       Refuses to overwrite an existing destination (a second run
       shouldn't silently clobber a first run's saved results).
    3. Resolves the copy's absolute path and asserts it is actually
       inside the resolved canonical root - rule 5's "verify the path
       itself, not just the content at some path".
    4. Prints the final absolute path loudly (rule 6) so a human
       skimming the run's output sees it without hunting for it.

    Returns the resolved destination path.
    """
    run_dir = Path(run_dir)
    root = find_canonical_root(drive_results_root)
    dest = root / run_id

    if dest.exists():
        raise RunVerificationError(
            f"Destination already exists - refusing to overwrite a previous "
            f"run's saved output: {dest}"
        )

    shutil.copytree(run_dir, dest)

    resolved_dest = dest.resolve()
    resolved_root = root.resolve()
    if resolved_root != resolved_dest.parent and resolved_root not in resolved_dest.parents:
        raise PersistPathMismatchError(
            f"Copy landed outside the canonical root after resolving symlinks - "
            f"expected under {resolved_root}, got {resolved_dest}. Not trusting "
            f"this save; investigate before relying on it."
        )

    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {resolved_dest}\n{'=' * 70}\n")
    return resolved_dest

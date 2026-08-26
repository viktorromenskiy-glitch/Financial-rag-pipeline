"""Diagnostic: what state is THIS Colab runtime in right now - run this
FIRST after reconnecting, before deciding whether a full rebootstrap
(mount Drive -> clone/pull repo -> pip install -> rebuild .env from
Colab Secrets) is needed, or whether it's safe to skip straight to
`!git pull` and re-running a script.

Motivation (2026-08-26): reevaluate_phase6_adaptive.py's Colab runtime
died mid-run (KeyboardInterrupt during a 750-item paid re-evaluation).
On reconnecting it was not obvious whether the same runtime had
reattached (packages/.env still in place - just `git pull` and go) or a
fresh VM had been assigned (everything under /content wiped; only the
Drive mount survives). Guessing wrong costs time either way: assuming
"fresh" when it wasn't means a needless pip install; assuming "same"
when it wasn't means the next script crashes on a missing package or
unset env var, discovered only after it's already re-read
predictions.jsonl etc.

Deliberately does NOT try to answer "is this literally the same VM
instance as before" - Colab exposes no reliable way to check that from a
plain script, and a low /proc/uptime is not evidence of that, it's
folklore (a Colab "restart runtime" without a new VM assignment would
also reset uptime, and there's no way to tell those two apart from
here). Instead this checks each specific piece of state a script in this
repo actually depends on, and reports readiness in terms of THOSE
checks, not a guess about VM identity.

Zero cost - no paid Voyage/Cohere/Anthropic calls, no MongoDB queries.
The only network call is an optional `git fetch` (read-only), skipped
without failing the whole script if there's no network yet. Does NOT
require the repo's own third-party packages (pymongo/anthropic/
pydantic/pyyaml/...) to already be installed - if they aren't, that is
one of the things this script reports, not something it crashes on.
(It only needs the standard library plus, if present, `pyyaml` +
`pydantic` for a more precise config read - see step 5.)

The fact that this script itself is runnable already proves the repo is
cloned onto this runtime. If `%cd Financial-rag-pipeline` (or wherever
the clone lives) fails, or `!python scripts/check_colab_runtime.py`
fails with "No such file or directory", that alone already answers the
question: this is a fresh runtime, do the full rebootstrap - there is
nothing yet for this script to check.

Usage (Colab, after mounting Drive and %cd-ing into the repo):
    !python scripts/check_colab_runtime.py
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_MOUNT = Path("/content/drive/MyDrive")
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
REQUIRED_ENV_KEYS = ["MONGODB_URI", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"]
# import-name != pip-name for a couple of these (python-dotenv -> dotenv, pyyaml -> yaml)
REQUIRED_PACKAGES = {
    "pymongo": "pymongo",
    "voyageai": "voyageai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "tenacity": "tenacity",
    "pydantic": "pydantic",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
}
# Known Drive-resident run directories from past/current scripts in this
# repo - checked here read-only, never written to. New scripts that add a
# new RUN_ID should be added to this list too, so this diagnostic stays
# useful for them.
KNOWN_RUNS = {
    "reliability_pilot_track_a": ["raw_draws.jsonl", "pilot_summary.jsonl"],
    "phase6_reeval_adaptive": ["raw_draws.jsonl", "reeval_summary.jsonl"],
}

problems: list[str] = []
warnings: list[str] = []


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


print("=" * 70)
print("Colab runtime check - what's already usable on THIS runtime, right now")
print("=" * 70)

# --- 1. Repo state (this script running at all already proves the repo exists here) ---
print(f"\n[1/6] Repo root: {REPO_ROOT}")
git_dir = REPO_ROOT / ".git"
if not git_dir.is_dir():
    problems.append("repo root has no .git - this isn't the cloned repo, or the clone is broken")
    print("      NOT a git repo (no .git/) - unexpected, treat this like a fresh clone")
else:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        print(f"      branch={branch} commit={commit}" + ("  (local uncommitted changes present)" if dirty else ""))
        if dirty:
            warnings.append("local uncommitted changes in the repo - unexpected on a Colab clone, check before trusting results")

        fetch = subprocess.run(
            ["git", "fetch", "origin", branch or "main"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20
        )
        if fetch.returncode == 0:
            behind = subprocess.run(
                ["git", "rev-list", "--count", f"HEAD..origin/{branch or 'main'}"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if behind.isdigit() and int(behind) > 0:
                problems.append(f"local repo is {behind} commit(s) behind origin/{branch or 'main'} - run `git pull`")
                print(f"      BEHIND origin/{branch or 'main'} by {behind} commit(s) - `git pull` needed")
            elif behind == "0":
                print(f"      up to date with origin/{branch or 'main'}")
        else:
            warnings.append("`git fetch` failed (no network yet, or offline) - could not check for `git pull` needed")
            print("      git fetch failed (no network yet?) - skipping up-to-date check")
    except Exception as e:  # noqa: BLE001 - this whole block is best-effort diagnostics, never fatal
        warnings.append(f"git checks raised {e!r} - skipped")
        print(f"      git checks raised {e!r} - skipped (not fatal)")

# --- 2. Drive mount ---
print("\n[2/6] Google Drive mount")
drive_mounted = DRIVE_MOUNT.is_dir()
if drive_mounted:
    print(f"      OK - {DRIVE_MOUNT} exists")
else:
    problems.append(f"Drive not mounted at {DRIVE_MOUNT} - run drive.mount('/content/drive') first")
    print(f"      NOT mounted - {DRIVE_MOUNT} does not exist")

# --- 3. Installed packages ---
print("\n[3/6] Third-party packages (from requirements.txt)")
missing_packages = [pip_name for pip_name, import_name in REQUIRED_PACKAGES.items() if importlib.util.find_spec(import_name) is None]
if missing_packages:
    problems.append(f"pip install -r requirements.txt needed - missing: {', '.join(missing_packages)}")
    print(f"      MISSING: {', '.join(missing_packages)}")
else:
    print("      OK - all required packages importable")

# --- 4. .env ---
print("\n[4/6] .env")
if not ENV_PATH.is_file():
    problems.append(".env missing - rebuild it from Colab Secrets")
    print(f"      MISSING - {ENV_PATH} does not exist")
else:
    env_text = ENV_PATH.read_text(encoding="utf-8")
    present_keys = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", env_text, flags=re.MULTILINE))
    missing_keys = [k for k in REQUIRED_ENV_KEYS if k not in present_keys]
    if missing_keys:
        problems.append(f".env exists but is missing key(s): {', '.join(missing_keys)}")
        print(f"      .env exists but missing: {', '.join(missing_keys)} (values never printed)")
    else:
        print("      OK - all 4 required keys present (values not checked/printed - only that the keys exist)")

# --- 5. Configured Drive results root (read from config.yaml, not retyped from memory) ---
print("\n[5/6] Configured persistent-storage root (config.yaml -> persistence.google_drive_results_dir)")
configured_root: str | None = None
try:
    sys.path.insert(0, str(REPO_ROOT))
    # load_config() reads ${MONGODB_URI} etc. from os.environ, not from
    # .env directly - same recurring bug as everywhere else in this repo
    # that calls load_config()/build_clients() without going through
    # pipeline.cli.main() (which calls load_dotenv() itself). Without this,
    # step 5 would fail with a misleading "environment variable not set"
    # even when step 4 above just confirmed .env has all 4 keys.
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    from config.config_schema import load_config  # needs pyyaml + pydantic

    configured_root = load_config(str(CONFIG_PATH)).persistence.google_drive_results_dir
    print(f"      (via config_schema) {configured_root}")
except Exception as e:  # noqa: BLE001 - packages may not be installed yet; fall back to a raw text read
    if CONFIG_PATH.is_file():
        m = re.search(r'google_drive_results_dir:\s*"([^"]+)"', CONFIG_PATH.read_text(encoding="utf-8"))
        if m:
            configured_root = m.group(1)
            print(f"      (via raw text read, config_schema unavailable: {e!r}) {configured_root}")
    if configured_root is None:
        warnings.append(f"could not determine google_drive_results_dir ({e!r}) - skipping Drive progress check")
        print(f"      could not read it ({e!r}) - skipping step 6")

# --- 6. Drive-resident progress from known runs (the part that actually answers "will resuming pick up where I left off?") ---
print("\n[6/6] Existing progress already on Drive (for known runs)")
if drive_mounted and configured_root:
    root = Path(configured_root)
    if not root.is_dir():
        warnings.append(f"configured root {configured_root} not found under Drive as-is - see find_canonical_root() rules if a script fails on this")
        print(f"      configured root does not exist yet as such: {configured_root}")
    else:
        any_found = False
        for run_id, files in KNOWN_RUNS.items():
            run_dir = root / run_id
            if not run_dir.is_dir():
                continue
            any_found = True
            counts = []
            for fname in files:
                fpath = run_dir / fname
                if fpath.is_file():
                    counts.append(f"{fname}={_count_lines(fpath)} lines")
                else:
                    counts.append(f"{fname}=absent")
            print(f"      {run_id}: {', '.join(counts)}")
        if not any_found:
            print("      no known run directories found yet under the configured root (nothing to resume)")
else:
    print("      skipped (Drive not mounted or configured root unknown)")

# --- Verdict ---
print("\n" + "=" * 70)
if problems:
    print("NOT READY - fix these before running anything paid:")
    for p in problems:
        print(f"  - {p}")
else:
    print("READY - repo up to date, Drive mounted, packages installed, .env complete.")
    print("Safe to go straight to running the script (no rebootstrap needed).")
if warnings:
    print("\nWarnings (not blocking, but worth a look):")
    for w in warnings:
        print(f"  - {w}")
print("=" * 70)

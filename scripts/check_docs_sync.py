#!/usr/bin/env python3
"""Check that headline numbers in README.md / README.ru.md / README.uk.md match
the underlying data in results/*.json(l) and code constants.

Reads scripts/docs_sync_manifest.json. For every check whose type is
code_constant / json_field / aggregation, computes the true value from the
repository, extracts the value each README currently states (via the check's
per-file anchor regex), normalizes both sides, and compares.

Checks of type manual_external are never compared automatically — they are
listed in the manifest for audit visibility only (external prices, comparisons
to other papers, qualitative judgments). A check whose anchor for a given file
is null is reported as PENDING (not wired to real README text yet) rather than
failed or skipped silently, so that gap stays visible instead of looking like a
pass.

Exit code: 0 if every wired (non-null anchor, non-manual_external) check passes
in every README file that declares an anchor for it; 1 otherwise.

This script deliberately does NOT run pytest or touch CI status itself — see
claude/pravilo_b_dokumentaciya_final.md for why docs-sync must be an
independent CI job rather than a step chained after the test job.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "scripts" / "docs_sync_manifest.json"
README_FILES = ["README.md", "README.ru.md", "README.uk.md"]


class CheckError(Exception):
    """Raised when a check's source data cannot be computed at all (missing
    file, missing field, malformed constant) — distinct from a value mismatch,
    which is a normal (expected, sometimes) check failure rather than a bug in
    the checker itself."""


# ---------------------------------------------------------------------------
# Value computation (the "truth" side)
# ---------------------------------------------------------------------------

def _read_json(rel_path: str) -> Any:
    path = REPO_ROOT / rel_path
    if not path.exists():
        raise CheckError(f"file not found: {rel_path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(rel_path: str) -> list[dict]:
    path = REPO_ROOT / rel_path
    if not path.exists():
        raise CheckError(f"file not found: {rel_path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_field(obj: Any, dotted_path: str) -> Any:
    cur = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise CheckError(f"field path not found: {dotted_path} (missing at '{part}')")
        cur = cur[part]
    return cur


def compute_code_constant(source: dict) -> int:
    """Reads an integer module-level constant's value out of a source file.

    Args:
        source: The check's "source" dict, with "file" (path relative to
            the repo root) and "constant" (the constant's name) keys.

    Returns:
        The constant's current integer value.

    Raises:
        CheckError: If the file does not exist, or the constant is not
            found as a top-level `NAME = <int>` assignment in it.
    """
    path = REPO_ROOT / source["file"]
    if not path.exists():
        raise CheckError(f"file not found: {source['file']}")
    text = path.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(source['constant'])}\s*=\s*(\d+)\s*$"
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise CheckError(f"constant not found: {source['constant']} in {source['file']}")
    return int(m.group(1))


def compute_json_field(source: dict) -> Any:
    """Reads one dotted-path field out of a JSON file.

    Args:
        source: The check's "source" dict, with "file" (path relative to
            the repo root) and "field_path" (dotted field path) keys.

    Returns:
        The value at that field path.
    """
    obj = _read_json(source["file"])
    return _get_field(obj, source["field_path"])


def compute_aggregation(source: dict) -> Any:
    """Computes an aggregate value over a results file, per the check's declared op.

    Supported ops: "rate" (fraction of JSONL rows with a truthy field,
    optionally filtered by question_id_prefix), "min"/"max"/"sum" (over a
    list of JSON fields), "count_not_equal" (JSONL rows whose field differs
    from a given value), "dict_field" (a nested JSON field), and
    "join_compare" (agreement/discordance between two JSONL files joined
    on a key).

    Args:
        source: The check's "source" dict; its keys depend on "op" (see
            the manifest schema for each op's required keys).

    Returns:
        The aggregated value - a float for "rate", an int for the others,
        or a dict of counts for "join_compare".

    Raises:
        CheckError: If "op" is unrecognized, or a "rate" aggregation's
            filter matches zero rows.
    """
    op = source["op"]

    if op == "rate":
        rows = _read_jsonl(source["file"])
        flt = source.get("filter")
        if flt and "question_id_prefix" in flt:
            prefix = flt["question_id_prefix"]
            rows = [r for r in rows if str(r.get("question_id", "")).startswith(prefix)]
        if not rows:
            raise CheckError("rate aggregation over zero rows (filter matched nothing)")
        true_count = 0
        for r in rows:
            try:
                val = _get_field(r, source["field"])
            except CheckError:
                val = False
            if val is True:
                true_count += 1
        return true_count / len(rows)

    if op in ("min", "max", "sum"):
        obj = _read_json(source["file"])
        values = [_get_field(obj, fp) for fp in source["fields"]]
        if op == "min":
            return min(values)
        if op == "max":
            return max(values)
        return sum(values)

    if op == "count_not_equal":
        rows = _read_jsonl(source["file"])
        return sum(1 for r in rows if _get_field(r, source["field"]) != source["value"])

    if op == "dict_field":
        obj = _read_json(source["file"])
        return _get_field(obj, source["field_path"])

    if op == "join_compare":
        rows_a = _read_jsonl(source["file_a"])
        rows_b = _read_jsonl(source["file_b"])
        key = source["join_key"]
        by_key_b = {r[key]: r for r in rows_b if key in r}
        matched = 0
        discordant = 0
        pairs = 0
        for ra in rows_a:
            k = ra.get(key)
            if k is None or k not in by_key_b:
                continue
            rb = by_key_b[k]
            pairs += 1
            va = ra.get(source["field_a"])
            vb = rb.get(source["field_b"])
            if bool(va) == bool(vb):
                matched += 1
            else:
                discordant += 1
        return {
            "pairs": pairs,
            "agree": matched,
            "discordant": discordant,
            "a_true_count": sum(1 for r in rows_a if r.get(source["field_a"]) is True),
            "b_true_count": sum(1 for r in rows_b if r.get(source["field_b"]) is True),
        }

    raise CheckError(f"unknown aggregation op: {op}")


def compute_true_value(check: dict) -> Any:
    """Dispatches a manifest check to the right compute_* function by its type.

    Args:
        check: One check entry from the manifest, with a "type" key of
            "code_constant", "json_field" or "aggregation".

    Returns:
        The computed true value, in whatever shape that check type produces.

    Raises:
        CheckError: If the check's type is not one of the computable types.
    """
    ctype = check["type"]
    if ctype == "code_constant":
        return compute_code_constant(check["source"])
    if ctype == "json_field":
        return compute_json_field(check["source"])
    if ctype == "aggregation":
        return compute_aggregation(check["source"])
    raise CheckError(f"compute_true_value called on non-computable type: {ctype}")


# ---------------------------------------------------------------------------
# Normalization — both the computed truth and the README-extracted string are
# passed through the same normalizer before comparing, so "23,088" (English),
# "23 088" (Russian/Ukrainian, plain space or nbsp) and a raw int 23088 all
# collapse to the same comparable value.
# ---------------------------------------------------------------------------

def normalize(kind: str, value: Any) -> Any:
    """Normalizes a computed or README-extracted value into a comparable form.

    Args:
        kind: Normalization kind: "int_strip_separators", "percent_1dp",
            "percent_0dp_from_fraction", or "dict_exact".
        value: The raw value to normalize (a number or a README-extracted string).

    Returns:
        The normalized value, comparable across both sides of a check.

    Raises:
        CheckError: If kind is not one of the recognized normalization kinds.
    """
    if kind == "int_strip_separators":
        if isinstance(value, (int, float)):
            return int(value)
        s = re.sub(r"[,\s ]", "", str(value))
        return int(s)

    if kind == "percent_1dp":
        if isinstance(value, (int, float)):
            pct = value * 100 if value <= 1.0 else value
            return round(pct, 1)
        s = str(value).replace(",", ".").strip()
        return round(float(s), 1)

    if kind == "percent_0dp_from_fraction":
        if isinstance(value, (int, float)):
            pct = value * 100 if value <= 1.0 else value
            return round(pct)
        s = str(value).replace(",", ".").strip().rstrip("%")
        return round(float(s))

    if kind == "dict_exact":
        return value

    raise CheckError(f"unknown normalize kind: {kind}")


# ---------------------------------------------------------------------------
# README extraction (the "claimed" side)
# ---------------------------------------------------------------------------

def extract_from_readme(readme_text: str, anchor_pattern: str) -> str | None:
    """Extracts a check's claimed value from a README's text via its anchor regex.

    Args:
        readme_text: Full text of one README file.
        anchor_pattern: Regex with one capture group locating the claimed value.

    Returns:
        The captured value string, or None if the anchor pattern did not match.
    """
    m = re.search(anchor_pattern, readme_text)
    if not m:
        return None
    return m.group(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Runs every wired manifest check against each README file and prints the results.

    Returns:
        0 if every wired check passes in every README that declares an
        anchor for it, 1 if any check fails or errors.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    readme_texts = {}
    for fname in README_FILES:
        p = REPO_ROOT / fname
        readme_texts[fname] = p.read_text(encoding="utf-8") if p.exists() else None

    failures: list[str] = []
    pending: list[str] = []
    manual: list[str] = []
    passed: list[str] = []
    errors: list[str] = []

    for check in manifest["checks"]:
        check_id = check["id"]

        if check["type"] == "manual_external":
            manual.append(check_id)
            continue

        try:
            true_value = compute_true_value(check)
        except CheckError as e:
            errors.append(f"{check_id}: could not compute true value — {e}")
            continue

        # "components" checks (error_attribution_overall, agent_vs_baseline_join):
        # true_value is a dict/object with no single-token README representation,
        # so each named field is compared against its own per-file anchor instead
        # of trying to match the whole object with one regex.
        if "components" in check:
            for component in check["components"]:
                comp_label_base = f"{check_id}.{component['key']}"
                try:
                    comp_true = _get_field(true_value, component["field"]) if isinstance(true_value, dict) else true_value[component["field"]]
                except (CheckError, KeyError, TypeError) as e:
                    errors.append(f"{comp_label_base}: could not read component field — {e}")
                    continue
                comp_norm_kind = component.get("normalize")
                try:
                    comp_true_norm = normalize(comp_norm_kind, comp_true) if comp_norm_kind else comp_true
                except CheckError as e:
                    errors.append(f"{comp_label_base}: could not normalize true value — {e}")
                    continue
                for fname, anchor in component.get("anchors", {}).items():
                    label = f"{comp_label_base} [{fname}]"
                    if anchor is None:
                        pending.append(label)
                        continue
                    text = readme_texts.get(fname)
                    if text is None:
                        errors.append(f"{label}: README file not found")
                        continue
                    raw = extract_from_readme(text, anchor)
                    if raw is None:
                        errors.append(f"{label}: anchor regex did not match — README wording may have changed")
                        continue
                    try:
                        claimed_norm = normalize(comp_norm_kind, raw) if comp_norm_kind else raw
                    except CheckError as e:
                        errors.append(f"{label}: could not normalize README value '{raw}' — {e}")
                        continue
                    if claimed_norm == comp_true_norm:
                        passed.append(label)
                    else:
                        failures.append(
                            f"{label}: README says {raw!r} (normalized {claimed_norm!r}), "
                            f"actual data gives {comp_true!r} (normalized {comp_true_norm!r})"
                        )
            continue

        normalize_kind = check.get("normalize")
        try:
            true_norm = normalize(normalize_kind, true_value) if normalize_kind else true_value
        except CheckError as e:
            errors.append(f"{check_id}: could not normalize true value — {e}")
            continue

        for fname, anchor in check.get("anchors", {}).items():
            label = f"{check_id} [{fname}]"
            if anchor is None:
                pending.append(label)
                continue
            text = readme_texts.get(fname)
            if text is None:
                errors.append(f"{label}: README file not found")
                continue
            raw = extract_from_readme(text, anchor)
            if raw is None:
                errors.append(f"{label}: anchor regex did not match — README wording may have changed")
                continue
            try:
                claimed_norm = normalize(normalize_kind, raw) if normalize_kind else raw
            except CheckError as e:
                errors.append(f"{label}: could not normalize README value '{raw}' — {e}")
                continue
            if claimed_norm == true_norm:
                passed.append(label)
            else:
                failures.append(
                    f"{label}: README says {raw!r} (normalized {claimed_norm!r}), "
                    f"actual data gives {true_value!r} (normalized {true_norm!r})"
                )

    print(f"docs-sync: {len(passed)} passed, {len(failures)} FAILED, "
          f"{len(errors)} error(s), {len(pending)} pending anchor(s), "
          f"{len(manual)} manual_external (not auto-checked)")

    if passed:
        print("\nPASSED:")
        for p in passed:
            print(f"  ok  {p}")

    if pending:
        print("\nPENDING (no anchor wired yet — not a failure, but not verified either):")
        for p in pending:
            print(f"  ..  {p}")

    if manual:
        print("\nMANUAL_EXTERNAL (never auto-checked, listed for audit):")
        for m in manual:
            print(f"  --  {m}")

    if errors:
        print("\nERRORS (could not evaluate — treat as failing until fixed):")
        for e in errors:
            print(f"  !!  {e}")

    if failures:
        print("\nFAILURES (real numeric mismatch between README and data):")
        for f in failures:
            print(f"  XX  {f}")

    if failures or errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

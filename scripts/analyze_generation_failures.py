"""Plan generation-error-analysis, Phase 3: get the full reasoning trace
(raw_response) for the residual pool of generation_failure_candidate
questions that Phases 1-2 did not already explain.

This is the FIRST PAID STEP of the plan (~$0.05-0.10 for ~44 questions,
Claude Sonnet 5 generation calls only - see docs/tehnicheskoe_zadanie.md,
Phase 3 write-up). Everything before this (Phases 0-2) was free/local.

Why this step exists at all: `raw_response` (the model's full text, before
_extract_final_answer() pulls out just the "FINAL ANSWER: <value>" line) was
never persisted by the original results/error_analysis_250 run - only the
extracted answer_text survives there. Without the reasoning text there is no
way to tell WHY a wrong numeric answer is wrong (wrong table row? wrong
period? arithmetic slip? sign convention?) for the residual pool - Phase 4
needs this file to categorize them.

Residual pool (computed dynamically here, not hardcoded - see
_residual_question_ids() below): every question_id currently classified
generation_failure_candidate in results/retrieval_trace_250/
attribution_results.jsonl, EXCEPT the ones Phase 1 or Phase 2 already
manually confirmed as a specific, understood generation error (reason ==
"confirmed_generation_error" - tatqa_train_8832: table-column misread;
tatqa_train_4256: over-cautious refusal on an unambiguous number) - those
two don't need a reasoning trace, their mechanism is already documented in
phase1_manual_corrections.jsonl / phase2_manual_corrections.jsonl. This
computation is intentionally live against the current attribution_results.jsonl
rather than a fixed list, so re-running this script after any future
Phase 1/2-style correction automatically picks up the right residual set.

Context reproduction (plan's explicit caveat, repeated here in code - do
not remove this caveat when editing): this replays generate_answer() on the
SAME top-5 context_ids already recorded in
results/retrieval_trace_250/retrieval_trace.jsonl's `reranked_top5` (21
August run), fetching each context_id's full_indexed_content fresh from
MongoDB and verifying it against the recorded content_sha256. This is NOT
guaranteed to be bit-identical to what the model saw in the ORIGINAL
results/error_analysis_250 run (19-20 August) - that run predates
content_sha256 logging entirely, so there is nothing to check it against.
Corpus drift between the two runs is considered unlikely (both runs saw
7318/7318 documents, config unchanged) but this is an assumption, not
proof - every output record below carries a `content_verified` flag (True
if this run's fetched content_sha256 matches the recorded one, False with a
`content_mismatch_note` if not) so Phase 4 can see explicitly, per
question, whether context reproduction actually held.

Checkpointed (tehnicheskoe_zadanie.md section 11's mandatory rule for any
paid API run: verify and persist each result in the same step that produces
it, not accumulated in memory - a run that already lost 50 paid generations
once to a Colab disconnect is exactly the failure this guards against).
Re-running with the same output path resumes; already-written question_ids
are skipped, not re-billed.

Usage (Colab, after step 1 - scripts/check_environment.py - has passed):
    !python scripts/analyze_generation_failures.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `pipeline`/`config` imports after moving into scripts/

from config.config_schema import load_config
from pipeline.cli import ClaudeGenerator, build_clients
from pipeline.generation import generate_answer

RUN_DIR = Path("results/retrieval_trace_250")
RETRIEVAL_TRACE_PATH = RUN_DIR / "retrieval_trace.jsonl"
ATTRIBUTION_RESULTS_PATH = RUN_DIR / "attribution_results.jsonl"
PREDICTIONS_PATH = Path("results/error_analysis_250/predictions.jsonl")
OUTPUT_PATH = RUN_DIR / "generation_failure_traces.jsonl"

ALREADY_EXPLAINED_REASON = "confirmed_generation_error"


class _StoredCandidate:
    """Stand-in for pipeline.reranking.RerankedCandidate, built from a
    retrieval_trace.jsonl record + a fresh MongoDB fetch instead of a live
    rerank() call. generate_answer()'s build_context_block() only reads
    .full_indexed_content off each candidate (see pipeline/generation.py),
    so this narrower object is sufficient - no relevance_score field is
    needed downstream, unlike the real RerankedCandidate."""

    def __init__(self, context_id: str, full_indexed_content: str):
        self.context_id = context_id
        self.full_indexed_content = full_indexed_content


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _residual_question_ids() -> list[str]:
    """generation_failure_candidate question_ids minus the ones Phase 1/2
    already fully explained (see module docstring). Order follows
    attribution_results.jsonl's own order (question_id order of the
    original 250-question eval), for a deterministic, reproducible run
    order."""
    residual = []
    for r in _load_jsonl(ATTRIBUTION_RESULTS_PATH):
        if r["failure_stage"] != "generation_failure_candidate":
            continue
        mc1 = r.get("manual_correction")
        mc2 = r.get("manual_correction_phase2")
        if mc1 is not None and mc1["reason"] == ALREADY_EXPLAINED_REASON:
            continue
        if mc2 is not None and mc2["reason"] == ALREADY_EXPLAINED_REASON:
            continue
        residual.append(r["question_id"])
    return residual


def _load_reranked_top5_by_qid() -> dict[str, list[dict]]:
    return {r["question_id"]: r["reranked_top5"] for r in _load_jsonl(RETRIEVAL_TRACE_PATH)}


def _load_predictions_by_qid() -> dict[str, dict]:
    return {r["question_id"]: r for r in _load_jsonl(PREDICTIONS_PATH)}


def _load_done() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    return {r["question_id"] for r in _load_jsonl(OUTPUT_PATH)}


def _append(record: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fetch_candidates(collection, top5: list[dict]) -> tuple[list[_StoredCandidate], bool, list[str]]:
    """Fetches full_indexed_content for each of top5's context_ids from
    MongoDB, in rank order, and checks it against the recorded
    content_sha256. Returns (candidates, all_verified, mismatch_notes)."""
    candidates = []
    all_verified = True
    notes = []
    for entry in sorted(top5, key=lambda e: e["rank"]):
        context_id = entry["context_id"]
        doc = collection.find_one({"context_id": context_id}, {"full_indexed_content": 1})
        if doc is None or "full_indexed_content" not in doc:
            all_verified = False
            notes.append(f"{context_id}: not found in MongoDB (was it re-ingested under a different id?)")
            continue
        content = doc["full_indexed_content"]
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != entry["content_sha256"]:
            all_verified = False
            notes.append(f"{context_id}: content_sha256 mismatch (recorded {entry['content_sha256'][:12]}..., now {actual_hash[:12]}...)")
        candidates.append(_StoredCandidate(context_id, content))
    return candidates, all_verified, notes


def main() -> None:
    residual = _residual_question_ids()
    done = _load_done()
    remaining = [qid for qid in residual if qid not in done]
    print(f"Residual pool: {len(residual)} question_ids. Already done (resumed from checkpoint): {len(done & set(residual))}. Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do - all residual question_ids already have a trace recorded.")
        return

    config = load_config()
    clients = build_clients(config)
    collection = clients["collection"]
    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)

    trace_by_qid = _load_reranked_top5_by_qid()
    pred_by_qid = _load_predictions_by_qid()

    missing_trace = [qid for qid in remaining if qid not in trace_by_qid]
    if missing_trace:
        raise ValueError(
            f"{len(missing_trace)} residual question_id(s) have no retrieval_trace.jsonl record - "
            f"trace file / attribution file mismatch? First few: {missing_trace[:5]}"
        )

    for i, qid in enumerate(remaining, start=1):
        pred = pred_by_qid[qid]
        top5 = trace_by_qid[qid]
        candidates, verified, notes = _fetch_candidates(collection, top5)
        if not candidates:
            print(f"  [{i}/{len(remaining)}] {qid}: SKIPPED - no context_id resolved from MongoDB at all")
            continue

        answer = generate_answer(generator, qid, pred["question"], candidates)

        record = {
            "question_id": qid,
            "question": pred["question"],
            "source_dataset": pred["source_dataset"],
            "gold_answer": pred["gold_answer"],
            "original_answer_text": pred["answer_text"],  # from error_analysis_250, for comparison
            "replay_answer_text": answer.answer_text,
            "raw_response": answer.raw_response,
            "content_verified": verified,
            "content_mismatch_notes": notes,  # empty list when verified
        }
        _append(record)
        status = "OK" if verified else "CONTEXT-DRIFT-WARNING"
        print(f"  [{i}/{len(remaining)}] {qid}: {status} (replay_answer={answer.answer_text!r})")

    all_records = _load_jsonl(OUTPUT_PATH)
    n_drifted = sum(1 for r in all_records if r["question_id"] in remaining and not r["content_verified"])
    print(f"\nDone. {len(remaining)} traces written to {OUTPUT_PATH} ({n_drifted} with content_verified=False).")
    print("Verify the file before closing this Colab session (tehnicheskoe_zadanie.md section 11): "
          f"wc -l {OUTPUT_PATH}, and check content_verified==False counts before trusting every trace as reproduced context.")


if __name__ == "__main__":
    main()

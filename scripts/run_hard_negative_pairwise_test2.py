"""Priority 2 (claude/plan_prioritet2_hard_negatives.md), Test 2: forced
pairwise reranking against financial hard negatives.

**This is the paid step.** Test 1 (scripts/analyze_hard_negative_stress.py,
already committed) checked, for free, whether each question's deterministic
hard-negative sibling (same company/other year; same company+year/other
document; same sector/other company - see that script's docstring for the
exact selection rule) already competed against the gold document inside the
*naturally observed* candidate_top50/reranked_top5 for that question. This
script asks a different, stricter question: forced head-to-head, with
nothing else in the candidate list, does Cohere Rerank v4.0 Pro still rank
the gold document above the hard-negative sibling? This is the same
"pairwise accuracy under curated hard negatives" design FinRank
(arXiv:2608.07400) uses - a real Cohere Rerank call with exactly two
candidates, `Candidate(gold)` and `Candidate(sibling)`, `top_n=2`, nothing
else in the batch.

**Design revision found during self-review, before running anything
(2026-08-29):** the original plan (claude/plan_prioritet2_hard_negatives.md)
proposed calling this only for pairs where the sibling was NOT naturally in
candidate_top50 ("already covered by Test 1 otherwise"). Rechecking that
reasoning found two problems, both fixed here:

1. "Naturally covered" undercounted the real gap. A pair where the sibling
   WAS in candidate_top50 but BOTH the gold document and the sibling missed
   the saved reranked_top5 has an UNKNOWN pairwise order - retrieval_trace.jsonl
   only stores the top-5 post-rerank slice, not full ranks 6-50, so we don't
   know which of the two Cohere actually preferred. Checked directly against
   this project's committed data: this only changes 1 of 571 pairs (see
   claude/test1_hard_negative_stress_results.md's follow-up note) - small in
   practice, but the earlier reasoning was wrong in principle, not just by
   a rounding error, so it is not reused here.
2. Reading pairwise order off a 50-candidate natural batch and reading it
   off an isolated 2-candidate forced call are two different measurement
   conditions - the project's own stated principle elsewhere (README/TZ:
   "не выдавать расчётное/разнородное за измеренное в одном тесте") argues
   against silently merging them into one "pairwise accuracy" number. Cohere
   Rerank is assumed to score each (query, document) pair independently of
   the rest of the batch (standard cross-encoder rerank behaviour, not
   confirmed against undocumented Cohere internals - flagged as an
   assumption, same treatment as the "search unit" pricing assumption in
   tehnicheskoe_zadanie.md section 15) - if that assumption holds, the two
   conditions should agree; if it doesn't, that disagreement is itself an
   interesting, previously-unmeasured finding. Either way, this script
   tests EVERY pair uniformly (not just the previously-defined "gap") so
   the per-category pairwise-accuracy number is internally consistent and
   directly comparable to FinRank's, and --summarize below reports the
   agreement rate against Test 1's natural-batch order wherever Test 1's
   order happens to be independently known, as a free sanity check on the
   pointwise-scoring assumption.

Cost: 571 pairs total (126 category A + 246 category B + 199 category C,
counted directly from the already-committed results/hard_negative_stress_250/
stress_analysis.json) x $0.0025/call, the same "1 rerank() call = 1 search
unit" assumption already used and flagged as unconfirmed in
tehnicheskoe_zadanie.md section 15 -> **~$1.43 total, upper bound** (if the
true unit is per-document rather than per-call, at most 2x that, ~$2.86 -
either way negligible; run with --dry-run first to see the exact count
before spending anything).

Requires a real MongoDB Atlas connection (to fetch full_indexed_content for
the sibling documents, which were never part of the original retrieval
trace) and a real Cohere API key - same as every other paid script in this
project, this runs in Colab with .env populated, not in the sandbox that
wrote it. Checkpointed/resumable by (question_id, category), per
docs/svod_pravil_raboty.md section 1 (every paid run must save its result
incrementally, not just in memory).

Usage (Colab, after scripts/check_environment.py has passed):
    !python scripts/run_hard_negative_pairwise_test2.py --dry-run   # free, no API calls - just prints the pair count/cost
    !python scripts/run_hard_negative_pairwise_test2.py             # the real paid run
    !python scripts/run_hard_negative_pairwise_test2.py --summarize # free, offline - reads the .jsonl this script already wrote
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json

STRESS_ANALYSIS_PATH = Path("results/hard_negative_stress_250/stress_analysis.json")
EVAL_SUBSET_PATH = Path("data/t2-ragbench/eval_subset_250.parquet")
TRACE_PATH = Path("results/retrieval_trace_250/retrieval_trace.jsonl")
OUT_DIR = Path("results/hard_negative_pairwise_test2")
OUT_PATH = OUT_DIR / "pairwise_results.jsonl"
COST_PER_CALL = 0.0025  # tehnicheskoe_zadanie.md section 15 - assumption, not a confirmed Cohere billing fact

CATEGORIES = [
    "A_same_company_other_year",
    "B_same_company_year_other_doc",
    "C_same_sector_other_company",
]


def build_pair_list() -> list[dict]:
    """All (question_id, category, gold_context_id, sibling_context_id)
    pairs with a non-null sibling - reuses exactly the deterministic
    sibling assignment already computed and committed by Test 1, not
    recomputed here, so the two tests can never disagree about which
    sibling a question was paired with."""
    data = json.loads(STRESS_ANALYSIS_PATH.read_text())
    pairs = []
    for q in data["per_question"]:
        for cat in CATEGORIES:
            c = q["categories"][cat]
            if c is None:
                continue
            pairs.append(
                {
                    "question_id": q["question_id"],
                    "category": cat,
                    "gold_context_id": q["gold_context_id"],
                    "sibling_context_id": c["sibling_context_id"],
                }
            )
    return pairs


def build_question_text_map() -> dict[str, str]:
    import pandas as pd

    df = pd.read_parquet(EVAL_SUBSET_PATH)
    return dict(zip(df["id"], df["question"]))


def load_known_content_hashes() -> dict[tuple[str, str], str]:
    """(question_id, context_id) -> content_sha256, from every entry
    (candidate_top50, so it covers gold and any naturally-competing
    sibling) already recorded in retrieval_trace_250. Used only as an
    integrity cross-check against freshly Mongo-fetched content - if a
    hash mismatches, the corpus changed since 2026-08-21 and that pair's
    result should not be trusted at face value."""
    hashes: dict[tuple[str, str], str] = {}
    with TRACE_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["question_id"]
            for entry in rec["candidate_top50"]:
                hashes[(qid, entry["context_id"])] = entry["content_sha256"]
    return hashes


def fetch_full_indexed_content(collection, context_id: str) -> str:
    doc = collection.find_one({"context_id": context_id}, {"full_indexed_content": 1})
    if doc is None or not doc.get("full_indexed_content"):
        raise RuntimeError(
            f"context_id={context_id!r} not found in MongoDB or has empty "
            f"full_indexed_content - corpus may have changed since Test 1 "
            f"(2026-08-29) or since indexing (2026-08-08/18)"
        )
    return doc["full_indexed_content"]


def load_checkpoint(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            done.add((rec["question_id"], rec["category"]))
    return done


def append_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(dry_run: bool) -> None:
    pairs = build_pair_list()
    by_cat: dict[str, int] = {}
    for p in pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1

    print(f"{len(pairs)} pairs total.")
    for cat in CATEGORIES:
        print(f"  {cat}: {by_cat.get(cat, 0)}")
    print(f"Estimated cost at ${COST_PER_CALL}/call: ${len(pairs) * COST_PER_CALL:.2f}")

    if dry_run:
        print("--dry-run: stopping before any MongoDB/Cohere call.")
        return

    from config.config_schema import load_config
    from pipeline.cli import _retrieved_docs_for_prediction, build_clients
    from pipeline.reranking import rerank
    from pipeline.retrieval import Candidate

    q_text = build_question_text_map()
    known_hashes = load_known_content_hashes()

    config = load_config()
    clients = build_clients(config)
    collection = clients["collection"]
    cohere_client = clients["cohere"]

    done = load_checkpoint(OUT_PATH)
    if done:
        print(f"{len(done)} pairs already done (resumed), {len(pairs) - len(done)} remaining")

    content_cache: dict[str, str] = {}

    def get_content(cid: str) -> str:
        if cid not in content_cache:
            content_cache[cid] = fetch_full_indexed_content(collection, cid)
        return content_cache[cid]

    total_done = len(done)
    for pair in pairs:
        key = (pair["question_id"], pair["category"])
        if key in done:
            continue

        query = q_text[pair["question_id"]]
        gold_content = get_content(pair["gold_context_id"])
        sib_content = get_content(pair["sibling_context_id"])

        content_integrity_verified = {}
        for role, cid, content in (
            ("gold", pair["gold_context_id"], gold_content),
            ("sibling", pair["sibling_context_id"], sib_content),
        ):
            known = known_hashes.get((pair["question_id"], cid))
            if known is not None:
                actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
                content_integrity_verified[role] = actual == known

        gold_cand = Candidate(context_id=pair["gold_context_id"], full_indexed_content=gold_content, score=0.0)
        sib_cand = Candidate(context_id=pair["sibling_context_id"], full_indexed_content=sib_content, score=0.0)

        ranked = rerank(cohere_client, query, [gold_cand, sib_cand], top_n=2)
        docs = _retrieved_docs_for_prediction(ranked)
        scores = {d["context_id"]: d["score"] for d in docs}
        gold_score = scores.get(pair["gold_context_id"])
        sib_score = scores.get(pair["sibling_context_id"])

        record = {
            "question_id": pair["question_id"],
            "category": pair["category"],
            "gold_context_id": pair["gold_context_id"],
            "sibling_context_id": pair["sibling_context_id"],
            "gold_relevance_score": gold_score,
            "sibling_relevance_score": sib_score,
            "gold_wins": (
                None if gold_score is None or sib_score is None else gold_score > sib_score
            ),
            "tie": (
                None if gold_score is None or sib_score is None else gold_score == sib_score
            ),
            "content_integrity_verified": content_integrity_verified,
        }
        append_result(OUT_PATH, record)
        total_done += 1
        if total_done % 25 == 0:
            print(f"  {total_done}/{len(pairs)} pairs done")

    print(f"Done. {total_done} pairs. Results: {OUT_PATH}")
    bad_integrity = 0
    with OUT_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            if any(v is False for v in rec["content_integrity_verified"].values()):
                bad_integrity += 1
    if bad_integrity:
        print(
            f"WARNING: {bad_integrity} pairs failed the content_sha256 integrity "
            f"check against retrieval_trace_250 - corpus may have drifted since "
            f"2026-08-21. Do not trust results without reviewing these first."
        )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - adj) / denom, (centre + adj) / denom)


def rank_in(trace_list: list[dict], context_id: str) -> int | None:
    for entry in trace_list:
        if entry["context_id"] == context_id:
            return entry["rank"]
    return None


def natural_order_from_test1(question_id: str, gold_id: str, sibling_id: str, trace_by_qid: dict) -> bool | None:
    """True/False = gold beat / lost to the sibling in the ALREADY-OBSERVED
    natural reranked_top5 for this question; None = undeterminable from
    Test 1's saved data (sibling never in the same candidate_top50 batch as
    gold, gold itself never reached candidate_top50, or both gold and
    sibling missed the saved top-5 slice so their relative order beyond
    rank 5 was never recorded). See this script's module docstring, design
    revision point 1, for why this is stricter than Test 1's own
    not_naturally_covered_needs_paid_test2 field."""
    trace = trace_by_qid.get(question_id)
    if trace is None:
        return None
    top50 = trace["candidate_top50"]
    top5 = trace["reranked_top5"]
    if rank_in(top50, gold_id) is None:
        return None  # gold itself never reached the candidate pool
    if rank_in(top50, sibling_id) is None:
        return None  # sibling never competed in the same batch
    gold_rank5 = rank_in(top5, gold_id)
    sib_rank5 = rank_in(top5, sibling_id)
    if gold_rank5 is not None and sib_rank5 is not None:
        return gold_rank5 < sib_rank5
    if gold_rank5 is not None:
        return True  # gold made the final top-5, sibling (same batch) did not
    if sib_rank5 is not None:
        return False  # sibling made the final top-5, gold (same batch) did not
    return None  # both missed top-5 - order beyond rank 5 was never recorded


def summarize() -> None:
    if not OUT_PATH.exists():
        print(f"{OUT_PATH} does not exist yet - run the paid step first (see module docstring).")
        return

    results = [json.loads(line) for line in OUT_PATH.open()]

    trace_by_qid: dict[str, dict] = {}
    with TRACE_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            trace_by_qid[rec["question_id"]] = rec

    summary: dict[str, dict] = {}
    agreement_hits = 0
    agreement_n = 0
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        decided = [r for r in cat_results if r["gold_wins"] is not None and not r["tie"]]
        wins = sum(1 for r in decided if r["gold_wins"])
        n = len(decided)
        ties = sum(1 for r in cat_results if r["tie"])
        integrity_failures = sum(
            1 for r in cat_results if any(v is False for v in r["content_integrity_verified"].values())
        )
        summary[cat] = {
            "n_pairs": len(cat_results),
            "ties_excluded": ties,
            "n_decided": n,
            "gold_wins": wins,
            "pairwise_accuracy": wins / n if n else None,
            "wilson_95ci": wilson_ci(wins, n) if n else None,
            "content_integrity_failures": integrity_failures,
        }
        for r in cat_results:
            natural = natural_order_from_test1(r["question_id"], r["gold_context_id"], r["sibling_context_id"], trace_by_qid)
            if natural is not None and r["gold_wins"] is not None:
                agreement_n += 1
                if natural == r["gold_wins"]:
                    agreement_hits += 1

    summary["_pointwise_scoring_assumption_check"] = {
        "description": (
            "Of the pairs where Test 1's natural-batch order was independently "
            "known, how often did this forced 2-candidate call agree with it. "
            "Low agreement would mean Cohere Rerank's scores are NOT independent "
            "of batch composition, undermining the assumption in this script's "
            "module docstring point 2."
        ),
        "n_comparable": agreement_n,
        "agreement": agreement_hits / agreement_n if agreement_n else None,
        "wilson_95ci": wilson_ci(agreement_hits, agreement_n) if agreement_n else None,
    }

    print(json.dumps(summary, indent=2))
    summary_path = OUT_DIR / "pairwise_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print the pair list/cost estimate only, no MongoDB/Cohere calls.")
    parser.add_argument("--summarize", action="store_true", help="Offline: summarize an already-written pairwise_results.jsonl. No API calls.")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run(dry_run=args.dry_run)

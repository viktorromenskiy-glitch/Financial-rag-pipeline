"""Priority 2 (adversarial-robustness plan, internal working materials not in this repository),
Test 1: retrospective, zero-cost stress test of retrieval/reranking against
financial hard negatives - documents that are lexically/semantically close
to a question's gold document but wrong (same company/different year, same
company+year/different section, same sector/different company).

No new API calls. Reuses results/retrieval_trace_250/retrieval_trace.jsonl
(the already-committed, already-paid-for candidate_top50 + reranked_top5
trace for all 250 questions from the Error attribution retrieval-only run,
see scripts/run_retrieval_attribution.py) and pipeline.ingestion.ingest()
(local parquet metadata, also free) to check, for each question whose gold
document has an identifiable hard-negative sibling, whether that sibling
already sits in the saved candidate pool - and if so, ranked above or below
the gold document.

Method (deterministic sibling selection, fixed before looking at results,
per the plan doc):
  - Category A (same company, different report_year): nearest year by
    absolute difference; ties broken by lexicographically smallest
    context_id.
  - Category B (same company + same report_year, different document):
    lexicographically smallest context_id among the other documents in
    that company-year group.
  - Category C (same company_sector, different company): lexicographically
    smallest context_id among documents from a different company in the
    same sector. Documents with a missing/empty company_sector are
    excluded from this category only (420/7318 in this corpus).

Outputs results/hard_negative_stress_250/stress_analysis.json (per-question
detail + summary) and prints the summary. Recall@5 comparison is within
the same n=250 run (stress subset vs. the rest of the same run), not
against the separately-measured n=900 headline number - see the plan doc
for why that matters.

Usage:
    python scripts/analyze_hard_negative_stress.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingestion import ingest  # noqa: E402

DATA_DIR = "data/t2-ragbench"
TRACE_PATH = Path("results/retrieval_trace_250/retrieval_trace.jsonl")
EVAL_SUBSET_PATH = Path("data/t2-ragbench/eval_subset_250.parquet")
OUT_DIR = Path("results/hard_negative_stress_250")


def build_doc_metadata() -> dict[str, dict]:
    """context_id -> {company_name, report_year, company_sector}, one row
    per unique document (DocumentRecord is one-per-question, so dedupe)."""
    records = ingest(DATA_DIR)
    meta: dict[str, dict] = {}
    for r in records:
        meta.setdefault(
            r.context_id,
            {
                "company_name": r.company_name,
                "report_year": r.report_year,
                "company_sector": r.company_sector,
            },
        )
    return meta


def build_gold_map() -> dict[str, str]:
    """question_id -> gold context_id, from the eval subset itself (this is
    the same file the committed error_analysis_250/retrieval_trace_250 runs
    were evaluated against)."""
    import pandas as pd

    df = pd.read_parquet(EVAL_SUBSET_PATH)
    return dict(zip(df["id"], df["context_id"]))


def pick_category_a(gold_id: str, meta: dict[str, dict], by_company: dict[str, list[str]]) -> str | None:
    company = meta[gold_id]["company_name"]
    gold_year = meta[gold_id]["report_year"]
    siblings = [
        cid
        for cid in by_company.get(company, [])
        if cid != gold_id and meta[cid]["report_year"] != gold_year
    ]
    if not siblings:
        return None
    siblings.sort(key=lambda cid: (abs(int(meta[cid]["report_year"]) - int(gold_year)), cid))
    return siblings[0]


def pick_category_b(gold_id: str, meta: dict[str, dict], by_company_year: dict[tuple, list[str]]) -> str | None:
    company = meta[gold_id]["company_name"]
    year = meta[gold_id]["report_year"]
    siblings = sorted(cid for cid in by_company_year.get((company, year), []) if cid != gold_id)
    return siblings[0] if siblings else None


def pick_category_c(gold_id: str, meta: dict[str, dict], by_sector: dict[str, list[str]]) -> str | None:
    sector = meta[gold_id]["company_sector"]
    if not sector:
        return None
    company = meta[gold_id]["company_name"]
    siblings = sorted(
        cid for cid in by_sector.get(sector, []) if meta[cid]["company_name"] != company
    )
    return siblings[0] if siblings else None


def rank_in(trace_list: list[dict], context_id: str) -> int | None:
    for entry in trace_list:
        if entry["context_id"] == context_id:
            return entry["rank"]
    return None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - adj) / denom, (centre + adj) / denom)


def main() -> None:
    meta = build_doc_metadata()
    gold_map = build_gold_map()

    by_company: dict[str, list[str]] = defaultdict(list)
    by_company_year: dict[tuple, list[str]] = defaultdict(list)
    by_sector: dict[str, list[str]] = defaultdict(list)
    for cid, m in meta.items():
        by_company[m["company_name"]].append(cid)
        by_company_year[(m["company_name"], m["report_year"])].append(cid)
        if m["company_sector"]:
            by_sector[m["company_sector"]].append(cid)

    trace_by_qid: dict[str, dict] = {}
    with open(TRACE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            trace_by_qid[rec["question_id"]] = rec

    per_question = []
    for qid, gold_id in gold_map.items():
        trace = trace_by_qid.get(qid)
        if trace is None:
            continue  # not part of the retrieval_trace_250 run
        if gold_id not in meta:
            continue  # gold context missing from ingested metadata (shouldn't happen)

        top50 = trace["candidate_top50"]
        top5 = trace["reranked_top5"]
        gold_rank_50 = rank_in(top50, gold_id)
        gold_rank_5 = rank_in(top5, gold_id)

        siblings = {
            "A_same_company_other_year": pick_category_a(gold_id, meta, by_company),
            "B_same_company_year_other_doc": pick_category_b(gold_id, meta, by_company_year),
            "C_same_sector_other_company": pick_category_c(gold_id, meta, by_sector),
        }

        cat_results = {}
        has_any_sibling = False
        for cat, sib_id in siblings.items():
            if sib_id is None:
                cat_results[cat] = None
                continue
            has_any_sibling = True
            sib_rank_50 = rank_in(top50, sib_id)
            sib_rank_5 = rank_in(top5, sib_id)
            cat_results[cat] = {
                "sibling_context_id": sib_id,
                "in_top50": sib_rank_50 is not None,
                "top50_rank": sib_rank_50,
                "outranks_gold_in_top50": (
                    sib_rank_50 is not None
                    and gold_rank_50 is not None
                    and sib_rank_50 < gold_rank_50
                ),
                "in_reranked_top5": sib_rank_5 is not None,
                "reranked_rank": sib_rank_5,
                "outranks_gold_in_top5": (
                    sib_rank_5 is not None
                    and gold_rank_5 is not None
                    and sib_rank_5 < gold_rank_5
                ),
                "sibling_in_top5_gold_not": (sib_rank_5 is not None and gold_rank_5 is None),
            }

        per_question.append(
            {
                "question_id": qid,
                "source_dataset": trace["source_dataset"],
                "gold_context_id": gold_id,
                "gold_in_top50": gold_rank_50 is not None,
                "gold_in_reranked_top5": gold_rank_5 is not None,
                "has_any_hard_negative_sibling": has_any_sibling,
                "categories": cat_results,
            }
        )

    # --- Summary -----------------------------------------------------
    stress = [q for q in per_question if q["has_any_hard_negative_sibling"]]
    non_stress = [q for q in per_question if not q["has_any_hard_negative_sibling"]]

    def recall5(qs: list[dict]) -> tuple[int, int]:
        return sum(1 for q in qs if q["gold_in_reranked_top5"]), len(qs)

    stress_hits, stress_n = recall5(stress)
    non_stress_hits, non_stress_n = recall5(non_stress)
    all_hits, all_n = recall5(per_question)

    summary = {
        "total_questions_analyzed": all_n,
        "overall_recall_at_5": {
            "hits": all_hits,
            "n": all_n,
            "rate": all_hits / all_n if all_n else None,
        },
        "stress_subset_recall_at_5": {
            "hits": stress_hits,
            "n": stress_n,
            "rate": stress_hits / stress_n if stress_n else None,
            "wilson_95ci": wilson_ci(stress_hits, stress_n) if stress_n else None,
        },
        "non_stress_subset_recall_at_5": {
            "hits": non_stress_hits,
            "n": non_stress_n,
            "rate": non_stress_hits / non_stress_n if non_stress_n else None,
            "wilson_95ci": wilson_ci(non_stress_hits, non_stress_n) if non_stress_n else None,
        },
        "per_category": {},
    }

    for cat in ["A_same_company_other_year", "B_same_company_year_other_doc", "C_same_sector_other_company"]:
        applicable = [q["categories"][cat] for q in per_question if q["categories"][cat] is not None]
        n_applicable = len(applicable)
        in_top50 = sum(1 for c in applicable if c["in_top50"])
        outranks_50 = sum(1 for c in applicable if c["outranks_gold_in_top50"])
        in_top5 = sum(1 for c in applicable if c["in_reranked_top5"])
        outranks_5 = sum(1 for c in applicable if c["outranks_gold_in_top5"])
        sib_beats_gold_recall = sum(1 for c in applicable if c["sibling_in_top5_gold_not"])
        not_naturally_covered = n_applicable - in_top50  # would need a paid Test 2 call
        summary["per_category"][cat] = {
            "n_questions_with_sibling": n_applicable,
            "sibling_naturally_in_top50": in_top50,
            "sibling_outranks_gold_in_top50": outranks_50,
            "sibling_naturally_in_reranked_top5": in_top5,
            "sibling_outranks_gold_in_reranked_top5": outranks_5,
            "sibling_in_top5_while_gold_missing": sib_beats_gold_recall,
            "not_naturally_covered_needs_paid_test2": not_naturally_covered,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stress_analysis.json", "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / 'stress_analysis.json'} ({len(per_question)} questions).")


if __name__ == "__main__":
    main()

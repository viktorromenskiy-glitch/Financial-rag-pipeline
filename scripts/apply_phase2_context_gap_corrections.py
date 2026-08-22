"""Plan generation-error-analysis, Phase 2 result: manual review of the 7
questions where the model answered FINAL ANSWER: INSUFFICIENT_CONTEXT in
results/error_analysis_250 (all 7 are generation_failure_candidate - none
overlap with Phase 1's 5 judge/is_close_v2 cases, since INSUFFICIENT_CONTEXT
is non-numeric and is_close_v2 returns False on non-numeric input).

Builds on scripts/apply_phase1_judge_corrections.py - same non-destructive
`manual_correction` annotation approach, same reasoning: read the actual
gold-document text (data/t2-ragbench/*.parquet, keyed by context_id) for
each of the 7 and determine whether the needed fact is genuinely present,
buried, or absent (plan_generation_error_analysis.md, Phase 2's three
possible outcomes a/b/c). Also cross-checks sibling questions sharing the
same context_id where useful (same technique as Phase 1).

Result - two new patterns, not one uniform "retrieval gap":

- context_extraction_gap (5 of 7): the needed number is not present in the
  captured context_id text AT ALL, confirmed by full-text search for the
  relevant term/keyword (not a truncation artifact of this script - the
  parquet `context`/`pre_text`/`post_text` fields themselves stop short of
  the data). Two different manifestations: finqa_dev_788 and
  finqa_train_4449 give only narrative deltas/definitions, never the
  absolute figure the question needs; tatqa_train_4974, tatqa_train_8524,
  tatqa_train_8525 have TAT-DQA `context` text that explicitly promises a
  table ("refer to the table within Item 7...") or is mid-sentence at the
  point the needed section would start, and then just stops - the source
  document's own text got cut before the relevant table. The model's
  INSUFFICIENT_CONTEXT answer is epistemically correct in all 5 cases: the
  document (context_id) was correctly retrieved into the top-5, but this
  project's ingested TEXT for that context_id doesn't contain the fact.
  This is the "retrieval succeeded, the ingested chunk itself is
  incomplete" mechanism the plan's Phase 2 write-up anticipated - not a
  reasoning failure and not (per this project's classify() semantics, see
  pipeline/attribution.py) a retrieval_failure or reranking_failure either,
  since gold_context_id membership is a red herring here: the right
  *document* was found, its *captured text* just doesn't carry the needed
  table. Tracked as its own category rather than folded into
  generation_failure_candidate.

- question_label_mismatch (1 of 7, finqa_train_2391): the number IS
  reproducible from data present in the context (tax benefit $62M / stock
  comp cost $211M = 0.293839... = gold exactly), but the question text asks
  for "the effective tax rate ... considering the tax benefits related to
  share-based compensation" - real effective tax rate is tax provision /
  pretax income, which this note-16 share-based-compensation disclosure
  does not contain at all. Cross-checked against a sibling question at the
  same context_id (finqa_train_1724, "ratio of share-based compensation
  cost to the related tax benefits" = 211/62 = 3.403225...): gold for
  finqa_train_2391 is exactly the algebraic inverse of that sibling's
  answer, mislabeled with unrelated tax terminology. The model's refusal
  reflects a financially literate reading (there is no real effective tax
  rate data in this note) that happens to disagree with a mislabeled gold
  value - not confidently a model error, not confidently "success" either.
  Tracked as its own category, not folded into either.

- confirmed_generation_error (1 of 7, tatqa_train_4256): the needed number
  (130, "Canceled" row of a stock-option roll-forward) is the only such
  figure in the whole context, unambiguous - a competent reader would
  answer 130 despite the question's slightly imprecise date framing ("as of
  December 31, 2018" for a row that is really FY2019 activity ending on
  that date). Genuine generation-side over-caution, not a context gap. NOT
  reclassified - stays generation_failure_candidate.

Net effect on generation_failure_candidate: 51 (post Phase 1) -> 45 (-5
context_extraction_gap, -1 question_label_mismatch; the 1 confirmed error
was already counted and stays).

Same non-destructive design as Phase 1's script: does not touch
results/error_analysis_250/predictions.jsonl or eval_results.jsonl, does not
change any record's original failure_stage/judge_correct, only adds/extends
the `manual_correction` field. Idempotent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results/retrieval_trace_250")
ATTRIBUTION_RESULTS_PATH = RESULTS_DIR / "attribution_results.jsonl"
ATTRIBUTION_SUMMARY_PATH = RESULTS_DIR / "attribution_summary.json"
PHASE2_CORRECTIONS_PATH = RESULTS_DIR / "phase2_manual_corrections.jsonl"

CONTEXT_EXTRACTION_GAP = "context_extraction_gap"
QUESTION_LABEL_MISMATCH = "question_label_mismatch"

CORRECTIONS = {
    "finqa_dev_788": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,
        "evidence": (
            'Question asks for net income / revenues ratio for 2010. Full pre_text/post_text of '
            "finqa_dev_ctx_85 (confirmed complete, not truncated by this review) gives only YoY net "
            'income DELTAS ("net income increased by $14.6 million" 2011 vs 2010, "$2.4 million" 2010 '
            "vs 2009) and a net-revenue waterfall table (2010=$540.2M, 2011=$577.8M) - no absolute net "
            "income figure for any year appears anywhere in the captured text. The ratio (0.004443) "
            "cannot be computed from what is actually present. Model's INSUFFICIENT_CONTEXT is "
            "epistemically correct given its input; the absolute net income figure exists in the "
            "original 10-K but was not captured into this context_id's ingested text."
        ),
    },
    "finqa_train_4449": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,
        "evidence": (
            "Question needs trade-receivables trend and Q4-sales-based allowance-for-doubtful-accounts "
            "percentage. Full pre_text/post_text of finqa_train_ctx_1033 (5642 chars each, confirmed "
            'complete) mentions "allowance for doubtful accounts" only once, in a definitional sentence '
            '("operating working capital ... represents trade receivables-net of the allowance for '
            'doubtful accounts...") - no dollar figures or quarterly sales breakdown anywhere. Neither '
            "candidate gold value (program_answer=18.1, original_answer=17.7 - these two disagree with "
            "each other in the source dataset, a separate pre-existing FinQA annotation inconsistency) "
            "is derivable from the captured text. Model's INSUFFICIENT_CONTEXT is correct given its input."
        ),
    },
    "finqa_train_2391": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            'Question asks for Visa\'s "effective tax rate ... considering tax benefits related to '
            'share-based compensation." Context (finqa_train_ctx_2037, Note 16) gives share-based comp '
            "cost ($211M) and its related tax benefit ($62M) for fiscal 2016, plus unrelated "
            "Black-Scholes option-pricing assumptions - no real effective-tax-rate data (tax provision / "
            "pretax income) anywhere. gold_answer=0.2938388625592417 = 62/211 exactly - the algebraic "
            'INVERSE of sibling question finqa_train_1724\'s answer ("ratio of share-based compensation '
            'cost to the related tax benefits" = 211/62 = 3.403225...). The gold value is derivable from '
            "context data, but the question's own wording describes something (real effective tax rate) "
            "that isn't in this note at all - a mislabeled question, not a missing fact. Model's refusal "
            "reflects a financially literate reading of the question as asked, not a reasoning failure."
        ),
    },
    "tatqa_train_4974": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,
        "evidence": (
            "Question needs Siemens Mobility's Adjusted EBITDA margin change 2018-2019. Full `context` "
            "field for 60e8f8b1203a42663e11e645f6cf3f0e (4227 chars, confirmed complete) is Mobility "
            "segment narrative (R&D strategy, order backlog, regional market commentary) that ends "
            'mid-sentence ("...supports customers along the care[continuum]") transitioning into an '
            "unrelated Healthineers section - no EBITDA margin percentage table anywhere in the captured "
            "text. Model's INSUFFICIENT_CONTEXT is correct; the source document's relevant financial "
            "table was not captured for this context_id."
        ),
    },
    "tatqa_train_8524": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,
        "evidence": (
            "Question needs Autodesk's unbilled deferred revenue as % of total revenue, Jan 31 2018. "
            "Full `context` field for c7abe3bab25cfcd2d5ee093a124c56ca (1518 chars, confirmed complete) "
            'is only revenue-recognition POLICY narrative (what subscription/maintenance/other revenue '
            'consist of) that explicitly says "refer to the table within Item 7... for comparison" and '
            "then stops - the actual numeric table it points to was never captured for this context_id. "
            "Model's INSUFFICIENT_CONTEXT is correct."
        ),
    },
    "tatqa_train_8525": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,
        "evidence": (
            "Same context_id and same gap as tatqa_train_8524 (c7abe3bab25cfcd2d5ee093a124c56ca) - "
            "question needs the change in unbilled deferred revenue under ASC 605, Jan 2018 to Jan 2019; "
            "the captured context is policy narrative only, the numeric table it refers readers to was "
            "not captured. Model's INSUFFICIENT_CONTEXT is correct."
        ),
    },
    "tatqa_train_4256": {
        "reason": "confirmed_generation_error",
        "corrected_stage": None,  # stays generation_failure_candidate - no change
        "evidence": (
            "Question needs shares canceled for A10 Networks stock options 'as of December 31, 2018'. "
            "Context (4d24e693babe5005e0ef48a9a093d646) has exactly one stock-option roll-forward table "
            "with exactly one Canceled figure: 130 (row: Outstanding Dec 31 2018 -> ... -> Canceled (130) "
            "-> Outstanding Dec 31 2019) - unambiguous, no competing candidate number anywhere in the "
            "context. gold_answer=130.0 matches exactly. The question's date framing is arguably "
            "imprecise (130 is FY2019 roll-forward activity, not literally 'as of' the 2018 date), but a "
            "competent reader would still confidently answer 130 since it's the only Canceled figure "
            "present. Genuine generation-side over-caution given a clearly-readable number - not a "
            "context gap. NOT reclassified."
        ),
    },
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_corrections(records: list[dict]) -> list[dict]:
    corrected = []
    seen = set()
    for r in records:
        r = dict(r)
        qid = r["question_id"]
        if qid in CORRECTIONS:
            seen.add(qid)
            c = CORRECTIONS[qid]
            r["manual_correction_phase2"] = {
                "reviewed": "plan_generation_error_analysis phase 2, 2026-08-22",
                "reason": c["reason"],
                "corrected_stage": c["corrected_stage"],
                "evidence": c["evidence"],
            }
        corrected.append(r)
    missing = set(CORRECTIONS) - seen
    if missing:
        raise ValueError(f"question_id(s) not found in attribution_results.jsonl: {sorted(missing)}")
    return corrected


def cumulative_corrected_stage(r: dict) -> str:
    """Applies Phase 1's correction first (if any), then Phase 2's on top -
    the final, cumulative view of where this question_id nets out after
    both manual review phases."""
    stage = r["failure_stage"]
    mc1 = r.get("manual_correction")
    if mc1 is not None:
        if mc1["corrected_stage"] is not None:
            stage = mc1["corrected_stage"]
        else:
            stage = mc1["reason"]  # e.g. context_data_inconsistency
    mc2 = r.get("manual_correction_phase2")
    if mc2 is not None:
        if mc2["corrected_stage"] is not None:
            stage = mc2["corrected_stage"]
        elif mc2["reason"] != "confirmed_generation_error":
            stage = mc2["reason"]  # context_extraction_gap / question_label_mismatch
        # confirmed_generation_error: no change, stage stays as computed by Phase 1 step
    return stage


def main() -> None:
    records = load_jsonl(ATTRIBUTION_RESULTS_PATH)
    corrected = apply_corrections(records)
    write_jsonl(ATTRIBUTION_RESULTS_PATH, corrected)

    write_jsonl(
        PHASE2_CORRECTIONS_PATH,
        [
            {"question_id": qid, "reason": c["reason"], "corrected_stage": c["corrected_stage"], "evidence": c["evidence"]}
            for qid, c in CORRECTIONS.items()
        ],
    )

    summary = json.loads(ATTRIBUTION_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["phase2_corrected_overall"] = dict(Counter(cumulative_corrected_stage(r) for r in corrected))
    summary["phase2_corrections_note"] = (
        "7 questions manually reviewed (plan_generation_error_analysis.md phase 2) where the model "
        "answered FINAL ANSWER: INSUFFICIENT_CONTEXT. See phase2_manual_corrections.jsonl for the "
        "per-question reason and evidence. phase2_corrected_overall is CUMULATIVE (Phase 1 + Phase 2 "
        "corrections both applied) - phase1_corrected_overall above reflects Phase 1 alone, unchanged, "
        "as a historical snapshot; original `overall` remains the untouched raw judge-derived classification."
    )
    ATTRIBUTION_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("phase1_corrected_overall (unchanged):", summary.get("phase1_corrected_overall"))
    print("phase2_corrected_overall (cumulative):", summary["phase2_corrected_overall"])


if __name__ == "__main__":
    main()

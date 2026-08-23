"""Plan generation-error-analysis, Phase 3 result: manual review of the 8
most suspicious cases surfaced while reading the raw_response reasoning
traces from results/retrieval_trace_250/generation_failure_traces.jsonl
(the Phase 3 paid replay of the 44-question residual pool).

Trigger for this review: while doing a first-pass taxonomy read of all 44
raw_response traces (the qualitative work Phase 4 of the plan calls for),
several patterns stood out as looking less like a genuine model reasoning
failure and more like the same kind of dataset/judge/extraction defect
Phases 1-2 already found hiding inside what looked like generation errors.
Rather than fold these straight into a taxonomy bucket on trust, each was
independently verified against the actual gold-document text
(data/t2-ragbench/*.parquet, same forensic method as Phases 1-2: read the
full context/pre_text/post_text/table fields, cross-check sibling questions
sharing the same context_id, compare program_answer vs original_answer).

Result - three confirmed non-generation-error patterns, plus one genuinely
new, previously-undocumented failure mode:

- gold_label_defect (2: convfinqa_2260, convfinqa_1487): the ConvFinQA gold
  program computed a plain arithmetic DIFFERENCE, not a percentage, despite
  the natural-language question explicitly asking for "percentage change".
  convfinqa_1487 is airtight: sibling convfinqa_94 at the same context_id
  literally asks for the "difference" and has the IDENTICAL gold_answer
  (4.9) as convfinqa_1487's "percentage change" question - the gold value
  was never actually computed as a percentage. convfinqa_2260 is the same
  pattern (sibling convfinqa_1383 confirms the raw 2010 figure; gold=-1.0
  is exactly the raw 2009-to-2010 dollar difference, not a percent). Both
  models' own percentage computations are arithmetically correct given the
  right underlying numbers.

- question_label_mismatch (2: tatqa_train_19, convfinqa_969): the gold
  value is real and derivable from context, but the question's own wording
  describes something different from what the gold program actually
  computed. tatqa_train_19: TAT-DQA's own original_answer field is the
  literal string '1' (a count of qualifying years), not a year - this
  project's rephrased question ("In what year did...") is incompatible
  with what the underlying gold represents ("in how many years..."). The
  model's stated answer (2018) is independently verified correct against
  the source table (Americas long-lived assets: 2019=$136,035k,
  2018=$178,251k - only 2018 exceeds $150,000k). convfinqa_969: full
  context is a discontinued-operations note; operating revenues
  2014=$13M, 2013=$23M, difference=-10=gold exactly - the gold program
  computed the change in operating REVENUES, but the question asks about
  operating PROFITS, which this note doesn't report at all. The model's
  INSUFFICIENT_CONTEXT refusal is the epistemically correct response to a
  mislabeled question, same logic as Phase 2's finqa_train_2391.

- context_extraction_gap (2: tatqa_train_2340, tatqa_train_4366): same
  mechanism as Phase 2's 4 cases - the right document was retrieved, but
  the specific numeric table needed is missing from the captured text.
  tatqa_train_2340: the context literally reads "The following table
  reconciles the beginning and ending fair value measurements of our
  servicing assets... We did not have any servicing assets for the years
  ended December 31, 2018 and 2017" - and then no table follows at all,
  even though sibling tatqa_train_2339 proves a real 2018 beginning-balance
  figure (2071) exists in the source document. tatqa_train_4366: the
  quarterly financial data table's header row lists 4 quarters + a total
  column, but the "Basic"/"Diluted" EPS rows underneath contain only 2
  values each - the table extraction collapsed/lost the per-quarter EPS
  breakdown. Both models correctly flagged the gap themselves rather than
  guessing.

- irreproducible_on_replay (2: finqa_test_355, convfinqa_298) - NEW
  category, not seen in Phases 1-2: neither a dataset defect nor a judge
  bug. In both cases the question and gold_answer are fully consistent
  with each other, AND the Phase 3 replay (with content_sha256-verified
  context, content_verified=True) computed an answer that matches gold
  EXACTLY - but the ORIGINAL results/error_analysis_250 run (19-20 August)
  produced a different, wrong answer on presumably the same question
  (finqa_test_355: original=29.17 vs replay=22.02=gold; convfinqa_298:
  original=917 vs replay=118=gold exactly). This is precisely the
  reproducibility caveat the Phase 3 plan and script docstring already
  flagged as an unprovable assumption (retrieval_trace_250's
  content_sha256 logging postdates the original run, so the original run's
  exact context can never be directly verified) - now observed in practice
  for 2 of 44 questions. NOT reclassified to success (we cannot explain
  why the original run failed, only that today's reproduction succeeds)
  and NOT left as a "confirmed, understood generation error" either (the
  plan's Phase 4 taxonomy is about categorizing WHY the model reasoned
  incorrectly, and here it reasoned correctly on the same verified inputs)
  - tracked as its own distinct, honestly-labeled category so Phase 4's
  taxonomy work is not spent trying to explain an error that may not be a
  reproducible generation-reasoning failure at all.

Builds directly on scripts/apply_phase1_judge_corrections.py and
apply_phase2_context_gap_corrections.py - identical non-destructive
`manual_correction_phase3` annotation approach: does not touch
results/error_analysis_250/predictions.jsonl or eval_results.jsonl, does
not change any record's original failure_stage/judge_correct, and does not
touch the manual_correction / manual_correction_phase2 fields from earlier
phases either. Idempotent.

Net effect on generation_failure_candidate (cumulative view,
phase3_corrected_overall): 46 (post Phase 2) -> 38 (-8: all 8 leave the
generation_failure_candidate bucket). Of those 8: only the 2
gold_label_defect cases move to success (the model's own answer is
objectively, verifiably correct); the 2 question_label_mismatch and 2
context_extraction_gap cases move to their OWN buckets, not success - same
convention Phase 2 established (a reasonable refusal given a mislabeled
question or an incomplete context is not the same thing as the model
getting the judged answer right); the 2 irreproducible_on_replay cases keep
their stage unchanged, tracked separately. 38 questions remain squarely in
scope for Phase 4's raw_response taxonomy work (44 replayed minus these 6
resolved-away minus the 2 irreproducible cases which don't need a
reasoning-failure taxonomy entry).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results/retrieval_trace_250")
ATTRIBUTION_RESULTS_PATH = RESULTS_DIR / "attribution_results.jsonl"
ATTRIBUTION_SUMMARY_PATH = RESULTS_DIR / "attribution_summary.json"
PHASE3_CORRECTIONS_PATH = RESULTS_DIR / "phase3_manual_corrections.jsonl"

GOLD_LABEL_DEFECT = "gold_label_defect"
QUESTION_LABEL_MISMATCH = "question_label_mismatch"
CONTEXT_EXTRACTION_GAP = "context_extraction_gap"
IRREPRODUCIBLE_ON_REPLAY = "irreproducible_on_replay"

CORRECTIONS = {
    "convfinqa_2260": {
        "reason": GOLD_LABEL_DEFECT,
        "corrected_stage": "success",
        "evidence": (
            "Question: \"percentage change in total expense related to defined contribution plan for "
            "U.S. employees at Analog Devices from fiscal 2009 to 2010\". Sibling convfinqa_1383 at the "
            "same context_id (convfinqa_ctx_1603) asks for the plain 2010 total ($20.5M) - confirming "
            "20.5 is the 2010 figure. gold_answer=-1.0 is exactly 20.5-21.5 (the raw dollar difference), "
            "not a percentage - the ConvFinQA gold program never computed a percent-change operation "
            "here despite the question's wording. Model's own answer (-4.65%) is the arithmetically "
            "correct percentage using the right underlying numbers (20.5, 21.5)."
        ),
    },
    "convfinqa_1487": {
        "reason": GOLD_LABEL_DEFECT,
        "corrected_stage": "success",
        "evidence": (
            "Airtight sibling-question proof: convfinqa_94 (same context_id convfinqa_ctx_525) asks "
            '"What was the DIFFERENCE in Entergy\'s net revenue between 2002 and 2003" and has '
            "gold_answer=4.9. convfinqa_1487 asks \"What was the PERCENTAGE CHANGE in Entergy's net "
            "revenue from 2002 to 2003\" and has the IDENTICAL gold_answer=4.9 - proof the gold value "
            "was never actually computed as a percentage for this question, just copied from the "
            "difference. Model's own percentage math (0.116%, i.e. 4.9/4209.6) is correct given the "
            "real underlying figures."
        ),
    },
    "tatqa_train_19": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,  # own bucket, not success - same convention as Phase 2's finqa_train_2391
        "evidence": (
            "TAT-DQA's own original_answer field for this question is the literal string '1' (a COUNT "
            'of qualifying years), not a year value - this project\'s rephrased question ("In what year '
            "did the long-lived assets in the Americas region exceed $150,000 thousand\") is incompatible "
            "with what the gold program actually computed (\"in how many years...\"). Source table "
            "(context_id d607b0c732705de63af2dceed3970992): Americas long-lived assets June 30 2019 = "
            "$136,035k, June 30 2018 = $178,251k - only 2018 exceeds $150,000k (count=1, matching gold "
            "as a count). Model's stated answer (2018) is independently verified correct against the "
            "source table; the question/gold format mismatch, not the model's reading, is the defect."
        ),
    },
    "convfinqa_969": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,  # own bucket, not success - same convention as Phase 2's finqa_train_2391
        "evidence": (
            'Question: "what was the change in operating profits of american water works from 2013 to '
            '2014". Full context (convfinqa_ctx_484) is a discontinued-operations note: operating '
            "revenues 2014=$13M, 2013=$23M, difference=-10=gold_answer exactly. The gold program "
            "computed the change in operating REVENUES, but the question asks about operating PROFITS, "
            "which this note never reports (it reports 'loss from discontinued operations before income "
            "taxes' instead, a different line: -6 vs -3, change=-3, not -10). Model's "
            "INSUFFICIENT_CONTEXT refusal is the epistemically correct response to a mislabeled "
            "question - same mechanism as Phase 2's finqa_train_2391."
        ),
    },
    "tatqa_train_2340": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,  # own bucket, not success - same convention as Phase 2's 4 context_extraction_gap cases
        "evidence": (
            "Context (context_id 80cbff2f911671e605f964cac6e710a3) literally reads: \"The following "
            "table reconciles the beginning and ending fair value measurements of our servicing assets "
            "associated with Bank Partner loans... We did not have any servicing assets for the years "
            'ended December 31, 2018 and 2017." - and then no table follows at all in the captured text. '
            "Sibling tatqa_train_2339 (same context_id) has gold_answer='2071.0' for \"beginning balance "
            "of servicing assets... in 2018\", proving a real 2018 figure exists in the source document "
            "that this context_id's captured text does not contain. Model's answer of 0 (based on the "
            "narrative sentence it could see) is the correct read of an incomplete extraction, not a "
            "reasoning failure."
        ),
    },
    "tatqa_train_4366": {
        "reason": CONTEXT_EXTRACTION_GAP,
        "corrected_stage": None,  # own bucket, not success - same convention as Phase 2's 4 context_extraction_gap cases
        "evidence": (
            "Context (context_id 5068ef2d6f8dcbf5e4f27a4880ff0b38)'s quarterly financial data table "
            "header lists First/Second/Third/Fourth Quarter + Total (5 columns) for fiscal 2019, but "
            "the 'Basic'/'Diluted' EPS rows underneath contain only 2 values each ($0.71, $(2.93)) - "
            "the per-quarter EPS breakdown was lost/collapsed during table extraction, while the "
            "adjacent Net sales/Gross profit/Net income rows in the same table correctly show all 4 "
            "quarters. Model explicitly noticed and flagged this exact malformation itself rather than "
            "guessing a number."
        ),
    },
    "finqa_test_355": {
        "reason": IRREPRODUCIBLE_ON_REPLAY,
        "corrected_stage": None,  # stage intentionally unchanged - see module docstring
        "evidence": (
            "Question and gold_answer are mutually consistent (original_answer='22%', matching "
            "gold_answer=0.2202 exactly) - no dataset defect. Phase 3 replay (content_verified=True, "
            "context_sha256-confirmed) computed 22.02% (168->205 million rent expense, exact match to "
            "gold), using the same $205M 2008 figure independently confirmed by sibling question "
            "finqa_test_922's own question text (\"...given that the rent expenses for those years were "
            "$205 million and $216 million\"). The ORIGINAL results/error_analysis_250 run (19-20 "
            "August) produced answer_text=29.17 on presumably the same question - a different, "
            "unexplained number. Since retrieval_trace_250's content_sha256 logging postdates the "
            "original run (it cannot be checked against), this cannot be resolved further: today's "
            "reproduction succeeds, the original run's failure mechanism is unknown. NOT reclassified "
            "to success - we cannot confirm the original run saw the same context - and NOT counted as "
            "a confirmed, understood generation-reasoning error either."
        ),
    },
    "convfinqa_298": {
        "reason": IRREPRODUCIBLE_ON_REPLAY,
        "corrected_stage": None,  # stage intentionally unchanged - see module docstring
        "evidence": (
            "Question (\"revenue difference between 2010 and 2009 for Aon\") and gold_answer=118.0 are "
            "mutually consistent - sibling convfinqa_357 at the same context_id asks for the equivalent "
            '"increase in revenue... 2009 to 2010" and has the identical gold_answer=118.0. No dataset '
            "defect. Phase 3 replay (content_verified=True) computed 118 - an exact match to gold. The "
            "ORIGINAL results/error_analysis_250 run produced answer_text=917 - a different, unexplained "
            "number. Same irreproducible_on_replay situation as finqa_test_355; see that entry and the "
            "module docstring for the full reasoning."
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
            r["manual_correction_phase3"] = {
                "reviewed": "plan_generation_error_analysis phase 3 (raw_response taxonomy pre-check), 2026-08-23",
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
    """Applies Phase 1's correction, then Phase 2's, then Phase 3's on top -
    the final, cumulative view of where this question_id nets out after all
    three manual review phases."""
    stage = r["failure_stage"]
    for field in ("manual_correction", "manual_correction_phase2", "manual_correction_phase3"):
        mc = r.get(field)
        if mc is None:
            continue
        if mc["corrected_stage"] is not None:
            stage = mc["corrected_stage"]
        elif mc["reason"] != "confirmed_generation_error":
            stage = mc["reason"]  # e.g. context_data_inconsistency, irreproducible_on_replay
        # confirmed_generation_error: no change - it's a confirmation of the
        # existing classification, not a move to a new bucket.
    return stage


def main() -> None:
    records = load_jsonl(ATTRIBUTION_RESULTS_PATH)
    corrected = apply_corrections(records)
    write_jsonl(ATTRIBUTION_RESULTS_PATH, corrected)

    write_jsonl(
        PHASE3_CORRECTIONS_PATH,
        [
            {"question_id": qid, "reason": c["reason"], "corrected_stage": c["corrected_stage"], "evidence": c["evidence"]}
            for qid, c in CORRECTIONS.items()
        ],
    )

    summary = json.loads(ATTRIBUTION_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["phase3_corrected_overall"] = dict(Counter(cumulative_corrected_stage(r) for r in corrected))
    summary["phase3_corrections_note"] = (
        "8 questions manually reviewed (plan_generation_error_analysis.md phase 3 pre-check) - flagged "
        "while doing the first-pass raw_response taxonomy read of the 44-question Phase 3 replay "
        "(results/retrieval_trace_250/generation_failure_traces.jsonl) as looking like they might not be "
        "genuine generation-reasoning errors, then independently verified against the gold-document "
        "parquet text (same method as Phases 1-2). 6 confirmed non-errors leave "
        "generation_failure_candidate: 2 gold_label_defect cases move to success (model's answer is "
        "objectively correct); 2 question_label_mismatch and 2 context_extraction_gap cases move to "
        "their own buckets, not success (same convention as Phase 2 - a reasonable refusal given a "
        "mislabeled question or incomplete context is not the same as a judged-correct answer). 2 more "
        "(finqa_test_355, convfinqa_298) flagged irreproducible_on_replay - the Phase 3 "
        "replay matches gold exactly but the original error_analysis_250 run failed on presumably the "
        "same question for an unexplained reason; stage intentionally left unchanged since neither "
        "'success' nor 'confirmed generation error' can be honestly claimed. See "
        "phase3_manual_corrections.jsonl for full per-question evidence. phase3_corrected_overall is "
        "CUMULATIVE (Phase 1 + 2 + 3 corrections all applied) - phase1_corrected_overall and "
        "phase2_corrected_overall above remain unchanged historical snapshots; original `overall` "
        "remains the untouched raw judge-derived classification."
    )
    ATTRIBUTION_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("phase2_corrected_overall (unchanged):", summary.get("phase2_corrected_overall"))
    print("phase3_corrected_overall (cumulative):", summary["phase3_corrected_overall"])


if __name__ == "__main__":
    main()

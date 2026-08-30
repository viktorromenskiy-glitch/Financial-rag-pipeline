"""Plan generation-error-analysis, Phase 4 result: manual review of the 15
most disputed cases out of the 36 questions squarely in scope for Phase 4's
raw_response taxonomy work (see scripts/apply_phase3_context_gap_corrections.py
docstring for how the 36-question pool was arrived at).

Trigger for this review: while doing the qualitative raw_response taxonomy
read Phase 4 of the plan calls for, 15 cases looked less like straightforward
model reasoning failures and more like further instances of the same
dataset/question defect pattern Phases 1-3 already found. Given repeated
self-caught quality problems while drafting a first-pass analysis alone
(an arithmetic slip fixed 2026-08-23, and three iterations needed to write a
document that actually stood on its own for an external reader - documented
in this project's internal working rules, not in this repository), this
round was not resolved by solo analysis. Instead it went through the process
now codified in this project's internal external-review procedure (also not
in this repository): multiple independent
external model reviews of the same primary-source material (raw
table/pre_text/post_text/context, full raw_response, and sibling questions
sharing the same context_id - the same forensic method Phases 1-3 used),
each subsequent reviewer given a neutral summary of prior reviewers' findings
and disagreements and instructed to verify independently before agreeing or
disputing, plus - for the two cases that stayed genuinely unresolved after
that - a check against the real, filed source document (SEC EDGAR / company
investor-relations site) where the working hypothesis was specifically "the
needed number is missing from the extracted fragment" rather than "which of
several present numbers was intended". Full per-case reasoning chains and the
review process are in this project's internal working materials (not in
this repository); this docstring summarizes the result.

Result, by category:

- gold_label_defect (3: finqa_train_2273, finqa_train_518, tatqa_train_1350):
  the eval program computed the SAME quantity the question asks for, but
  arithmetically wrong (summed an extra year, summed two dates instead of
  reading the value at one date, took one year's figure instead of averaging
  two). Model's own answer is objectively, independently verifiable correct
  from the same source text -> corrected_stage="success".

- question_label_mismatch (7: finqa_test_690, tatqa_train_6282,
  tatqa_train_6820, tatqa_train_7624, tatqa_train_8654, finqa_train_932,
  tatqa_train_7746): the eval program computed a real, source-derivable
  value, but for a different period/table/entity/metric than the question
  literally asks (wrong year, wrong acquisition, wrong reconciling line,
  probable wrong table scope, or - confirmed via two independent
  same-context_id sibling rows, tatqa_train_7746 and its sibling
  tatqa_train_7745 - a systematic question-label bug: "defined benefit
  obligation" in the question text where the eval program actually computes
  "Total contributions"). Same convention as Phases 1-3: not success, even
  where the model's answer is independently the objectively correct read of
  the literal question, because the benchmark case itself asked/answered two
  different things - corrected_stage=None.

- context_data_inconsistency (1: finqa_train_6066): the dataset's own two
  representations of gold for this question do not agree with each other
  (original_answer text says "decreased 29333", program_answer is -28933,
  a 400 discrepancy) - and neither matches reality. Advance Auto Parts'
  actual filed FY2012 10-K (SEC EDGAR, accession 0001158449-13-000069,
  Selected Consolidated Financial Data) reports Cost of sales of
  $2,963,888 thousand in fiscal 2010 and $3,106,967 thousand in fiscal 2012
  - an INCREASE of $143,079 thousand, not a decrease of any size. Neither
  gold value nor the model's own answer (5,467, the change in the LIFO
  reduction effect - the only cost-of-sales-adjacent figures present in the
  captured text) is the real answer to the literal question -
  corrected_stage=None.

- judge_tolerance_gap (1: tatqa_train_5282) - NEW category, not seen in
  Phases 1-3, distinct from the judge-calibration issue Phase 1 already
  flagged for a different reason: the source table's own "Change" column
  states 22.6% (171,819 vs 140,176 revenue, exact calculation =
  22.573764...%); gold_answer is 23.0. Both are legitimate representations
  of the identical underlying value at different rounding precision - 22.6%
  is that value rounded to one decimal place, 23% is the SAME value
  correctly rounded to the nearest whole percent (0.573764 > 0.5 rounds up).
  This is not a computational defect in gold (unlike the gold_label_defect
  cases above, no arithmetic operation was done wrong) and not a genuine
  question/label mismatch (both figures answer the same, literal question) -
  it is a precision/tolerance question for the judge, not a dataset defect
  requiring correction. corrected_stage=None; not counted as
  confirmed_generation_error either, since the model's answer is not wrong.

- confirmed_generation_error (2: finqa_train_4334, tatqa_train_4978): stage
  intentionally left unchanged - reason recorded for the audit trail, but
  per the established convention this reason keyword means "no bucket
  change". Both cases also have a separately-flagged dataset-side issue
  (documented in the evidence field) but that does NOT make the model's
  specific answer in this run correct: finqa_train_4334's raw_response does
  the right operation (853.5 x 1.049) but miscomputes the result as 895.5
  instead of 895.3215; tatqa_train_4978's raw_response invents a company
  ("Teradyne") and figures (2,805 / 2,587) that do not appear anywhere in
  the provided CTS Corporation document - an ungrounded hallucination, not
  a defensible alternate reading of a real table.

- unresolved (1: tatqa_train_2109): no combination of numbers in the
  provided context reproduces gold_answer=3472, and a direct look at
  Vodafone's real FY2019 report did not resolve it either (the needed
  reconciliation note, if it exists, is deeper in the document than the
  available fetch tooling could reliably reach - confirmed on two separate
  large annual-report/10-K documents, see pravila_vneshney_ekspertizy.md).
  Left unresolved rather than guessed at, per this project's standing rule
  against fabricating explanations that do not check out arithmetically.

Builds directly on scripts/apply_phase1_judge_corrections.py,
apply_phase2_context_gap_corrections.py and
apply_phase3_context_gap_corrections.py - identical non-destructive
`manual_correction_phase4` annotation approach: does not touch
results/error_analysis_250/predictions.jsonl or eval_results.jsonl, does not
change any record's original failure_stage/judge_correct, and does not touch
the manual_correction / manual_correction_phase2 / manual_correction_phase3
fields from earlier phases either. Idempotent.

Net effect on generation_failure_candidate (cumulative view,
phase4_corrected_overall): 38 (post Phase 3) -> 25 (-13: the 3
gold_label_defect, 7 question_label_mismatch, 1 context_data_inconsistency,
and 1 judge_tolerance_gap cases all leave the bucket; the 1 unresolved case
ALSO leaves - "unresolved" is its own named bucket, not a synonym for
"still a confirmed generation failure" - only the 2 confirmed_generation_error
cases stay). Of the 13 that leave: only the 3 gold_label_defect cases move to
success; the other 10 move to their own named buckets, not success, per the
standing convention that a reasonable-but-mismatched, ambiguous-scope, or
unresolved-origin answer is not the same thing as the model getting the
judged question right.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results/retrieval_trace_250")
ATTRIBUTION_RESULTS_PATH = RESULTS_DIR / "attribution_results.jsonl"
ATTRIBUTION_SUMMARY_PATH = RESULTS_DIR / "attribution_summary.json"
PHASE4_CORRECTIONS_PATH = RESULTS_DIR / "phase4_manual_corrections.jsonl"

GOLD_LABEL_DEFECT = "gold_label_defect"
QUESTION_LABEL_MISMATCH = "question_label_mismatch"
CONTEXT_DATA_INCONSISTENCY = "context_data_inconsistency"
JUDGE_TOLERANCE_GAP = "judge_tolerance_gap"
CONFIRMED_GENERATION_ERROR = "confirmed_generation_error"
UNRESOLVED = "unresolved"

CORRECTIONS = {
    "finqa_train_4334": {
        "reason": CONFIRMED_GENERATION_ERROR,
        "corrected_stage": None,
        "evidence": (
            "Question asks for 2017 smokeless-products volume assuming the same growth rate as 2016 "
            "(\"853.5 million units in 2016, an increase of 4.9% from 2015\"). Correct operation: "
            "853.5 x 1.049 = 895.3215. Model's raw_response does the right operation but states the "
            "result as 895.5 - its own arithmetic is off by ~0.2. Separately, gold_answer=853.549 is "
            "853.5+0.049 - the eval program added 4.9% as the raw number 0.049 instead of applying it "
            "as a growth rate, a genuine gold-side defect - but that does not make the model's own "
            "895.5 the correct answer either. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #1)."
        ),
    },
    "finqa_test_690": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: \"percentage increase in the NASDAQ Composite Index from January 1, 2011, to "
            "December 31, 2011\". Table: 1/1/2011=117.61, 12/31/2011=118.7 -> (118.7-117.61)/117.61 = "
            "0.93% (model's answer). gold_answer=0.1761 = (117.61-100)/100 - the change over the PRIOR "
            "period (1/2/2010 to 1/1/2011), not the period literally named in the question. Confirmed "
            "via 3-reviewer external check (claude/phase4_final_taxonomy.md, case #2)."
        ),
    },
    "finqa_train_2273": {
        "reason": GOLD_LABEL_DEFECT,
        "corrected_stage": "success",
        "evidence": (
            "Question asks for BlackRock's total 2017 increase in deferred carried interest liability "
            "\"due to acquisitions and unrealized allocations\". Table: 2017 unrealized allocations=75, "
            "2017 acquisition=13 -> 75+13=88 (model's answer, exactly matching the two named 2017 line "
            "items). gold_answer=125 = 75+13+37, where 37 is 2016's unrealized-allocations figure - the "
            "eval program summed in an extra year the question never asked about. Confirmed via "
            "3-reviewer external check (claude/phase4_final_taxonomy.md, case #3)."
        ),
    },
    "finqa_train_518": {
        "reason": GOLD_LABEL_DEFECT,
        "corrected_stage": "success",
        "evidence": (
            "Question: \"As of December 31, 2008, what was the total amount of liabilities acquired... "
            "for the BFI post-retirement healthcare plan\". Text: \"liabilities acquired for this plan "
            "were $1.2 million and $1.3 million, respectively, at the acquisition date and at december "
            "31, 2008\" - the value AS OF the requested date is explicitly 1.3 (model's answer). "
            "gold_answer=2.5 = 1.2+1.3, summing the value at two different dates as if they were "
            "additive amounts rather than reading the single point-in-time value the question asks for. "
            "Confirmed via 3-reviewer external check (claude/phase4_final_taxonomy.md, case #4)."
        ),
    },
    "tatqa_train_4978": {
        "reason": CONFIRMED_GENERATION_ERROR,
        "corrected_stage": None,
        "evidence": (
            "Question: percentage change in company contributions to the U.S. pension plans, 2018 to "
            "2019. Model's raw_response abandons the CTS Corporation table actually in front of it "
            "(U.S. Pension Plans company contributions: 2019=103, 2018=103, i.e. 0% change) and instead "
            "answers 8.43% using a company name (\"Teradyne\") and figures (2,805 / 2,587) that do not "
            "appear anywhere in the provided document - an ungrounded hallucination, not a defensible "
            "alternate reading. Separately, gold_answer=-7.64 exactly matches the SAME document's "
            "Post-Retirement Life Insurance Plan contributions ((145-157)/157=-7.64%), a different table "
            "than the one the question names - a real, separate question_label_mismatch-shaped issue on "
            "the gold side - but that does not make the model's own 8.43% correct. Confirmed via "
            "3-reviewer external check (claude/phase4_final_taxonomy.md, case #5)."
        ),
    },
    "tatqa_train_6282": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: difference in D&A between Software Solutions and Corporate and Other \"for the "
            "year ended December 31, 2019\". 2019 figures: 123.9 vs 96.4 -> 27.5 (model's answer). "
            "gold_answer=22.9 = 112.9-90.0, the 2018 figures - the eval program used the wrong year's "
            "column entirely, despite the question naming 2019 explicitly. Confirmed via 3-reviewer "
            "external check (claude/phase4_final_taxonomy.md, case #6)."
        ),
    },
    "tatqa_train_6820": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: change in TOTAL stock-based compensation expense, net of tax, 2018 to 2019. "
            "Table extraction is damaged (duplicated row labels), but two distinct rows are legible: "
            "\"Stock-based compensation expense, net of tax\" (2019=6962, 2018=7155 -> -193 = gold) and "
            "\"Total stock-based compensation expense, net of tax\" (2019=5303, 2018=5723 -> -420 = "
            "model's answer). The model's row match is exact on the word \"Total\" in the question. "
            "Sibling tatqa_train_6821 (same context_id, asking for the percentage change in the same "
            "\"Total\" line 2017-2018) has gold=-0.19%, which only reconciles against the \"Total\" row "
            "(5723 vs 5734) - independent proof the eval program's own convention for this table is the "
            "\"Total\" row, making this case's own non-Total-row gold an inconsistent application of that "
            "same convention. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #7)."
        ),
    },
    "tatqa_train_5282": {
        "reason": JUDGE_TOLERANCE_GAP,
        "corrected_stage": None,
        "evidence": (
            "Question: percentage change in Altium's total product revenue 2018 to 2019. Source table's "
            "own \"Change\" column literally states 22.6% (171,819 vs 140,176 -> exact calculation "
            "22.573764...%) - model's answer. gold_answer=23.0. Both figures are the SAME underlying "
            "value at different rounding precision: 22.6% is that value to one decimal place, 23% is "
            "the correct nearest-whole-percent rounding (0.573764 > 0.5 rounds up) - not two competing "
            "interpretations, and not a computational defect in gold. This is a judge precision-tolerance "
            "question (same general issue class as Phase 1's <=0.5-point tolerance finding, but a "
            "distinct instance/mechanism), not a dataset defect requiring a corrected_stage change. "
            "Confirmed via 3-reviewer external check where the majority view (2 of 3 reviewers) argued "
            "for success/gold_label_defect but did so using an arithmetically incorrect rebuttal "
            "(asserted \"standard rounding gives 22.6, not 23\" without checking whole-percent rounding "
            "specifically) - resolved by direct verification rather than vote count, per this project's "
            "internal rule that direct, unambiguously checkable math takes precedence over review consensus."
        ),
    },
    "tatqa_train_7624": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: how many liability components across BOTH the 'Accounts payable, accrued "
            "expenses and other current liabilities' table AND the 'Other liabilities' table exceed "
            "$2,000 thousand. Literal count across both named tables at the stated $2,000 threshold: "
            "5 (32,878 / 10,092 / 5,616 / 2,595 / 3,244) - model's answer, fully itemized. "
            "gold_answer=2, which is exactly reproduced by counting only the 'Other liabilities' table "
            "(2,595 + 3,244) at the SAME stated $2,000 threshold - no invented threshold needed, only an "
            "eval-program scope that covers one of the two tables the question names, not both. Best-"
            "supported hypothesis after external review (not proven with full certainty - flagged as "
            "such by all reviewers); the eval program produced a real, source-derivable number under a "
            "narrower scope than the literal question asks for, which is the same shape of defect as the "
            "other question_label_mismatch cases in this batch. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #9)."
        ),
    },
    "tatqa_train_1350": {
        "reason": GOLD_LABEL_DEFECT,
        "corrected_stage": "success",
        "evidence": (
            "Question: average Q3 dividend per share, 2018 and 2019. Table: Q3 2019=$0.01, Q3 2018="
            "$0.02 -> average=(0.01+0.02)/2=0.015 (model's answer, arithmetic matches the literal "
            "question). gold_answer=0.01 is just the 2019 figure alone - the eval program never actually "
            "averaged the two years the question asks for. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #10)."
        ),
    },
    "tatqa_train_8654": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: difference between current assets and fixed assets acquired in the Meta Networks "
            "acquisition. Meta Networks table: current=356, fixed=68 -> 288 (model's answer, exactly the "
            "acquisition the question names). gold_answer=22390 = 23,344-954, the current/fixed assets "
            "of Wombat Security - a DIFFERENT acquisition described in the same source document. "
            "Confirmed via 3-reviewer external check (claude/phase4_final_taxonomy.md, case #11)."
        ),
    },
    "finqa_train_6066": {
        "reason": CONTEXT_DATA_INCONSISTENCY,
        "corrected_stage": None,
        "evidence": (
            "Question: change in cost of sales, fiscal 2010 to fiscal 2012, Advance Auto Parts. The "
            "dataset's own two representations of gold disagree with each other: original_answer (text) "
            "says \"decreased 29333\", program_answer (number) is -28933, a 400 discrepancy. Checked "
            "against the real filed 10-K (SEC EDGAR CIK 1158449, accession 0001158449-13-000069, "
            "'Selected Consolidated Financial Data'): actual Cost of sales was $2,963,888 thousand in "
            "fiscal 2010 (ended Jan 1, 2011) and $3,106,967 thousand in fiscal 2012 (ended Dec 29, "
            "2012) - an INCREASE of $143,079 thousand, not a decrease of any size. Neither gold "
            "representation nor the model's own answer (5,467, the change in the LIFO reduction effect - "
            "the only cost-of-sales-adjacent figures present in the captured text fragment) matches the "
            "real reported change. Confirmed via 3-reviewer external check plus direct primary-source "
            "verification (claude/phase4_final_taxonomy.md, case #12)."
        ),
    },
    "finqa_train_932": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: percentage difference between carrying value and fair value of 3M's long-term "
            "debt AS OF December 31, 2012. Table has duplicated-looking column headers for 2012 and 2011 "
            "(carrying=4916/fair=5363 for 2012; carrying=4484/fair=5002 for 2011). Literal question, "
            "2012 only: (5363-4916)/4916 = 9.09% (model's answer). gold_answer=0.09634255129348795 "
            "matches (4916-4484)/4484 to 14 significant figures exactly - the year-over-year GROWTH of "
            "carrying value 2011->2012, an entirely different metric than the carrying-vs-fair-value "
            "spread the question asks for. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #13)."
        ),
    },
    "tatqa_train_2109": {
        "reason": UNRESOLVED,
        "corrected_stage": None,
        "evidence": (
            "Question: average adjusted profit before tax for Vodafone, 2019. No combination of numbers "
            "in the provided context (adjusted profit attributable to owners=1,451; taxation adjustment="
            "792; impairment=3,525; India=3,535; non-controlling interests=-5; adjusted income tax "
            "expense=2,613) reproduces gold_answer=3472 under any tested formula. Model's own raw_response "
            "mislabels the 2,613 adjusted-income-tax-expense figure as \"loss before tax\", which it is "
            "not, per the source text. A direct check against Vodafone's real FY2019 annual report/results "
            "documents did not resolve this either - the specific reconciliation note, if present, sits "
            "deeper in the document than available fetch tooling could reliably reach (confirmed on two "
            "separate large source documents in this batch, per this project's internal external-review "
            "procedure). "
            "Separately, the question's use of \"average\" for a single fiscal year is suspicious - "
            "sibling tatqa_train_2108 gives a standalone, non-averaged 2018 figure (4,408) - but this is "
            "an unproven observation, not a resolution, and is recorded as such rather than treated as an "
            "explanation. Left unresolved per claude/phase4_final_taxonomy.md, case #14."
        ),
    },
    "tatqa_train_7746": {
        "reason": QUESTION_LABEL_MISMATCH,
        "corrected_stage": None,
        "evidence": (
            "Question: percentage increase in Cedrik Neike's defined benefit obligation (DBO), 2018 to "
            "2019. DBO: 1,239,785 -> 1,862,660 = 50.24% (model's answer, matches the literal question). "
            "gold_answer=2.0 exactly matches the percentage change in a DIFFERENT column, Total "
            "contributions (604,800 -> 616,896 = +2.0%). Independently confirmed via sibling "
            "tatqa_train_7745 (same context_id, Joe Kaeser): its gold_answer=24,360 is not his DBO "
            "difference (~1.33M) but exactly his Total contributions difference "
            "(1,234,800-1,210,440=24,360) - the same systematic question-label bug in two independent "
            "rows of the same table. Confirmed via 3-reviewer external check "
            "(claude/phase4_final_taxonomy.md, case #15)."
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
            r["manual_correction_phase4"] = {
                "reviewed": (
                    "plan_generation_error_analysis phase 4 precheck (15 most-disputed cases, "
                    "multi-reviewer external process), 2026-08-23"
                ),
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
    """Applies Phase 1's correction, then Phase 2's, then Phase 3's, then
    Phase 4's on top - the final, cumulative view of where this question_id
    nets out after all four manual review phases."""
    stage = r["failure_stage"]
    for field in (
        "manual_correction",
        "manual_correction_phase2",
        "manual_correction_phase3",
        "manual_correction_phase4",
    ):
        mc = r.get(field)
        if mc is None:
            continue
        if mc["corrected_stage"] is not None:
            stage = mc["corrected_stage"]
        elif mc["reason"] != "confirmed_generation_error":
            stage = mc["reason"]  # e.g. question_label_mismatch, judge_tolerance_gap, unresolved
        # confirmed_generation_error: no change - it's a confirmation of the
        # existing classification, not a move to a new bucket.
    return stage


def main() -> None:
    records = load_jsonl(ATTRIBUTION_RESULTS_PATH)
    corrected = apply_corrections(records)
    write_jsonl(ATTRIBUTION_RESULTS_PATH, corrected)

    write_jsonl(
        PHASE4_CORRECTIONS_PATH,
        [
            {"question_id": qid, "reason": c["reason"], "corrected_stage": c["corrected_stage"], "evidence": c["evidence"]}
            for qid, c in CORRECTIONS.items()
        ],
    )

    summary = json.loads(ATTRIBUTION_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["phase4_corrected_overall"] = dict(Counter(cumulative_corrected_stage(r) for r in corrected))
    summary["phase4_corrections_note"] = (
        "15 questions manually reviewed (plan_generation_error_analysis.md phase 4 precheck) - the "
        "most disputed cases out of the 36-question pool squarely in scope for Phase 4's raw_response "
        "taxonomy work. Reviewed via a multi-reviewer external process (claude/pravila_vneshney_"
        "ekspertizy.md in the project) rather than solo analysis, after solo drafts of this same review "
        "needed repeated self-correction. 13 of 15 leave generation_failure_candidate: 3 gold_label_defect "
        "cases move to success (model's answer is objectively, independently verifiable correct); "
        "7 question_label_mismatch, 1 context_data_inconsistency, and 1 judge_tolerance_gap case move to "
        "their own buckets, not success (an answer that is reasonable, or even objectively correct, for a "
        "DIFFERENT question/scope/precision than the eval program actually computed is not the same as "
        "the model getting the judged case right); 1 more case (tatqa_train_2109) moves to its own "
        "unresolved bucket - no source-text combination nor a direct check of the real filed document "
        "reproduced gold_answer, so it is recorded as genuinely unexplained rather than left implicitly "
        "counted as a confirmed model error. Only 2 confirmed_generation_error cases (a genuine own-"
        "arithmetic slip, and a hallucinated company/figures not present in the source) stay in "
        "generation_failure_candidate - each also has a separately-documented, but non-exculpatory, "
        "dataset-side issue. Full per-question evidence in "
        "phase4_manual_corrections.jsonl and claude/phase4_final_taxonomy.md. phase4_corrected_overall is "
        "CUMULATIVE (Phase 1 + 2 + 3 + 4 corrections all applied) - phase1/2/3_corrected_overall above "
        "remain unchanged historical snapshots; original `overall` remains the untouched raw "
        "judge-derived classification."
    )
    ATTRIBUTION_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("phase3_corrected_overall (unchanged):", summary.get("phase3_corrected_overall"))
    print("phase4_corrected_overall (cumulative):", summary["phase4_corrected_overall"])


if __name__ == "__main__":
    main()

"""Plan generation-error-analysis, Phase 1 result: manual correction of the
5 questions where the LLM judge and is_close_v2 disagreed
(deterministic_match=True, judge_correct=False) in results/error_analysis_250.

This is a one-off, hand-reviewed correction, not a reusable classifier - each
of the 5 cases required reading the actual gold-document text and (for two of
them) cross-referencing a human-annotated `original_answer` on a sibling
question sharing the same context_id, to determine WHY the judge and
is_close_v2 disagreed. See docs/tehnicheskoe_zadanie.md, the Phase 1 write-up
(new section following the plan-2 item 7 error-attribution section), for the
full evidence for each case; short version:

- finqa_dev_569: gold-document text literally says "serving 192 countries" -
  the model's answer (192) is verbatim correct. The dataset's own gold_answer
  field (193.0) appears to be an annotation defect, not a model or judge
  error. Reclassified success; reason=gold_label_defect.
- finqa_dev_451, finqa_train_2226: the judge was shown a raw fraction as
  "gold" (e.g. 1.424025...) and the model correctly followed its own prompt
  instruction to express percentages as a plain 0-100 number (142.4). This is
  exactly the "percentage vs fraction" equivalence JUDGE_PROMPT
  (pipeline/evaluation.py) explicitly tells the judge to accept - confirmed
  against a sibling question's human-annotated original_answer ("142.4%" /
  "1%" respectively) in the same source table. The judge violated its own
  stated instruction. Reclassified success; reason=judge_calibration_error.
- tatqa_train_8832: gold (119655.67) reproduces exactly from the three
  values actually present in the model's retrieved context (160320, 109454,
  89193 - all three verbatim-confirmed in the chunk text). The model's answer
  (118989.67) does not reproduce from any combination of those visible
  numbers - most likely a column-reading slip in a confusingly laid-out
  side-by-side table (two year-pairs sharing one flattened header row). This
  is a genuine generation-stage error; is_close_v2's blanket 1% relative
  tolerance was just loose enough to call it "close". NOT reclassified -
  stays generation_failure_candidate, reason=confirmed_generation_error.
- tatqa_train_1740: the model's answer (50226.67) is the EXACT average of
  three values literally present in its own retrieved context (70723, 47879,
  32078). The gold value (50158.67) implies a different 2017 figure (~31874)
  that is not present in that chunk at all - and a THIRD variant of the same
  underlying line item (32178) appears in a different chunk of the same
  filing. This is a table-extraction inconsistency in the underlying
  T2-RAGBench source data (the same figure reads differently across chunks),
  not a model reasoning failure and not simply "judge miscalibration" either.
  NOT reclassified as success (the model didn't reproduce the official gold
  value) and NOT counted as a confirmed generation error (the model was
  arithmetically correct given its own input) - tracked as a new third
  category, reason=context_data_inconsistency.

Net effect on the 55 generation_failure_candidate count: 55 -> 51 confirmed
generation errors, +3 reclassified to success (186 -> 189), +1 moved to a new
context_data_inconsistency bucket that is deliberately not folded into either
success or generation_failure_candidate.

Does NOT touch results/error_analysis_250/eval_results.jsonl (the raw judge
output stays as an honest historical record of what the judge actually
said - the correction is applied as an annotation layer on top, not by
rewriting history) or pipeline/attribution.py's classify() logic (that
function is, and remains, a purely deterministic function of retrieval
membership + raw judge_correct - the manual Phase 1 review is out of its
scope by design).

Idempotent: running this twice produces the same output (it recomputes
manual_correction from the CORRECTIONS table below and overwrites the
derived files, it does not accumulate).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results/retrieval_trace_250")
ATTRIBUTION_RESULTS_PATH = RESULTS_DIR / "attribution_results.jsonl"
ATTRIBUTION_SUMMARY_PATH = RESULTS_DIR / "attribution_summary.json"
PHASE1_CORRECTIONS_PATH = RESULTS_DIR / "phase1_manual_corrections.jsonl"

CONTEXT_DATA_INCONSISTENCY = "context_data_inconsistency"

# question_id -> correction record. `corrected_stage` is None for the one
# case that is deliberately NOT folded into success or
# generation_failure_candidate (see module docstring, tatqa_train_1740).
CORRECTIONS = {
    "finqa_dev_569": {
        "reason": "gold_label_defect",
        "corrected_stage": "success",
        "evidence": (
            'Gold-document text (finqa_dev_ctx_20) reads verbatim: "48 located in the '
            'united states and 123 located outside the united states, serving 192 '
            'countries." Model answered 192 - matches the source document exactly. '
            "Dataset's gold_answer field (193.0) does not match its own source text; "
            "treated as an annotation defect, not a model or judge error."
        ),
    },
    "finqa_dev_451": {
        "reason": "judge_calibration_error",
        "corrected_stage": "success",
        "evidence": (
            "gold_answer shown to the judge was the raw fraction 1.424025457438345. "
            "Model answered 142.4 (= gold x 100), following PROMPT_TEMPLATE's own "
            "instruction to express percentages as a plain 0-100 number. A sibling "
            "question at the same context_id (finqa_dev_ctx_236, id=finqa_dev_131..764) "
            "has human-annotated original_answer=\"142.4%\" for this exact figure - "
            "confirming 142.4 is the correct percentage-form answer. JUDGE_PROMPT "
            "(pipeline/evaluation.py) explicitly instructs accepting 'equivalent "
            "expression as percentage vs fraction' - the judge did not apply its own rule."
        ),
    },
    "finqa_train_2226": {
        "reason": "judge_calibration_error",
        "corrected_stage": "success",
        "evidence": (
            "gold_answer shown to the judge was the raw fraction 0.005470916481712618. "
            "Model answered 0.55 (= gold x 100, rounded to 2dp), the same "
            "percentage-vs-fraction pattern as finqa_dev_451. Sibling question at the "
            "same context_id (finqa_train_ctx_1068) has human-annotated "
            'original_answer="1%" for this exact figure (0.547%, rounded to the nearest '
            "whole percent by the human annotator) - confirms the fraction/percentage "
            "correspondence. Judge again did not apply its own stated equivalence rule."
        ),
    },
    "tatqa_train_8832": {
        "reason": "confirmed_generation_error",
        "corrected_stage": None,  # stays generation_failure_candidate - no change
        "evidence": (
            "gold=119655.67 reproduces exactly as the average of 160320 (2019), 109454 "
            "(2018), 89193 (2017) - all three verbatim-present in the model's own "
            "retrieved context (context_id=cf16db5516b9b12a1b9cbaabe955a922). Model's "
            "answer 118989.67 does not reproduce from any combination of those visible "
            "numbers (implied 2017 figure ~87195 is absent from the context entirely). "
            "The source table lays out two side-by-side year-pair sub-tables (2019-vs-2018 "
            "and 2018-vs-2017) sharing one flattened header row with '2018' appearing "
            "twice - consistent with a column-reading slip. Confirmed genuine "
            "generation-stage error; is_close_v2's 1% relative tolerance was simply loose "
            "enough (0.56% actual deviation) to flag it as numerically close anyway."
        ),
    },
    "tatqa_train_1740": {
        "reason": CONTEXT_DATA_INCONSISTENCY,
        "corrected_stage": None,  # NOT success, NOT counted as confirmed generation error
        "evidence": (
            "Model's answer 50226.67 is the exact average of 70723, 47879, 32078 - all "
            "three verbatim-present in its own retrieved context "
            "(context_id=6e13cba60fe2c0425f78aa9eb8fdfa15, 'Other' row under Cost of "
            "revenues). gold=50158.67 requires a 2017 figure of ~31874, which is absent "
            "from this chunk. A THIRD variant of the same underlying line item (32178) "
            "appears in a different chunk of the same filing "
            "(context_id=cf16db5516b9b12a1b9cbaabe955a922). Three different values for "
            "what should be one reported number (31874 gold / 32078 here / 32178 "
            "elsewhere) indicates a table-extraction inconsistency in the underlying "
            "T2-RAGBench source data, not a model reasoning failure - the model is "
            "arithmetically correct given what it was actually shown. Deliberately not "
            "folded into success (doesn't match official gold) or generation error "
            "(model's own arithmetic on its own input is correct) - tracked separately."
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
    """Returns a new list with a `manual_correction` field added to the 5
    corrected records (all other records unchanged, and the original
    `failure_stage`/`judge_correct` fields on the corrected records are
    LEFT AS-IS - the raw deterministic classification stays visible; the
    correction is an explicit annotation on top, not a silent rewrite)."""
    corrected = []
    seen = set()
    for r in records:
        r = dict(r)
        qid = r["question_id"]
        if qid in CORRECTIONS:
            seen.add(qid)
            c = CORRECTIONS[qid]
            r["manual_correction"] = {
                "reviewed": "internal error-analysis plan, phase 1, 2026-08-22",
                "reason": c["reason"],
                "corrected_stage": c["corrected_stage"],
                "evidence": c["evidence"],
            }
        corrected.append(r)
    missing = set(CORRECTIONS) - seen
    if missing:
        raise ValueError(f"question_id(s) not found in attribution_results.jsonl: {sorted(missing)}")
    return corrected


def corrected_stage_counts(records: list[dict]) -> Counter:
    """The Phase-1-corrected view of failure_stage counts: applies
    corrected_stage where a manual_correction says to move a question,
    otherwise keeps the original deterministic failure_stage. The one
    context_data_inconsistency case gets its own bucket (corrected_stage is
    None but reason is set) rather than silently staying counted as
    generation_failure_candidate."""
    counts: Counter = Counter()
    for r in records:
        mc = r.get("manual_correction")
        if mc is None:
            counts[r["failure_stage"]] += 1
        elif mc["corrected_stage"] is not None:
            counts[mc["corrected_stage"]] += 1
        elif mc["reason"] == CONTEXT_DATA_INCONSISTENCY:
            counts[CONTEXT_DATA_INCONSISTENCY] += 1
        else:
            counts[r["failure_stage"]] += 1  # confirmed_generation_error: no change
    return counts


def main() -> None:
    records = load_jsonl(ATTRIBUTION_RESULTS_PATH)
    corrected = apply_corrections(records)
    write_jsonl(ATTRIBUTION_RESULTS_PATH, corrected)

    write_jsonl(
        PHASE1_CORRECTIONS_PATH,
        [
            {"question_id": qid, **{"reason": c["reason"], "corrected_stage": c["corrected_stage"], "evidence": c["evidence"]}}
            for qid, c in CORRECTIONS.items()
        ],
    )

    original_summary = json.loads(ATTRIBUTION_SUMMARY_PATH.read_text(encoding="utf-8"))
    original_summary["phase1_corrected_overall"] = dict(corrected_stage_counts(corrected))
    original_summary["phase1_corrections_note"] = (
        "5 questions manually reviewed (internal error-analysis plan, not in this repository, phase 1) where "
        "judge_correct=False but is_close_v2 deterministic_match=True. See "
        "phase1_manual_corrections.jsonl for the per-question reason and evidence. Original "
        "`overall` above is left untouched (raw deterministic classification from judge "
        "output, unmodified); phase1_corrected_overall reflects the reviewed correction."
    )
    ATTRIBUTION_SUMMARY_PATH.write_text(json.dumps(original_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("original overall:", dict(Counter(r["failure_stage"] for r in records)))
    print("phase1_corrected_overall:", dict(corrected_stage_counts(corrected)))


if __name__ == "__main__":
    main()

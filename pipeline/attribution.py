"""Deterministic retrieval/generation error attribution (plan doradotki-2,
item 7 - see docs/tehnicheskoe_zadanie.md for the full writeup).

Origin and scope correction (important - read before using this module).
ChatGPT's original recommendation (see the re-uploaded audit doc, section
"2. Как сделать автоматическую attribution") scoped this to FinQA only,
reasoning that FinQA is the one T2-RAGBench source with an official
sub-document "gold_inds" (supporting facts) field, while ConvFinQA/TAT-DQA
have no comparable gold-evidence annotation. That premise turned out not to
apply to this project's actual data: this pipeline chunks one document per
context_id (see README "Chunking (one document = one chunk)"), and
data/t2-ragbench/eval_subset_250.parquet already carries an authoritative
`context_id` column per question for ALL THREE sources - the document that
question was written against - not just FinQA. So the "gold_inds -> your
internal context_id" mapping problem ChatGPT worried about (gold_inds are
sub-document supporting facts, not directly usable as a chunk id) does not
arise at this project's document-level chunking granularity: gold_context_id
IS just eval_subset_250.parquet's `context_id` column, verbatim, for every
source. This module therefore attributes all three sources, not FinQA-only
- see docs/tehnicheskoe_zadanie.md for the full writeup of this correction.

What this module does NOT do, deliberately (per the same source document's
own later self-correction): it does not attempt a fuzzy/semantic match
between gold and retrieved context - only exact context_id membership in
the top-50 (pre-rerank) and top-5 (post-rerank) pools. And it does not
build a single "one classifier fits all three datasets" black box beyond
this one binary in/out-of-pool check - the underlying gold-evidence
granularity (whole document) is uniform across sources here, which is what
makes a shared classifier honest in this specific case, unlike the
sub-document "gold_inds"-based classifier ChatGPT's own document warned
against building as one universal thing.

What this module needs that no committed run currently has: `candidate_top50`
(the pre-rerank pool). Neither results/error_analysis_250 (predates
per-question retrieval logging entirely) nor an ordinary `eval` run before
`--retrieval-only` existed saves that - only the post-rerank top-5
(`retrieved_docs`, see pipeline.cli._retrieved_docs_for_prediction). Real
numbers require a new `eval --retrieval-only` run (cheap - no
generation/judge API calls) producing retrieval_trace.jsonl, joined here
against an existing run's eval_results.jsonl for judge_correct. See
`attribute_run()` below.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

# The four categories a question can fall into. Terminology follows the
# audit document's own corrected terms: "generation_failure_candidate", not
# "generation_failure" - a wrong answer with the gold document already in
# the generation context is evidence generation went wrong, not proof (the
# document could contain the right number in a place the model still read
# incorrectly for an unrelated reason). See module docstring.
RETRIEVAL_FAILURE = "retrieval_failure"
RERANKING_FAILURE = "reranking_failure"
GENERATION_FAILURE_CANDIDATE = "generation_failure_candidate"
SUCCESS = "success"
UNKNOWN_OUTCOME = "unknown_generation_outcome"  # judge_correct not available for this question_id


def load_gold_context_ids(questions_path: str | Path) -> dict[str, str]:
    """Loads {question_id: gold_context_id} straight from the eval parquet's
    own `id`/`context_id` columns - no per-source raw-dataset lookup or
    gold_inds mapping needed, see module docstring for why.
    """
    df = pd.read_parquet(questions_path)
    if "id" not in df.columns or "context_id" not in df.columns:
        raise ValueError(f"{questions_path} is missing 'id' and/or 'context_id' - cannot recover gold evidence")
    return dict(zip(df["id"], df["context_id"]))


def classify(gold_context_id: str, candidate_top50: list[dict], reranked_top5: list[dict], judge_correct: bool | None) -> str:
    """The deterministic decision tree from the audit doc (section "Логика"),
    unchanged except for the terminology correction noted in the module
    docstring. Never uses an LLM - purely set membership on context_id.
    """
    top50_ids = {c["context_id"] for c in candidate_top50}
    top5_ids = {c["context_id"] for c in reranked_top5}

    if gold_context_id not in top50_ids:
        return RETRIEVAL_FAILURE
    if gold_context_id not in top5_ids:
        return RERANKING_FAILURE
    if judge_correct is None:
        return UNKNOWN_OUTCOME
    return SUCCESS if judge_correct else GENERATION_FAILURE_CANDIDATE


def attribute_run(retrieval_trace_path: str | Path, eval_results_path: str | Path, questions_path: str | Path) -> list[dict]:
    """Joins a retrieval_trace.jsonl (from `eval --retrieval-only`) with an
    existing run's eval_results.jsonl (judge_correct) by question_id, plus
    gold_context_id from the questions parquet, and classifies each
    question. Both jsonl files must cover the same question_id set (i.e.
    the same --questions file) - this is a deliberate requirement, not
    inferred or fuzzy-matched, since retrieval is only valid to reuse
    against a previous run's judge verdicts if it was computed for the
    exact same questions against the exact same indexed corpus.
    """
    gold = load_gold_context_ids(questions_path)

    judge_correct_by_qid: dict[str, bool] = {}
    with open(eval_results_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            judge_correct_by_qid[rec["question_id"]] = rec["judge_scores"]["judge_correct"]

    records = []
    with open(retrieval_trace_path, encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)
            qid = trace["question_id"]
            if qid not in gold:
                raise ValueError(f"{qid}: no gold context_id in {questions_path} - trace file/questions file mismatch?")
            gold_context_id = gold[qid]
            judge_correct = judge_correct_by_qid.get(qid)  # None if this qid wasn't judged in eval_results_path
            failure_stage = classify(gold_context_id, trace["candidate_top50"], trace["reranked_top5"], judge_correct)
            records.append(
                {
                    "question_id": qid,
                    "source_dataset": trace["source_dataset"],
                    "gold_context_id": gold_context_id,
                    "gold_in_top50": failure_stage != RETRIEVAL_FAILURE,
                    "gold_in_top5": failure_stage not in (RETRIEVAL_FAILURE, RERANKING_FAILURE),
                    "judge_correct": judge_correct,
                    "failure_stage": failure_stage,
                }
            )
    return records


def summarize_attribution(records: list[dict]) -> dict:
    """Overall + per-source_dataset counts of each failure_stage - the same
    mandatory stratification convention used everywhere else in this
    project (docs/tehnicheskoe_zadanie.md, section 10).
    """
    summary = {"n": len(records), "overall": dict(Counter(r["failure_stage"] for r in records))}
    by_source: dict[str, Counter] = {}
    for r in records:
        by_source.setdefault(r["source_dataset"], Counter())[r["failure_stage"]] += 1
    summary["by_source_dataset"] = {source: dict(counts) for source, counts in by_source.items()}
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run-id", required=True, help="run_id of an `eval --retrieval-only` run (has retrieval_trace.jsonl)")
    parser.add_argument("--judged-run-id", required=True, help="run_id of a full `eval` run to take judge_correct from (e.g. error_analysis_250)")
    parser.add_argument("--questions", required=True, help="The eval parquet both runs were computed against")
    args = parser.parse_args(argv)

    retrieval_trace_path = Path("results") / args.retrieval_run_id / "retrieval_trace.jsonl"
    eval_results_path = Path("results") / args.judged_run_id / "eval_results.jsonl"

    records = attribute_run(retrieval_trace_path, eval_results_path, args.questions)
    summary = summarize_attribution(records)

    out_dir = Path("results") / args.retrieval_run_id
    with (out_dir / "attribution_results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "attribution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

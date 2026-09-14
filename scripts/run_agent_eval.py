"""Day 2 agent evaluation harness - plan_rabot_posle_ekspertizy_agent_profil.md,
День 2, "Evaluation harness": runs BOTH the baseline pipeline (modules 6-9,
unchanged) and the bounded agent (agent/) over the SAME matched, stratified
sample of T2-RAGBench questions, plus 3 fixed prompt-injection canary
questions (agent/canary.py), so the two systems can be compared with a
paired test (scripts/mcnemar_agent_eval.py, run separately afterward on
this script's saved output).

This is a real-API-cost script, same category as scripts/run_eval.py and
scripts/run_unanswerable_probe.py - not runnable in a dev sandbox without
real MONGODB_URI/VOYAGE_API_KEY/ANTHROPIC_API_KEY/COHERE_API_KEY (see
.env.example). Every piece it calls (agent/, pipeline/, config/) is unit-
tested offline with fakes; this script itself has no test file, matching
the existing convention that scripts/*.py (thin real-API orchestration) is
not unit-tested while the library code it calls is (see tests/ - there is
no test_run_eval.py or test_mcnemar_phase6.py either).

Usage (Colab, after the usual %cd + secrets-loading cells, after `git
pull`):
    !python scripts/run_agent_eval.py

Idempotent/resumable: re-running with the same RUN_ID skips any
question_id/canary_id already present in the relevant checkpoint file and
does not re-spend money on it - same pattern as
scripts/run_unanswerable_probe.py and pipeline/cli.py's cmd_eval
generation checkpoint.

Global safety limits (agent/safety.py, config.agent_eval): if the run's
LLM-call/wall-clock/cost budget is exceeded partway through, this script
stops issuing new calls, prints a clear message, and leaves everything
already checkpointed on disk - re-running (after raising the budget in
config.yaml, if that was the intent) resumes rather than starting over.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ID = "agent_eval_day2"  # change for a fresh, non-resuming run
SAMPLE_SIZE = 30  # plan: "20-30 questions from T2-RAGBench"
RANDOM_SEED = 20260914  # date this harness was written - fixed for reproducibility, not re-rolled per run
QUESTIONS_PATH = REPO_ROOT / "data" / "t2-ragbench" / "eval_subset_250.parquet"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / RUN_ID

from dotenv import load_dotenv  # noqa: E402 - same load_dotenv() placement as scripts/run_unanswerable_probe.py

load_dotenv(REPO_ROOT / ".env")

from agent.canary import CANARY_CASES, contains_injection_marker  # noqa: E402
from agent.loop import ClaudeEvidenceAssessor, run_agent_query  # noqa: E402
from agent.run_metadata import build_agent_eval_run_metadata  # noqa: E402
from agent.safety import BudgetedGenerator, BudgetedJudge, BudgetExceededError, RunBudgetTracker, RunSafetyLimits  # noqa: E402
from agent.success import evaluate_agent_success  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, ClaudeJudge, build_clients, load_eval_questions, _resolve_embedding_model  # noqa: E402
from pipeline.common.paired_stats import compare_paired_binary_outcomes  # noqa: E402
from pipeline.common.run_config import write_run_config  # noqa: E402
from pipeline.common.sampling import stratified_sample  # noqa: E402
from pipeline.evaluation import evaluate_answer  # noqa: E402
from pipeline.generation import build_context_block, generate_answer  # noqa: E402
from pipeline.indexing import validate_startup_indexes  # noqa: E402
from pipeline.retrieval import Candidate  # noqa: E402


def _load_jsonl_checkpoint(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec[key]] = rec
    return done


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _is_routed(config, source_dataset: str) -> bool:
    routing = config.embedding.routing
    return routing.enabled and source_dataset in routing.routed_sources


def _canary_search_fn(document_content: str, canary_id: str):
    """Returns every reformulated query the SAME fixed candidate (the
    canary's injected document) - the point of a canary question is to
    guarantee the injected content actually reaches the assessor/generator
    prompts, not to test whether real hybrid retrieval would surface it.
    Only the search step is fixed; the assessment and generation calls
    below are real Claude API calls, same as for the regular sample."""
    from agent.tools import SearchToolCall

    candidate = Candidate(context_id=canary_id, full_indexed_content=document_content, score=1.0)

    def _search(query: str):
        return SearchToolCall(query=query, candidates=(candidate,))

    return _search


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS_DIR / "baseline_results.jsonl"
    agent_path = RESULTS_DIR / "agent_results.jsonl"
    canary_path = RESULTS_DIR / "canary_results.jsonl"
    trace_path = RESULTS_DIR / "agent_trace.jsonl"

    from agent.tracing import TraceWriter

    trace_writer = TraceWriter(trace_path)

    all_questions = load_eval_questions(str(QUESTIONS_PATH))
    sample = stratified_sample(all_questions, SAMPLE_SIZE, key="source_dataset", seed=RANDOM_SEED)
    print(f"Sampled {len(sample)} questions (stratified by source_dataset, seed={RANDOM_SEED}) from {QUESTIONS_PATH}")

    limits = RunSafetyLimits(
        max_llm_calls=config.agent_eval.max_llm_calls,
        max_wall_clock_seconds=config.agent_eval.max_wall_clock_seconds,
        max_estimated_cost_usd=config.agent_eval.max_estimated_cost_usd,
    )
    tracker = RunBudgetTracker(limits, cost_per_llm_call_usd=config.agent_eval.cost_per_llm_call_usd)

    raw_generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    raw_judge = ClaudeJudge(clients["anthropic"], config.judge.model, config.judge.temperature)
    generator = BudgetedGenerator(raw_generator, tracker)
    judge = BudgetedJudge(raw_judge, tracker)
    assessor = ClaudeEvidenceAssessor(generator)

    baseline_done = _load_jsonl_checkpoint(baseline_path, "question_id")
    agent_done = _load_jsonl_checkpoint(agent_path, "question_id")
    canary_done = _load_jsonl_checkpoint(canary_path, "canary_id")
    print(
        f"Resuming: {len(baseline_done)} baseline, {len(agent_done)} agent, "
        f"{len(canary_done)} canary result(s) already on disk"
    )

    try:
        for item in sample:
            question_id = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]
            source_dataset = item["source_dataset"]
            routed = _is_routed(config, source_dataset)
            embedding_model = _resolve_embedding_model(config, source_dataset)

            search_fn = functools.partial(
                search_documents,
                clients["voyage"],
                collection,
                clients["cohere"],
                pool_size=config.retrieval.pool_size,
                vector_weight=config.retrieval.weights.vector,
                text_weight=config.retrieval.weights.text,
                reranker_enabled=config.reranker.enabled,
                reranker_top_n=config.reranker.top_n,
                embedding_model=embedding_model,
                source_dataset=source_dataset if routed else None,
                exclude_source_datasets=list(config.embedding.routing.routed_sources) if not routed else None,
            )

            if question_id not in baseline_done:
                baseline_call = search_fn(question)
                if baseline_call.candidates:
                    generated = generate_answer(generator, question_id, question, list(baseline_call.candidates))
                    context_text = build_context_block(list(baseline_call.candidates))
                    # pipeline.evaluation.evaluate_answer below handles this
                    # normally (no special-casing needed - it just compares
                    # generated.answer_text, whatever it is, against
                    # gold_answer, same as pipeline.cli.cmd_eval would).
                    eval_result = evaluate_answer(judge, question_id, question, context_text, generated.answer_text, gold_answer)
                    baseline_record = {
                        "question_id": question_id,
                        "source_dataset": source_dataset,
                        "gold_answer": gold_answer,
                        "answer_text": generated.answer_text,
                        "judge_correct": eval_result.judge_scores["judge_correct"],
                        "deterministic_match": eval_result.deterministic_match,
                    }
                else:
                    # Defensive only - build_context_block()/generate_answer()
                    # would raise ValueError on empty candidates (unlike
                    # pipeline.cli.cmd_eval, which never guards this because
                    # empty retrieval essentially never happens on this
                    # already-indexed T2-RAGBench corpus). Recorded as a
                    # judged failure unless gold_answer itself is genuinely
                    # "INSUFFICIENT_CONTEXT", rather than crashing the whole
                    # matched-sample run over one edge case.
                    baseline_record = {
                        "question_id": question_id,
                        "source_dataset": source_dataset,
                        "gold_answer": gold_answer,
                        "answer_text": "INSUFFICIENT_CONTEXT",
                        "judge_correct": str(gold_answer).strip().upper() == "INSUFFICIENT_CONTEXT",
                        "deterministic_match": False,
                    }
                _append_jsonl(baseline_path, baseline_record)
                baseline_done[question_id] = baseline_record

            if question_id not in agent_done:
                agent_answer = run_agent_query(
                    question_id,
                    question,
                    search_fn,
                    assessor,
                    generator,
                    max_additional_tool_calls=config.agent.max_additional_tool_calls,
                    max_wall_clock_seconds=config.agent.max_wall_clock_seconds,
                    trace_writer=trace_writer,
                )
                # agent_answer.context_documents carries the exact
                # accumulated candidates the agent actually saw (Day 2
                # addition to AgentAnswer) - re-running search_fn(question)
                # here instead would silently miss anything a reformulated
                # query alone surfaced, which is wrong input for
                # evaluate_agent_success's insufficiency judge.
                context_text = build_context_block(list(agent_answer.context_documents)) if agent_answer.context_documents else ""
                success_result = evaluate_agent_success(
                    judge, question_id, question, context_text, agent_answer.answer_text, gold_answer
                )
                agent_record = {
                    "question_id": question_id,
                    "source_dataset": source_dataset,
                    "gold_answer": gold_answer,
                    "answer_text": agent_answer.answer_text,
                    "stopped_reason": agent_answer.stopped_reason,
                    "additional_calls_used": agent_answer.additional_calls_used,
                    "forced_insufficient": agent_answer.forced_insufficient,
                    "success": success_result.success,
                    "success_reason": success_result.reason,
                }
                _append_jsonl(agent_path, agent_record)
                agent_done[question_id] = agent_record

        for canary in CANARY_CASES:
            if canary.canary_id in canary_done:
                continue
            canary_search_fn = _canary_search_fn(canary.document_content, canary.canary_id)
            agent_answer = run_agent_query(
                canary.canary_id,
                canary.question,
                canary_search_fn,
                assessor,
                generator,
                max_additional_tool_calls=config.agent.max_additional_tool_calls,
                max_wall_clock_seconds=config.agent.max_wall_clock_seconds,
                trace_writer=trace_writer,
            )
            leaked = contains_injection_marker(agent_answer.answer_text, canary.injection_markers)
            canary_record = {
                "canary_id": canary.canary_id,
                "question": canary.question,
                "gold_answer": canary.gold_answer,
                "answer_text": agent_answer.answer_text,
                "forced_insufficient": agent_answer.forced_insufficient,
                "leaked_injection_marker": leaked,
                "note": canary.note,
            }
            _append_jsonl(canary_path, canary_record)
            canary_done[canary.canary_id] = canary_record
    except BudgetExceededError as exc:
        print(f"\n{'!' * 70}\nSTOPPED: evaluation-run safety budget exceeded: {exc}\n{'!' * 70}\n")
        print(f"Progress so far ({tracker.llm_calls} LLM calls, ~${tracker.estimated_cost_usd:.2f}) is saved to {RESULTS_DIR} - re-run to resume.")
        return

    print(f"\nDone. {len(baseline_done)} baseline, {len(agent_done)} agent, {len(canary_done)} canary result(s).")
    print(f"Budget used: {tracker.llm_calls} LLM calls, ~${tracker.estimated_cost_usd:.2f}, {tracker.elapsed_seconds:.0f}s")

    both_correct = a_only = b_only = both_wrong = 0
    matched = 0
    for question_id, baseline_record in baseline_done.items():
        agent_record = agent_done.get(question_id)
        if agent_record is None:
            continue
        matched += 1
        a_correct = agent_record["success"]  # agent = "a" throughout this comparison
        b_correct = baseline_record["judge_correct"]
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            both_wrong += 1

    comparison = None
    if matched > 0:
        comparison = compare_paired_binary_outcomes(both_correct, a_only, b_only, both_wrong)
        print(
            f"\nMatched n={matched}: agent {comparison.p_a:.3f}, baseline {comparison.p_b:.3f}, "
            f"difference {comparison.difference:+.3f} (95% CI [{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}]), "
            f"McNemar exact p={comparison.mcnemar_exact_pvalue:.4f}"
        )

    canary_leaks = sum(1 for r in canary_done.values() if r["leaked_injection_marker"])
    print(f"Canaries: {len(canary_done)} run, {canary_leaks} leaked an injection marker")

    write_run_config(
        {
            "embedding": config.embedding.model_dump(),
            "enrichment": config.enrichment.model_dump(),
            "retrieval": config.retrieval.model_dump(),
            "reranker": config.reranker.model_dump(),
            "generation": config.generation.model_dump(),
            "judge": config.judge.model_dump(),
        },
        RUN_ID,
        results_dir=str(REPO_ROOT / "results"),
    )
    metadata = build_agent_eval_run_metadata(config, RUN_ID)
    metadata["matched_n"] = matched
    metadata["comparison"] = None if comparison is None else comparison.__dict__
    metadata["canary_leaks"] = canary_leaks
    (RESULTS_DIR / "agent_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

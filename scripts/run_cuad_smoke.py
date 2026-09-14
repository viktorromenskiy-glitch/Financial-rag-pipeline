"""Day 3 CUAD smoke test - plan_rabot_posle_ekspertizy_agent_profil.md,
День 3: "Смок-тест на CUAD (Contract Understanding Atticus Dataset, CC BY
4.0): 3-5 вопросов, без изменения кода агента - только конфигурация
retrieval-индекса и индексация новых документов."

*** This is a robustness/regression-shaped smoke check on 4 real legal
*** contracts and 5 real clause-extraction questions (see
*** data/cuad_smoke/cuad_smoke_questions.json and its _license_notice key
*** for the CC BY 4.0 attribution) - it is NOT a claim that this project's
*** pipeline is a usable legal-document product. n=5 has no statistical
*** power, CUAD's "Highlight the parts... that should be reviewed by a
*** lawyer" questions are span-extraction, not the exact-numeric-answer
*** task this pipeline was designed and validated for, and no reranker/
*** embedding/prompt tuning was done for this domain. See README.md's
*** Agent section for the exact wording this project uses to describe
*** what a pass here does and does not demonstrate.

No agent/pipeline CODE changes were needed for this - only
config/config_cuad_smoke.yaml (a separate MongoDB collection/index pair,
so CUAD documents never mix with the real t2_ragbench_full corpus) and
pipeline/cuad_smoke.py (a fixture loader that turns the 4 CUAD contracts
into ordinary pipeline.ingestion.DocumentRecord objects - see that
module's docstring). Indexing below reuses
pipeline.ingestion.dedupe_documents, pipeline.enrichment.enrich_documents,
pipeline.embedding.embed_documents, and pipeline.indexing.index_corpus
completely unchanged - the same functions pipeline/cli.py's cmd_index
calls for the real corpus.

Real-API-cost script, same category as scripts/run_agent_eval.py - not
runnable in a dev sandbox without real MONGODB_URI/VOYAGE_API_KEY/
ANTHROPIC_API_KEY/COHERE_API_KEY. pipeline/cuad_smoke.py (the only new
library code this needs) is unit-tested offline with the real committed
fixture; this script itself has no test file, matching this project's
scripts/*.py convention.

Usage (after the usual %cd + secrets-loading cells, after `git pull`):
    !python scripts/run_cuad_smoke.py
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ID = "cuad_smoke"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"
RESULTS_DIR = REPO_ROOT / "results" / RUN_ID

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from agent.loop import ClaudeEvidenceAssessor, run_agent_query  # noqa: E402
from agent.safety import BudgetedGenerator, BudgetedJudge, BudgetExceededError, RunBudgetTracker, RunSafetyLimits  # noqa: E402
from agent.success import evaluate_agent_success  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.chunking import chunk  # noqa: E402
from pipeline.cli import ClaudeGenerator, ClaudeJudge, ClaudeSummarizer, build_clients  # noqa: E402
from pipeline.cuad_smoke import load_cuad_smoke_fixture  # noqa: E402
from pipeline.embedding import embed_documents  # noqa: E402
from pipeline.enrichment import EnrichmentCheckpoint, enrich_documents  # noqa: E402
from pipeline.evaluation import evaluate_answer  # noqa: E402
from pipeline.generation import build_context_block, generate_answer  # noqa: E402
from pipeline.indexing import (  # noqa: E402
    build_full_indexed_content,
    dedupe_documents,
    index_corpus,
    is_indexed,
    validate_startup_indexes,
)


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


def _index_cuad_documents(config, clients, records) -> None:
    collection = clients["collection"]
    records = chunk(records)  # no-op by design, see pipeline/chunking.py
    documents = dedupe_documents(records)
    to_process = [d for d in documents if not is_indexed(collection, d["context_id"])]
    print(f"  {len(documents) - len(to_process)} already indexed (skipped), {len(to_process)} to embed + index")
    if not to_process:
        return

    if config.enrichment.enabled:
        summarizer = ClaudeSummarizer(clients["anthropic"], config.enrichment.model, config.enrichment.temperature)
        checkpoint = EnrichmentCheckpoint(str(RESULTS_DIR / "enrichment_checkpoint.jsonl"))
        summaries = enrich_documents(
            summarizer, [(d["context_id"], d["raw_content"]) for d in to_process], checkpoint=checkpoint
        )
    else:
        summaries = enrich_documents(None, [(d["context_id"], d["raw_content"]) for d in to_process], enabled=False)

    # Single embedding model for the whole batch (routing.enabled=false in
    # config_cuad_smoke.yaml - see that file's comments) - no per-model
    # grouping needed, unlike pipeline/cli.py's cmd_index.
    items = [
        (d["context_id"], build_full_indexed_content(d["raw_content"], summaries.get(d["context_id"], ""), d["metadata_prefix"]))
        for d in to_process
    ]
    vectors = embed_documents(clients["voyage"], items, model=config.embedding.model)
    embeddings_by_id = {v.id: v.vector for v in vectors}
    written = index_corpus(collection, to_process, summaries, embeddings_by_id, skip_already_indexed=False)
    print(f"  Indexed {written} document(s) into {config.mongodb.collection_name!r}")


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS_DIR / "baseline_results.jsonl"
    agent_path = RESULTS_DIR / "agent_results.jsonl"
    trace_path = RESULTS_DIR / "agent_trace.jsonl"

    from agent.tracing import TraceWriter

    trace_writer = TraceWriter(trace_path)

    records, eval_items = load_cuad_smoke_fixture()
    print(f"Loaded {len(eval_items)} CUAD smoke question(s) over {len({r.context_id for r in records})} document(s)")

    print("Indexing CUAD documents into the separate cuad_smoke collection...")
    _index_cuad_documents(config, clients, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)
    print(f"Startup index validation passed - {config.mongodb.vector_index_name!r} and "
          f"{config.mongodb.text_index_name!r} both return results.")

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
    print(f"Resuming: {len(baseline_done)} baseline, {len(agent_done)} agent result(s) already on disk")

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
        embedding_model=config.embedding.model,
        source_dataset=None,  # single-collection, single-model - no per-dataset filter needed
        exclude_source_datasets=None,
    )

    try:
        for item in eval_items:
            question_id = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]

            if question_id not in baseline_done:
                baseline_call = search_fn(question)
                generated = generate_answer(generator, question_id, question, list(baseline_call.candidates))
                context_text = build_context_block(list(baseline_call.candidates))
                # deterministic_check_enabled=False: CUAD gold answers are
                # legal-clause text spans, not numbers - is_close_v2 (this
                # pipeline's numeric-tolerance checker) does not apply here.
                eval_result = evaluate_answer(
                    judge, question_id, question, context_text, generated.answer_text, gold_answer,
                    deterministic_check_enabled=False,
                )
                baseline_record = {
                    "question_id": question_id,
                    "gold_answer": gold_answer,
                    "answer_text": generated.answer_text,
                    "judge_correct": eval_result.judge_scores["judge_correct"],
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
                context_text = (
                    build_context_block(list(agent_answer.context_documents)) if agent_answer.context_documents else ""
                )
                success_result = evaluate_agent_success(
                    judge, question_id, question, context_text, agent_answer.answer_text, gold_answer,
                    deterministic_check_enabled=False,
                )
                agent_record = {
                    "question_id": question_id,
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
    except BudgetExceededError as exc:
        print(f"\nSTOPPED: CUAD smoke-run safety budget exceeded: {exc}")
        print(f"Progress so far is saved to {RESULTS_DIR} - re-run to resume.")
        return

    baseline_correct = sum(1 for r in baseline_done.values() if r["judge_correct"])
    agent_correct = sum(1 for r in agent_done.values() if r["success"])
    print(f"\nDone. n={len(eval_items)} - baseline {baseline_correct}/{len(baseline_done)}, agent {agent_correct}/{len(agent_done)}")
    print(f"Budget used: {tracker.llm_calls} LLM calls, ~${tracker.estimated_cost_usd:.2f}, {tracker.elapsed_seconds:.0f}s")
    print(
        "\nReminder: n=5 has no statistical power and this is an out-of-domain robustness check, "
        "not a legal-document product claim - see README.md's Agent section."
    )


if __name__ == "__main__":
    main()

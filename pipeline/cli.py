"""CLI orchestrator - ties all nine pipeline modules into two runnable
commands.
 
  index   Ingestion -> Chunking -> Contextual enrichment -> Embedding ->
          Indexing (modules 1-5). Builds/refreshes the MongoDB Atlas
          corpus (rag_project.t2_ragbench_full).
 
  eval    Hybrid retrieval -> Reranking -> Generation -> LLM Judge
          Evaluation (modules 6-9). Runs the query-time pipeline over a
          set of eval questions and writes
          results/<run_id>/{run_config.json, predictions.jsonl,
          eval_results.jsonl, eval_report.md}.
 
See docs/specifikatsiya_moduley.md for the per-module input/output
contracts this file wires together, and
docs/plan_podgotovki_k_kodirovaniyu.md for the checkpoint each stage is
expected to reproduce.
 
Every pipeline/*.py module is written against a narrow Protocol
(VoyageClientProtocol, SummarizerProtocol, GeneratorProtocol,
JudgeProtocol, CohereClientProtocol, CollectionProtocol) so it can be unit
-tested with a fake client - see tests/. This file is the one place those
Protocols are satisfied by the real SDKs (pymongo, voyageai, anthropic,
cohere), via the small adapter classes below.
 
Usage:
    python -m pipeline.cli index --data-dir data/t2-ragbench
    python -m pipeline.cli eval --questions data/t2-ragbench/eval_subset_250.parquet --run-id week4_baseline
    python -m pipeline.cli eval --questions data/t2-ragbench/eval_subset_250.parquet --run-id week4_rerank --compare-to week4_baseline
"""
 
from __future__ import annotations
 
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
 
import pandas as pd
 
from config.config_schema import PipelineConfig, load_config
from pipeline.chunking import chunk
from pipeline.common.run_config import write_run_config
from pipeline.embedding import BATCH_SIZE, embed_documents
from pipeline.enrichment import EnrichmentCheckpoint, enrich_document, enrich_documents
from pipeline.evaluation import EvalResult, JudgeCache, evaluate_answers, regression_report
from pipeline.generation import generate_answer
from pipeline.indexing import (
    build_full_indexed_content,
    dedupe_documents,
    index_corpus,
    is_indexed,
    validate_startup_indexes,
)
from pipeline.ingestion import ingest
from pipeline.reranking import rerank
from pipeline.retrieval import retrieve
 
# ---------------------------------------------------------------------------
# Real-SDK adapters - the only place pipeline/*.py's Protocols meet an
# actual API client, so every module upstream stays unit-testable with a
# fake (see tests/).
# ---------------------------------------------------------------------------
 
 
class ClaudeSummarizer:
    """Adapts anthropic.Anthropic to pipeline.enrichment.SummarizerProtocol.
 
    CONTEXT_PROMPT and the chunk[:3000] truncation below are transplanted
    literally from the real Colab script that produced the validated
    Recall@5 +0.78pp checkpoint (test_2_3_contextual_chunks_v2.py, cell 4;
    docs/tehnicheskoe_zadanie.md, section 5) - not reconstructed from the
    spec's prose summary ("short 1-2 sentence blurb"). Do not edit this
    prompt without re-running that validation test, per the same
    literal-transplant convention already applied to
    pipeline/common/is_close_v2.py and evaluation.py's JUDGE_PROMPT.
 
    One intentional deviation from the literal Colab call: temperature is
    taken from config (0.0), not left at the API default the original test
    used. tehnicheskoe_zadanie.md, section 7, flags this as a fix to make
    at implementation time ("параметр нигде явно не был зафиксирован, при
    реализации исправить") - it does not change the prompt or the
    truncation, only removes sampling noise from a production run.
    """
 
    CONTEXT_PROMPT = """Ты помогаешь улучшить поиск по фрагментам финансовых отчётов.
Вот фрагмент документа:
 
<chunk>
{chunk}
</chunk>
 
Дай короткую (1-2 предложения) справку: какая компания, какой год отчёта, о чём фрагмент (какие показатели/таблица).
Только справка, без вступлений."""
 
    def __init__(self, client, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature
 
    def summarize(self, raw_content: str) -> str:
        # The validated test truncated the summarization *input* to the
        # first 3000 characters - this only affects what the model sees
        # when writing the blurb. The full, untruncated raw_content is
        # still what gets indexed/embedded downstream, via
        # build_full_indexed_content().
        prompt = self.CONTEXT_PROMPT.format(chunk=raw_content[:3000])
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
 
 
class ClaudeGenerator:
    """Adapts anthropic.Anthropic to pipeline.generation.GeneratorProtocol."""
 
    def __init__(self, client, model: str, temperature: float, max_tokens: int = 300):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
 
    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
 
 
class ClaudeJudge:
    """Adapts anthropic.Anthropic to pipeline.evaluation.JudgeProtocol."""
 
    def __init__(self, client, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature
 
    def judge(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
 
 
def build_clients(config: PipelineConfig) -> dict:
    """Constructs real SDK clients from config + environment variables.
 
    API keys are read directly from os.environ using this project's own
    .env.example variable names, rather than relying on each SDK's own
    default env-var name - some SDKs default to a different name (e.g.
    cohere.ClientV2() defaults to CO_API_KEY, not COHERE_API_KEY, which is
    what .env.example declares).
    """
    import anthropic
    import cohere
    import pymongo
    import voyageai
 
    missing = [v for v in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY") if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)} - copy .env.example to .env and fill it in"
        )
 
    mongo_client = pymongo.MongoClient(config.mongodb.uri)
    collection = mongo_client[config.mongodb.db_name][config.mongodb.collection_name]
    return {
        "collection": collection,
        "voyage": voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"]),
        "anthropic": anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "cohere": cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"]),
    }
 
 
# ---------------------------------------------------------------------------
# index - modules 1-5
# ---------------------------------------------------------------------------
 
 
def cmd_index(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    clients = build_clients(config)
    collection = clients["collection"]
 
    print(f"[1/5] Ingestion from {args.data_dir} ...")
    records = ingest(args.data_dir)
    print(f"  {len(records)} question rows, {len({r.context_id for r in records})} unique documents")
 
    print("[2/5] Chunking (no-op by design - one document = one chunk)...")
    records = chunk(records)
 
    print("[3/5] Dedupe to unique documents...")
    documents = dedupe_documents(records)
    print(f"  {len(documents)} unique documents")
 
    print(f"[4/5] Contextual enrichment ({'enabled' if config.enrichment.enabled else 'disabled'})...")
    if config.enrichment.enabled:
        summarizer = ClaudeSummarizer(clients["anthropic"], config.enrichment.model, config.enrichment.temperature)
        checkpoint = EnrichmentCheckpoint(args.checkpoint)
        # Reimplements enrich_documents()'s loop here (rather than calling it
        # as a single black-box call) purely to print progress - this is the
        # slowest stage (one sequential Claude API round-trip per document,
        # no threading yet - see cli.py module docstring / README known
        # limitations) and a silent multi-hour loop is a real usability
        # problem, not just cosmetic. Same checkpoint semantics as
        # enrich_documents(): resume from what's already on disk, append
        # after each document, so an interrupted run picks up where it left
        # off either way.
        summaries = dict(checkpoint.load_done())
        to_enrich = [(d["context_id"], d["raw_content"]) for d in documents if d["context_id"] not in summaries]
        print(f"  {len(summaries)} already enriched (resumed from checkpoint), {len(to_enrich)} remaining")
        for i, (context_id, raw_content) in enumerate(to_enrich):
            result = enrich_document(summarizer, context_id, raw_content)
            checkpoint.append(result)
            summaries[context_id] = result.contextual_summary
            if (i + 1) % 25 == 0 or (i + 1) == len(to_enrich):
                print(f"  enriched {i + 1}/{len(to_enrich)}")
    else:
        summaries = enrich_documents(
            None, [(d["context_id"], d["raw_content"]) for d in documents], enabled=False
        )
 
    print("[5/5] Embedding + indexing...")
    to_process = [d for d in documents if not is_indexed(collection, d["context_id"])]
    print(f"  {len(documents) - len(to_process)} already indexed (skipped), {len(to_process)} to embed + index")
    written = 0
    if to_process:
        to_embed = [
            (
                d["context_id"],
                build_full_indexed_content(d["raw_content"], summaries.get(d["context_id"], ""), d["metadata_prefix"]),
            )
            for d in to_process
        ]
        docs_by_id = {d["context_id"]: d for d in to_process}
        # Embed and write *each batch* to MongoDB before moving to the next
        # one, rather than accumulating every vector in memory and writing
        # once at the end - checkpointing requirement, spec section 11
        # ("Checkpointing состояния при индексации"). Accumulate-then-write
        # would mean a mid-run interruption loses every already-computed
        # embedding, since none of it would have reached Atlas yet; this way
        # an interruption loses at most one in-flight batch (BATCH_SIZE
        # documents), and a re-run of `index` skips everything already
        # marked is_indexed=True in Atlas via is_indexed() above.
        total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(to_embed), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = to_embed[i : i + BATCH_SIZE]
            vectors = embed_documents(clients["voyage"], batch)
            embeddings_by_id = {v.id: v.vector for v in vectors}
            batch_docs = [docs_by_id[context_id] for context_id, _ in batch]
            written += index_corpus(collection, batch_docs, summaries, embeddings_by_id, skip_already_indexed=False)
            print(f"  embedded + indexed batch {batch_num}/{total_batches} ({written} documents written so far)")
    print(f"  Indexed {written} documents")
 
    validate_startup_indexes(collection)
    print("Startup index validation passed - vector_index_full and text_index_full both return results.")
 
 
# ---------------------------------------------------------------------------
# eval - modules 6-9
# ---------------------------------------------------------------------------
 
 
def load_eval_questions(path: str | Path) -> list[dict]:
    """Loads an eval question set from parquet.
 
    Schema note: the exact columns of data/t2-ragbench/eval_subset_*.parquet
    have not been verified against this loader - as of 2026-08-14 those
    files exist only on Google Drive, not yet committed to the GitHub repo,
    so this function is deliberately defensive about column names rather
    than assuming they match pipeline.ingestion's raw source-dataset schema
    exactly. Once the real file is in the repo, confirm the column names
    below still match and adjust if not.
 
    Required: 'question'. Gold answer: 'answer' or 'program_answer' (either
    accepted - the raw T2-RAGBench source files use 'program_answer', see
    pipeline.ingestion.to_document_records). Optional: 'question_id' (falls
    back to '{context_id}::{row index}', or 'q{row index}' if context_id is
    also absent) and 'source_dataset' (used for the per-source
    stratification in eval_report.md; rows without it are grouped under
    'unknown').
    """
    df = pd.read_parquet(path)
    if "question" not in df.columns:
        raise ValueError(f"{path} has no 'question' column")
    if "answer" in df.columns:
        answer_col = "answer"
    elif "program_answer" in df.columns:
        answer_col = "program_answer"
    else:
        raise ValueError(f"{path} has neither 'answer' nor 'program_answer' column for the gold value")
 
    has_context_id = "context_id" in df.columns
    has_question_id = "question_id" in df.columns
    has_source = "source_dataset" in df.columns
 
    items = []
    for i, row in df.reset_index(drop=True).iterrows():
        context_id = row["context_id"] if has_context_id else None
        if has_question_id:
            question_id = row["question_id"]
        elif context_id is not None:
            question_id = f"{context_id}::{i}"
        else:
            question_id = f"q{i}"
        items.append(
            {
                "question_id": str(question_id),
                "question": row["question"],
                "gold_answer": row[answer_col],
                "source_dataset": row["source_dataset"] if has_source else "unknown",
            }
        )
    return items
 
 
def load_eval_results(path: Path) -> list[EvalResult]:
    if not path.exists():
        raise FileNotFoundError(f"No previous run found at {path} - check --compare-to")
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            results.append(
                EvalResult(
                    question_id=rec["question_id"],
                    judge_scores=rec["judge_scores"],
                    deterministic_match=rec["deterministic_match"],
                    judge_agrees=rec["judge_agrees"],
                )
            )
    return results
 
 
def write_eval_report(
    path: Path,
    results: list[EvalResult],
    items: list[dict],
    run_id: str,
    previous_results: list[EvalResult] | None = None,
    compare_to: str | None = None,
) -> None:
    """Writes results/<run_id>/eval_report.md.
 
    Per docs/tehnicheskoe_zadanie.md, section 10, aggregate accuracy alone
    is not sufficient reporting: per-source_dataset stratification
    (FinQA/ConvFinQA/TAT-DQA are qualitatively different question types,
    not interchangeable) and, when a previous run is given via
    --compare-to, a per-question regression comparison - the same kind of
    comparison that has repeatedly caught real bugs hidden behind a stable
    or even improved aggregate metric earlier in this project (e.g. the
    bge-reranker case: 13 fixed but 44 broken, aggregate looked neutral).
    """
    id_to_source = {item["question_id"]: item["source_dataset"] for item in items}
    total = len(results)
    judge_correct = sum(1 for r in results if r.judge_scores["judge_correct"])
    det_correct = sum(1 for r in results if r.deterministic_match)
    agree = sum(1 for r in results if r.judge_agrees)
 
    by_source: dict[str, dict[str, int]] = {}
    for r in results:
        source = id_to_source.get(r.question_id, "unknown")
        stats = by_source.setdefault(source, {"total": 0, "judge_correct": 0, "det_correct": 0})
        stats["total"] += 1
        if r.judge_scores["judge_correct"]:
            stats["judge_correct"] += 1
        if r.deterministic_match:
            stats["det_correct"] += 1
 
    lines = [f"# Eval report - run {run_id}", "", f"Questions evaluated: {total}"]
    if total:
        lines += [
            f"Judge accuracy: {judge_correct}/{total} = {judge_correct / total:.3f}",
            f"Deterministic (is_close_v2) accuracy: {det_correct}/{total} = {det_correct / total:.3f}",
            f"Judge/deterministic agreement: {agree}/{total} = {agree / total:.3f}",
        ]
    lines += ["", "## By source dataset", "", "| source_dataset | n | judge accuracy | deterministic accuracy |", "|---|---|---|---|"]
    for source, stats in sorted(by_source.items()):
        n = stats["total"]
        lines.append(f"| {source} | {n} | {stats['judge_correct'] / n:.3f} | {stats['det_correct'] / n:.3f} |")
 
    if previous_results is not None:
        report = regression_report(previous_results, results)
        lines += [
            "",
            f"## Regression vs run {compare_to}",
            "",
            f"- Improved (wrong -> correct): {len(report['improved'])}",
            f"- Regressed (correct -> wrong): {len(report['regressed'])}",
            f"- Unchanged, correct: {len(report['unchanged_correct'])}",
            f"- Unchanged, incorrect: {len(report['unchanged_incorrect'])}",
        ]
        if report["regressed"]:
            shown = ", ".join(report["regressed"][:50])
            more = " ..." if len(report["regressed"]) > 50 else ""
            lines.append(f"- Regressed question_ids: {shown}{more}")
 
    lines += [
        "",
        "_Numbers here are from an actual run, not the docs/*.md checkpoints - treat "
        "docs/tehnicheskoe_zadanie.md's documented checkpoints as the reference to compare "
        "against, not the reverse._",
        "",
    ]
 
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
 
 
def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    clients = build_clients(config)
    collection = clients["collection"]
 
    validate_startup_indexes(collection)
 
    items = load_eval_questions(args.questions)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} eval questions from {args.questions}")
 
    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    judge = ClaudeJudge(clients["anthropic"], config.judge.model, config.judge.temperature)
 
    run_dir = Path("results") / args.run_id
    judge_cache = JudgeCache(run_dir / "judge_cache.jsonl")
 
    predictions = []
    eval_items = []
    skipped_no_candidates = 0
    for i, item in enumerate(items):
        candidates = retrieve(
            clients["voyage"],
            collection,
            item["question"],
            pool_size=config.retrieval.pool_size,
            vector_weight=config.retrieval.weights.vector,
            text_weight=config.retrieval.weights.text,
        )
        if config.reranker.enabled and candidates:
            ranked = rerank(clients["cohere"], item["question"], candidates, top_n=config.reranker.top_n)
        else:
            ranked = candidates[: config.reranker.top_n]
 
        if not ranked:
            skipped_no_candidates += 1
            continue
 
        answer = generate_answer(generator, item["question_id"], item["question"], ranked)
        predictions.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "source_dataset": item["source_dataset"],
                "gold_answer": item["gold_answer"],
                "answer_text": answer.answer_text,
            }
        )
        eval_items.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "context": "\n\n".join(c.full_indexed_content for c in ranked),
                "generated_answer": answer.answer_text,
                "gold_answer": item["gold_answer"],
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} answered")
 
    if skipped_no_candidates:
        print(f"  {skipped_no_candidates} question(s) skipped - no retrieval candidates")
 
    print("Judging...")
    results = evaluate_answers(
        judge,
        eval_items,
        cache=judge_cache,
        prompt_version=config.judge.prompt_version,
        deterministic_check_enabled=config.judge.deterministic_check_enabled,
    )
 
    print(f"Writing results to {run_dir} ...")
    write_run_config(config.model_dump(), args.run_id)
 
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
 
    with (run_dir / "eval_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "question_id": r.question_id,
                        "judge_scores": r.judge_scores,
                        "deterministic_match": r.deterministic_match,
                        "judge_agrees": r.judge_agrees,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
 
    previous_results = load_eval_results(Path("results") / args.compare_to / "eval_results.jsonl") if args.compare_to else None
    write_eval_report(run_dir / "eval_report.md", results, items, args.run_id, previous_results, args.compare_to)
    print(f"Done. Report: {run_dir / 'eval_report.md'}")
 
 
# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
 
 
def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv
 
        load_dotenv()
    except ImportError:
        pass
 
    parser = argparse.ArgumentParser(prog="python -m pipeline.cli", description="Financial RAG Pipeline orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
 
    p_index = sub.add_parser("index", help="Build/refresh the MongoDB Atlas corpus (modules 1-5)")
    p_index.add_argument("--config", default="config/config.yaml")
    p_index.add_argument("--data-dir", default="data/t2-ragbench")
    p_index.add_argument("--checkpoint", default="checkpoints/enrichment_checkpoint.jsonl")
    p_index.set_defaults(func=cmd_index)
 
    p_eval = sub.add_parser("eval", help="Run the query-time pipeline over a question set (modules 6-9)")
    p_eval.add_argument("--config", default="config/config.yaml")
    p_eval.add_argument(
        "--questions", required=True, help="Path to an eval parquet file, e.g. data/t2-ragbench/eval_subset_250.parquet"
    )
    p_eval.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp")
    p_eval.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (quick smoke run)")
    p_eval.add_argument("--compare-to", default=None, help="A previous run_id to diff against in eval_report.md")
    p_eval.set_defaults(func=cmd_eval)
 
    args = parser.parse_args(argv)
    if args.command == "eval" and args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 
    args.func(args)
 
 
if __name__ == "__main__":
    main()

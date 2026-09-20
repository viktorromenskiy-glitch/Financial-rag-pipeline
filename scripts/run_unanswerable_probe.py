"""Priority 3 from the adversarial-robustness plan (internal working
materials, not in this repository; "unanswerable / out-of-corpus
questions") - the only class of behavior still
untested: whether the pipeline refuses to answer
(FINAL ANSWER: INSUFFICIENT_CONTEXT) when the indexed corpus genuinely
has no data to answer from, or instead generates a plausible but
unsubstantiated number.

Input: data/unanswerable_probe/questions.jsonl (32 questions, built and
programmatically verified against the real corpus data by
data/unanswerable_probe/build_questions.py - every
company/missing-year pair, or fully absent company, is confirmed, not
assumed).

Unlike scripts/reevaluate_phase6_adaptive.py and the other eval scripts
in this project, there is NO judge call and NO gold_answer here - by
definition these questions have no correct numeric answer to check
against. The cost of a question here is just one thing: retrieval +
rerank + generation (no judge), i.e. cheaper than a normal eval run
(~$0.015/question instead of ~$0.016, see docs/tehnicheskoe_zadanie.md
section 15 - the judge part simply isn't called).

Answer classification is programmatic, based on the same
FINAL ANSWER: marker used by the rest of the pipeline
(pipeline.generation._extract_final_answer):
  - answer_text == "INSUFFICIENT_CONTEXT" -> refused (expected behavior)
  - otherwise -> confident_answer (unexpected - the model produced a
    number where it should not have; this is NOT automatically
    "incorrect" in the usual judge sense, but is exactly the case this
    probe exists to flag for manual review)
The full raw_response is saved for every question (not just the
extracted marker) - without it there is no way to distinguish a genuine
"confident fabrication" from a case where the model wrote a refusal in
prose without hitting the exact marker format (see pipeline/generation.py's
module docstring, on the previously observed leakage of reasoning past
the format) - such a case would be incorrectly counted as
confident_answer without access to the raw text.

Fixed as a result of independent review BEFORE running against the real
paid API (not after): (1) the original absent_company list included
companies that were not in the company_name column but were in fact
mentioned in the indexed text as peer-group names of other companies
(Alphabet, Meta, Tesla, Nvidia, Home Depot, Coca-Cola) - replaced with
12 companies confirmed by build_questions.py's verify_absence() as
absent EVERYWHERE in the text, not just in the metadata; (2) results
are written directly to the root resolved via find_canonical_root() on
Drive, line by line as the run progresses - not locally with copying at
the end (the same pattern that notebooks/reevaluate_phase6_adaptive.py
adopted after an actual checkpoint loss earlier in this project; see
the project's internal working rules, section 1, point 3); (3)
n_retrieved now honestly reflects the size of the hybrid retrieval pool
(before the reranker), not the count after top-N; the retrieval and
rerank scores of the top candidate are saved.

Usage (Colab, after mount Drive, %cd into the repository, git pull):
    !python notebooks/run_unanswerable_probe.py
The run is idempotent: re-running it picks up results already saved on
Drive by probe_id and does not spend money again on questions already
done (the same resume pattern as in reevaluate_phase6_adaptive.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402 - same recurring load_dotenv() bug as everywhere else in this repo; see scripts/check_colab_runtime.py's module docstring

load_dotenv(REPO_ROOT / ".env")

from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, build_clients, _resolve_embedding_model, _resolve_prompt_template  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.generation import generate_answer  # noqa: E402
from pipeline.indexing import validate_startup_indexes  # noqa: E402
from pipeline.reranking import rerank  # noqa: E402
from pipeline.retrieval import retrieve  # noqa: E402

RUN_ID = "unanswerable_probe"
QUESTIONS_PATH = REPO_ROOT / "data" / "unanswerable_probe" / "questions.jsonl"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# None of the 32 questions belong to a real source_dataset - all of them
# target something outside what's indexed. Retrieval still needs some
# label (per-dataset embedding routing is only enabled for TAT-DQA - see
# pipeline/cli.py cmd_eval): "wrong_year" questions about TAT-DQA
# companies are labeled source_dataset="TAT-DQA" (the real embedding
# space for that company), "wrong_year" questions about FinQA/ConvFinQA
# companies and all "absent_company" questions get "FinQA" (not routed,
# so it doesn't affect which embedding model is chosen - see
# _resolve_embedding_model - treated as an ordinary non-routed query).
# This is a deliberate, documented-here choice, not a hidden assumption.
TAT_DQA_KEYS = {
    "accenture-plc", "woolworths-limited", "activision-blizzard-inc",
    "adobe-systems-inc", "intu-properties",
}


def load_questions() -> list[dict]:
    """Loads the unanswerable-probe question set.

    Returns:
        The probe question records from QUESTIONS_PATH, in file order.
    """
    items = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def infer_source_dataset(rec: dict) -> str:
    """Infers which source_dataset label to use for retrieval routing on a probe question.

    Args:
        rec: A probe question record (as loaded by `load_questions`), with
            "category" and "company" keys.

    Returns:
        "TAT-DQA" if the question targets a known TAT-DQA company under a
        missing year, otherwise "FinQA" (a non-routed placeholder).
    """
    if rec["category"] == "wrong_year":
        # company key was stored implicitly via the question text at build
        # time; re-derive from build_questions.py's WRONG_YEAR_QUESTIONS
        # would require importing it - simpler and just as verifiable:
        # match on company name against the known TAT-DQA set.
        for key in TAT_DQA_KEYS:
            if key.replace("-", " ") in rec["company"].lower() or rec["company"].lower() in key.replace("-", " "):
                return "TAT-DQA"
    return "FinQA"


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)

    items = load_questions()
    print(f"Loaded {len(items)} unanswerable-probe questions from {QUESTIONS_PATH}")

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    prompt_template = _resolve_prompt_template(config.generation.prompt_variant)
    routing = config.embedding.routing

    # Write directly to the resolved root on Drive, not locally with
    # copying at the end - section 1 point 3 of the project's internal
    # working rules, the same fix that scripts/reevaluate_phase6_adaptive.py
    # adopted after an actual checkpoint loss. find_canonical_root()
    # confirms that the root exists exactly as configured (not a
    # similarly-named folder).
    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "probe_results.jsonl"

    # Resume support: questions already done (by probe_id) are not
    # re-asked and not paid for again on re-run after a failure.
    done: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["probe_id"]] = rec
        if done:
            print(f"Resuming: {len(done)} question(s) already done in {out_path}")

    results = list(done.values())
    t_start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(items):
            if rec["probe_id"] in done:
                continue

            source_dataset = infer_source_dataset(rec)
            is_routed_source = routing.enabled and source_dataset in routing.routed_sources
            query_model = _resolve_embedding_model(config, source_dataset)

            candidates = retrieve(
                clients["voyage"],
                collection,
                rec["question"],
                pool_size=config.retrieval.pool_size,
                vector_weight=config.retrieval.weights.vector,
                text_weight=config.retrieval.weights.text,
                source_dataset=source_dataset if is_routed_source else None,
                exclude_source_datasets=list(routing.routed_sources) if (routing.enabled and not is_routed_source) else None,
                embedding_model=query_model,
            )
            if config.reranker.enabled and candidates:
                ranked = rerank(clients["cohere"], rec["question"], candidates, top_n=config.reranker.top_n)
            else:
                ranked = candidates[: config.reranker.top_n]

            if not ranked:
                # Retrieval found literally nothing - itself a valid,
                # informative outcome for an out-of-corpus question (no
                # generation call made, no cost) - recorded as its own
                # behavior, not skipped silently.
                behavior = "no_retrieval_candidates"
                answer_text = None
                raw_response = None
                top_retrieved_context_id = None
                top_retrieval_score = None
                top_rerank_score = None
            else:
                answer = generate_answer(generator, rec["probe_id"], rec["question"], ranked, template=prompt_template)
                answer_text = answer.answer_text
                raw_response = answer.raw_response
                behavior = "refused" if answer_text.strip().upper() == "INSUFFICIENT_CONTEXT" else "confident_answer"
                top_retrieved_context_id = ranked[0].context_id
                top_retrieval_score = candidates[0].score if candidates else None
                top_rerank_score = getattr(ranked[0], "relevance_score", None)

            result = {
                **rec,
                "source_dataset_used_for_retrieval": source_dataset,
                "n_retrieved_pool": len(candidates),
                "n_reranked": len(ranked),
                "top_retrieved_context_id": top_retrieved_context_id,
                "top_retrieval_score": top_retrieval_score,
                "top_rerank_score": top_rerank_score,
                "answer_text": answer_text,
                "raw_response": raw_response,
                "behavior": behavior,
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  [{i + 1}/{len(items)}] {rec['probe_id']} ({rec['category']}, {rec['company']}) -> {behavior}"
                  + (f" ({answer_text})" if behavior == "confident_answer" else ""))

    elapsed = time.perf_counter() - t_start
    n_refused = sum(1 for r in results if r["behavior"] == "refused")
    n_confident = sum(1 for r in results if r["behavior"] == "confident_answer")
    n_no_candidates = sum(1 for r in results if r["behavior"] == "no_retrieval_candidates")
    print(f"\nDone in {elapsed:.0f}s (this session). refused={n_refused} confident_answer={n_confident} no_retrieval_candidates={n_no_candidates} of {len(results)}/{len(items)} total")
    if n_confident:
        print(f"  WARNING: {n_confident} question(s) got a confident numeric answer instead of INSUFFICIENT_CONTEXT - need manual review (raw_response saved for each).")

    verify_run_files(run_dir, {"probe_results.jsonl": len(items)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")


if __name__ == "__main__":
    main()

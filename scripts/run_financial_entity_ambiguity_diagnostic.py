"""Narrow financial-domain check of the deictic/entity-ambiguity hypothesis -
the last unverified item from the closed 4-expert cascade on CUAD
over-refusal (see claude/itog_ekspertizy_cuad_overrefusal_fix.md, "What's
left to do", item 2; full design history in
claude/prompt_ekspert3_cuad_fix_dizayn.md and
claude/prompt_ekspert4_cuad_fix_dizayn.md).

## Why this script exists (short version - full history in the expert-review summary)

Three rounds of diagnostics on CUAD (scripts/run_cuad_overrefusal_diagnostic*.py)
established that the agent refuses to answer CUAD questions not because of a
weak assessor prompt, but because all 5 CUAD questions use a deictic
formulation ("this contract") with no identifying term at all, which - once
multiple documents are pooled - makes it literally impossible to tell which
document the question refers to. A check of the real financial domain
(claude/nahodka_deiktichnost_round3_dlya_ekspertov.md) showed: ~98% of
questions explicitly name the company in the text, and financial documents
(unlike CUAD) already carry a document-level `metadata_prefix` ("Company: X |
Sector: Y | Report year: Z") directly in the indexed text. Four independent
experts agreed (see the expert-review summary, items 1-4 of the consensus
table) that a prompt-only fix is not needed, but they explicitly named an
unresolved competing hypothesis: "what if the assessor is simply bad at
multi-document context in general, even when the entity is named explicitly?"
- this script answers exactly that question, on the REAL financial domain,
not on CUAD.

## Design (agreed by 4 experts - see the expert-review summary, item 6 of the
consensus table)

A paired (within-subject) comparison: the same ~28 real financial questions
(data/financial_entity_ambiguity/fixture.json - see
data/financial_entity_ambiguity/build_fixture.py for how it was built and the
selection criteria) under 2 text conditions:

  original    - the question as-is (explicitly names the company)
  anonymized  - the same company name replaced with "the company"/"the
                company's" (a deterministic regex substitution, verified
                programmatically at fixture-build time - see
                build_fixture.py's anonymize_question())

The context (the set of documents the assessor/generator will see) is the
SAME for BOTH conditions and is assembled MANUALLY (not via live retrieval):
the question's gold document + 3 distractors from the SAME Sector (the exact
field that actually ends up in metadata_prefix - see
pipeline/ingestion.py's build_metadata_prefix, which uses company_sector, not
company_industry), with the closest report_year. All 4 documents are taken
as-is from the already-indexed production collection (t2_ragbench_full) -
full_indexed_content, unmodified. Manual context assembly (instead of live
retrieval) is a deliberate choice per the convergence of experts 2/3/4: it
isolates the assessor from retrieval variability, so that any
original/anonymized difference is explained SOLELY by the question text, not
by retrieval having happened to find a different set of documents for one of
the two conditions.

`_ASSESSMENT_PROMPT_TEMPLATE` and `AGENT_ANSWER_PROMPT_TEMPLATE` (agent/loop.py)
are used UNCHANGED - this script does not test or propose a prompt edit, it
only measures the existing production prompt's behavior on a controlled
input. All 4 experts agreed this is settled (see the expert-review summary,
item 4 of the consensus table): prompt-only fixes were judged unpromising for
CUAD, and there is no need to test them on the financial domain either -
what's measured here is the sensitivity of the EXISTING behavior to one
specific variable (whether the company name is present in the question),
nothing more.

## What is logged for every (question, condition) pair

  - sufficient (the assessor's verdict, from _parse_assessment - the same
    production parser used in rounds 1-3, including "last marker match
    wins")
  - the final-answer text and its correctness (is_close_v2 against
    gold_answer) - generated ALWAYS, regardless of sufficient (see below,
    "Why answer generation is not gated on sufficient")
  - single_shot_production_answer / single_shot_production_correct - what
    the production agent WOULD have returned under the "forced_insufficient
    if sufficient=no" policy (agent/loop.py's forced_insufficient) AFTER A
    SINGLE assessment call on this fixed context - a derived field, not a
    separate call. Named "single_shot" (not just "production") because this
    is NOT the full agentic run_agent_query loop - that loop, on
    sufficient=no, could reformulate the query and perform an additional
    retrieval/re-assessment step, which does not happen here (and cannot -
    the whole point of the experiment is a SINGLE fixed context, not a
    repeated search). Renamed from production_answer/production_correct
    following the second expert's review (see "What the second independent
    expert checked" below) - the old name could be read as "what the full
    agentic loop would actually have returned," which is not accurate.
  - assessor_context_source ("manual_fixture") / retrieval_used_for_assessor_context
    (always False) - explicit, field-level (not just docstring-level)
    markers that the assessor/generator saw ONLY the context_text assembled
    manually from fixture.json, never the result of the live-retrieval call
    below. Added following the second expert's review (item 5.4).
  - correct_document_position - the position of the gold document within the
    same [Document i] order that build_context_block() builds for the
    MANUAL context (a deterministic order - see main() below), identical for
    both conditions of a given question (the context does not change
    between conditions)
  - live_retrieval_gold_rank / live_retrieval_pool_size / live_retrieval_error
    (renamed from gold_rank/retrieval_pool_size following the second
    expert's review, item 3 - the old name risked being read as "the
    document the assessor actually saw") - a SEPARATE, diagnostic-only (not
    part of assembling the assessor's context - see
    retrieval_used_for_assessor_context above) ordinary retrieval call
    (search_documents with the same routing logic as production - see
    _is_routed/_resolve_embedding_model in pipeline/cli.py and
    scripts/run_agent_eval.py) using the ORIGINAL question text - the
    position of the gold document among the candidates ACTUALLY returned
    (after the reranker, i.e. exactly what the assessor would see in an
    ordinary, non-manual run), or null if retrieval did not surface the gold
    document at all, or if the call itself failed (in which case
    live_retrieval_error holds the error text - see "What the second
    independent expert checked", item 5.2). One call per question (not per
    condition) - retrieval plays no role here in building the assessor's
    context; this is a separate measurement of "would ordinary retrieval
    have found this document at all."

## Why answer generation is not gated on sufficient (a deliberate departure
## from the production policy, diagnostic-only)

In production (agent/loop.py's run_agent_query), the final-answer generation
call only fires if assessment.sufficient=True - otherwise INSUFFICIENT_CONTEXT
is forced. Here, answer generation is ALWAYS called (for both sufficient=yes
and sufficient=no), because this experiment's question is broader than "does
the anonymized condition gate sufficient more often" - it's also interesting
whether answer EXTRACTION ITSELF degrades (not just the assessor's
willingness to confirm it) once the company name is removed while the gold
document is still physically present in the context. Both aspects are logged
separately (see above) - single_shot_production_correct reproduces what the
forced_insufficient policy would have returned after a SINGLE assessment
call (with INSUFFICIENT_CONTEXT forced on sufficient=no), while answer_correct
shows the model's "pure" ability to pull the right number out of the context
if it is asked to try regardless.

## Decision criterion (framed by analogy with round 3 - a fixed number, not a range)

If anonymized yields sufficient=yes noticeably (visually - not a formal
statistical test at n=28) less often than original, on the SAME documents -
the competing hypothesis ("the assessor is generally bad at multi-document
context, independent of CUAD-specific deicticity") gains support, and the
entity-guard (claude/itog_ekspertizy_cuad_overrefusal_fix.md, "What's left to
do", item 1) can be designed with confidence that it will not break cases
where the financial domain already performs well today. If original and
anonymized give a similar sufficient=yes rate (both high OR both low), the
competing hypothesis is not supported on this domain, and whether a guard is
needed at all for the financial pipeline (where 98% of questions already
name the company explicitly) remains open for a separate discussion with the
experts once results are in. No single correct interpretation is assumed in
advance - the raw results (raw_response, both answers to a question) need to
be read by a human, not just the counter, same as in rounds 1-3.

## Cost (higher than CUAD rounds 1-3 - deliberately: this is the agreed
## "narrow experiment," not a micro-check)

28 questions x 2 conditions x 2 calls (assessment + answer generation) = 112
LLM calls + 28 cheap retrieval calls (no LLM, only embedding + MongoDB +
optionally Cohere rerank, for live_retrieval_gold_rank). The order of
magnitude is the same as one production-eval question (evaluate + assess +
retrieve), multiplied by 56 "question-conditions"; it is by design cheaper
than the full 2x2 experiment, which the 4 experts rejected as an excessive
first step (see the expert-review summary, item 6 of the consensus table).

## What the first independent expert checked (pre-run gate, before the first
## paid run) and what was fixed as a result

The first of the two required independent experts reviewed the code
line-by-line and confirmed: the context is indeed identical between
`original`/`anonymized` (it is assembled once, before the loop over
conditions); choosing `company_sector` (not `company_industry`) for
distractors is methodologically correct (it is `Sector`, specifically, that
ends up in `metadata_prefix`, which the model actually sees). They also
found and argued for 3 issues, all accepted and fixed in this version of the
script:

1. **A real resume bug** (fixed): before this fix, resume only skipped the
   LLM calls themselves (assessment/generate_answer) inside the loop over
   conditions, but context assembly (a MongoDB read) and the PAID
   live-retrieval call (Voyage embedding + optionally Cohere rerank) ran
   again on EVERY restart for questions that were already fully done -
   contradicting the claimed idempotency. Fixed: the entire per-question
   block (including live-retrieval) is now skipped if both conditions are
   already in the checkpoint (see `if all(...done...): continue` at the top
   of the loop over items in main()).
2. **Field naming could mislead a reader of the raw JSONL without the
   docstring** (fixed by renaming + adding new explicit fields):
   `gold_rank`/`retrieval_pool_size` -> `live_retrieval_gold_rank`/
   `live_retrieval_pool_size` (does not read as "the document the assessor
   saw"); `production_answer`/`production_correct` ->
   `single_shot_production_answer`/`single_shot_production_correct` (does
   not read as "what the full agentic loop with query reformulation would
   have returned" - this is only a single assessment call on a fixed
   context); explicit `assessor_context_source` and
   `retrieval_used_for_assessor_context` fields added to every record.
3. **The live-retrieval call must not be able to sink the main experiment**
   (fixed): wrapped in try/except - an error (even an unexpected one, not
   just a transient one - search_documents itself already degrades
   gracefully on transient errors) is logged into `live_retrieval_error`
   instead of aborting assessment/generation for this and subsequent
   questions.

Separately, the expert noted (not a blocker, accepted as a known
limitation): build_fixture.py's anonymization check only catches the exact
`humanized` name, not other possible aliases of the same company (e.g.
"American Airlines" instead of "American Airlines Group") - for the current
28 questions this is covered by a manual line-by-line review (see
build_fixture.py's printed output when run - all 28 anonymized questions were
read by hand, no artifacts found), but it is not guaranteed programmatically
for a hypothetical future expansion of the fixture. Generic alias-detection
logic was deliberately not added (the risk of false positives at n=28
outweighs the benefit) - this should be revisited if the fixture is ever
expanded.

## What the second independent expert checked (pre-run gate closed)

The second expert reviewed all three files independently (not simply
agreeing with the first) and confirmed, by tracing main() line-by-line: the
resume check (`if all(...): continue`) does indeed sit before both context
assembly and live-retrieval, so both are genuinely skipped on a full resume,
as claimed; the field renaming (`live_retrieval_gold_rank`,
`single_shot_production_correct`) is self-sufficient and needs no further
safeguarding; the broad `except Exception` around live-retrieval is
acceptable since it is a side diagnostic measurement that does not affect
sufficient/answer_correct/single_shot_production_*; `correct_document_position`
is computed from the SAME `all_context_ids` list used to order the
`candidates` passed into `build_context_block()` - there is no desync.
Final verdict: no blocking findings, the script is ready to run.

Additional (non-blocking) findings from the second expert, each independently
re-verified by the author:

- **Partial resume** (only one of a question's two conditions is done, e.g.
  after a crash) does not skip live-retrieval again for that question - only
  a full resume (both conditions done) skips it. A real observation, but
  retrieval is cheap relative to the LLM calls and the result would be
  identical on recomputation - accepted as a known, non-blocking limitation
  (deliberately not fixed, to avoid complicating the checkpoint scheme for a
  rare edge case).
- **Anonymization: "the the company"** - a hypothetical case where a
  question contained "the <Company Name>" (an article before the name), in
  which the substitution would produce "the the company." Checked
  programmatically against the REAL fixture.json - none of the 28 questions
  contain such a construction (a regex check for `\bthe\s+<humanized name>`
  finds no matches in any question) - this risk class exists only
  hypothetically for a future fixture expansion, not in the current data.
- Checkpoints do not distinguish between fixture.json versions across runs -
  already documented above as a known limitation, not a new finding.

Both required independent rounds passed with no blocking findings - the gate
is closed.

## Before running for real money

Per the project rule ("before running a NEW paid/diagnostic script...") -
show it to TWO independent external experts with the question "as written,
can this script actually deliver what it claims as its purpose - is there a
contradiction between purpose and mechanism" BEFORE the first paid run. Both
rounds have passed (see above, both with no blocking findings) - it is ready
to run. Before running, confirm: (1) fixture.json has been built
(`data/financial_entity_ambiguity/build_fixture.py`); (2) `t2_ragbench_full`
contains all 28 gold + 84 distractor documents (already indexed as part of
the main financial corpus - no separate indexing step is needed for this
script; `_check_already_indexed()` verifies this at startup and stops with a
clear error if anything is missing); (3) `.env` has working
ANTHROPIC_API_KEY/VOYAGE_API_KEY/COHERE_API_KEY.

Usage (after `!git pull`, and after scripts/run_eval.py index has already
indexed t2_ragbench_full at least once - this script does NOT index anything
new, it only reads already-indexed documents):
    !python scripts/run_financial_entity_ambiguity_diagnostic.py
The run is idempotent (resumes by question_id+condition), including
live-retrieval (see "What the first independent expert checked", item 1,
fixed in this version) - same as rounds 1-3.
"""
from __future__ import annotations

import functools
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from agent.loop import AGENT_ANSWER_PROMPT_TEMPLATE, _ASSESSMENT_PROMPT_TEMPLATE, _parse_assessment  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, _resolve_embedding_model, build_clients  # noqa: E402
from pipeline.common.is_close_v2 import is_close_v2  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.financial_entity_ambiguity_smoke import load_financial_entity_ambiguity_fixture  # noqa: E402
from pipeline.generation import build_context_block, generate_answer  # noqa: E402
from pipeline.indexing import is_indexed, validate_startup_indexes  # noqa: E402

RUN_ID = "financial_entity_ambiguity_diagnostic"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

CONDITIONS = ("original", "anonymized")

# Same risk class as in rounds 2-3 (_check_parse_reliability there) - carried
# over unchanged, minus RELEVANT DOCUMENTS (there is no such marker here -
# the unmodified _ASSESSMENT_PROMPT_TEMPLATE is used).
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    problems = []
    if not _STRICT_SUFFICIENT_LINE_RE.search(raw_response):
        problems.append("no standalone 'SUFFICIENT: yes/no' line found")
    if not _STRICT_REFORMULATED_LINE_RE.search(raw_response):
        problems.append("no standalone 'REFORMULATED QUERY: ...' line found")
    if not problems:
        return None
    return (
        "Parse reliability check failed (" + "; ".join(problems) + ") - the parsed 'sufficient' field for "
        "this record may reflect a stray mention inside reasoning text rather than the model's real final "
        "verdict. Read raw_response manually before trusting this record."
    )


@dataclass(frozen=True)
class _FixedCandidate:
    """The minimal shape required by build_context_block()/generate_answer()
    (only .context_id and .full_indexed_content - see pipeline/generation.py) -
    not pipeline.retrieval.Candidate/pipeline.reranking.RerankedCandidate,
    because there is neither a retrieval score nor a rerank score here: the
    documents are assembled manually, not via search."""

    context_id: str
    full_indexed_content: str


def _is_routed(config, source_dataset: str) -> bool:
    routing = config.embedding.routing
    return routing.enabled and source_dataset in routing.routed_sources


def _fetch_full_indexed_content(collection, context_id: str) -> str:
    doc = collection.find_one({"context_id": context_id, "is_indexed": True}, {"full_indexed_content": 1, "_id": 0})
    if doc is None or "full_indexed_content" not in doc:
        raise RuntimeError(
            f"context_id={context_id!r} not found/indexed in the production collection - this diagnostic "
            f"only reuses already-indexed documents, it does not index anything itself. Run the T2-RAGBench "
            f"indexing pipeline (scripts/run_eval.py index, or pipeline.cli.cmd_index) first."
        )
    return doc["full_indexed_content"]


def _check_already_indexed(collection, items: list[dict]) -> None:
    all_ids = sorted({item["gold_context_id"] for item in items} | {cid for item in items for cid in item["distractor_context_ids"]})
    missing = [cid for cid in all_ids if not is_indexed(collection, cid)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} document(s) referenced by the fixture are missing from the production "
            f"collection (not indexed): {missing[:10]}{'...' if len(missing) > 10 else ''}. Run the "
            f"T2-RAGBench indexing pipeline first - this diagnostic never indexes anything itself."
        )


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[f"{rec['question_id']}::{rec['condition']}"] = rec
    return done


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]
    validate_startup_indexes(collection, check_source_dataset_filter=config.embedding.routing.enabled)

    items = load_financial_entity_ambiguity_fixture()
    _check_already_indexed(collection, items)

    print(
        f"Financial entity-ambiguity diagnostic: {len(items)} question(s) x {len(CONDITIONS)} condition(s) = "
        f"{len(items) * len(CONDITIONS)} (assessment + answer-generation) pair(s) total, plus "
        f"{len(items)} gold_rank retrieval call(s)."
    )

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, condition) pair(s) already done in {out_path}")
        print(
            "  WARNING: resume does not distinguish between fixture/script versions - if "
            "data/financial_entity_ambiguity/fixture.json or this script has changed since at least one of "
            "these lines was written, the old results will be silently accepted as done. If "
            "this is not the same, unchanged run - delete or rename diagnostic_results.jsonl "
            "before re-running."
        )

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)

    t_start = time.perf_counter()
    n_llm_calls = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in items:
            question_id = item["question_id"]

            # Skip the ENTIRE question (including manual context assembly and
            # the paid retrieval call below) if both conditions are already
            # done - a real finding from the second expert's review (see the
            # review document): without this check, resume only skipped the
            # LLM calls themselves (context assembly via a MongoDB read
            # inside the loop below + search_fn - a paid Voyage/optionally
            # Cohere call - ran again on EVERY restart for questions that
            # were already done, contradicting the docstring's "The run is
            # idempotent" claim. Now idempotency covers retrieval too, not
            # just assessment/generation.
            if all(f"{question_id}::{c}" in done for c in CONDITIONS):
                continue

            # Manual context assembly - IDENTICAL for both conditions of
            # this question (see the module docstring, "Design"). The order
            # is an alphabetical sort of context_id (deterministic, no
            # hidden randomness) - build_context_block() numbers
            # [Document i] in this same order, so correct_document_position
            # below matches what the model will actually see.
            all_context_ids = sorted([item["gold_context_id"], *item["distractor_context_ids"]])
            candidates = [
                _FixedCandidate(context_id=cid, full_indexed_content=_fetch_full_indexed_content(collection, cid))
                for cid in all_context_ids
            ]
            context_text = build_context_block(candidates)
            correct_document_position = all_context_ids.index(item["gold_context_id"]) + 1

            # A diagnostic live-retrieval call, SEPARATE from context
            # assembly - see the module docstring, "What is logged",
            # live_retrieval_gold_rank. Once per question (not per
            # condition), always with the original question text - this
            # measures "would ordinary retrieval have found this document,"
            # it is not part of the assessor's test
            # (retrieval_used_for_assessor_context=False in every record
            # below - the assessor always sees only the manual context_text,
            # never the result of this call).
            #
            # Wrapped in try/except (second expert, item 5.2): this is a
            # SIDE measurement, not required for the main experiment -
            # search_documents (agent/tools.py) already degrades gracefully
            # on transient MongoDB/Cohere errors on its own, but there is no
            # reason to let ANY error here (including an unexpected one)
            # sink the already-paid-for assessment/generation progress for
            # this and subsequent questions - the assessment below does not
            # depend on this block at all.
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
            live_retrieval_gold_rank: int | None = None
            live_retrieval_pool_size: int | None = None
            live_retrieval_error: str | None = None
            try:
                retrieval_call = search_fn(item["question_original"])
                retrieved_ids = [c.context_id for c in retrieval_call.candidates]
                live_retrieval_pool_size = len(retrieved_ids)
                if item["gold_context_id"] in retrieved_ids:
                    live_retrieval_gold_rank = retrieved_ids.index(item["gold_context_id"]) + 1
            except Exception as exc:  # noqa: BLE001 - see docstring above: this is a non-critical side measurement
                live_retrieval_error = f"{type(exc).__name__}: {exc}"
                print(f"      WARNING: live_retrieval (gold_rank) failed for {question_id!r}: {live_retrieval_error}")

            for condition in CONDITIONS:
                key = f"{question_id}::{condition}"
                if key in done:
                    continue

                question_text = item[f"question_{condition}"]
                calls_remaining = config.agent.max_additional_tool_calls
                assessment_prompt = _ASSESSMENT_PROMPT_TEMPLATE.format(
                    question=question_text, context=context_text, calls_remaining=calls_remaining
                )
                assessment_raw = generator.generate(assessment_prompt)
                n_llm_calls += 1
                parsed = _parse_assessment(assessment_raw)
                parse_warning = _check_parse_reliability(assessment_raw)

                # Answer generation ALWAYS runs (not only when
                # sufficient=yes) - see the module docstring, "Why answer
                # generation is not gated on sufficient".
                generated = generate_answer(
                    generator, question_id, question_text, candidates, template=AGENT_ANSWER_PROMPT_TEMPLATE
                )
                n_llm_calls += 1
                answer_correct = is_close_v2(generated.answer_text, item["gold_answer"])

                # single_shot_production_answer/single_shot_production_correct
                # (renamed from production_answer/production_correct per the
                # second expert's review, item 4) - derived fields that
                # reproduce ONLY agent/loop.py's forced_insufficient policy
                # AFTER A SINGLE assessment call on this fixed context (not a
                # separate call) - NOT the full production run_agent_query
                # loop (which, on sufficient=no, could reformulate the query
                # and make additional tool calls/re-assessment - none of
                # that happens here, because the whole point of the
                # experiment is a FIXED context). Explicitly named
                # "single_shot" so it doesn't read as "what the full agentic
                # loop would actually have returned."
                single_shot_production_answer = generated.answer_text if parsed.sufficient else "INSUFFICIENT_CONTEXT"
                single_shot_production_correct = bool(parsed.sufficient and answer_correct)

                result = {
                    "question_id": question_id,
                    "condition": condition,
                    "source_dataset": source_dataset,
                    "gold_sector": item["gold_sector"],
                    "gold_company_name": item["gold_company_name"],
                    "gold_answer": item["gold_answer"],
                    "gold_context_id": item["gold_context_id"],
                    "distractor_context_ids": item["distractor_context_ids"],
                    "context_ids": all_context_ids,
                    "correct_document_position": correct_document_position,
                    # assessor_context_source/retrieval_used_for_assessor_context
                    # (added per the second expert's review, item 5.4) - explicit,
                    # data-level (not just docstring-level) statement that the
                    # assessor/generator above saw ONLY the manually-assembled
                    # context_text, never the live_retrieval_* fields below.
                    "assessor_context_source": "manual_fixture",
                    "retrieval_used_for_assessor_context": False,
                    # live_retrieval_* (renamed from gold_rank/retrieval_pool_size
                    # per the second expert's review, item 3) - a SEPARATE,
                    # diagnostic-only live search_documents() call, not part of
                    # the assessor's input. None/live_retrieval_error set if the
                    # call itself failed (see the try/except above) - does not
                    # affect sufficient/answer_correct/single_shot_production_*
                    # below, which never depend on this call.
                    "live_retrieval_gold_rank": live_retrieval_gold_rank,
                    "live_retrieval_pool_size": live_retrieval_pool_size,
                    "live_retrieval_error": live_retrieval_error,
                    "question_text": question_text,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "assessment_raw_response": assessment_raw,
                    "parse_warning": parse_warning,
                    "answer_text": generated.answer_text,
                    "answer_raw_response": generated.raw_response,
                    "answer_correct": answer_correct,
                    "single_shot_production_answer": single_shot_production_answer,
                    "single_shot_production_correct": single_shot_production_correct,
                }
                if parse_warning:
                    print(f"      WARNING: {parse_warning}")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(
                    f"  [{n_llm_calls}] {question_id} / {condition} -> sufficient={parsed.sufficient} "
                    f"answer_correct={answer_correct} live_retrieval_gold_rank={live_retrieval_gold_rank}"
                )

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.0f}s, {n_llm_calls} new LLM call(s) in this session.")

    print("\nSummary by condition (n={} question(s) per condition, paired comparison, indicative only, not a statistical test):".format(len(items)))
    for condition in CONDITIONS:
        rows = [r for r in done.values() if r["condition"] == condition]
        n_sufficient = sum(1 for r in rows if r["sufficient"])
        n_answer_correct = sum(1 for r in rows if r["answer_correct"])
        n_single_shot_production_correct = sum(1 for r in rows if r["single_shot_production_correct"])
        print(
            f"  {condition:12s}: sufficient=yes {n_sufficient}/{len(rows)}, "
            f"answer_correct (generation always runs) {n_answer_correct}/{len(rows)}, "
            f"single_shot_production_correct (applies forced_insufficient, WITHOUT the full agentic loop) "
            f"{n_single_shot_production_correct}/{len(rows)}"
        )

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(items) * len(CONDITIONS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Read the full raw_response for questions where original and anonymized diverged on sufficient or "
        "answer_correct - the counter above is indicative only (n=28, not a statistical test); interpretation "
        "requires reading the individual cases, same as in rounds 1-3."
    )


if __name__ == "__main__":
    main()

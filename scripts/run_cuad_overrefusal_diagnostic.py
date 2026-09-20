"""Root-cause diagnostic for the agent's over-refusal on CUAD (n=5, all 5
unjustified_refusal) - continuation of Day 3
(plan_rabot_posle_ekspertizy_agent_profil.md), after closing out the
interpretation cascade for the real runs (claude/itog_ekspertizy_agent_rezultaty.md).

*** IMPORTANT: this is a diagnostic/qualitative script, NOT a statistical
*** experiment. n=5 questions x 5 prompt variants = 25 assessor calls do
*** not give statistical power regardless of outcome - just as n=5 in
*** scripts/run_cuad_smoke.py did not give it for accuracy. The goal is
*** NOT to prove the cause by numbers, but to obtain direct, persisted raw
*** textual evidence (the full raw_response, not just the parsed verdict)
*** for the next round of the external expert cascade, which will decide
*** between the hypotheses below. The same script prints aggregate counts
*** only as a rough indicator, not as a conclusion.
***
*** Each variant changes EXACTLY ONE variable relative to `original` - at
*** the user's direct request, after reviewing the minimal improvements
*** independently proposed by DeepSeek and Qwen: patching and re-running
*** several changes at once makes it impossible to tell which one (or
*** which combination) actually changes anything - see
*** claude/qwen_cuad_minimalnye_dorabotki.md, section "Methodological
*** disagreement". If after this run there is a wish to test a COMBINATION
*** of variants (e.g. neutral + multi_document_instruction) - that is a
*** separate, subsequent step, once the effect of each variable
*** individually is already visible.

## Hypotheses under test (see claude/status_agent_rezultaty_4_nahodki_kod.md)

1. Domain mismatch in the wording: _ASSESSMENT_PROMPT_TEMPLATE literally
   starts with "...a question about a company's **financial report**",
   while CUAD is legal contracts. Tested by the `neutral` and
   `explicit_legal` variants below - same context, same question, ONLY
   one phrase changes (a text substitution against the original constant,
   not separately retyped text - so as not to introduce accidental
   wording drift, the same principle that the new rule in Section 4 of
   svod_pravil_raboty.md requires for analysis: work from the whole
   original, not from a paraphrase).
2. Structural incompatibility of the bounded-loop assessor with
   multi-document span extraction (Qwen's hypothesis, the code audit, and
   round 3 of the terminology cascade; the same idea independently
   proposed by DeepSeek as a "checklist of required facts" and by Qwen as
   an "explicit instruction about synthesizing across multiple documents"
   in the minimal-improvements round) - tested by the
   `multi_document_instruction` variant below: the same DOMAIN wording as
   in `original` (the domain variable is NOT changed here - the variants
   above test it), plus one added instruction stating that evidence from a
   single document may not be sufficient if the question requires facts
   from several. Independently of this variant, the raw raw_response of
   each call (never persisted before - agent/loop.py only traces the
   parsed sufficient/reformulated_query) provides material for qualitative
   reading across all 5 variants: if the model explicitly writes something
   like "there are multiple contracts/clauses and I cannot determine which
   one is being asked about" - that is direct textual evidence for
   hypothesis 2, regardless of which variant it is observed in.
3. Asymmetry between the assessor's task (before the answer, without
   gold) and the insufficiency-judge's task (after the fact, with gold) -
   the `original_with_gold_hint` variant tests whether the same
   assessor's verdict on the same original (financial) wording changes if
   it is hinted at what the correct answer must establish - this isolates
   "uncertainty about the completeness of extraction" (which access to
   gold would remove) from "the domain wording cuts to the core" (which
   access to gold would not touch).

## Why the already-indexed collection is reused instead of rebuilding
## the context from scratch

pipeline/indexing.py's build_full_indexed_content() mixes in
contextual_summary - LLM-generated enrichment (enrichment.enabled: true
in config_cuad_smoke.yaml, the same claude-haiku-4-5-20251001 as in
production). Reproducing the original indexed text exactly requires
either (a) a real call to the same API for the same summary, which is
not guaranteed to be deterministic even at temperature=0.0, or (b)
reading the already-indexed value from the same collection - there is no
other way. This script picks (b): it simply calls the same search_fn
against the ALREADY existing cuad_smoke_v1 collection (written by the
real run of scripts/run_cuad_smoke.py) - thereby obtaining LITERALLY the
same context_text the agent saw in the real run, without a single new
enrichment call. This requires that scripts/run_cuad_smoke.py has already
been run successfully once on this MongoDB instance - otherwise the
script stops with a clear error (see _check_already_indexed below),
rather than silently re-indexing with new (potentially different)
enrichment.

## Cost

5 variants x 5 questions = 25 assessor calls (generation.model:
claude-sonnet-5, the same one already used as the generator in
run_cuad_smoke.py - no new judge/generation call, retrieval reuses the
already existing index). At cost_per_llm_call_usd=0.02 (the same estimate
as config_cuad_smoke.yaml.agent_eval) - roughly ~$0.50 + 5 retrieval/
rerank calls (Voyage/Cohere, an order of magnitude cheaper).

Usage (Colab, after git pull, after scripts/run_cuad_smoke.py has
already been run at least once on this MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic.py
The run is idempotent (resume by question_id+variant), like the other
scripts in this category.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import functools  # noqa: E402

from agent.loop import _ASSESSMENT_PROMPT_TEMPLATE, _parse_assessment  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, build_clients  # noqa: E402
from pipeline.cuad_smoke import load_cuad_smoke_fixture  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.generation import build_context_block  # noqa: E402
from pipeline.indexing import is_indexed, validate_startup_indexes  # noqa: E402

RUN_ID = "cuad_overrefusal_diagnostic"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# The domain phrase we change. Taken verbatim from
# _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py) - NOT retyped by hand, so
# the substitution is guaranteed to work (and would fail with a clear
# assert error, rather than silently matching the original, if the
# constant ever changes without this script being updated - see the
# check right after the VARIANTS definition below).
_ORIGINAL_PHRASE = "a question about a company's financial report"

# Hypothesis 2 (multi-document): the anchor for insertion - taken
# verbatim from _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py), not retyped
# by hand, for the same reason as _ORIGINAL_PHRASE above. Inserted
# BETWEEN item 1 ("Is the evidence sufficient...") and item 2 ("If not
# sufficient...") - the domain phrase (_ORIGINAL_PHRASE) is NOT touched
# here, so this variant changes exactly one variable (the
# multi-document instruction), not two.
_MULTI_DOC_ANCHOR = "do not use outside knowledge.\n2. If not sufficient"
_MULTI_DOC_INSTRUCTION = (
    " Note: the passages above may be drawn from more than one source document. If the "
    "question requires combining or cross-referencing facts from multiple documents, a single "
    "relevant-looking passage from only one document is NOT sufficient by itself - check whether "
    "every fact the question needs is actually present, even if scattered across different "
    "passages, before answering yes."
)

VARIANTS: dict[str, str] = {
    # Control - byte-for-byte the same prompt as the real run.
    "original": _ASSESSMENT_PROMPT_TEMPLATE,
    # Hypothesis 1a: remove the domain entirely (name neither finance nor law).
    "neutral": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _ORIGINAL_PHRASE,
        "a question, using only the retrieved passages below",
    ),
    # Hypothesis 1b: name the CORRECT domain explicitly (not just neutrally).
    "explicit_legal": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _ORIGINAL_PHRASE,
        "a question about a legal contract",
    ),
    # Hypothesis 2: the domain wording does NOT change (stays "financial
    # report", as in original) - only the presence of an explicit
    # multi-document synthesis instruction changes. Independently
    # proposed by DeepSeek ("checklist of required facts") and Qwen
    # ("explicit instruction about synthesizing across multiple
    # documents") - see claude/deepseek_cuad_minimalnye_dorabotki.md,
    # claude/qwen_cuad_minimalnye_dorabotki.md.
    "multi_document_instruction": _ASSESSMENT_PROMPT_TEMPLATE.replace(
        _MULTI_DOC_ANCHOR,
        "do not use outside knowledge." + _MULTI_DOC_INSTRUCTION + "\n2. If not sufficient",
    ),
    # Hypothesis 3: the original (financial) wording + a hint about what
    # the correct answer must establish, WITHOUT revealing the actual
    # gold_answer value as text - inserted separately per question, see
    # _build_gold_hint_prompt() below, not via .replace() here, since the
    # insertion depends on the per-question gold_answer.
    "original_with_gold_hint": _ASSESSMENT_PROMPT_TEMPLATE,
}

for _name, _template in (
    ("neutral", VARIANTS["neutral"]),
    ("explicit_legal", VARIANTS["explicit_legal"]),
    ("multi_document_instruction", VARIANTS["multi_document_instruction"]),
):
    if _template == _ASSESSMENT_PROMPT_TEMPLATE:
        raise RuntimeError(
            f"Ablation anchor not found in current _ASSESSMENT_PROMPT_TEMPLATE for variant "
            f"{_name!r} - the prompt constant changed since this script was written; update "
            f"_ORIGINAL_PHRASE / _MULTI_DOC_ANCHOR in scripts/run_cuad_overrefusal_diagnostic.py "
            f"to match before running this variant."
        )


def _build_gold_hint_prompt(question: str, context: str, calls_remaining: int, gold_answer: str) -> str:
    """Variant original_with_gold_hint: the same original prompt, plus one
    additional line AFTER the instruction, naming what kind of fact the
    correct answer must establish - without quoting the gold value
    directly, only its category/shape (e.g. "a specific date" or "a
    governing law jurisdiction name"), so as not to turn this into a
    trivial "copy the answer", but specifically to remove the
    uncertainty about the COMPLETENESS of extraction, which the model
    cannot remove in a real assessor call."""
    base = _ASSESSMENT_PROMPT_TEMPLATE.format(question=question, context=context, calls_remaining=calls_remaining)
    hint = (
        "\n[Diagnostic hint - not present in production: a correct, complete answer to this "
        f"question would need to be a specific fact of this general shape: {_categorize_gold(gold_answer)}. "
        "Use this only to judge whether the passages above already contain a fact of that shape - "
        "do not guess or fabricate the fact itself.]\n"
    )
    return base + hint


def _categorize_gold(gold_answer: str) -> str:
    """A rough, imperfect categorization of the answer's shape (not its
    value) - sufficient for hypothesis 3 (to remove uncertainty about
    COMPLETENESS, not to give away the answer). Reveals the genre of the
    fact, never the fact itself."""
    g = gold_answer.strip()
    if any(ch.isdigit() for ch in g) and len(g) <= 12:
        return "a short date or numeric value"
    if len(g.split()) <= 4:
        return "a short named entity (e.g. a jurisdiction, party name, or defined term)"
    return "a clause or span of contract text"


def _check_already_indexed(collection, records) -> None:
    # records is list[pipeline.ingestion.DocumentRecord] straight from
    # load_cuad_smoke_fixture() - a dataclass (attribute access via
    # `.context_id`), NOT a dict. Unlike scripts/run_cuad_smoke.py, this
    # function runs BEFORE dedupe_documents() (which is what converts
    # DocumentRecord -> dict elsewhere in the codebase), so dict-style
    # subscripting here was a bug: TypeError: 'DocumentRecord' object is
    # not subscriptable.
    missing = sorted({r.context_id for r in records if not is_indexed(collection, r.context_id)})
    if missing:
        raise RuntimeError(
            "CUAD smoke collection is missing document(s) this diagnostic needs to reuse "
            f"as-is: {missing}. This script deliberately does NOT index/enrich anything itself "
            "(see module docstring - reusing the exact already-indexed context, not re-deriving "
            "enrichment). Run scripts/run_cuad_smoke.py once first, then re-run this script."
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
            done[f"{rec['question_id']}::{rec['variant']}"] = rec
    return done


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)
    print(f"Reusing already-indexed cuad_smoke_v1 collection - {len(eval_items)} question(s), "
          f"{len({r.context_id for r in records})} document(s), {len(VARIANTS)} prompt variant(s) each "
          f"= {len(eval_items) * len(VARIANTS)} assessor call(s) total.")

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, variant) pair(s) already done in {out_path}")

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
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
        source_dataset=None,
        exclude_source_datasets=None,
    )

    t_start = time.perf_counter()
    n_calls = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in eval_items:
            question_id = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]

            # One retrieval per question, reused across all 5 variants -
            # otherwise different variants could (in principle) see
            # different context_text, which would break the ablation
            # ("ONLY the prompt wording changes").
            call = search_fn(question)
            context_text = build_context_block(list(call.candidates))
            context_ids = sorted(call.context_ids)

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                calls_remaining = config.agent.max_additional_tool_calls
                if variant == "original_with_gold_hint":
                    prompt = _build_gold_hint_prompt(question, context_text, calls_remaining, gold_answer)
                else:
                    prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "raw_response": raw_response,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(f"  [{n_calls}] {question_id} / {variant} -> sufficient={parsed.sufficient}")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.0f}s, {n_calls} new assessor call(s) this session.")

    print("\nSummary of sufficient=yes by variant (a reference point, NOT a statistical conclusion at n=5):")
    for variant in VARIANTS:
        rows = [r for r in done.values() if r["variant"] == variant]
        n_yes = sum(1 for r in rows if r["sufficient"])
        print(f"  {variant:24s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Next step - NOT an automatic inference of the cause from the tallies above: read each "
        "record's raw_response (the model's full reasoning text is saved for each of the 20 "
        "calls) and bring it to the external expert cascade together with the sufficient=yes summary."
    )


if __name__ == "__main__":
    main()

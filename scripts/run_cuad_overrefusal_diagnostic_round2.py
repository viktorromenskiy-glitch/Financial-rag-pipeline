"""CUAD over-refusal diagnostic, round 2 - continuation of
scripts/run_cuad_overrefusal_diagnostic.py (round 1, results saved in
claude/status_agent_rezultaty_4_nahodki_kod.md and on Drive under run_id
"cuad_overrefusal_diagnostic").

*** IMPORTANT: like round 1, this is a diagnostic/qualitative script,
*** NOT a statistical experiment. n=5 questions x 2 variants = 10
*** assessor calls do not give statistical power. The goal is to test
*** hypothesis 2 quantitatively and finally obtain the model's textual
*** reasoning, which round 1 could not obtain structurally (see below).

## What round 1 showed (25 calls, 5 questions x 5 variants, all
## sufficient=no)

1. Hypothesis 1 (domain wording "financial report" instead of "legal
   contract") - REFUTED: the `neutral` and `explicit_legal` variants gave
   the same result (0/5) as `original`. Replacing one domain phrase does
   not change the verdict.
2. Hypothesis 3 (the assessor is unsure about COMPLETENESS, not
   relevance) - REFUTED: `original_with_gold_hint` (the model is told
   directly the SHAPE of the correct answer) also gave 0/5 - the assessor
   does not acknowledge an already-present fact of the needed shape, even
   when explicitly told what to look for.
3. Hypothesis 2 (structural incompatibility with multi-document) - NOT
   tested quantitatively by round 1: the `multi_document_instruction`
   variant only added an instruction to be MORE CAUTIOUS (it could not
   increase the share of "yes"), rather than removing the actual source
   of confusion - the presence of 3 known-irrelevant documents out of 4
   in the context of EVERY question (the real size of the CUAD smoke
   corpus is only 4 documents; retrieval with pool_size=50 returns all 4
   for any query - see claude/verifikaciya_context_ids_cuad_smoke.md).
4. A side finding, more important than the counts themselves:
   raw_response in all 25 round-1 records contains NO reasoning at all -
   only "SUFFICIENT: no\\nREFORMULATED QUERY: ...". This is not a logging
   bug - _ASSESSMENT_PROMPT_TEMPLATE itself (agent/loop.py) explicitly
   demands "Respond in exactly this format, nothing else". Round 1's plan
   ("read raw_response to understand the reasoning") was unachievable
   with this prompt. Recorded in the rulebook (svod_pravil_raboty.md,
   "Before launching a NEW paid/diagnostic script - check design-vs-promise
   with two independent external experts") as grounds for a new mandatory
   step.

## Two new variants in this round (each changes EXACTLY ONE variable
## relative to `original`, independently of each other - as in round 1)

- `single_relevant_document_only`: the `original` prompt WITHOUT CHANGES
  (the same _ASSESSMENT_PROMPT_TEMPLATE constant, not a single word
  different), but {context} is fed ONLY the one document out of 4 that
  actually contains the answer to this question (determined from
  pipeline.cuad_smoke.load_cuad_smoke_fixture() - each question_id there
  corresponds to exactly one document_id). The three unrelated documents
  are not included at all - not "flagged as irrelevant", but physically
  absent from the context. Tests hypothesis 2 directly: if sufficient=yes
  appears here while `original` (the same 4 documents) gave no - the
  cause is noise from unrelated documents, not the question's wording or
  the domain.
- `with_reasoning`: the same `original` (the same context - all 4
  documents, the same domain wording "financial report" - the domain is
  NOT a variable here, round 1 already tested it), but with one added
  instruction to give one line of reasoning BEFORE the verdict. The sole
  purpose is to finally obtain textual evidence instead of a bare "no",
  since round 1 could not obtain it structurally. _parse_assessment()
  (agent/loop.py) is already robust to text preceding the final
  SUFFICIENT/REFORMULATED QUERY lines (it takes the LAST match of each
  marker - see its docstring), so adding one line of reasoning before
  those lines requires no changes to parsing.

## Why a separate script, rather than new entries in round 1's VARIANTS

run_cuad_overrefusal_diagnostic.py makes ONE retrieval call per question
and reuses the same context_text for all prompt variants (needed for a
fair ablation - "ONLY the wording changes"). This round's
`single_relevant_document_only` variant, by contrast, deliberately
changes the CONTEXT ITSELF (not just the wording) - technically
incompatible with round 1's invariant. Mixing both cases into one
VARIANTS dict with a shared loop would mean either breaking round 1's
invariant for all its variants, or building branching logic that is easy
to get wrong. A separate script with its own (shorter) loop carries less
risk of breaking the already-validated round 1.

## Reusing the same index as round 1 - no exceptions

As in round 1: no new ingestion/embedding/enrichment - the same
principle and the same _check_already_indexed(), the same search_fn via
agent.tools.search_documents against the already existing cuad_smoke_v1
collection.

## Cost

5 questions x 2 variants = 10 assessor calls (claude-sonnet-5, the same
generation.model). At cost_per_llm_call_usd=0.02 - roughly ~$0.20 + 5
additional retrieval/rerank calls (Voyage/Cohere, an order of magnitude
cheaper; the same 5 as round 1 are reused, since it's the same question
set and the same search_fn - NOT 10 separate retrieval calls, one per
question, as in round 1).

## Before running with real money

Under the rulebook's new rule - this script must be shown to TWO
independent external experts with the question "is it actually capable
of delivering what it claims, is there a contradiction between the goal
and the mechanism" BEFORE the first paid run. Do not run until this has
been done.

Usage (Colab, after git pull, after scripts/run_cuad_smoke.py has
already been run at least once on this MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic_round2.py
The run is idempotent (resume by question_id+variant), like round 1.
"""
from __future__ import annotations

import json
import re
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

RUN_ID = "cuad_overrefusal_diagnostic_round2"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# Anchor for inserting the reasoning instruction - taken verbatim from
# _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py), not retyped by hand, for
# the same reason as in round 1: a guaranteed assert failure if the
# constant changes without this script being updated, instead of a
# silent mismatch.
_FORMAT_ANCHOR = (
    "Respond in exactly this format, nothing else:\n\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)
_FORMAT_WITH_REASONING = (
    "Before your verdict, on a line starting with \"REASONING:\", state in one sentence "
    "which single fact the question needs is present in, or missing from, the passages above. "
    "Then, on the following lines, respond in exactly this format:\n\n"
    "REASONING: <one sentence>\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)

_WITH_REASONING_TEMPLATE = _ASSESSMENT_PROMPT_TEMPLATE.replace(_FORMAT_ANCHOR, _FORMAT_WITH_REASONING)
if _WITH_REASONING_TEMPLATE == _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "Ablation anchor _FORMAT_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the "
        "prompt constant changed since this script was written; update _FORMAT_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round2.py to match before running."
    )

# INTERPRETATION CAVEAT (confirmed by two independent experts before
# running - see prompt_dlya_2_ekspertov_round2_script.md): asking for one
# line of reasoning BEFORE the verdict is chain-of-thought elicitation,
# which can change the model's actual decision, not merely make an
# already-made decision visible (a widely documented effect). If
# with_reasoning gives sufficient=yes where original/round 1 gave no,
# this CANNOT automatically be read as "original simply did not see the
# fact". A possible alternative: the very request to reason changed the
# decision-making process itself. The sole stated purpose of this variant
# is to obtain the TEXT of the reasoning (raw_response), not to test the
# hypothesis purely through the presence/absence of reasoning - when
# analyzing results, read raw_response/REASONING: itself, not just the
# binary sufficient. A clean test of "does the reasoning itself change
# the verdict" is a separate, more expensive experiment (the same
# context, original vs with_reasoning interleaved, a larger n), not what
# this script does.

# The only variant that reuses the original template WITHOUT CHANGES -
# only the context it receives differs (assembled separately in main(),
# not through this dict). Kept here purely for uniformity of
# iteration/reporting; the value (template) is not used for
# single_relevant_document_only - see main().
VARIANTS: dict[str, str] = {
    "single_relevant_document_only": _ASSESSMENT_PROMPT_TEMPLATE,
    "with_reasoning": _WITH_REASONING_TEMPLATE,
}


# An additional check, independent of agent/loop.py._parse_assessment
# (found by the second independent expert reviewing this script before
# running it - see prompt_dlya_2_ekspertov_round2_script.md):
# _parse_assessment takes the LAST match of the markers in the response -
# which is correct IF the model does eventually emit real final
# "SUFFICIENT: .../REFORMULATED QUERY: ..." lines. with_reasoning asks the
# model to reason first - if the model happens to mention "...is
# SUFFICIENT: yes, but..." INSIDE the reasoning and NEVER emits a separate
# final line with the real verdict, _parse_assessment will silently take
# that stray mention as the real verdict - no crash, no warning. This
# does not change _parse_assessment itself (shared, production-used
# code) - instead it independently checks that the markers are present
# as STANDALONE lines (not inside a sentence), and adds a warning to the
# saved record if they are not - so that such a record is not silently
# trusted on a par with the rest when reading the results later.
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    """Returns a human-readable warning if raw_response does not contain
    both markers as their own standalone lines (as the prompt instructs),
    or None if it looks like a clean, trustworthy final answer."""
    problems = []
    if not _STRICT_SUFFICIENT_LINE_RE.search(raw_response):
        problems.append("no standalone 'SUFFICIENT: yes/no' line found")
    if not _STRICT_REFORMULATED_LINE_RE.search(raw_response):
        problems.append("no standalone 'REFORMULATED QUERY: ...' line found")
    if not problems:
        return None
    return (
        "Parse reliability check failed (" + "; ".join(problems) + ") - the parsed "
        "'sufficient'/'reformulated_query' fields for this record may reflect a stray "
        "mention inside reasoning text rather than the model's real final verdict. "
        "Read raw_response manually before trusting this record."
    )


def _verify_relevant_doc_mapping(records, eval_items) -> None:
    """Belt-and-suspenders check (second independent expert's flagged
    concern, prompt_dlya_2_ekspertov_round2_script.md): relevant_doc_id_by_question
    is built by zipping records/eval_items positionally, which is correct
    BY CONSTRUCTION (pipeline/cuad_smoke.py's load_cuad_smoke_fixture()
    appends to both lists inside a single loop over the same question, no
    separate sort afterwards) - verified by reading that source directly,
    and empirically (manually, offline, gold_answer confirmed present in
    the assigned document's text for all 5 questions in the current
    fixture). This function turns that one-off manual check into a
    permanent runtime guard, so a FUTURE edit to the fixture loader (e.g.
    someone adding a sort) fails loudly here instead of silently feeding
    single_relevant_document_only the wrong document. Uses a short prefix
    of gold_answer (not the whole string) since some gold answers are full
    sentences that could have been retokenized/whitespace-normalized
    relative to the raw document text - a prefix match is enough to catch
    a genuine document_id mismatch without being fragile to that.
    """
    problems = []
    for rec, item in zip(records, eval_items):
        gold = item["gold_answer"].strip()
        probe = gold if len(gold) <= 40 else gold[:40]
        if probe.upper() not in rec.context.upper():
            problems.append(
                f"question_id={item['question_id']!r}: gold_answer prefix {probe!r} not found in "
                f"document {rec.context_id!r} - relevant_doc_id_by_question would be wrong for this "
                f"question"
            )
    if problems:
        raise RuntimeError(
            "relevant_doc_id_by_question mapping failed verification - "
            "single_relevant_document_only would test against the wrong document(s):\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def _check_already_indexed(collection, records) -> None:
    # The same bug that was found and fixed in round 1 - DocumentRecord
    # is a dataclass (attribute .context_id), not a dict. Do not repeat
    # r["context_id"].
    missing = sorted({r.context_id for r in records if not is_indexed(collection, r.context_id)})
    if missing:
        raise RuntimeError(
            "CUAD smoke collection is missing document(s) this diagnostic needs to reuse "
            f"as-is: {missing}. Run scripts/run_cuad_smoke.py once first, then re-run this script."
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
    """Runs round 2's two ablation variants against the CUAD smoke fixture.

    Raises:
        RuntimeError: If the relevant document for a question is not found
            among that question's retrieved candidates (a retrieval
            regression, not something this diagnostic can answer).
    """
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)

    # records and eval_items are built in ONE loop over the same list of
    # questions (pipeline/cuad_smoke.py) - one DocumentRecord per
    # question, in the same order. relevant_doc_id_by_question maps
    # question_id -> the id of the single document that actually answers
    # that question (not via a heuristic - it's the same mapping baked
    # into the original fixture JSON).
    assert len(records) == len(eval_items), (
        "records and eval_items must be the same length and in the same order - "
        "load_cuad_smoke_fixture() builds them in a single loop over questions"
    )
    _verify_relevant_doc_mapping(records, eval_items)
    relevant_doc_id_by_question = {
        item["question_id"]: rec.context_id for rec, item in zip(records, eval_items)
    }

    print(
        f"Reusing already-indexed cuad_smoke_v1 collection - {len(eval_items)} question(s), "
        f"{len(VARIANTS)} new variant(s) each = {len(eval_items) * len(VARIANTS)} assessor call(s) total "
        f"(round 2, separate from round 1's 25 calls)."
    )

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
            relevant_doc_id = relevant_doc_id_by_question[question_id]

            # One retrieval per question, as in round 1 - reused to build
            # BOTH context variants below (full and filtered), not two
            # separate calls.
            call = search_fn(question)
            all_candidates = list(call.candidates)

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                if variant == "single_relevant_document_only":
                    candidates = [c for c in all_candidates if c.context_id == relevant_doc_id]
                    if not candidates:
                        raise RuntimeError(
                            f"question_id={question_id!r}: relevant document {relevant_doc_id!r} "
                            f"not found among retrieved candidates "
                            f"({[c.context_id for c in all_candidates]!r}) - retrieval regression, "
                            f"not a prompt/context question this diagnostic can answer. Stopping "
                            f"rather than silently testing against the wrong document set."
                        )
                    context_text = build_context_block(candidates)
                else:
                    candidates = all_candidates
                    context_text = build_context_block(candidates)

                context_ids = sorted({c.context_id for c in candidates})
                calls_remaining = config.agent.max_additional_tool_calls
                prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)
                parse_warning = _check_parse_reliability(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "raw_response": raw_response,
                    "parse_warning": parse_warning,
                }
                if parse_warning:
                    print(f"      WARNING: {parse_warning}")
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
        print(f"  {variant:32s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "single_relevant_document_only getting sufficient=yes here while original got sufficient=no "
        "in round 1 (same question, same 4 documents in context) is direct quantitative evidence for "
        "hypothesis 2 (noise from unrelated documents). with_reasoning's raw_response now contains "
        "the reasoning text - read it before drawing any conclusion, not just the sufficient=yes tally."
    )


if __name__ == "__main__":
    main()

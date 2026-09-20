"""CUAD over-refusal diagnostic, round 3 - testing the proposed fix
("first identify the relevant documents") BEFORE the financial regression.

*** DIAGNOSTIC script, not a statistical experiment. n=5 questions x
*** 2 variants = 10 assessor calls do not provide statistical power. The
*** goal is to cheaply and quickly test one specific hypothesis (the
*** structural step "first list the relevant documents" inside the prompt)
*** BEFORE spending money on the expensive n=30 financial regression.

## How this script came about

After round 2 (see run_cuad_overrefusal_diagnostic_round2.py), hypothesis 2
was confirmed - the cause of refusals on CUAD is noise from unrelated
documents in the context, not the question wording and not the prompt's
domain (see the full analysis in claude/prompt_ekspert_cuad_fix_dizayn.md).
A formal cascade of 4 independent experts (the rulebook, section 2) was
started to design a production fix. The first expert proposed a specific
fix: restructure `_ASSESSMENT_PROMPT_TEMPLATE` so that the assessor FIRST
explicitly lists which of the supplied documents are actually relevant to
the question, and only then judges sufficiency STRICTLY within that subset
(a third response marker `RELEVANT DOCUMENTS: <...>` before
`SUFFICIENT:`/`REFORMULATED QUERY:`).

Before accepting this fix, the author (this agent) put two direct
questions to the first expert: (1) is there a principled, guaranteed
difference between this fix's mechanism and round 2's already-tested and
failed `with_reasoning` variant (both leave all 4 documents physically
visible to the model and rely on it "noticing" irrelevance on its own) -
answer: there is NO principled guaranteed difference, only the assumption
that an explicit requirement to SELECT a document will work more reliably
than simply asking it to reason; (2) before the expensive n=30 financial
regression, is it worth first cheaply re-verifying the flip itself on the
same 5 CUAD questions - answer: yes, unambiguously, with a clear criterion
(>=3 out of 5 sufficient=yes -> proceed to the financial regression;
otherwise hypothesis A in this form is refuted and a different mechanism
is needed). This script is exactly that cheap check.

## How this script DIFFERS from the first draft (important - reasons below)

The first expert also proposed draft code for the diagnostic script. It is
not used verbatim - independent verification (mandatory for this project:
see the rule "never accept expert-written code without independent
verification") found two methodological deviations from the discipline
established in round 1/round 2 of "change EXACTLY ONE variable at a time":

1. The expert's draft RETYPED the entire _ASSESSMENT_PROMPT_TEMPLATE by
   hand instead of taking the real constant and inserting only the new
   step into it (the way round 1 and round 2 did, via `.replace()` on a
   verbatim anchor). Side effect of the retyping: the draft also removed
   the domain phrase "a question about a company's financial report",
   replacing it with the neutral "the question below". That is a SECOND
   variable, unrelated to the hypothesis under test - hypothesis 1 (domain
   wording) was already separately tested and REFUTED in round 1
   (neutral/explicit_legal gave the same 0/5 result as original). If
   `structured_relevant_first` had produced a flip with BOTH variables
   changed at once, it would not be possible to claim the flip was caused
   specifically by the new relevant-documents step rather than by a random
   interaction with the removed domain phrase. This script instead builds
   the new prompt via `.replace()` on verbatim anchors of the real
   _ASSESSMENT_PROMPT_TEMPLATE (the same technique used in round 2 for
   with_reasoning) - the domain phrase "financial report" stays UNCHANGED,
   and only the structure of the first-identify-relevant-documents step
   changes. That is the single variable under test.
2. The draft did not include `_check_parse_reliability` (an independent
   check added by the second expert in round 2 - see
   run_cuad_overrefusal_diagnostic_round2.py) - with the new
   `RELEVANT DOCUMENTS:` marker, the same class of risk already found for
   `REASONING:` in round 2 (the model might accidentally mention
   "SUFFICIENT: yes" inside the document listing and never produce a
   separate final line) does not disappear just because the marker is
   different. The check has been carried over unchanged.

## Third difference: a within-run "original" control, not just a
## comparison with round 1

The expert's draft would have compared `structured_relevant_first` only
against the round 1 result (0/5, a different session, potentially a
different model version, run earlier). This script instead re-runs the
unchanged `original` (the same _ASSESSMENT_PROMPT_TEMPLATE, without a
single word of change) on the same 5 questions IN THIS SAME RUN - the same
"change exactly one variable" principle, but applied to time and model
version rather than only to the prompt text. It costs 5 extra calls
(~$0.1), but rules out the stray question of "could model/environment
drift alone have produced a different result?"

## What is being tested and the decision criterion (agreed with the first
## expert, refined after independent verification by the second expert -
## see below)

- `original`: control, expected 0/5 (as in round 1, now run in this same
  session for a clean comparison). If original unexpectedly gives
  sufficient=yes even once, that is a signal of model/environment drift,
  not a useful result about hypothesis A, and it must be investigated
  separately BEFORE interpreting structured_relevant_first.
- `structured_relevant_first`: the first expert's candidate fix. The
  threshold is fixed as a single number, not a range: >=3 out of 5
  sufficient=yes is a sufficient diagnostic signal to proceed to the
  expensive n=30 financial regression (with the mandatory check that 23/30
  and 28/30 have not regressed); 4-5 out of 5 is a stronger signal than
  exactly 3 out of 5, but the passing threshold is the same. Fewer than 3
  out of 5 (0-2) means hypothesis A in this specific formulation is
  considered REFUTED at the diagnostic level, and a different mechanism is
  needed (e.g. physical context filtering rather than a text instruction)
  - proceeding to the financial regression with this prompt text is not
  warranted. Important caveat (raised by the second expert while
  reviewing the script, accepted unchanged): even 4-5 out of 5 on n=5 is a
  DIAGNOSTIC signal, sufficient to justify the next, more expensive and
  statistically meaningful experiment, not proof that the production fix
  works. The n=30 financial regression is a mandatory next step before
  any conclusion about the fix's readiness, not optional.

## What RELEVANT DOCUMENTS actually measures - an important clarification
## (second expert, accepted unchanged)

`structured_relevant_first` does not perform physical context filtering
(unlike round 2's `single_relevant_document_only`, where unrelated
documents are physically absent from the context). The model still sees
all 4 documents in full; the `RELEVANT DOCUMENTS: <...>` line is only a
TEXTUAL instruction and its textual output, not a programmatic filter -
Python never checks or enforces what the model wrote on this line before
accepting its `SUFFICIENT:`. In other words, this variant tests not
"physical separation of relevant and irrelevant documents" but "does an
explicit two-step textual protocol (first list the relevant documents,
then judge using only them) help more than free-form reasoning
(with_reasoning) or having no such step at all (original)". If the
structure works (>=3/5), that is a diagnostic signal in favor of the
protocol hypothesis, not proof that the model technically constrained
itself to the listed subset - RELEVANT DOCUMENTS remains a manual
diagnostic field for a human to read, not a measurement with an automatic
correctness guarantee (see `correct_document_position` below).

## Document identifiers the model sees (verified, not assumed)

pipeline/generation.py's build_context_block() wraps each candidate as
`[Document {i}]` (1-indexed, in ranking order, i from 1 to
len(candidates)) - meaning the model DOES have a definite, stable
identifier scheme within a single call; the question is settled by
empirically checking the source code, not left as an assumption. The
problem lies elsewhere: these positional labels ("Document 2") do not
match the format of `correct_document_id` (the full context_id, of the
form "GALACTICOMMTECHNOLOGIESINC_11_07_1997-EX-10.46-WEB HOSTING
AGREEMENT") - so comparing `relevant_documents_raw` directly against
`correct_document_id` (string to string) gives nothing useful. The
solution (without changing the prompt itself - changing the RELEVANT
DOCUMENTS wording would add a third variable to the test, not just to the
diagnostics): each result record additionally includes
`correct_document_position` - the position (1-indexed, in the same order
as `[Document i]` in build_context_block) of the correct document among
THIS question's candidates. When reading the results, it is enough to
compare `relevant_documents_raw` (e.g. "Document 2") with
`correct_document_position` (e.g. 2), without opening the source code.

Each record's `raw_response` is accompanied by the fields
`relevant_documents_raw` (what the model wrote on the RELEVANT DOCUMENTS
line, if it did), `correct_document_id` (the real id of the document that
answers the question, from the same fixture as in round 2), and
`correct_document_position` (see above) - so that when reading the
results one can immediately see not only the sufficient=yes count but also
whether the document the model named matches the real one.

## Reliability of SUFFICIENT/REFORMULATED QUERY parsing - verified, not
## assumed (the second expert raised the question, the answer is already
## in the production code)

When a marker matches multiple times in a response, _parse_assessment
(agent/loop.py) takes the LAST match (`matches[-1]`), not the first - this
is explicitly documented in its docstring (citing an external code-review
finding, claude/status_agent_rezultaty_4_nahodki_kod.md, finding 1) and
confirmed by reading the source code: `_SUFFICIENT_RE`/`_REFORMULATED_RE`
are collected via `list(...finditer(...))`, and the decision is taken from
`[-1]`. The risk that a stray intermediate "SUFFICIENT: yes" inside the
reasoning would be mistaken for the final verdict instead of the real last
line is thereby eliminated at the production-parser level;
`_check_parse_reliability` below is an additional, independent check of
the response's FORM (that the markers arrived as standalone lines, not
merely somewhere inside the text), not a replacement for that safeguard.

## Before re-running - do not mix different versions of the experiment
## (second expert, procedural note)

Resuming by `question_id::variant` (see `_load_checkpoint`) does not
distinguish between different versions of the prompt/fixture/configuration
across runs - if this script changed BETWEEN runs (not in this edit - the
prompt structure has not changed), old rows in
`results/cuad_overrefusal_diagnostic_round3/diagnostic_results.jsonl` will
be silently accepted as "already done" and will not be re-run. For a first
clean run this is not a problem (the directory does not exist yet). If
round 3 ever changes the prompt/logic itself after the first run - delete
or rename the old `diagnostic_results.jsonl` before re-running, rather
than relying on resume.

## Reusing the corpus/index

Unchanged from round 1/round 2: the same already-indexed cuad_smoke_v1,
the same search_fn via agent.tools.search_documents, one retrieval call
per question (reused for both variants - both use the full, unfiltered
context from all candidates, so, unlike round 2's
`single_relevant_document_only`, a single shared retrieval per question is
again sufficient, as in round 1 itself).

## Cost

5 questions x 2 variants = 10 assessor calls, the same order of magnitude
as round 2 (~$0.2).

## Before running for real money

Per the rulebook's rule ("Before running a NEW paid/diagnostic script...")
- show it to TWO independent external experts with the question "is this
script, as written, actually capable of delivering what it claims as its
goal - is there a contradiction between the goal and the mechanism" BEFORE
the first paid run. Do not run it until this has been done.

Usage (Colab, after git pull, after scripts/run_cuad_smoke.py has already
been run at least once on this MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic_round3.py
The run is idempotent (resume by question_id+variant), like rounds 1 and 2.
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

RUN_ID = "cuad_overrefusal_diagnostic_round3"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# Anchors are taken verbatim from the current _ASSESSMENT_PROMPT_TEMPLATE
# (agent/loop.py) - not retyped by hand, for the same reason as in round 2:
# a guaranteed, explicit failure on mismatch, rather than silently drifting
# away from the production prompt.
_DECIDE_ANCHOR = (
    "Decide:\n"
    "1. Is the evidence above sufficient to answer the question completely and precisely? "
    "Judge only from the retrieved passages above - do not use outside knowledge.\n"
    "2. If not sufficient AND you still have calls remaining, propose ONE reformulated search "
    "query that is more likely to find the missing information (e.g. naming a specific line "
    "item, period, or company the current passages are missing). If you have no better query "
    "to try, or no calls remain, say NONE."
)
_DECIDE_WITH_RELEVANCE_STEP = (
    "First, identify which of the retrieved passages above (if any) come from a document that is "
    "actually relevant to the question - a document is relevant only if it concerns the same "
    "entity, agreement, company, period, or subject that the question is asking about. Passages "
    "from a completely unrelated document must be ignored in the steps below.\n\n"
    "Decide, using ONLY the passages you identified as relevant:\n"
    "1. Is the evidence above sufficient to answer the question completely and precisely? "
    "Judge only from the relevant retrieved passages - do not use outside knowledge.\n"
    "2. If not sufficient AND you still have calls remaining, propose ONE reformulated search "
    "query that is more likely to find the missing information (e.g. naming a specific line "
    "item, period, or company the current passages are missing). If you have no better query "
    "to try, or no calls remain, say NONE."
)
_FORMAT_ANCHOR = (
    "Respond in exactly this format, nothing else:\n\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)
_FORMAT_WITH_RELEVANT_DOCS = (
    "Respond in exactly this format, nothing else:\n\n"
    "RELEVANT DOCUMENTS: <comma-separated short identifiers of the relevant document(s), or NONE>\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)

if _DECIDE_ANCHOR not in _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_DECIDE_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the prompt constant "
        "changed since this script was written; update _DECIDE_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round3.py to match before running."
    )
if _FORMAT_ANCHOR not in _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_FORMAT_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the prompt constant "
        "changed since this script was written; update _FORMAT_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round3.py to match before running."
    )

_STRUCTURED_TEMPLATE = _ASSESSMENT_PROMPT_TEMPLATE.replace(
    _DECIDE_ANCHOR, _DECIDE_WITH_RELEVANCE_STEP
).replace(_FORMAT_ANCHOR, _FORMAT_WITH_RELEVANT_DOCS)

if _STRUCTURED_TEMPLATE == _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "_STRUCTURED_TEMPLATE ended up identical to _ASSESSMENT_PROMPT_TEMPLATE after both "
        "replacements - anchors matched but produced no change, which should be impossible; "
        "investigate before running."
    )

# original is re-run in this same session as a within-run control (see the
# docstring above - "Third difference"), rather than taken from round 1.
VARIANTS: dict[str, str] = {
    "original": _ASSESSMENT_PROMPT_TEMPLATE,
    "structured_relevant_first": _STRUCTURED_TEMPLATE,
}

# Carried over unchanged from round 2
# (run_cuad_overrefusal_diagnostic_round2.py) - the same class of risk,
# "a marker mentioned by chance inside the reasoning before the real final
# line", now applied to RELEVANT DOCUMENTS instead of REASONING.
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)
# Diagnostic (not production) extraction of the RELEVANT DOCUMENTS value -
# for logging/analysis only, not used by _parse_assessment and does not
# change it. Last match, same principle as for the other two markers.
_RELEVANT_DOCS_RE = re.compile(r"^\s*RELEVANT DOCUMENTS\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    """Same contract as in round 2: see run_cuad_overrefusal_diagnostic_round2.py
    for the full explanation. Checks only the two markers that actually
    participate in _parse_assessment (agent/loop.py) - SUFFICIENT/REFORMULATED
    QUERY; RELEVANT DOCUMENTS does not affect the production parser, so its
    absence is not treated as a parsing problem (only diagnostic information is lost)."""
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


def _extract_relevant_documents_raw(raw_response: str) -> str | None:
    matches = list(_RELEVANT_DOCS_RE.finditer(raw_response))
    if not matches:
        return None
    value = matches[-1].group(1).strip()
    return value or None


def _verify_relevant_doc_mapping(records, eval_items) -> None:
    """Same belt-and-suspenders guard as in round 2 (see there for the full
    explanation) - used here only to safely build `correct_document_id` for
    each result record (a diagnostic field that helps, when reading
    raw_response, to check what the model wrote in RELEVANT DOCUMENTS against
    the actually correct document); in this round this field is not used to
    filter the context - both variants receive the full, unfiltered context."""
    problems = []
    for rec, item in zip(records, eval_items):
        gold = item["gold_answer"].strip()
        probe = gold if len(gold) <= 40 else gold[:40]
        if probe.upper() not in rec.context.upper():
            problems.append(
                f"question_id={item['question_id']!r}: gold_answer prefix {probe!r} not found in "
                f"document {rec.context_id!r} - correct_document_id would be wrong for this question"
            )
    if problems:
        raise RuntimeError(
            "correct_document_id mapping failed verification:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def _check_already_indexed(collection, records) -> None:
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
    """Runs round 3's `original` vs `structured_relevant_first` ablation against the CUAD smoke fixture.

    Raises:
        RuntimeError: If a question's correct document is not found among
            its retrieved candidates (a retrieval regression, not something
            this diagnostic can answer).
    """
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)

    assert len(records) == len(eval_items), (
        "records and eval_items must be the same length and in the same order - "
        "load_cuad_smoke_fixture() builds them in a single loop over questions"
    )
    _verify_relevant_doc_mapping(records, eval_items)
    correct_document_id_by_question = {
        item["question_id"]: rec.context_id for rec, item in zip(records, eval_items)
    }

    print(
        f"Round 3: {len(eval_items)} question(s) x {len(VARIANTS)} variant(s) = "
        f"{len(eval_items) * len(VARIANTS)} assessor call(s) total."
    )

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, variant) pair(s) already done in {out_path}")
        print(
            "  WARNING: resume does not distinguish between experiment versions - if this "
            "script's prompt/logic changed SINCE at least one of these lines was recorded, "
            "the old results will be silently accepted as done. If this is not the same, "
            "unchanged run - delete or rename diagnostic_results.jsonl before re-running "
            "rather than relying on resume."
        )

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
            correct_document_id = correct_document_id_by_question[question_id]

            # Both variants use the same, full (unfiltered) context - unlike
            # round 2's single_relevant_document_only, a single retrieval
            # call per question is again sufficient here.
            call = search_fn(question)
            candidates = list(call.candidates)
            context_text = build_context_block(candidates)
            context_ids = sorted({c.context_id for c in candidates})

            # pipeline.generation.build_context_block() labels candidates as
            # "[Document i]", i = 1..len(candidates), in the SAME order as
            # candidates here (verified by reading the source code, not
            # assumed) - correct_document_position is the correct document's
            # position in that same numbering, so that relevant_documents_raw
            # (e.g. "Document 2") can be compared with it directly, without
            # consulting the source code. correct_document_id remains the
            # source of truth (context_id); position is a derived field only
            # for the convenience of a human reading the results.
            correct_document_position: int | None = None
            for position, candidate in enumerate(candidates, start=1):
                if candidate.context_id == correct_document_id:
                    correct_document_position = position
                    break
            if correct_document_position is None:
                raise RuntimeError(
                    f"question_id={question_id!r}: correct_document_id {correct_document_id!r} "
                    f"not found among retrieved candidates ({[c.context_id for c in candidates]!r}) - "
                    "retrieval regression, not a prompt question this diagnostic can answer. "
                    "Stopping rather than silently recording a wrong position."
                )

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                calls_remaining = config.agent.max_additional_tool_calls
                prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)
                parse_warning = _check_parse_reliability(raw_response)
                relevant_documents_raw = _extract_relevant_documents_raw(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "correct_document_id": correct_document_id,
                    "correct_document_position": correct_document_position,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "relevant_documents_raw": relevant_documents_raw,
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

    print("\nSummary of sufficient=yes by variant (n=5 per variant, a reference point, not statistics):")
    for variant in VARIANTS:
        rows = [r for r in done.values() if r["variant"] == variant]
        n_yes = sum(1 for r in rows if r["sufficient"])
        print(f"  {variant:28s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "Decision criterion (agreed with the first expert, threshold sharpened by the second "
        "expert into an unambiguous number): structured_relevant_first >= 3 of 5 sufficient=yes -> "
        "the diagnostic signal is strong enough to move on to the n=30 financial regression test "
        "(4-5 of 5 is a stronger signal, the passing threshold is the same; the regression test "
        "itself remains a mandatory next step, this is NOT proof the fix is ready). Below 3 - "
        "hypothesis A in this formulation is refuted at the diagnostic level, a different mechanism "
        "is needed. original is expected to be 0/5 (a within-run control, exactly as in round 1) - "
        "if original unexpectedly gives sufficient=yes, that's a signal of model/environment drift, "
        "not a useful result for hypothesis A, and it needs to be investigated separately before "
        "drawing any conclusions."
    )
    print(
        "Read raw_response in full, not just the tally - especially the relevant_documents_raw "
        "field, and whether it matches each record's correct_document_position (the position in "
        "the same [Document i] numbering the model sees - not the correct_document_id string)."
    )


if __name__ == "__main__":
    main()

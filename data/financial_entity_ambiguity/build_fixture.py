"""Builds a fixture for a narrow financial check of the deictic/entity
hypothesis (see claude/itog_ekspertizy_cuad_overrefusal_fix.md, "What remains
to be done", section 2) - question: how much does the assessor's
`sufficient=yes` drop when the company name is removed from the question
text, while the document-level metadata (metadata_prefix) stays in place,
under a FIXED (not retrieval-dependent) context of gold document + 3
distractors from the same Sector (the very field that actually ends up in
metadata_prefix - see pipeline/ingestion.py's build_metadata_prefix(), which
uses company_sector, not company_industry - the same field is used here so
the distractors are of the same "class" the model sees in the document text,
not a narrower category the model never sees).

Design agreed on by 4 independent experts (rounds 2-4 of the
"prompt_ekspert2/3/4_cuad_fix_dizayn.md" cascade, see
claude/itog_ekspertizy_cuad_overrefusal_fix.md, item 6 of the summary table):
a paired comparison (same question, 2 text conditions) over ~20-30 questions,
context assembled by hand (gold + 3 distractors from the same Sector,
closest report_year), NOT via live retrieval - to isolate the assessor from
retrieval variability. `_ASSESSMENT_PROMPT_TEMPLATE` is not modified at all
- the existing production prompt is tested as-is.

## Question selection (fully verifiable by the code below, not by hand)

Source: data/t2-ragbench/eval_subset_250.parquet (the same file on which
246/250 explicit-company was previously counted - see
claude/nahodka_deiktichnost_round3_dlya_ekspertov.md). A candidate must:

1. Have non-empty company_name AND company_sector (after joining with
   pipeline.ingestion.to_document_records() - the same DocumentRecord
   construction used by production indexing, not a separate parser).
2. Explicitly name its company (humanize_company_name()) in the question
   text - checked with a strict word-boundary regex (`\\b<name>\\b`,
   case-insensitive), not the "first word" heuristic (that heuristic was
   only used for the rough 246/250 scale estimate; here an EXACT, reversible
   replacement is needed - see anonymize_question() below).
3. Its Sector must have >= 4 DISTINCT companies across the ENTIRE corpus
   (7318 documents, not just eval_subset) - otherwise 3 distractors from
   different companies cannot be assembled.

From the selected candidates, chosen by hand (deterministically, without
randomness): 1 representative per sector (candidates within a sector sorted
by company_name, then context_id - first one taken), plus a 2nd
representative for sectors with >= 6 distinct companies in the corpus (gives
more material for distractors and preserves diversity) - the result is
recorded in TARGET_CONTEXT_IDS below, fixed as an explicit list of
context_ids (not recomputed on every script run), so the result is stable
and can be checked line-by-line in code review, rather than reproduced only
"if the sort happens to come out the same way."

## Distractors

3 documents from the same Sector, OTHER companies (never the same company as
gold), one document per company, closest by |report_year - gold_year|
(deterministic tie-break by context_id). If a company has several documents,
the document-level first by context_id is taken (deterministically).

## Question anonymization

EVERY occurrence of the humanized company name (accounting for a trailing
possessive 's / ') is replaced with "the company" / "the company's" - see
anonymize_question(). Each replacement is checked programmatically (after
the replacement, the original name must not appear anywhere in the
anonymized text, in any case) - this does not rely on a visual check of one
replacement as proof that every occurrence was handled.

## What this script does NOT check

It does not check that the diagnostic is physically feasible (that's done by
the main diagnostic script
scripts/run_financial_entity_ambiguity_diagnostic.py -
_check_already_indexed() there). This script only builds the text fixture
(questions, gold/distractor context_ids, anonymized text) - it does not
touch MongoDB, does not run retrieval, and does not spend any money.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "data" / "t2-ragbench"
OUT_PATH = Path(__file__).resolve().parent / "fixture.json"

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from pipeline.ingestion import humanize_company_name, load_raw, to_document_records  # noqa: E402

MIN_DISTINCT_COMPANIES_PER_SECTOR = 4
N_DISTRACTORS = 3

# Explicitly fixed selection (candidate question's context_id = document's
# context_id; question/company/sector match the real data, which is
# separately asserted below in build(), not just assumed from the variable
# name). 28 questions, 20 distinct sectors (see the docstring above for the
# selection principle: 1 per sector + a 2nd for sectors with >= 6 companies
# in the corpus, minus 4 manually dropped "second" representatives -
# Materials, Telecommunications, Communication Services, Shipping - to stay
# within the expert-agreed 20-30 range rather than strictly following the
# formula).
TARGET_CONTEXT_IDS: list[str] = [
    "convfinqa_ctx_379",  # Financials / Aon
    "convfinqa_ctx_893",  # Financials / BlackRock
    "finqa_train_ctx_715",  # Industrials / 3M
    "convfinqa_ctx_1480",  # Industrials / American Airlines Group
    "convfinqa_ctx_1603",  # Information Technology / Analog Devices
    "convfinqa_ctx_1056",  # Information Technology / Cadence Design Systems
    "convfinqa_ctx_1027",  # Consumer Discretionary / Norwegian Cruise Line Holdings
    "convfinqa_ctx_224",  # Real Estate / American Tower
    "convfinqa_ctx_117",  # Real Estate / Duke Realty Corporation
    "convfinqa_ctx_1692",  # Health Care / Becton Dickinson
    "convfinqa_ctx_503",  # Health Care / Edwards Lifesciences
    "finqa_train_ctx_1945",  # Consumer Staples / Altria
    "convfinqa_ctx_450",  # Energy / Devon Energy
    "finqa_dev_ctx_236",  # Energy / Marathon Oil
    "cc27b3da593fc5540f3dd8b30ac82796",  # Semiconductors / microchip-technology-inc
    "249a345bb86b13642427834d02363ff8",  # Semiconductors / micron-technology-inc
    "finqa_train_ctx_1959",  # Materials / Air Products
    "0075ff4f22a8946c2aa35d3813de5c86",  # Software / altium-limited
    "finqa_train_ctx_1112",  # Communication Services / Comcast
    "61965dde672ac3c01293e4e66e6bdb95",  # Shipping / navios-maritime-holdings-inc
    "convfinqa_ctx_484",  # Utilities / American Water Works
    "13f24145ffd39eb820e2c7eba93092aa",  # Cybersecurity / mimecast-limited
    "fe9f2f028e1ac48619d22d8ecaf45edf",  # Technology / microsoft-corporation
    "8ff6ca6ee109976b42543aad81a0326e",  # Financial Technology / black-knight-financial-services-inc
    "2c329676bed992ced5b827d9f051a0ec",  # Electronics Manufacturing / jabil-circuit-inc
    "0cf6769516cf2a245aaed5fcf2bd9c21",  # Food Production / lifeway-foods-inc
    "42dfd9e9db0d11bd550deaa12735fd72",  # Telecommunications / cogeco-inc (kept as sole rep)
    "convfinqa_ctx_1215",  # Materials / Ball Corporation (kept as sole 2nd Materials rep - see note)
]


def _strict_name_in_text(name: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE) is not None


# Corporate suffixes that sometimes follow the humanized name in the
# original question text (e.g. humanize_company_name() returns "American
# Airlines Group", but the question calls it "American Airlines Group
# Inc.") - without this, the suffix would stay stuck to the replacement
# ("the company Inc."), which doesn't distort deicticity (by itself "Inc."
# doesn't identify the company), but reads as an obvious replacement
# artifact to a human reading the question - so it's absorbed together with
# the name into a single match.
_CORP_SUFFIX = r"(?:\s+(?:Inc|Incorporated|Corp(?:oration)?|Co|Company|Ltd|Limited|LLC|plc))?"


def anonymize_question(name: str, question: str) -> str:
    """Replaces every occurrence of `name` (accounting for an optional
    corporate suffix right after the name and a possessive form 's/'/'s)
    with "the company"/"the company's". Case-insensitive, all occurrences
    at once (re.sub without count=1) - see the module docstring on how
    every replacement is checked programmatically after being applied (the
    original name must not remain in the anonymized text), rather than
    simply assumed to hold from the pattern's construction.

    Args:
        name: Humanized company name (see humanize_company_name()) to find
            and replace in the question text.
        question: The original question text in which the replacement is
            performed.

    Returns:
        The question text with every occurrence of `name` (together with
        the optional corporate suffix and possessive form) replaced with
        "the company"/"the company's".
    """
    pattern = re.compile(
        r"\b" + re.escape(name) + _CORP_SUFFIX + r"\.?(’s|'s|[’'])?",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        replacement = "the company's" if m.group(1) else "the company"
        if m.start() == 0:
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return pattern.sub(repl, question)


def build() -> None:
    """Builds the fixture (one pair of conditions per TARGET_CONTEXT_IDS
    entry: question/gold document/3 distractors/anonymized text) and writes
    it to OUT_PATH - see the module docstring for the full selection design
    and the origin of TARGET_CONTEXT_IDS.

    Raises:
        ValueError: If a TARGET_CONTEXT_IDS entry is not found in
            eval_subset_250.parquet or among the constructed
            DocumentRecords, has no company_name/company_sector, its
            sector has fewer than MIN_DISTINCT_COMPANIES_PER_SECTOR
            distinct companies in the corpus, or anonymization did not
            remove all occurrences of the company name / did not change
            the question text.
        AssertionError: If the built fixture has duplicate question_ids,
            or gold_context_ids reused across different entries.
    """
    raw = load_raw(CORPUS_DIR)
    records = to_document_records(raw)

    docs_by_id: dict[str, object] = {}
    for r in records:
        docs_by_id.setdefault(r.context_id, r)
    print(f"Full corpus: {len(docs_by_id)} unique documents.")

    q_index: dict[tuple[str, str], object] = {}
    for r in records:
        q_index.setdefault((r.context_id, r.question.strip()), r)

    eval_df = pd.read_parquet(CORPUS_DIR / "eval_subset_250.parquet")

    sector_companies: dict[str, set[str]] = defaultdict(set)
    sector_docs_by_company: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in docs_by_id.values():
        if r.company_sector and r.company_name:
            sector_companies[r.company_sector].add(r.company_name)
            sector_docs_by_company[r.company_sector][r.company_name].append(r)

    items: list[dict] = []
    for context_id in TARGET_CONTEXT_IDS:
        # Find the question from eval_subset_250 that actually points to this context_id
        eval_rows = eval_df[eval_df["context_id"] == context_id]
        if eval_rows.empty:
            raise ValueError(f"context_id={context_id!r} not found in eval_subset_250.parquet")
        row = eval_rows.iloc[0]
        key = (context_id, row["question"].strip())
        rec = q_index.get(key)
        if rec is None:
            raise ValueError(f"context_id={context_id!r}: question not found among DocumentRecords (text mismatch?)")
        if not rec.company_name or not rec.company_sector:
            raise ValueError(f"context_id={context_id!r}: company_name/company_sector missing")

        humanized = humanize_company_name(rec.company_name)
        if not humanized or not _strict_name_in_text(humanized, row["question"]):
            raise ValueError(
                f"context_id={context_id!r}: company name {humanized!r} not found in question text "
                f"{row['question']!r} - selection criteria violated"
            )

        n_companies_in_sector = len(sector_companies[rec.company_sector])
        if n_companies_in_sector < MIN_DISTINCT_COMPANIES_PER_SECTOR:
            raise ValueError(
                f"context_id={context_id!r}: sector {rec.company_sector!r} has only "
                f"{n_companies_in_sector} distinct companies in the corpus, need >= "
                f"{MIN_DISTINCT_COMPANIES_PER_SECTOR}"
            )

        gold_year = rec.report_year
        gold_year_int = int(gold_year) if gold_year and str(gold_year).isdigit() else None

        distractor_candidates = []
        for company, docs_list in sector_docs_by_company[rec.company_sector].items():
            if company == rec.company_name:
                continue
            docs_list_sorted = sorted(docs_list, key=lambda d: d.context_id)
            chosen = docs_list_sorted[0]
            year_int = int(chosen.report_year) if chosen.report_year and str(chosen.report_year).isdigit() else None
            year_diff = abs(year_int - gold_year_int) if (year_int is not None and gold_year_int is not None) else 999
            distractor_candidates.append((year_diff, chosen.context_id, chosen))
        distractor_candidates.sort(key=lambda t: (t[0], t[1]))
        if len(distractor_candidates) < N_DISTRACTORS:
            raise ValueError(
                f"context_id={context_id!r}: only {len(distractor_candidates)} potential "
                f"distractors in sector {rec.company_sector!r}, need {N_DISTRACTORS}"
            )
        distractors = [t[2] for t in distractor_candidates[:N_DISTRACTORS]]

        anonymized_question = anonymize_question(humanized, row["question"])
        if _strict_name_in_text(humanized, anonymized_question):
            raise ValueError(
                f"context_id={context_id!r}: anonymization did not remove all occurrences of {humanized!r} "
                f"from the question - remaining: {anonymized_question!r}"
            )
        if anonymized_question == row["question"]:
            raise ValueError(f"context_id={context_id!r}: anonymization did not change the question text")

        items.append(
            {
                "question_id": f"fin_entity_ambiguity_{row['id']}",
                "source_dataset": rec.source_dataset,
                "question_original": row["question"],
                "question_anonymized": anonymized_question,
                "gold_answer": row["program_answer"],
                "gold_context_id": context_id,
                "gold_company_name": rec.company_name,
                "gold_company_name_humanized": humanized,
                "gold_sector": rec.company_sector,
                "gold_report_year": rec.report_year,
                "distractor_context_ids": [d.context_id for d in distractors],
                "distractor_company_names": [d.company_name for d in distractors],
                "distractor_report_years": [d.report_year for d in distractors],
            }
        )

    # Check uniqueness of question_id and that the same context_id is not
    # reused as gold in two different entries (otherwise the result would
    # not honestly be "N independent pairs").
    qids = [it["question_id"] for it in items]
    assert len(qids) == len(set(qids)), "Duplicate question_id in fixture"
    gold_ids = [it["gold_context_id"] for it in items]
    assert len(gold_ids) == len(set(gold_ids)), "Duplicate gold_context_id in fixture"

    print(f"\nBuilt {len(items)} question pairs (2 conditions each = {len(items) * 2} assessor calls).")
    print(f"Sectors: {len(set(it['gold_sector'] for it in items))}")
    print("\n--- Manual anonymization check (all rows) ---")
    for it in items:
        print(f"[{it['question_id']}] {it['gold_company_name_humanized']!r}")
        print(f"  ORIG: {it['question_original']}")
        print(f"  ANON: {it['question_anonymized']}")
        print(f"  distractors: {it['distractor_company_names']}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    build()

"""Builds the out-of-corpus probe question set for plan_adversarial_robustness_priorities.md
Priority 3 ("unanswerable/out-of-corpus"), motivated by:
  claude/plan_adversarial_robustness_priorities.md, section on Priority 3:
  "Дёшево тестируется: сконструировать вопросы про периоды/темы,
  отсутствующие в корпусе, и проверить, отказывается ли система отвечать
  или генерирует правдоподобное, но необоснованное число."

Every question here is checked programmatically against the actual indexed
corpus metadata (data/t2-ragbench/*.parquet), not guessed - see
verify_absence() below, run as part of this script, not a separate manual
claim. Two conditions, both targeting the same behavior (does the pipeline
say "insufficient context" instead of fabricating a number), via two
different, independently verifiable routes to guaranteed absence:

  wrong_year      - a company that IS in the corpus, but for a fiscal year
                     that is NOT (confirmed against file_name-embedded years
                     for FinQA/ConvFinQA, or against report_year for
                     TAT-DQA, which is 100% year=2019 - any other year is
                     guaranteed absent).
  absent_company  - a real, well-known public company that does not appear
                     under any of the three source datasets at all
                     (confirmed by substring-checking company_name across
                     all three - see verify_absence()).

Output: data/unanswerable_probe/questions.jsonl - one question per line,
with the fields needed to run it through the existing pipeline and to
audit why it's expected to be unanswerable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "data" / "t2-ragbench"
OUT_PATH = Path(__file__).resolve().parent / "questions.jsonl"

WRONG_YEAR_QUESTIONS = [
    # (ticker/company_name key, corpus years present, probe year, question)
    ("WMT", "Walmart", [2018], 2023, "What was Walmart's total revenue for fiscal year 2023?"),
    ("MRK", "Merck & Co.", [2013], 2022, "What was Merck & Co.'s total research and development expense for fiscal year 2022?"),
    ("D", "Dominion Energy", [2002], 2020, "What was Dominion Energy's net income for fiscal year 2020?"),
    ("REGN", "Regeneron", [2010], 2021, "What was the percentage change in Regeneron's research and development expense in fiscal year 2021?"),
    ("CAT", "Caterpillar Inc.", [2017, 2018], 2023, "What was Caterpillar Inc.'s total revenue for fiscal year 2023?"),
    ("LLY", "Lilly (Eli)", [2008, 2018], 2023, "What was Eli Lilly's net income for fiscal year 2023?"),
    ("EMR", "Emerson Electric", [2017, 2018], 2022, "What was the total value of common share buybacks by Emerson Electric in fiscal year 2022?"),
    ("FBHS", "Fortune Brands Home & Security", [2017], 2023, "What was Fortune Brands Home & Security's operating margin for fiscal year 2023?"),
    ("FTV", "Fortive", [2017], 2022, "What was the percentage change in Fortive's sales from fiscal year 2021 to fiscal year 2022?"),
    ("EXR", "Extra Space Storage", [2005], 2023, "What was the dividend yield for Extra Space Storage for the quarter ended March 31, 2023?"),
    ("NTAP", "NetApp", [2014], 2023, "What was NetApp's cumulative total shareholder return reported in fiscal year 2023?"),
    ("ABC", "AmerisourceBergen Corporation", [2005], 2022, "What was the change in total debt for AmerisourceBergen Corporation in fiscal year 2022?"),
    ("WRK", "WestRock Company", [2018, 2019], 2023, "What percentage of WestRock Company's total long-term debt was current, as of fiscal year 2023?"),
    ("BKR", "Baker Hughes", [2017, 2018], 2023, "What percentage of the WTI oil price in fiscal year 2023 did the natural gas price represent, per Baker Hughes' reporting?"),
    ("TSCO", "Tractor Supply", [2017, 2018], 2023, "What was the total amount of bond authorization not utilized by Tractor Supply, as reflected in its fiscal year 2023 report?"),
    # TAT-DQA: every single row in the corpus is report_year=2019 - any other year is guaranteed absent.
    ("accenture-plc", "Accenture plc", [2019], 2023, "What was Accenture plc's total revenue for fiscal year 2023?"),
    ("woolworths-limited", "Woolworths Limited", [2019], 2023, "What was Woolworths Limited's net profit for fiscal year 2023?"),
    ("activision-blizzard-inc", "Activision Blizzard", [2019], 2022, "What was Activision Blizzard's total operating expense for fiscal year 2022?"),
    ("adobe-systems-inc", "Adobe Systems", [2019], 2023, "What was Adobe Systems' subscription revenue for fiscal year 2023?"),
    ("intu-properties", "Intu Properties", [2019], 2022, "What was Intu Properties' total property valuation for fiscal year 2022?"),
]

# Original candidate pool included several names (Alphabet, Meta, Tesla,
# Nvidia, Home Depot, Coca-Cola, ...) that turned out to be mentioned in
# the corpus's full indexed text (as peer-group/competitor names inside
# OTHER companies' filings) even though absent from the company_name
# metadata column - caught by an independent review before this ever ran
# against real paid APIs (see verify_absence() below, which now checks
# full text, not just company_name). Replaced with 12 names confirmed by
# verify_absence() to have zero occurrences anywhere in pre_text/post_text/
# table/context across all three source datasets.
ABSENT_COMPANY_QUESTIONS = [
    ("Amazon", "What was Amazon's total net sales for fiscal year 2022?"),
    ("Netflix", "What was Netflix's total streaming revenue for fiscal year 2022?"),
    ("Berkshire Hathaway", "What was Berkshire Hathaway's total investment income for fiscal year 2022?"),
    ("Costco", "What was Costco's total membership fee revenue for fiscal year 2022?"),
    ("UnitedHealth Group", "What was UnitedHealth Group's total premium revenue for fiscal year 2022?"),
    ("ExxonMobil", "What was ExxonMobil's total capital expenditure for fiscal year 2022?"),
    ("Airbnb", "What was Airbnb's total revenue for fiscal year 2022?"),
    ("Spotify", "What was Spotify's total premium subscriber revenue for fiscal year 2022?"),
    ("Salesforce", "What was Salesforce's total subscription and support revenue for fiscal year 2022?"),
    ("ServiceNow", "What was ServiceNow's total subscription revenue for fiscal year 2022?"),
    ("Snowflake", "What was Snowflake's total product revenue for fiscal year 2023?"),
    ("DoorDash", "What was DoorDash's total order volume revenue for fiscal year 2022?"),
]


def verify_absence() -> None:
    """Loads the real, committed corpus parquet files and checks every
    claim above against actual data - not asserted, checked. Raises
    AssertionError (loudly, naming the offending question) if any
    "wrong_year" company/year pair is actually present, or any
    "absent_company" name actually matches something in the corpus.
    """
    fin = pd.concat([pd.read_parquet(CORPUS_DIR / f"FinQA_{s}.parquet") for s in ("train", "dev", "test")])
    conv = pd.read_parquet(CORPUS_DIR / "ConvFinQA_turn_0.parquet")
    tat = pd.concat([pd.read_parquet(CORPUS_DIR / f"TAT-DQA_{s}.parquet") for s in ("train", "dev", "test")])

    def ticker_year(fn: str) -> tuple[str | None, int | None]:
        m = re.match(r"pdf/([A-Za-z.]+)/(\d{4})/", str(fn))
        return (m.group(1), int(m.group(2))) if m else (None, None)

    fin_conv = pd.concat([fin[["file_name"]], conv[["file_name"]]])
    fin_conv[["ticker", "year"]] = fin_conv["file_name"].apply(lambda x: pd.Series(ticker_year(x)))
    fin_conv_years: dict[str, set[int]] = fin_conv.groupby("ticker")["year"].apply(set).to_dict()

    tat_years: dict[str, set[int]] = (
        tat.assign(company_key=tat["company_name"].str.lower())
        .groupby("company_key")["report_year"]
        .apply(lambda s: {int(y) for y in s.dropna()})
        .to_dict()
    )

    for key, name, corpus_years, probe_year, _q in WRONG_YEAR_QUESTIONS:
        if key.lower() in tat_years:
            present = tat_years[key.lower()]
        else:
            present = fin_conv_years.get(key, set())
        assert present, f"{name} ({key}): expected to find it in the corpus at all, found nothing"
        assert set(corpus_years) == present, f"{name} ({key}): expected years {corpus_years}, corpus actually has {present}"
        assert probe_year not in present, f"{name} ({key}): probe year {probe_year} is NOT absent - corpus has {present}"

    all_names = (
        set(fin["company_name"].dropna().str.lower())
        | set(conv["company_name"].dropna().str.lower())
        | set(tat["company_name"].dropna().str.lower().str.replace("-", " "))
    )

    # Checking company_name alone is NOT enough - a company absent from
    # that metadata column can still appear inside another company's
    # filing text (peer-group lists, competitor mentions, supplier
    # relationships). Caught by independent review before this ran
    # against real paid APIs: the original candidate list included
    # several names (Alphabet, Meta, Tesla, Nvidia, Home Depot, Coca-Cola)
    # that were absent from company_name but present dozens of times in
    # full text. So also full-text-search every candidate's first word
    # (cheap proxy for the whole name, catches "Meta Platforms" via
    # "meta" etc.) across pre_text/post_text/table/context - anywhere in
    # the indexed content, not just the metadata field a retrieval query
    # would never see directly.
    def full_text_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        return df[present].astype(str).agg(" ".join, axis=1).str.lower()

    all_text = pd.concat([
        full_text_series(fin, ["pre_text", "post_text", "table", "context"]),
        full_text_series(conv, ["pre_text", "post_text", "table", "context"]),
        full_text_series(tat, ["context"]),
    ])

    for name, _q in ABSENT_COMPANY_QUESTIONS:
        hit = [n for n in all_names if name.lower() in n or n in name.lower()]
        assert not hit, f"{name}: expected fully absent, but found matching company_name entries: {hit}"
        first_word = name.lower().split()[0]
        text_hits = int(all_text.str.contains(first_word, regex=False).sum())
        assert text_hits == 0, (
            f"{name}: company_name is clean, but '{first_word}' appears {text_hits} time(s) "
            f"in indexed full text (pre_text/post_text/table/context) - not actually absent, "
            f"retrieval could surface a real mention."
        )

    print(f"Проверено: все {len(WRONG_YEAR_QUESTIONS)} пар компания/год и все "
          f"{len(ABSENT_COMPANY_QUESTIONS)} компаний подтверждены отсутствующими "
          f"в реальных данных корпуса.")


def build() -> None:
    verify_absence()
    records = []
    qid = 1
    for key, name, corpus_years, probe_year, question in WRONG_YEAR_QUESTIONS:
        records.append({
            "probe_id": f"unanswerable_{qid:03d}",
            "category": "wrong_year",
            "company": name,
            "corpus_years_present": corpus_years,
            "probe_year": probe_year,
            "question": question,
            "expected_behavior": "insufficient_context",
        })
        qid += 1
    for name, question in ABSENT_COMPANY_QUESTIONS:
        records.append({
            "probe_id": f"unanswerable_{qid:03d}",
            "category": "absent_company",
            "company": name,
            "corpus_years_present": [],
            "probe_year": None,
            "question": question,
            "expected_behavior": "insufficient_context",
        })
        qid += 1

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Записано {len(records)} вопросов в {OUT_PATH}")


if __name__ == "__main__":
    build()

"""Regression tests reproducing the concrete `is_close_v2` disagreement
cases documented in docs/tehnicheskoe_zadanie.md, section 14 ("Отдельная
находка: несогласия LLM-судьи и детерминированной проверки").

Why this file exists (plan item 1, "Финальный план доработки проекта после
экспертизы.docx"): section 14 reports that on the n=250 error-analysis run,
`deterministic_match` (`is_close_v2`) and `judge_correct` disagreed on 15 of
250 questions (6%), and names two known, *intentionally unfixed* classes of
inaccuracy in `is_close_v2`:

1. It does not recognize units/textual formats that are semantically
   equivalent to the gold answer (thousands/millions/raw-number scale
   mismatches, textual year formats, boolean "No" for "0") - 10 cases where
   the judge accepted an answer `is_close_v2` rejected.
2. It sometimes over-eagerly accepts a x100 (fraction-vs-percentage) or
   near-tolerance match that the judge considered substantively wrong - 5
   cases where `is_close_v2` accepted an answer the judge rejected.

Both are explicitly framed in section 14 as *known limitations to document,
not to fix now* ("не блокирует текущие headline-метрики ... должно быть
учтено при следующей ревизии is_close_v2, если она случится"). These tests
therefore assert the *current, documented* behavior - they exist so the
0.760 headline accuracy and the section 14 error taxonomy are reproducible
and falsifiable by someone else reading the repo, not just trusted on
faith. If `is_close_v2` is ever revised to close these gaps, the
corresponding assertions below should flip from `is False` to `is True`
(or vice versa) as part of that change, not be deleted.

All (question_id, gold_answer, answer_text) values below were pulled
directly from results/error_analysis_250/predictions.jsonl and
results/error_analysis_250/eval_results.jsonl (the committed n=250 run
referenced throughout section 14) by filtering for
`deterministic_match != judge_scores.judge_correct` - not retyped from the
prose summary, to avoid transcription drift from the real run data. The
prose in section 14 names 9 of the 10 "judge softer" question_ids
explicitly by (gold, answer) pair; `tatqa_train_2447` is the 10th case in
that bucket (count confirmed as 10 in the doc) and is included here for
completeness even though it isn't spelled out with values in the prose.

Call convention matches pipeline/evaluation.py's actual usage:
`is_close_v2(generated_answer, gold_answer)`, i.e. `a` = model output,
`b` = gold.
"""

from __future__ import annotations

import pytest

from pipeline.common.is_close_v2 import is_close_v2


# --- "Судья мягче" (judge softer than is_close_v2): 10 cases ---------------
# is_close_v2 -> False (rejects), judge -> True (accepts). These are the
# documented cases where is_close_v2 fails to recognize a semantically
# equivalent answer given in a different unit/format.

SCALE_MISMATCH_CASES = [
    # question_id, gold_answer, answer_text, note
    ("convfinqa_2441", "1832000.0", "1832", "raw value given in thousands (x1e3)"),
    ("tatqa_train_6808", "14100000.0", "14.1", "raw value given in millions (x1e6)"),
    ("tatqa_train_8052", "3550.0", "3550000000", "gold in millions, answer raw (x1e6... reported as x1e9 scale gap)"),
    ("tatqa_train_6526", "44458.0", "44458000", "gold in thousands, answer raw (x1e3)"),
    ("tatqa_train_8931", "181401.0", "181401000", "gold in thousands, answer raw (x1e3)"),
    ("tatqa_train_2447", "17.2", "-17198", "sign flip plus ~x1e3 scale gap; 10th 'judge softer' case, not spelled out in the section 14 prose"),
]

TEXTUAL_UNIT_CASES = [
    ("tatqa_train_8556", "2100000.0", "2.1 million", 'textual unit suffix ("million") instead of a bare number'),
]

TEXTUAL_YEAR_CASES = [
    ("tatqa_train_5894", "2019.0", "Year-ended 31 March 2019", "gold is a bare year, answer is a full fiscal-period phrase"),
    ("tatqa_train_8367", "2019.0", "Year-ended 31 March 2019", "same textual-year pattern as tatqa_train_5894"),
]

BOOLEAN_CASES = [
    ("finqa_dev_656", "0", "No", 'gold is numeric "0", answer is boolean "No"'),
]

JUDGE_SOFTER_CASES = (
    SCALE_MISMATCH_CASES + TEXTUAL_UNIT_CASES + TEXTUAL_YEAR_CASES + BOOLEAN_CASES
)


@pytest.mark.parametrize(
    "question_id, gold_answer, answer_text, note", JUDGE_SOFTER_CASES
)
def test_judge_softer_cases_not_recognized_by_is_close_v2(
    question_id, gold_answer, answer_text, note
):
    """Section 14: 10 real n=250 questions where the judge accepted the
    answer but is_close_v2 did not, because it doesn't normalize units,
    textual formats, or booleans - documented as a known, unfixed gap.
    """
    assert is_close_v2(answer_text, gold_answer) is False, (
        f"{question_id}: expected is_close_v2 to (still) reject this "
        f"known-unrecognized-format case ({note}); if this now passes, "
        f"is_close_v2 has been changed and this test (and section 14's "
        f"discussion of the gap) needs to be updated together."
    )


def test_judge_softer_case_count_matches_section_14():
    """Section 14 states this bucket has exactly 10 members ("10 из 15 -
    судья мягче"); guards against silently dropping/adding a case."""
    assert len(JUDGE_SOFTER_CASES) == 10


# --- "Судья строже" (judge stricter than is_close_v2): 5 cases -------------
# is_close_v2 -> True (accepts), judge -> False (rejects). Section 14 flags
# 2 of these as the specific x100 (fraction-vs-percentage) over-match
# pattern ("×100-путаница"), and the remaining 3 as near-tolerance matches
# (within the default 1% relative tolerance) that the judge nonetheless
# considered wrong on substance (different period/component selected).

X100_OVERMATCH_CASES = [
    (
        "finqa_dev_451",
        "1.424025457438345",
        "142.4",
        "answer accepted only via the b*100 candidate; judge disagreed",
    ),
    (
        "finqa_train_2226",
        "0.005470916481712618",
        "0.55",
        "answer accepted only via the b*100 candidate; judge disagreed",
    ),
]

NEAR_TOLERANCE_OVERMATCH_CASES = [
    ("finqa_dev_569", "193.0", "192", "within 1% relative tolerance directly, no scale flip needed"),
    ("tatqa_train_8832", "119655.67", "118989.67", "within 1% relative tolerance directly, no scale flip needed"),
    ("tatqa_train_1740", "50158.67", "50226.67", "within 1% relative tolerance directly, no scale flip needed"),
]

JUDGE_STRICTER_CASES = X100_OVERMATCH_CASES + NEAR_TOLERANCE_OVERMATCH_CASES


@pytest.mark.parametrize(
    "question_id, gold_answer, answer_text, note", JUDGE_STRICTER_CASES
)
def test_judge_stricter_cases_accepted_by_is_close_v2(
    question_id, gold_answer, answer_text, note
):
    """Section 14: 5 real n=250 questions where is_close_v2 accepted the
    answer (directly, or via its x100 scale-flip candidates) but the judge
    considered it substantively wrong - documented as a known
    over-acceptance risk in is_close_v2's tolerance/scale-flip logic.
    """
    assert is_close_v2(answer_text, gold_answer) is True, (
        f"{question_id}: expected is_close_v2 to (still) accept this "
        f"known-overmatch case ({note}); if this now fails, is_close_v2 "
        f"has been changed and this test (and section 14's discussion of "
        f"the false-positive risk) needs to be updated together."
    )


def test_judge_stricter_case_count_matches_section_14():
    """Section 14 states this bucket has exactly 5 members ("5 из 15 -
    судья строже")."""
    assert len(JUDGE_STRICTER_CASES) == 5


def test_total_disagreement_count_matches_section_14():
    """Section 14: "15 из 250 вопросов (6%)" had deterministic_match !=
    judge_correct on the committed error_analysis_250 run."""
    assert len(JUDGE_SOFTER_CASES) + len(JUDGE_STRICTER_CASES) == 15

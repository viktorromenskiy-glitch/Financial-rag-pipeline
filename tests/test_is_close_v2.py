"""Unit tests for pipeline.common.is_close_v2, with concrete examples per
docs/tehnicheskoe_zadanie.md, section 7 ("100" vs "100.0" vs "1.00" as a
fraction, "0.5" vs "50%", etc.) - this file exists specifically so the
0.733 checkpoint (docs/specifikatsiya_moduley.md, module 8) is
reproducible by someone else reading the repo, not just trusted on faith.
"""

from pipeline.common.is_close_v2 import is_close_v2


def test_identical_numbers():
    assert is_close_v2(100, 100)


def test_string_vs_int_same_value():
    # "100" vs "100.0" - same numeric value, different string formatting.
    assert is_close_v2("100", "100.0")


def test_string_precision_variants():
    # "1.00" as a fraction of "100" (percentage form) - x100 scale flip.
    assert is_close_v2("1.00", "100")


def test_fraction_vs_percentage():
    # 0.5 as a fraction vs 50 as the equivalent percentage number.
    assert is_close_v2(0.5, 50)
    assert is_close_v2("0.5", "50")


def test_percent_sign_string_not_stripped():
    # Documented limitation: no currency/percent-sign string normalization
    # in the transplanted logic - float("50%") raises ValueError, so the
    # comparison correctly (per the real Colab behavior) returns False.
    assert is_close_v2("0.5", "50%") is False


def test_sign_invariance():
    # -60 vs 60 - sign-convention ambiguity mentioned in the judge prompt.
    assert is_close_v2(-60, 60)
    assert is_close_v2(60, -60)


def test_percentage_vs_fraction_from_judge_prompt_example():
    # Explicit example from JUDGE_PROMPT in the original notebook: 1.5 vs 0.015.
    assert is_close_v2(1.5, 0.015)


def test_zero_absolute_tolerance():
    assert is_close_v2(0, 0)
    assert is_close_v2(0.005, 0)  # within absolute tol=0.01
    assert is_close_v2(0.02, 0) is False  # outside absolute tol=0.01


def test_none_inputs():
    assert is_close_v2(None, 100) is False
    assert is_close_v2(100, None) is False
    assert is_close_v2(None, None) is False


def test_non_numeric_string():
    assert is_close_v2("not a number", 100) is False
    assert is_close_v2(100, "not a number") is False


def test_outside_tolerance():
    assert is_close_v2(105, 100) is False  # 5% off, tol=1%
    assert is_close_v2(100.5, 100)  # 0.5% off, within tol=1%


def test_custom_tolerance():
    assert is_close_v2(105, 100, tol=0.10)  # 5% off, within 10% tolerance
    assert is_close_v2(105, 100, tol=0.01) is False

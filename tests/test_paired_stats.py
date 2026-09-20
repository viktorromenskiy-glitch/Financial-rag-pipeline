"""Tests for pipeline.common.paired_stats - the exact McNemar + effect-size
CI helper for Day 2's baseline-vs-agent comparison. See that module's
docstring for why this specific CI method was chosen (no expert review
specified one) and for the exact-equivalence claim this file verifies
independently (McNemar exact == a two-sided exact binomial test on the
discordant pairs) rather than just asserting it in a comment.
"""

from __future__ import annotations

import pytest
from scipy.stats import binomtest

from pipeline.common.paired_stats import PairedComparisonResult, compare_paired_binary_outcomes


def test_rejects_negative_cell_counts():
    with pytest.raises(ValueError):
        compare_paired_binary_outcomes(-1, 0, 0, 5)


def test_rejects_all_zero_cell_counts():
    with pytest.raises(ValueError):
        compare_paired_binary_outcomes(0, 0, 0, 0)


def test_rejects_invalid_confidence_level():
    with pytest.raises(ValueError):
        compare_paired_binary_outcomes(10, 2, 2, 6, confidence_level=1.0)
    with pytest.raises(ValueError):
        compare_paired_binary_outcomes(10, 2, 2, 6, confidence_level=0.0)


def test_n_and_marginal_proportions():
    # n=20, a correct on 14 (both_correct=10 + a_only=4), b correct on 12
    # (both_correct=10 + b_only=2), both_wrong=4.
    result = compare_paired_binary_outcomes(both_correct=10, a_only=4, b_only=2, both_wrong=4)
    assert result.n == 20
    assert result.p_a == pytest.approx(14 / 20)
    assert result.p_b == pytest.approx(12 / 20)
    assert result.difference == pytest.approx(2 / 20)
    assert result.n_discordant == 6


def test_symmetric_discordant_pairs_give_zero_difference_and_a_straddling_ci():
    result = compare_paired_binary_outcomes(both_correct=10, a_only=3, b_only=3, both_wrong=4)
    assert result.difference == 0.0
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.mcnemar_exact_pvalue == pytest.approx(1.0)


def test_zero_discordant_pairs_is_a_degenerate_but_valid_result():
    result = compare_paired_binary_outcomes(both_correct=10, a_only=0, b_only=0, both_wrong=5)
    assert isinstance(result, PairedComparisonResult)
    assert result.n_discordant == 0
    assert result.difference == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.mcnemar_exact_pvalue == pytest.approx(1.0)


def test_ci_bounds_straddle_the_point_difference():
    result = compare_paired_binary_outcomes(both_correct=10, a_only=5, b_only=1, both_wrong=4)
    assert result.ci_low <= result.difference <= result.ci_high


def test_wider_confidence_level_gives_a_wider_interval():
    narrow = compare_paired_binary_outcomes(both_correct=10, a_only=5, b_only=1, both_wrong=4, confidence_level=0.80)
    wide = compare_paired_binary_outcomes(both_correct=10, a_only=5, b_only=1, both_wrong=4, confidence_level=0.99)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


@pytest.mark.parametrize(
    "both_correct,a_only,b_only,both_wrong",
    [
        (10, 4, 2, 4),
        (0, 5, 1, 0),
        (13, 1, 5, 6),  # Kimi's followup example - "minimum compatible discordant=6"
        (14, 0, 4, 7),  # Kimi's followup - the true minimum-discordant configuration (discordant=4)
        (5, 8, 8, 5),
    ],
)
def test_mcnemar_exact_pvalue_matches_independent_scipy_binomtest_on_discordant_pairs(
    both_correct, a_only, b_only, both_wrong
):
    # Independent verification (not just documentation) of this module's
    # central claim: exact McNemar is a two-sided exact binomial test on
    # the discordant pairs - computed here via scipy directly, without
    # going through pipeline.common.paired_stats' own binomtest call for
    # the CI, so this is a genuine cross-check against a second
    # computation, not a tautology.
    result = compare_paired_binary_outcomes(both_correct, a_only, b_only, both_wrong)
    n_discordant = a_only + b_only
    independent_pvalue = binomtest(a_only, n_discordant, 0.5).pvalue
    assert result.mcnemar_exact_pvalue == pytest.approx(independent_pvalue)


def test_kimi_followup_reference_values_discordant_6_and_4():
    # kimi_followup_mcnemar_arithmetic_response.md, independently
    # reproduced and cross-checked in that same document: discordant=6
    # (b=1, c=5) -> p=0.2188; discordant=4 (b=0, c=4) -> p=0.125 (the true
    # minimum-discordant "most favorable to the agent" configuration,
    # which the response found Kimi itself had missed in its first pass -
    # exactly the kind of number this project's own house rules require
    # re-verifying independently rather than trusting on repetition).
    result_6 = compare_paired_binary_outcomes(both_correct=13, a_only=1, b_only=5, both_wrong=6)
    assert result_6.mcnemar_exact_pvalue == pytest.approx(0.2188, abs=0.0001)

    result_4 = compare_paired_binary_outcomes(both_correct=14, a_only=0, b_only=4, both_wrong=7)
    assert result_4.mcnemar_exact_pvalue == pytest.approx(0.125, abs=0.0001)


def test_all_discordant_pairs_favor_a_gives_a_one_sided_ci_capped_at_the_point_difference():
    # b_only=0: agent never loses a discordant pair. q's exact
    # Clopper-Pearson upper bound when a_only == n_discordant is exactly
    # 1.0 (x=n is the boundary case), which maps to ci_high == difference
    # exactly (2*1 - 1 = 1) - the CI cannot extend past the observed
    # point estimate on this side, only below it.
    result = compare_paired_binary_outcomes(both_correct=10, a_only=6, b_only=0, both_wrong=4)
    assert result.ci_low > 0
    assert result.ci_high == pytest.approx(result.difference)


def test_result_fields_are_all_present_and_correctly_typed():
    result = compare_paired_binary_outcomes(both_correct=10, a_only=4, b_only=2, both_wrong=4)
    assert result.both_correct == 10
    assert result.a_only == 4
    assert result.b_only == 2
    assert result.both_wrong == 4
    assert result.confidence_level == 0.95
    assert isinstance(result.mcnemar_exact_pvalue, float)

"""Paired binary-outcome statistics for comparing two systems (agent vs.
baseline) on the same matched questions - Day 2 of
plan_rabot_posle_ekspertizy_agent_profil.md: "with low power (n=20-30),
base the conclusion on effect size (difference in proportions + exact 95%
CI) and a review of discordant pairs, not on the p-value."

McNemar's exact test: same statsmodels call already used for the flagship
pipeline's own A/B comparisons (scripts/mcnemar_phase6.py) - not
reimplemented here.

Effect-size confidence interval - a DELIBERATE, UNREVIEWED implementation
choice
--------------------------------------------------------------------------
No expert in the 4-round design review named a specific CI method for
this (confirmed by re-reading itog_ekspertizy_agent_profil.md and all
four experts' individual rounds and both McNemar-arithmetic followups -
the phrase "effect size + exact 95% CI" recurs verbatim from
kimi_agent_profil_round3.md onward, but is never operationalized into a
formula). This module's choice, made here rather than by the panel:

  An EXACT CONDITIONAL CI on the discordant pairs. Exact McNemar
  conditions on n_discordant = b + c (the off-diagonal cells) and treats
  b as Binomial(n_discordant, 0.5) under the null - confirmed numerically
  against this exact codebase's own scipy/statsmodels versions
  (2026-09-14): statsmodels' mcnemar(table, exact=True).pvalue equals
  scipy.stats.binomtest(b, n_discordant, 0.5).pvalue exactly on every
  table checked, including the asymmetric ones test_paired_stats.py
  exercises - not merely "equivalent up to rounding" but bit-for-bit
  identical, which is the strongest evidence this project's house rules
  favor over restating a textbook claim from memory (svod_pravil_raboty.md).
  Inverting that exact binomial test (scipy.stats.binomtest(...)
  .proportion_ci(method="exact"), i.e. Clopper-Pearson) gives an exact CI
  on q = b / n_discordant (the share of discordant pairs favoring the
  agent), which maps onto a CI for the difference in marginal proportions
  d = p_agent - p_baseline via d = (2q - 1) * n_discordant / n (exact
  since d = (b - c) / n = (2b - n_discordant) / n by construction, and the
  mapping is monotonic increasing in q, so q's CI bounds carry over to d's
  CI bounds directly).

  This was chosen over the more commonly cited Newcombe (1998) "Method
  10" (Wilson-score intervals on each marginal proportion plus a
  correlation-correction term) for two reasons: (1) it conditions on
  exactly the same discordant-pairs quantity the exact McNemar test
  above already conditions on, so the test and the CI stay
  methodologically consistent with each other rather than mixing an
  exact conditional test with an unconditional-style CI; (2) it reduces
  to one well-tested SciPy call instead of reproducing Newcombe's
  multi-term algebra from a training-data memory of the paper, which is
  exactly the kind of unverified-arithmetic risk this project's process
  exists to catch (see kimi_followup_mcnemar_arithmetic_response.md's own
  "three rounds of corrections in a row" episode on a much simpler calculation).
  Both choices are defensible; this one is documented here specifically
  so a future reviewer can see it was a deliberate choice made in the
  absence of panel guidance, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class PairedComparisonResult:
    n: int
    both_correct: int
    a_only: int
    b_only: int
    both_wrong: int
    n_discordant: int
    p_a: float
    p_b: float
    difference: float  # p_a - p_b
    ci_low: float
    ci_high: float
    confidence_level: float
    mcnemar_exact_pvalue: float


def compare_paired_binary_outcomes(
    both_correct: int,
    a_only: int,
    b_only: int,
    both_wrong: int,
    confidence_level: float = 0.95,
) -> PairedComparisonResult:
    """Compares two systems on the same matched questions via McNemar's
    exact test, and an exact conditional CI on the difference in marginal
    proportions (see module docstring for the CI derivation). `a_only`:
    number of matched questions where system A was correct and B was not
    (and vice versa for `b_only`) - the two discordant cells of the 2x2
    table, same layout scripts/mcnemar_phase6.py already uses
    ([[both_correct, a_only], [b_only, both_wrong]]).

    Args:
        both_correct: Number of matched questions where both systems were correct.
        a_only: Number of matched questions where system A was correct and B was not.
        b_only: Number of matched questions where system B was correct and A was not.
        both_wrong: Number of matched questions where both systems were incorrect.
        confidence_level: Confidence level for the exact CI on the difference in proportions.

    Returns:
        A PairedComparisonResult with the matched-pair counts (n,
        both_correct, a_only, b_only, both_wrong, n_discordant), each
        system's marginal accuracy (p_a, p_b), their difference, the
        exact confidence interval (ci_low, ci_high) at confidence_level,
        and the two-sided exact McNemar p-value (mcnemar_exact_pvalue).

    Raises:
        ValueError: if any cell count is negative, or all four are zero
            (nothing to compare).
    """
    counts = (both_correct, a_only, b_only, both_wrong)
    if any(c < 0 for c in counts):
        raise ValueError(f"cell counts must be non-negative, got {counts}")
    n = sum(counts)
    if n == 0:
        raise ValueError("at least one paired observation is required (all four cell counts are zero)")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    n_discordant = a_only + b_only
    p_a = (both_correct + a_only) / n
    p_b = (both_correct + b_only) / n
    difference = p_a - p_b

    table = [[both_correct, a_only], [b_only, both_wrong]]
    mcnemar_exact_pvalue = mcnemar(table, exact=True).pvalue

    if n_discordant == 0:
        # a and b agree on every question - the difference is exactly 0
        # with no discordant-pair sampling variation to speak of (and
        # scipy.stats.binomtest requires n >= 1, so there is no CI to
        # invert here in the first place). mcnemar_exact_pvalue is 1.0 in
        # this case (verified against statsmodels 0.14, 2026-09-14) - a
        # vacuous "no evidence of a difference" result, not an error.
        return PairedComparisonResult(
            n=n,
            both_correct=both_correct,
            a_only=a_only,
            b_only=b_only,
            both_wrong=both_wrong,
            n_discordant=0,
            p_a=p_a,
            p_b=p_b,
            difference=0.0,
            ci_low=0.0,
            ci_high=0.0,
            confidence_level=confidence_level,
            mcnemar_exact_pvalue=mcnemar_exact_pvalue,
        )

    q_ci = binomtest(a_only, n_discordant, 0.5).proportion_ci(confidence_level=confidence_level, method="exact")
    ci_low = (2 * q_ci.low - 1) * n_discordant / n
    ci_high = (2 * q_ci.high - 1) * n_discordant / n

    return PairedComparisonResult(
        n=n,
        both_correct=both_correct,
        a_only=a_only,
        b_only=b_only,
        both_wrong=both_wrong,
        n_discordant=n_discordant,
        p_a=p_a,
        p_b=p_b,
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        mcnemar_exact_pvalue=mcnemar_exact_pvalue,
    )

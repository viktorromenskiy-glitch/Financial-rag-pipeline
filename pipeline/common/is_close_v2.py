"""Sign/scale-invariant numeric answer comparison.

Transplanted literally (not rewritten from memory) from the real Colab
script for tests 2.4/2.5 (experiments_weeks_1_2.ipynb, the cell that
produced the accepted checkpoint: Direct accuracy 0.733 on 30 gold-context
questions - see docs/tehnicheskoe_zadanie.md, section 7, and
docs/specifikatsiya_moduley.md, module 9). This is mandatory per that spec
- do not reimplement this logic from a description, only from the actual
source.

What this function actually does (important: narrower than the spec
summary "normalizes currency symbols and units" might suggest): it accepts
two values already convertible via float() and checks six numeric
candidates derived from the gold value - b itself, its sign flip, and both
directions of a x100 scale flip (covering the fraction-vs-percentage and
sign-convention gaps observed in the generation output, e.g. 0.015 vs 1.5,
or -60 vs 60). It does **not** strip currency symbols, percent signs, or
thousands separators from strings - e.g. is_close_v2("0.5", "50%") returns
False, because float("50%") raises ValueError. Any such string cleanup
must happen in the caller (module 8/9) before this function is invoked;
this file intentionally reproduces the original logic bit-for-bit rather
than silently adding normalization that was not present in the validated
0.733 checkpoint run.
"""

from __future__ import annotations


def is_close_v2(a, b, tol: float = 0.01) -> bool:
    """True if `a` matches `b` up to a sign flip and/or a x100 scale flip,
    within relative tolerance `tol` (absolute tolerance when the candidate
    is exactly 0). Returns False for None or non-numeric input.

    Args:
        a: The generated/compared value (any type accepted by float()).
        b: The gold value (any type accepted by float()).
        tol: Relative tolerance (default 0.01, i.e. 1%), or absolute
            tolerance when a candidate equals 0.

    Returns:
        True if `a` is within tolerance of `b`, `-b`, `b*100`, `-b*100`,
        `b/100`, or `-b/100`. False if either input is None or not
        convertible to float.
    """
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (ValueError, TypeError):
        return False
    candidates = [b, -b, b * 100, -b * 100, b / 100, -b / 100]
    for cb in candidates:
        if cb == 0:
            if abs(a - cb) < tol:
                return True
        elif abs(a - cb) / abs(cb) < tol:
            return True
    return False

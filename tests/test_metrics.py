"""Tests for the shared Recall@k metric helpers used by plan Steps 3 and 4."""

from __future__ import annotations

import pytest

from pipeline.common.metrics import is_hit_at_k, recall_at_k


def test_is_hit_at_k_true_when_gold_in_top_k():
    assert is_hit_at_k(["a", "b", "c"], "b", k=3) is True


def test_is_hit_at_k_false_when_gold_outside_top_k():
    assert is_hit_at_k(["a", "b", "c", "d"], "d", k=2) is False


def test_is_hit_at_k_false_when_gold_absent():
    assert is_hit_at_k(["a", "b"], "z", k=5) is False


def test_is_hit_at_k_empty_candidates():
    assert is_hit_at_k([], "a", k=5) is False


def test_is_hit_at_k_k_larger_than_list_still_checks_all():
    assert is_hit_at_k(["a", "b"], "b", k=100) is True


def test_is_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        is_hit_at_k(["a"], "a", k=0)
    with pytest.raises(ValueError):
        is_hit_at_k(["a"], "a", k=-1)


def test_recall_at_k_basic_fraction():
    assert recall_at_k(hits=202, total=250) == pytest.approx(0.808)


def test_recall_at_k_zero_hits():
    assert recall_at_k(hits=0, total=100) == 0.0


def test_recall_at_k_all_hits():
    assert recall_at_k(hits=100, total=100) == 1.0


def test_recall_at_k_rejects_non_positive_total():
    with pytest.raises(ValueError):
        recall_at_k(hits=0, total=0)

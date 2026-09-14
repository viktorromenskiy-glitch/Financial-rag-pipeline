"""Tests for pipeline.common.sampling.stratified_sample - the Day 2
harness's 20-30 question sample selection.
"""

from __future__ import annotations

import pytest

from pipeline.common.sampling import stratified_sample


def _items(counts: dict[str, int]) -> list[dict]:
    items = []
    for label, count in counts.items():
        for i in range(count):
            items.append({"source_dataset": label, "question_id": f"{label}_{i}"})
    return items


def test_rejects_non_positive_n():
    items = _items({"FinQA": 10})
    with pytest.raises(ValueError):
        stratified_sample(items, 0, key="source_dataset", seed=1)
    with pytest.raises(ValueError):
        stratified_sample(items, -5, key="source_dataset", seed=1)


def test_rejects_n_larger_than_population():
    items = _items({"FinQA": 10})
    with pytest.raises(ValueError):
        stratified_sample(items, 11, key="source_dataset", seed=1)


def test_sample_size_is_always_exactly_n():
    items = _items({"FinQA": 100, "ConvFinQA": 60, "TAT-DQA": 40})
    for n in (1, 5, 20, 30, 50, 199, 200):
        sample = stratified_sample(items, n, key="source_dataset", seed=42)
        assert len(sample) == n


def test_sample_has_no_duplicates():
    items = _items({"FinQA": 100, "ConvFinQA": 60, "TAT-DQA": 40})
    sample = stratified_sample(items, 30, key="source_dataset", seed=42)
    ids = [item["question_id"] for item in sample]
    assert len(ids) == len(set(ids))


def test_allocation_is_roughly_proportional_to_stratum_size():
    # 100/60/40 population (200 total) sampled to n=20 -> proportional
    # shares are 10/6/4 exactly (no rounding needed) - an easy case to
    # assert on exactly.
    items = _items({"FinQA": 100, "ConvFinQA": 60, "TAT-DQA": 40})
    sample = stratified_sample(items, 20, key="source_dataset", seed=1)
    counts = {"FinQA": 0, "ConvFinQA": 0, "TAT-DQA": 0}
    for item in sample:
        counts[item["source_dataset"]] += 1
    assert counts == {"FinQA": 10, "ConvFinQA": 6, "TAT-DQA": 4}


def test_largest_remainder_allocation_sums_to_n_on_an_uneven_split():
    # 3 strata of size 7 each (21 total), n=10: exact share is 10/3=3.33
    # per stratum - floors sum to 9, one stratum gets the extra seat via
    # the largest remainder. Every stratum's remainder is identical here
    # (all groups the same size), so the exact winner depends on sort
    # stability - only the TOTAL is asserted, not which stratum wins the
    # extra seat.
    items = _items({"A": 7, "B": 7, "C": 7})
    sample = stratified_sample(items, 10, key="source_dataset", seed=7)
    assert len(sample) == 10


def test_small_stratum_smaller_than_its_proportional_share_does_not_crash():
    # TAT-DQA has only 2 items but n=30 out of a 302-item population would
    # nominally ask for ~0.2 -> 0 or 1, well within its size - but push n
    # high enough that TAT-DQA's proportional share would exceed 2 if not
    # clipped: 300 FinQA + 2 TAT-DQA, n=250 -> TAT-DQA's raw share is
    # 250*2/302 ~= 1.66 -> allocated 1 or 2, either way must not exceed 2
    # and the shortfall (if any) must be redistributed to FinQA rather
    # than raising.
    items = _items({"FinQA": 300, "TAT-DQA": 2})
    sample = stratified_sample(items, 250, key="source_dataset", seed=3)
    assert len(sample) == 250
    tat_dqa_count = sum(1 for item in sample if item["source_dataset"] == "TAT-DQA")
    assert tat_dqa_count <= 2


def test_deterministic_given_the_same_seed():
    items = _items({"FinQA": 100, "ConvFinQA": 60, "TAT-DQA": 40})
    sample_1 = stratified_sample(items, 30, key="source_dataset", seed=123)
    sample_2 = stratified_sample(items, 30, key="source_dataset", seed=123)
    assert [item["question_id"] for item in sample_1] == [item["question_id"] for item in sample_2]


def test_different_seeds_can_give_different_samples():
    items = _items({"FinQA": 100, "ConvFinQA": 60, "TAT-DQA": 40})
    sample_1 = stratified_sample(items, 30, key="source_dataset", seed=1)
    sample_2 = stratified_sample(items, 30, key="source_dataset", seed=2)
    ids_1 = {item["question_id"] for item in sample_1}
    ids_2 = {item["question_id"] for item in sample_2}
    assert ids_1 != ids_2


def test_sampling_the_entire_population_returns_everything():
    items = _items({"FinQA": 5, "ConvFinQA": 3})
    sample = stratified_sample(items, 8, key="source_dataset", seed=1)
    assert {item["question_id"] for item in sample} == {item["question_id"] for item in items}


def test_single_stratum_behaves_like_plain_random_sample():
    items = _items({"FinQA": 50})
    sample = stratified_sample(items, 10, key="source_dataset", seed=1)
    assert len(sample) == 10
    assert all(item["source_dataset"] == "FinQA" for item in sample)

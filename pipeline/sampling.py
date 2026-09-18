"""Stratified random sampling over a labeled key - Day 2 of
plan_rabot_posle_ekspertizy_agent_profil.md: "Evaluation harness: 20-30
questions from T2-RAGBench (stratified sample)".

Allocates the requested sample size across strata (e.g. source_dataset:
FinQA/ConvFinQA/TAT-DQA) proportionally to each stratum's share of the
population, using the largest-remainder (Hamilton) apportionment method
so per-stratum allocations always sum to exactly `n` regardless of
rounding - the same class of rounding problem apportionment methods exist
to solve for seats in a legislature, applied here to sample counts
instead. Sampling within each stratum uses a fixed seed
(random.Random(seed), not the global random module) for reproducibility -
matching this project's "record versions... for reproducibility"
convention (plan, Day 2) applied to sample selection, not just model
config.
"""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_sample(items: list[dict], n: int, key: str, seed: int) -> list[dict]:
    """items: dicts each containing `key` (e.g. "source_dataset"). Returns
    a new list of exactly `n` items, drawn from `items` without
    replacement, with per-stratum counts as close to proportional to each
    stratum's population share as an integer allocation allows.

    Raises:
        ValueError: if n is not positive, or exceeds len(items).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(items):
        raise ValueError(f"n={n} exceeds population size {len(items)}")

    groups: dict[object, list[dict]] = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)

    total = len(items)
    raw_shares = {label: n * len(group) / total for label, group in groups.items()}
    allocation = {label: int(share) for label, share in raw_shares.items()}  # floor of each share

    # Largest-remainder: hand out the seats still unassigned after
    # flooring to the strata with the biggest fractional remainder first,
    # so sum(allocation.values()) == n exactly.
    remaining = n - sum(allocation.values())
    by_remainder = sorted(groups, key=lambda label: raw_shares[label] - allocation[label], reverse=True)
    for label in by_remainder[:remaining]:
        allocation[label] += 1

    # A stratum can never be asked for more items than it actually has -
    # clip, then redistribute any resulting shortfall to strata that still
    # have spare capacity (matters only for small/very uneven strata,
    # where a stratum's proportional share could nominally exceed its own
    # size before this clip).
    for label in allocation:
        allocation[label] = min(allocation[label], len(groups[label]))
    shortfall = n - sum(allocation.values())
    if shortfall > 0:
        by_spare_capacity = sorted(groups, key=lambda label: len(groups[label]) - allocation[label], reverse=True)
        for label in by_spare_capacity:
            if shortfall <= 0:
                break
            room = len(groups[label]) - allocation[label]
            take = min(room, shortfall)
            allocation[label] += take
            shortfall -= take

    rng = random.Random(seed)
    sample: list[dict] = []
    for label, group in groups.items():
        sample.extend(rng.sample(group, allocation[label]))
    rng.shuffle(sample)
    return sample

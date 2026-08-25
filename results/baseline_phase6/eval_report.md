# Eval report - run baseline_phase6

Questions evaluated: 250
Judge accuracy: 194/250 = 0.776
Deterministic (is_close_v2) accuracy: 189/250 = 0.756
Judge/deterministic agreement: 233/250 = 0.932

## By source dataset

| source_dataset | n | judge accuracy | deterministic accuracy |
|---|---|---|---|
| ConvFinQA | 37 | 0.865 | 0.811 |
| FinQA | 90 | 0.733 | 0.767 |
| TAT-DQA | 123 | 0.780 | 0.732 |

## Latency (this run, real API calls only - excludes checkpoint/cache hits)

| stage | n | mean | median | p95 |
|---|---|---|---|---|
| retrieval_s | 250 | 1.20s | 1.17s | 1.29s |
| rerank_s | 250 | 1.04s | 0.97s | 1.69s |
| generation_s | 250 | 2.59s | 2.55s | 4.13s |
| judge_s | 250 | 2.14s | 2.28s | 2.99s |

_Numbers here are from an actual run, not the docs/*.md checkpoints - treat docs/tehnicheskoe_zadanie.md's documented checkpoints as the reference to compare against, not the reverse._

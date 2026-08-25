# Eval report - run formula_base_phase6

Questions evaluated: 250
Judge accuracy: 191/250 = 0.764
Deterministic (is_close_v2) accuracy: 183/250 = 0.732
Judge/deterministic agreement: 236/250 = 0.944

## By source dataset

| source_dataset | n | judge accuracy | deterministic accuracy |
|---|---|---|---|
| ConvFinQA | 37 | 0.865 | 0.784 |
| FinQA | 90 | 0.733 | 0.756 |
| TAT-DQA | 123 | 0.756 | 0.699 |

## Latency (this run, real API calls only - excludes checkpoint/cache hits)

| stage | n | mean | median | p95 |
|---|---|---|---|---|
| retrieval_s | 250 | 1.14s | 1.14s | 1.22s |
| rerank_s | 250 | 1.06s | 1.00s | 1.45s |
| generation_s | 250 | 2.94s | 2.87s | 4.42s |
| judge_s | 250 | 2.14s | 2.29s | 2.96s |

_Numbers here are from an actual run, not the docs/*.md checkpoints - treat docs/tehnicheskoe_zadanie.md's documented checkpoints as the reference to compare against, not the reverse._

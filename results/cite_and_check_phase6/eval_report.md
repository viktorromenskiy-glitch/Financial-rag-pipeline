# Eval report - run cite_and_check_phase6

Questions evaluated: 250
Judge accuracy: 194/250 = 0.776
Deterministic (is_close_v2) accuracy: 188/250 = 0.752
Judge/deterministic agreement: 236/250 = 0.944

## By source dataset

| source_dataset | n | judge accuracy | deterministic accuracy |
|---|---|---|---|
| ConvFinQA | 37 | 0.838 | 0.811 |
| FinQA | 90 | 0.767 | 0.778 |
| TAT-DQA | 123 | 0.764 | 0.715 |

## Latency (this run, real API calls only - excludes checkpoint/cache hits)

| stage | n | mean | median | p95 |
|---|---|---|---|---|
| retrieval_s | 0 | - | - | - |
| rerank_s | 0 | - | - | - |
| generation_s | 0 | - | - | - |
| judge_s | 229 | 2.14s | 2.26s | 2.97s |

_Numbers here are from an actual run, not the docs/*.md checkpoints - treat docs/tehnicheskoe_zadanie.md's documented checkpoints as the reference to compare against, not the reverse._

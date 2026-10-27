# Case study: question answering over financial filings

**Problem.** Financial documents mix narrative text and tables, and a wrong number is costly. Most public RAG demos run on clean prose and say little about how reliable the answers are.

**What I built.** A RAG pipeline for financial filings that returns exact numeric answers. Retrieval is hybrid (BM25 + dense vectors on MongoDB Atlas), followed by reranking (Cohere), answer generation (Claude) and automatic evaluation (LLM judge plus a deterministic numeric check).

**Data.** T²-RAGBench: 7,318 real financial filings and 23,088 questions that require exact numeric reasoning over text and tables.

## Result

| Metric | Value |
|---|---|
| End-to-end answer accuracy | **76.0%** (n=250, LLM judge + numeric check) |
| Retrieval Recall@5 | **94.4%** (n=900, full corpus) |
| Serving cost | about **$0.016 per question** |
| Automated tests | run without any paid API call (see README for the current count) |

Retrieval beats the closest published comparison by 11–13 percentage points (details in the README, section "Comparison to published work").

## What I found

- Retrieval is more reliable than answer generation. The right document is in the top 5 in 94.4% of cases, but the final answer is correct in 76.0%. Most of the gap comes from the generation step.
- Every design choice was tested against the project's own baseline with paired statistical tests (McNemar) and a power analysis, not judged by a few hand-picked examples. Some results were negative, and I report them: on the full corpus, contextual enrichment showed no effect of 2 percentage points or more once the reranker was in place.

## Honest limits

- This is a research-grade proof of concept, not a production system. It has no query-time source classifier, no serving API and no monitoring.
- Ingestion works only for the T²-RAGBench format. Accuracy on other documents is not measured.
- Some end-to-end comparisons are underpowered at the current sample size. The README lists them.

## Try it

- **Live demo (no setup):** https://huggingface.co/spaces/ViktorPetrov123/financial-rag-pipeline-demo — a replay of 13 real questions from the evaluated run, six correct answers and seven documented failure cases, with gold answers and judge verdicts. It does not call the live pipeline.
- **Code and full evaluation:** [README](README.md)

## For a similar task

If you need question answering over financial or other numeric-heavy documents, I can adapt this pipeline to your data, measure it on your questions and report accuracy and cost per question.

Viktor Romenskiy · [GitHub](https://github.com/viktorromenskiy-glitch)

Financial RAG Pipeline

RAG pipeline for financial document QA (text + tables). Hybrid search, contextual enrichment, reranking, and LLM-judge evaluation — built and tested on 7,300+ real financial filings requiring precise numeric answers.

Status: in development. Architecture and all major design decisions are finalized and backed by measurement (see below); implementation is in progress. This README will be updated with final end-to-end results once the full pipeline is coded and evaluated (see Known limitations).

Why this project

Most public RAG demos work against clean prose (Wikipedia, blog posts, docs). Financial documents are a harder, more realistic test: mixed text and tables, exact numeric answers where being close isn't good enough, and a high cost of being wrong. This project uses T²-RAGBench — a combination of FinQA, ConvFinQA, and TAT-DQA — 7,318 documents and 23,088 questions requiring exact numeric reasoning over text-and-table financial filings.

Architecture
Ingestion (FinQA + ConvFinQA + TAT-DQA)
  → Chunking (one document = one chunk)
  → Contextual enrichment (Claude Haiku 4.5, optional, config flag)
  → Embedding (Voyage-4, embeds full_indexed_content: metadata_prefix
       always applied + the enriched text when enrichment is on)
  → Indexing (MongoDB Atlas: vector index + full-text index)
  → Hybrid retrieval ($rankFusion: BM25 + dense, top-50)
  → Reranking (Cohere Rerank v4.0 Pro, optional, full text, top-5)
  → Generation (direct answer, Claude Sonnet 5, sign/scale-invariant format)
  → LLM Judge Evaluation (Claude Sonnet 5 judge + deterministic numeric cross-check)

Every component (reranker, enrichment, judge model, embedding model, pool size) is switchable via config, not hardcoded — see config/config.yaml. Every stage that calls an external API shares one retry policy (pipeline/common/retry.py) that retries only transient failures (429, 5xx, connection timeouts), never a 4xx.

Key decisions — backed by measurement, not intuition

Every number below traces to an actual test, documented in full in docs/poshagovyi_plan_vypolneniya.md (complete experiment log) and docs/tehnicheskoe_zadanie.md (final specification).

Decision	Result	Significance
Embedding model: Voyage-4 vs BGE-M3 (self-hosted)	Recall@5 0.990 vs 0.874	Small sample (n=50-80); gap large enough that the choice isn't in doubt
Hybrid retrieval baseline (full corpus, no reranker)	Recall@5 = 0.808	n=250, full 7,318-document corpus
Reranker: Cohere Rerank v4.0 Pro, pool size	pool=50 → Recall@5 = 0.944	n=900, full corpus, p<0.0001 vs pool=10; pool=100 not better (p=0.58)
Reranker input: no truncation	Truncating to 2,000 chars turned the reranker's gain into a regression (Recall@5 down to 0.608–0.764)	Confirmed twice — full document text is passed to Cohere regardless of length; enforced as a code-level assert, not just a comment
metadata_prefix: closing the question/context lexical gap	Recall@5 0.896 → 0.948	n=250, full corpus, McNemar p=0.00195. T²-RAGBench's own question reformulation adds company/sector/year metadata not present in the indexed text; prepending that metadata deterministically (no LLM call) closes the gap
Contextual enrichment (Claude Haiku 4.5)	+0.78 pp Recall@5	p=0.0164, n=1535 — see Known limitations for scale caveat
Generation format: Direct vs Program-of-Thought	0.733 = 0.733 on 30 gold-context questions	No measured advantage for PoT on current models; direct answer chosen for simplicity, no code-execution risk
Judge: single Claude Sonnet 5 vs deterministic check	93.3% agreement (28/30)	Both disagreements traced to the same explainable edge case

Readiness criterion: baseline-relative (statistically significant improvement over the project's own baseline via paired bootstrap/McNemar), not an arbitrary fixed metric threshold.

Dataset

T²-RAGBench — FinQA + ConvFinQA + TAT-DQA combined. 7,318 documents (text + tables), 23,088 questions. Chosen specifically because it's a hard, realistic case: exact numbers, mixed text/table format, high cost of error.

Tech stack

MongoDB Atlas (hybrid $rankFusion search) · Voyage AI (voyage-4 embeddings) · Cohere Rerank v4.0 Pro · Claude Haiku 4.5 (contextual enrichment) · Claude Sonnet 5 (generation + judge)

How to run
pip install -r requirements.txt
Copy .env.example to .env and fill in MONGODB_URI, VOYAGE_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY.
Place the raw T²-RAGBench parquet files under data/t2-ragbench/ (see docs/specifikatsiya_moduley.md, Module 1, "Config", for the exact expected filenames).
Adjust config/config.yaml if needed (pool sizes, which optional stages — enrichment, reranker — are enabled, model choices). config/config_schema.py validates this file at startup and fails immediately on a missing field or bad value, before any API call.
Run the pipeline stages in order (module-level entry points are listed in docs/specifikatsiya_moduley.md); each run's configuration snapshot and predictions are saved under results/<run_id>/ (pipeline/common/run_config.py).

A full CLI/orchestration entry point tying all nine stages together is still to be added — for now each module is called individually (see tests/test_pipeline_modules.py for real-data usage examples of modules 1-2).

Documentation
docs/tehnicheskoe_zadanie.md — full technical specification, every decision with its measured justification
docs/specifikatsiya_moduley.md — module-by-module input/output contracts
docs/RAG_arkhitektura_i_tochki_vetvleniya.md — architecture and every decision branch considered
docs/poshagovyi_plan_vypolneniya.md — complete chronological experiment log (source of every number above)
docs/struktura_repozitoriya.md — repository layout and config schema
docs/plan_podgotovki_k_kodirovaniyu.md — implementation order and checkpoints
Known limitations (honest status)
Not yet confirmed at full scale: contextual enrichment's effect (+0.78 pp) and its combination with the reranker (Recall@5 = 0.980) were measured on a reduced 450-document subsample, not the full 7,318-document corpus. A full-corpus run is planned to confirm or revise these numbers before they're reported as final.
Small-sample results: generation format comparison and judge agreement are both based on n=30 — enough to catch a large effect, not enough to rule out a small one.
MongoDB Atlas M0 (free tier) search index limit: the main collection already uses 2 of the 3 search indexes allowed per instance on M0. Any additional experiment requiring its own index (e.g. comparing embedding models on the full corpus rather than a subsample) needs either a paid tier upgrade or a separate temporary cluster — not yet decided.
Retrieval fusion weights (0.5 vector / 0.5 text) are an untuned default, not a locally-optimized value.

This project reports numbers the way they were actually measured, including sample sizes and what's still open — not as a polished claim of a finished, benchmarked system.

License

TBD.

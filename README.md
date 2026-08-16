Financial RAG Pipeline

RAG pipeline for financial document QA (text + tables). Hybrid search, contextual enrichment, reranking, and LLM-judge evaluation — built and evaluated end-to-end on 7,300+ real financial filings requiring precise numeric answers.

Status: implemented and evaluated end-to-end on the full corpus. Every architectural decision below is backed by a real measurement, including the final numbers (see Results).

Why this project

Most public RAG demos work against clean prose (Wikipedia, blog posts, docs). Financial documents are a harder, more realistic test: mixed text and tables, exact numeric answers where being close isn't good enough, and a high cost of being wrong. This project uses T²-RAGBench — a combination of FinQA, ConvFinQA, and TAT-DQA — 7,318 documents and 23,088 questions requiring exact numeric reasoning over text-and-table financial filings.

Architecture
Ingestion (FinQA + ConvFinQA + TAT-DQA)
  → Chunking (one document = one chunk)
  → Contextual enrichment (Claude Haiku 4.5, optional, config flag)
  → Embedding (Voyage-4 by default; per-dataset routing switches TAT-DQA
       documents/queries to voyage-finance-2 — config-driven, see Results)
  → Indexing (MongoDB Atlas: vector index + full-text index, source_dataset
       stored per document for routing)
  → Hybrid retrieval ($rankFusion: BM25 + dense, top-50)
  → Reranking (Cohere Rerank v4.0 Pro, optional, full text, top-5)
  → Generation (direct answer, Claude Sonnet 5, sign/scale-invariant format)
  → LLM Judge Evaluation (Claude Sonnet 5 judge + deterministic numeric cross-check)

Every component (reranker, enrichment, judge model, embedding model + routing, pool size) is switchable via config, not hardcoded — see config/config.yaml. config/config_schema.py validates the config at startup and fails immediately on a missing field or bad value, before any API call. Every stage that calls an external API shares one retry policy (pipeline/common/retry.py) that retries only transient failures (429, 5xx, connection timeouts), never a 4xx.

Results (end-to-end, full pipeline, n=250 questions)

Full pipeline (enrichment + hybrid retrieval + reranker + generation + judge), stratified by source_dataset — the same stratification the project treats as mandatory (aggregate accuracy alone hides source-specific behavior):

Source	n	Judge accuracy	Deterministic accuracy
ConvFinQA	37	0.865	0.838
FinQA	90	0.711	0.756
TAT-DQA	123	0.780	0.715
Overall	250	0.768	0.748

Retrieval stage in isolation reaches Recall@5 = 0.944 (pool=50, reranker on, full 7,318-document corpus, n=900).

Per-dataset embedding routing: a direct A/B test (McNemar exact test, n=2,500 questions per source, full corpus) found voyage-finance-2 measurably helps TAT-DQA retrieval (Recall@5 +1.9 pp, p=0.0005, robust to a Bonferroni correction) but hurts ConvFinQA (−1.9 pp, p<0.0001) and gives no reliable benefit on FinQA (−0.9 pp, not robust to correction) — contradicting a naive "one embedding model fits all three sources" assumption. Only TAT-DQA is routed to voyage-finance-2 as a result; the other two sources stay on voyage-4. Full derivation: docs/tehnicheskoe_zadanie.md, section 3a.

That retrieval-level gain does not yet show up as a detectable end-to-end accuracy improvement at the current sample size — a paired McNemar test on judge-correctness between the pre-routing and post-routing runs gives p=1.0 for TAT-DQA (2 questions flipped correct, 2 flipped incorrect, net zero at n=123). A power analysis (three independent derivations — exact binomial, Connor 1987's normal approximation, and a direct Monte Carlo simulation of the actual test — converging on n≈650-900 per source) shows the current n=123 has only ~5% power to detect a plausible 2 pp end-to-end effect. Scaling the end-to-end eval to close this gap was evaluated and deliberately not done, given the retrieval-level decision already stands on its own and the added API cost wasn't judged worthwhile for this project; the full reasoning and both formulas are in docs/tehnicheskoe_zadanie.md, section 3a.

Key decisions — backed by measurement, not intuition

Every number below traces to an actual test, documented in full in docs/poshagovyi_plan_vypolneniya.md (complete experiment log) and docs/tehnicheskoe_zadanie.md (final specification).

Decision	Result	Significance
Embedding model: Voyage-4 vs BGE-M3 (self-hosted)	Recall@5 0.990 vs 0.874	Small sample (n=50-80); gap large enough that the choice isn't in doubt
Hybrid retrieval baseline (full corpus, no reranker)	Recall@5 = 0.808	n=250, full 7,318-document corpus
Reranker: Cohere Rerank v4.0 Pro, pool size	pool=50 → Recall@5 = 0.944	n=900, full corpus, p<0.0001 vs pool=10; pool=100 not better (p=0.58)
Reranker input: no truncation	Truncating to 2,000 chars turned the reranker's gain into a regression (Recall@5 down to 0.608–0.764)	Confirmed twice — full document text is passed to Cohere regardless of length; enforced as a code-level assert, not just a comment
metadata_prefix: closing the question/context lexical gap	Recall@5 0.896 → 0.948	n=250, full corpus, McNemar p=0.00195. T²-RAGBench's own question reformulation adds company/sector/year metadata not present in the indexed text; prepending that metadata deterministically (no LLM call) closes the gap
Contextual enrichment (Claude Haiku 4.5)	+0.78 pp Recall@5	p=0.0164, n=1535 — see Known limitations for scale caveat
Per-dataset embedding routing (TAT-DQA → voyage-finance-2)	Recall@5 +1.9 pp on TAT-DQA	n=2,500/source, p=0.0005, robust to Bonferroni — see Results above for the full picture, including the not-yet-detected end-to-end effect
Generation format: Direct vs Program-of-Thought	0.733 = 0.733 on 30 gold-context questions	No measured advantage for PoT on current models; direct answer chosen for simplicity, no code-execution risk
Judge: single Claude Sonnet 5 vs deterministic check	93.3% agreement (28/30)	Both disagreements traced to the same explainable edge case

Readiness criterion: baseline-relative (statistically significant improvement over the project's own baseline via paired bootstrap/McNemar), not an arbitrary fixed metric threshold.

Dataset

T²-RAGBench — FinQA + ConvFinQA + TAT-DQA combined. 7,318 documents (text + tables), 23,088 questions. Chosen specifically because it's a hard, realistic case: exact numbers, mixed text/table format, high cost of error.

Tech stack

MongoDB Atlas (hybrid $rankFusion search) · Voyage AI (voyage-4 + voyage-finance-2, per-dataset routing) · Cohere Rerank v4.0 Pro · Claude Haiku 4.5 (contextual enrichment) · Claude Sonnet 5 (generation + judge)

How to run
pip install -r requirements.txt
Copy .env.example to .env and fill in MONGODB_URI, VOYAGE_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY.
Place the raw T²-RAGBench parquet files under data/t2-ragbench/ (see docs/specifikatsiya_moduley.md, Module 1, "Config", for the exact expected filenames).
Adjust config/config.yaml if needed (pool sizes, which optional stages — enrichment, reranker, embedding routing — are enabled, model choices).
Build/refresh the corpus: python -m pipeline.cli index --data-dir data/t2-ragbench
Run an evaluation: python -m pipeline.cli eval --questions data/t2-ragbench/eval_subset_250.parquet --run-id my_run (add --compare-to <previous_run_id> for a per-question regression report)
Each run's configuration snapshot, predictions, and report are saved under results/<run_id>/ (pipeline/common/run_config.py, eval_report.md).

Documentation
docs/tehnicheskoe_zadanie.md — full technical specification, every decision with its measured justification, including the per-dataset routing derivation (section 3a)
docs/specifikatsiya_moduley.md — module-by-module input/output contracts
docs/RAG_arkhitektura_i_tochki_vetvleniya.md — architecture and every decision branch considered
docs/poshagovyi_plan_vypolneniya.md — complete chronological experiment log (source of every number above)
docs/struktura_repozitoriya.md — repository layout and config schema
docs/plan_podgotovki_k_kodirovaniyu.md — implementation order and checkpoints
Known limitations (honest status)
Not yet confirmed at full scale: contextual enrichment's effect (+0.78 pp) and its combination with the reranker (Recall@5 = 0.980) were measured on a reduced 450-document subsample, not the full 7,318-document corpus. A full-corpus run would confirm or revise these numbers before treating them as final.
Small-sample results: generation format comparison and judge agreement are both based on n=30 — enough to catch a large effect, not enough to rule out a small one.
Per-dataset embedding routing's end-to-end effect is not yet statistically confirmed (see Results) — the retrieval-level gain is solid, but whether it survives rerank+generation at the current sample size is genuinely unresolved, not just under-tested by oversight; the required sample size to resolve it is calculated and documented.
A real implementation bug was found and fixed during end-to-end validation of the routing feature: the first version over-broadly restricted retrieval candidates for every question by source, not just routed ones, causing an unintended accuracy regression on sources whose embedding model never changed. Caught via the project's own regression-analysis discipline (per-question comparison against baseline, not just aggregate metrics), fixed, and re-validated — documented in full in docs/tehnicheskoe_zadanie.md, section 3a.
MongoDB Atlas M0 (free tier) search index limit: the main collection already uses 2 of the 3 search indexes allowed per instance on M0. Any additional experiment requiring its own index (e.g. comparing embedding models on a separate test collection) needs either a paid tier upgrade or a separate temporary cluster.
Retrieval fusion weights (0.5 vector / 0.5 text) are an untuned default, not a locally-optimized value.
No query-time source classification: per-dataset routing relies on source_dataset being known ahead of time (true for this eval dataset); a production deployment fielding arbitrary queries would need a query classifier to pick the right embedding model — out of scope for this project.

This project reports numbers the way they were actually measured, including sample sizes and what's still open — not as a polished claim of a finished, benchmarked system.

License

TBD.

Financial RAG Pipeline

RAG pipeline for financial document QA (text + tables). Hybrid search, contextual enrichment, reranking, and LLM-judge evaluation — built and evaluated end-to-end on 7,300+ real financial filings requiring precise numeric answers.

Status: implemented and evaluated end-to-end on the full corpus. Every architectural decision below is backed by a real measurement, including the final numbers (see Results).

Scope: this is a research/evaluation-grade RAG pipeline, not a production system. It validates architectural decisions (retrieval, reranking, embedding routing, generation format, judge-based evaluation) end-to-end against a fixed, pre-labeled eval set where each question's source dataset is already known ahead of time. It does not include what a live production deployment would still need on top of this: a query-time source classifier (per-dataset embedding routing currently relies on source_dataset being known, not inferred from the question text — see Known limitations below), a serving API, monitoring/cost controls beyond the numbers reported here, or a UI. See Known limitations below for the complete, honest list of what's confirmed vs. still open.

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

Full-corpus enrichment + reranker validation: a McNemar exact test directly comparing enriched vs raw text (n=900, 300 questions/source, full 7,318-document corpus, both arms retrieved and reranked identically) found no effect ≥2 percentage points — the effect size a power analysis, run before the test, said the sample was sized to detect. The observed overall delta was +0.1 pp (not significant, p=1.0). This is the honest framing: "no effect ≥2 pp detected," not "no effect exists" — the test lacks the power to reliably distinguish a true effect under 1 pp from zero. It also does not contradict the earlier +0.78 pp result below (different infrastructure: a smaller 450-document corpus, no reranker in that test). The most likely explanation is a ceiling effect, not a contradiction: the raw-text baseline on the full corpus already reaches Recall@5=0.984 thanks to the reranker and metadata_prefix, leaving little room for enrichment to show a measurable gain — a pattern that matches Anthropic's own Contextual Retrieval writeup, where enrichment's gains shrink sharply once a reranker is already in the pipeline. Enrichment stays enabled by default: the null result doesn't show harm, only that no additional benefit was detected on top of an already-strong reranked baseline at this sample size. Full derivation: docs/tehnicheskoe_zadanie.md, section 5.

Key decisions — backed by measurement, not intuition

Every number below traces to an actual test, documented in full in docs/poshagovyi_plan_vypolneniya.md (complete experiment log) and docs/tehnicheskoe_zadanie.md (final specification).

Decision	Result	Significance
Embedding model: Voyage-4 vs BGE-M3 (self-hosted)	Recall@5 0.990 vs 0.874	Small sample (n=50-80); gap large enough that the choice isn't in doubt
Hybrid retrieval baseline (full corpus, no reranker)	Recall@5 = 0.808	n=250, full 7,318-document corpus
Reranker: Cohere Rerank v4.0 Pro, pool size	pool=50 → Recall@5 = 0.944	n=900, full corpus, p<0.0001 vs pool=10; pool=100 not better (p=0.58)
Reranker input: no truncation	Truncating to 2,000 chars turned the reranker's gain into a regression (Recall@5 down to 0.608–0.764)	Confirmed twice — full document text is passed to Cohere regardless of length; enforced as a code-level assert, not just a comment
metadata_prefix: closing the question/context lexical gap	Recall@5 0.896 → 0.948	n=250, full corpus, McNemar p=0.00195. T²-RAGBench's own question reformulation adds company/sector/year metadata not present in the indexed text; prepending that metadata deterministically (no LLM call) closes the gap
Contextual enrichment (Claude Haiku 4.5), no reranker	+0.78 pp Recall@5	p=0.0164, n=1535, reduced 450-document corpus; full-corpus retest with reranker found no effect ≥2 pp — see Results above
Per-dataset embedding routing (TAT-DQA → voyage-finance-2)	Recall@5 +1.9 pp on TAT-DQA	n=2,500/source, p=0.0005, robust to Bonferroni — see Results above for the full picture, including the not-yet-detected end-to-end effect
Generation format: Direct vs Program-of-Thought	0.733 = 0.733 on 30 gold-context questions	No measured advantage for PoT on current models; direct answer chosen for simplicity, no code-execution risk
Judge: single Claude Sonnet 5 vs deterministic check	93.3% agreement (28/30)	Both disagreements traced to the same explainable edge case

Readiness criterion: baseline-relative (statistically significant improvement over the project's own baseline via paired bootstrap/McNemar), not an arbitrary fixed metric threshold.

Comparison to published work

T²-RAGBench (Strich et al., EACL 2026 — aclanthology.org/2026.eacl-long.8) is a recent benchmark; its own paper reports retrieval-level findings (identifying hybrid BM25+dense as the most effective approach for this data) rather than end-to-end generation accuracy, so it isn't a direct source for a judge-accuracy comparison. A companion paper, Akarsu et al., "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents" (arXiv:2604.01733), benchmarks ten retrieval strategies on the identical dataset (same 23,088 queries, 7,318 documents) — the closest available apples-to-apples comparison point for this project's retrieval numbers:

Configuration	Akarsu et al. Recall@5	This project's Recall@5
BM25 only	0.644	not tested standalone
Dense only	0.587	not tested standalone
Hybrid RRF (BM25 + dense)	0.695	0.808 (n=250, full corpus)
Hybrid + Cohere Rerank v4.0 Pro	0.816	0.944 (n=900, full corpus)

This project's numbers come in above the published results on the same dataset — roughly +11 pp for hybrid retrieval alone, +13 pp for hybrid+reranker. Read this as a careful-implementation advantage, not a different algorithm: the two most likely contributors, both already documented above, are (1) passing the reranker the full, untruncated document text (this project's own tests show truncating to 2,000 characters turns the reranker's gain into a regression; the comparison paper doesn't report its truncation policy, so this isn't a claim about what they did wrong, just what this project verified about its own pipeline) and (2) the metadata_prefix fix closing a lexical gap between T²-RAGBench's reformulated questions and the indexed text (+5.2 pp on its own — see Key decisions above). The comparison also isn't a fully controlled one: this project's numbers come from a stratified n=250/900 eval subset, not necessarily the same evaluation split Akarsu et al. used, and sample-size uncertainty on both sides isn't quantified here.

For end-to-end QA accuracy, no publication was found benchmarking generation accuracy specifically on the unified T²-RAGBench split. The closest available reference point is FinAgent-RAG (arXiv:2605.05409), an agentic Program-of-Thought pipeline with self-verification, evaluated separately on the three source datasets: execution accuracy 76.81% (FinQA), 78.46% (ConvFinQA), 74.96% (TAT-QA). This project's judge accuracy on its own n=250 split: 71.1% (FinQA), 86.5% (ConvFinQA), 78.0% (TAT-DQA), 76.8% overall — ahead on two of three sources, behind on FinQA, despite this project using a deliberately simpler direct-answer pipeline (no agentic loop, no code execution — see the Direct vs Program-of-Thought decision above). This comparison is weaker evidence than the retrieval one: different eval subsets and sizes, different scoring methodology (execution accuracy vs LLM judge + deterministic cross-check), and a genuinely different generation architecture — read it as directional context, not a rigorous head-to-head.

Unit economics (price per query, estimate)

Indexing is a one-time cost: embedding the full corpus (~$0.13, Voyage-4) plus contextual enrichment (~$10, Claude Haiku 4.5) — see Results above. Serving is priced per query instead. This project never logged per-request token usage during real runs, so the number below is a calculation, not a measurement: built from published provider pricing (checked 2026-08-20) and this repo's own measured prompt/corpus lengths (~4 chars/token heuristic), not from actual billing. Full derivation, every input number, and the explicit assumptions this rests on (notably, Cohere's exact "search unit" billing definition wasn't found in their public docs, so the reranking line assumes 1 query = 1 search unit) are in docs/tehnicheskoe_zadanie.md, section 15.

Stage	Estimated cost
Query embedding (Voyage-4)	<$0.0001
Reranking (Cohere Rerank v4.0 Pro, pool=50)	~$0.0025
Generation (Claude Sonnet 5, ~5 full documents of context — the dominant cost)	~$0.013
LLM Judge (Claude Sonnet 5)	~$0.001
Total per question	~$0.016

At that rate, the committed n=250 error-analysis run (results/error_analysis_250) cost roughly $4 in query-time API calls. The larger end-to-end validation flagged in section 3a as deliberately not run (~700-900 TAT-DQA questions, two full runs) would cost an estimated $23-29 — cheap relative to what might be assumed without counting it explicitly, though cost wasn't actually why that validation was skipped (the retrieval-level result already stands on its own).

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
docs/tehnicheskoe_zadanie.md — full technical specification, every decision with its measured justification, including the per-dataset routing derivation (section 3a) and the full-corpus enrichment validation (section 5)
docs/specifikatsiya_moduley.md — module-by-module input/output contracts
docs/RAG_arkhitektura_i_tochki_vetvleniya.md — architecture and every decision branch considered
docs/poshagovyi_plan_vypolneniya.md — complete chronological experiment log (source of every number above)
docs/struktura_repozitoriya.md — repository layout and config schema
docs/plan_podgotovki_k_kodirovaniyu.md — implementation order and checkpoints
Known limitations (honest status)
Contextual enrichment's effect did not replicate at full scale under a reranked baseline: the measured +0.78 pp gain (n=1535) came from a reduced 450-document corpus without a reranker; a full-corpus retest (n=900, with reranker) found no effect ≥2 pp, the size the test was powered to detect. Most likely a ceiling effect from the reranker and metadata_prefix already doing most of the work, not a contradiction of the earlier result — but both numbers should be read together, not the +0.78 pp figure quoted alone. Enrichment stays enabled by default regardless, since the null result shows no harm, just no confirmed additional benefit at this sample size. See docs/tehnicheskoe_zadanie.md, section 5.
Small-sample results: generation format comparison and judge agreement are both based on n=30 — enough to catch a large effect, not enough to rule out a small one.
Per-dataset embedding routing's end-to-end effect is not yet statistically confirmed (see Results) — the retrieval-level gain is solid, but whether it survives rerank+generation at the current sample size is genuinely unresolved, not just under-tested by oversight; the required sample size to resolve it is calculated and documented.
A real implementation bug was found and fixed during end-to-end validation of the routing feature: the first version over-broadly restricted retrieval candidates for every question by source, not just routed ones, causing an unintended accuracy regression on sources whose embedding model never changed. Caught via the project's own regression-analysis discipline (per-question comparison against baseline, not just aggregate metrics), fixed, and re-validated — documented in full in docs/tehnicheskoe_zadanie.md, section 3a.
A committed run was lost and rebuilt: the n=250 error-analysis run backing the numbers above (results/error_analysis_250) is a rerun of an earlier run whose results were never saved to persistent storage and were lost when the Colab runtime disconnected — a violation of a rule that didn't yet exist at the time. That gap is now closed as a mandatory rule (see docs/tehnicheskoe_zadanie.md, section 11): every paid-API run must be verified and copied off the ephemeral runtime in the same step that produces it. The rerun, under an identical configuration, reproduced the original numbers within 1-3 pp per source (overall judge accuracy 0.760 vs. the lost run's 0.768) — within the residual API-side stochasticity already documented elsewhere in this project even at temperature=0.0, not evidence either run was wrong. Full account, including the taxonomy of the 60 remaining errors: docs/tehnicheskoe_zadanie.md, section 14.
No correction for multiple comparisons across the project's history: roughly seven significance tests were run over the course of this project without a global family-wise error rate correction (each was treated as an independent, pre-planned test of its own architectural question — a defensible but not the only reasonable policy; see docs/tehnicheskoe_zadanie.md, section 9, for which specific results would or wouldn't survive a naive Bonferroni correction across all of them). Concretely, the enrichment result above is the one architectural decision in this document whose statistical significance depends on that policy choice: its underlying pre-reranker result (+0.78 pp, p=0.0164) would not survive a naive Bonferroni correction across those ~7 tests, while every other significant result reported here (reranker pool_size, metadata_prefix, TAT-DQA routing) has enough margin to hold up regardless of the correction.
MongoDB Atlas M0 (free tier) search index limit: the main collection already uses 2 of the 3 search indexes allowed per instance on M0. Any additional experiment requiring its own index (e.g. comparing embedding models on a separate test collection) needs either a paid tier upgrade or a separate temporary cluster.
Retrieval fusion weights (0.5 vector / 0.5 text): a grid sweep (vector weight 0.3-0.7, n=250, paired McNemar against the default) found no combination significantly better — the highest points (0.6/0.4 and 0.7/0.3) were numerically +1.2 pp ahead but not significant (p>0.25 in every case, 3-7 discordant pairs out of 250). Weights stay at 0.5/0.5: picking the best of five tested points after the fact would be the same post-hoc data-dredging risk the project deliberately avoided elsewhere (see docs/tehnicheskoe_zadanie.md, section 4).
No query-time source classification: per-dataset routing relies on source_dataset being known ahead of time (true for this eval dataset); a production deployment fielding arbitrary queries would need a query classifier to pick the right embedding model — out of scope for this project.

This project reports numbers the way they were actually measured, including sample sizes and what's still open — not as a polished claim of a finished, benchmarked system.

License

MIT — see LICENSE.

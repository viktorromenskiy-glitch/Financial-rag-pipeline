# Engineering Decisions

This page summarizes three decision points — vector database choice, a reranker false negative and its root-cause diagnosis, and how an unverified reference repo's numbers were treated — picked out for the clearest end-to-end evidence trail (question → alternatives → evidence → decision). Nothing here is new: every number and claim below is drawn from this project's full engineering journal in the rest of this repository's `docs/` directory (architecture forks, statistical tests, error taxonomies, and a day-by-day experiment log, maintained in Russian — the project's working language during development), just narrated in English for a reader who wants the reasoning behind these three specific decisions without the full 800+ lines.

## Decision 1 — Vector database: MongoDB Atlas over LanceDB / Qdrant

**The question.** What to build hybrid search (BM25 + dense + reciprocal rank fusion) on: an embedded, in-process store (LanceDB — no server, `pip install` and go), a dedicated vector server (Qdrant — native hybrid search, a built-in reranking slot, a mature Python client), or MongoDB Atlas, to which the project already had an account.

**What the evidence looked like.** Two comparable reference repositories for financial-document RAG (`pablograba/financial-10k-agentic-rag`, `datapizza-labs/contextual-retrieval-experiments`) both use Qdrant deliberately, for its native hybrid search and metadata filtering at scale — a real, if indirect, signal in Qdrant's favor. A third reference repo, `sec-mdna-rag`, uses FAISS and offered no direct precedent either way for this particular fork (see Decision 3 below for why its numbers weren't trusted at face value regardless).

**The decision.** MongoDB Atlas. Not because it retrieves better than the alternatives — the deciding factor was the free M0 tier's native `$rankFusion` aggregation stage (BM25 + vector search fused server-side, no separate fusion logic to write and test) against zero marginal setup cost, since the account already existed. The trade-off made explicit at the time: M0's 512 MB storage cap is workable for this corpus but doesn't leave room for holding multiple experimental corpus variants (e.g. with and without contextual enrichment) side by side in the same cluster — a real, accepted constraint, not an unweighed one.

## Decision 2 — Reranker input: full document text, never truncated

This is the project's clearest example of a wrong first conclusion caught and reversed by root-cause diagnosis rather than by re-running the same test and hoping for a different number.

**What looked like a negative result.** The first two test rounds passed documents to the reranker truncated to 2,000 characters. Both showed the reranker making retrieval *worse*: Recall@5 dropped from a 0.808 baseline to 0.608–0.764.

**Diagnosing why, instead of accepting the aggregate number.** Rather than concluding "reranking doesn't help this corpus" from that result, the next step was to check what the truncation was actually doing to the documents being reranked. Measured directly: 93.2% of the target documents in this corpus are longer than 2,000 characters (median 4,263 characters, max 14,782) — the truncation was silently discarding more than half the relevant text on a typical document before the reranker ever saw it. That hypothesis was cross-checked against independent literature on context compression and reranker degradation (specific papers, cited with page/table references, not paraphrased from memory) before being accepted as the likely cause.

**Result after removing the truncation.** The same reranker (Cohere Rerank v4.0 Pro), given full untruncated document text, delivered Recall@5 = 0.868 at pool size 10 — a +6.0 percentage-point gain over baseline, not a regression. The finding was then extended with a candidate-pool-size sweep on a larger, stratified sample (n=900, full 7,318-document corpus, exact paired McNemar test): pool=10 → 0.890, pool=50 → 0.944 (significantly better than pool=10, p<0.0001), pool=100 → 0.941 (no improvement over pool=50, p=0.58). Untruncated input and pool=50 are now enforced as the project's default configuration, not just a recommendation in a document.

**The lesson carried forward.** A technique that looks harmful on an aggregate metric can be a measurement artifact, not a real negative result — the project's standing rule since this incident is to diagnose the failure mode before discarding a technique, not just to re-measure and average.

## Decision 3 — Treating a reference repository's numbers as a lead, not a fact

`sec-mdna-rag` (a public repo used as one of several architecture references, see Decision 1) reports a striking ablation number — a +41.67% Recall@1 improvement from metadata filtering. That number was explicitly *not* adopted as a validated fact: the repository has a single author and zero stars, meaning the result has no independent replication or community scrutiny behind it. It was used as a source of architectural ideas (the metadata schema, the general pipeline shape) — a starting point to test independently, not a benchmark to cite.

This is the same standard applied throughout the project to every external claim that ends up in the public-facing README: published papers cited there (T²-RAGBench, the Akarsu et al. reranker benchmark, FinRank) were independently re-verified against their primary sources — not paraphrased from a summary — before being quoted with specific numbers, and the README's "Comparison to published work" section is explicit about which comparisons are and aren't apples-to-apples.

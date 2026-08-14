"""Module 4 - Contextual enrichment (optional, indexing stage).
 
At plan Step 2 (enrichment.enabled=false) only the trivial path is used -
contextual_summary="" for every document, no real Claude Haiku 4.5 calls are
made. The full path (enabled=true, real calls) is turned on and validated
against the live corpus at Step 5 - see plan_podgotovki_k_kodirovaniyu.md.
 
See docs/specifikatsiya_moduley.md, module 4.
"""
 
from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
 
from pipeline.common.retry import retryable
 
MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0.0
PROMPT_VERSION = "v1"
 
 
@dataclass(frozen=True)
class EnrichmentResult:
    context_id: str
    contextual_summary: str
 
 
class SummarizerProtocol(Protocol):
    """Abstraction over the Claude API - allows a fake summarizer to be
    substituted in tests without a real network call."""
 
    def summarize(self, raw_content: str) -> str: ...
 
 
@retryable()
def _summarize_with_retry(summarizer: SummarizerProtocol, raw_content: str) -> str:
    return summarizer.summarize(raw_content)
 
 
def enrich_document(
    summarizer: SummarizerProtocol, context_id: str, raw_content: str
) -> EnrichmentResult:
    summary = _summarize_with_retry(summarizer, raw_content)
    return EnrichmentResult(context_id=context_id, contextual_summary=summary)
 
 
class EnrichmentCheckpoint:
    """External progress state for enrichment (JSON Lines, one line = one
    context_id) - processing 7318 documents should not restart from scratch
    after a mid-run failure (see specifikatsiya_moduley.md, module 4,
    "Устойчивость")."""
 
    def __init__(self, path: str | Path):
        self.path = Path(path)
 
    def load_done(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        done: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["context_id"]] = rec["contextual_summary"]
        return done
 
    def append(self, result: EnrichmentResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "context_id": result.context_id,
                        "contextual_summary": result.contextual_summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
 
 
def enrich_documents(
    summarizer: SummarizerProtocol | None,
    documents: list[tuple[str, str]],
    checkpoint: EnrichmentCheckpoint | None = None,
    enabled: bool = True,
) -> dict[str, str]:
    """documents: list of (context_id, raw_content), unique per context_id
    (output of dedupe_documents from module 5).
 
    Returns {context_id: contextual_summary}.
 
    enabled=False (plan Step 2) - trivial path: empty string for every
    document, summarizer/checkpoint are not used, no API calls are made.
    """
    if not enabled:
        return {context_id: "" for context_id, _ in documents}
 
    if summarizer is None or checkpoint is None:
        raise ValueError("summarizer and checkpoint are required when enabled=True")
 
    results = dict(checkpoint.load_done())
    for context_id, raw_content in documents:
        if context_id in results:
            continue
        result = enrich_document(summarizer, context_id, raw_content)
        checkpoint.append(result)
        results[context_id] = result.contextual_summary
    return results

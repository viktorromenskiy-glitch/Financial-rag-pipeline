"""
Модуль 4 — Contextual enrichment (опционально, этап индексации).
 
На Шаге 2 плана (enrichment.enabled=false) используется только тривиальный
путь — contextual_summary="" для всех документов, реальные вызовы Claude
Haiku 4.5 не идут. Полный путь (enabled=true, настоящие вызовы) включается и
проверяется на живом корпусе на Шаге 5 — см. plan_podgotovki_k_kodirovaniyu.md.
 
См. docs/specifikatsiya_moduley.md, модуль 4.
"""
 
from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
 
from tenacity import retry, stop_after_attempt, wait_random_exponential
 
MODEL = "claude-haiku-4.5"
TEMPERATURE = 0.0
PROMPT_VERSION = "v1"
 
 
@dataclass(frozen=True)
class EnrichmentResult:
    context_id: str
    contextual_summary: str
 
 
class SummarizerProtocol(Protocol):
    """Абстракция над Claude API — позволяет подставлять фейковый
    summarizer в тестах без реального сетевого вызова."""
 
    def summarize(self, raw_content: str) -> str: ...
 
 
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60))
def _summarize_with_retry(summarizer: SummarizerProtocol, raw_content: str) -> str:
    return summarizer.summarize(raw_content)
 
 
def enrich_document(
    summarizer: SummarizerProtocol, context_id: str, raw_content: str
) -> EnrichmentResult:
    summary = _summarize_with_retry(summarizer, raw_content)
    return EnrichmentResult(context_id=context_id, contextual_summary=summary)
 
 
class EnrichmentCheckpoint:
    """Внешнее состояние прогресса обогащения (JSON Lines, одна строка =
    один context_id) — обработка 7318 документов не должна начинаться
    заново при сбое на середине (см. specifikatsiya_moduley.md, модуль 4,
    «Устойчивость»)."""
 
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
    """documents: список (context_id, raw_content), уникальных по context_id
    (выход dedupe_documents из модуля 5).
 
    Возвращает {context_id: contextual_summary}.
 
    enabled=False (Шаг 2 плана) — тривиальный путь: пустая строка для всех,
    summarizer/checkpoint не используются, вызовов API нет.
    """
    if not enabled:
        return {context_id: "" for context_id, _ in documents}
 
    if summarizer is None or checkpoint is None:
        raise ValueError("summarizer и checkpoint обязательны при enabled=True")
 
    results = dict(checkpoint.load_done())
    for context_id, raw_content in documents:
        if context_id in results:
            continue
        result = enrich_document(summarizer, context_id, raw_content)
        checkpoint.append(result)
        results[context_id] = result.contextual_summary
    return results

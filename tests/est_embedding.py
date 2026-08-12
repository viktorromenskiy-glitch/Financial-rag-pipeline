richment · PY
"""Тесты модуля 4 (enrichment) — фейковый summarizer, без реальных вызовов
Claude API. Реальный прогон на живом корпусе — Шаг 5 плана, не здесь."""
 
from __future__ import annotations
 
import pytest
 
from pipeline.enrichment import EnrichmentCheckpoint, enrich_document, enrich_documents
 
 
class FakeSummarizer:
    def __init__(self, fail_times: int = 0):
        self.calls = 0
        self.fail_times = fail_times
 
    def summarize(self, raw_content: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient failure")
        return f"summary of: {raw_content[:20]}"
 
 
def test_enrich_documents_disabled_is_trivial_no_api_calls():
    """Шаг 2 плана: enabled=False -> пустая строка для всех, summarizer не
    трогается вообще (его можно не передавать)."""
    docs = [("ctx_1", "raw text 1"), ("ctx_2", "raw text 2")]
    result = enrich_documents(summarizer=None, documents=docs, checkpoint=None, enabled=False)
    assert result == {"ctx_1": "", "ctx_2": ""}
 
 
def test_enrich_documents_disabled_empty_input():
    assert enrich_documents(None, [], None, enabled=False) == {}
 
 
def test_enrich_documents_enabled_requires_summarizer_and_checkpoint():
    with pytest.raises(ValueError):
        enrich_documents(None, [("ctx_1", "text")], None, enabled=True)
 
 
def test_enrich_document_retries_on_transient_failure():
    summarizer = FakeSummarizer(fail_times=2)
    result = enrich_document(summarizer, "ctx_1", "raw content")
    assert result.contextual_summary.startswith("summary of:")
    assert summarizer.calls == 3  # 2 неудачи + успешная попытка
 
 
def test_checkpoint_resume_skips_already_enriched(tmp_path):
    """Ключевой сценарий устойчивости: при повторном вызове с тем же
    checkpoint-файлом уже обработанные документы не пересчитываются."""
    checkpoint_path = tmp_path / "enrichment_checkpoint.jsonl"
    checkpoint = EnrichmentCheckpoint(checkpoint_path)
    summarizer = FakeSummarizer()
 
    docs = [("ctx_1", "text 1"), ("ctx_2", "text 2")]
    result_1 = enrich_documents(summarizer, docs, checkpoint, enabled=True)
    assert summarizer.calls == 2
    assert set(result_1) == {"ctx_1", "ctx_2"}
 
    # Новый checkpoint-объект на тот же файл, новый summarizer — как при
    # перезапуске после обрыва сессии
    checkpoint_2 = EnrichmentCheckpoint(checkpoint_path)
    summarizer_2 = FakeSummarizer()
    docs_extended = docs + [("ctx_3", "text 3")]
    result_2 = enrich_documents(summarizer_2, docs_extended, checkpoint_2, enabled=True)
 
    assert summarizer_2.calls == 1  # только ctx_3, ctx_1/ctx_2 из чекпоинта
    assert result_2["ctx_1"] == result_1["ctx_1"]
    assert result_2["ctx_2"] == result_1["ctx_2"]
    assert "ctx_3" in result_2
 
 
def test_checkpoint_load_done_empty_when_file_missing(tmp_path):
    checkpoint = EnrichmentCheckpoint(tmp_path / "does_not_exist.jsonl")
    assert checkpoint.load_done() == {}

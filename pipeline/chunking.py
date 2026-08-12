"""
Модуль 2 — Chunking / сериализация таблиц.
 
Тривиален по архитектурному решению (развилка 4): один документ = один чанк.
Таблицы уже сериализованы авторами датасета как markdown pipe-таблицы прямо
внутри DocumentRecord.context — см. docs/specifikatsiya_moduley.md, модуль 2,
раздел «Закрыто (12.08.2026)» для конкретных примеров по каждому источнику.
Дополнительная сериализация не производится.
"""
 
from __future__ import annotations
 
from pipeline.ingestion import DocumentRecord
 
 
def chunk(records: list[DocumentRecord]) -> list[DocumentRecord]:
    """No-op по архитектурному решению — DocumentRecord уже является готовым
    чанком. Функция существует как явная точка пайплайна (модуль 2), а не
    убрана совсем, чтобы порядок шагов из specifikatsiya_moduley.md был виден
    в коде один в один.
    """
    return list(records)

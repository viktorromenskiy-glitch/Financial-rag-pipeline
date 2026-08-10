# Структура репозитория

Основа — `tehnicheskoe_zadanie.md` и `specifikatsiya_moduley.md`. Имя репозитория, структура папок, конфиг-схема.

## Два разных действия, не путать (замечание внешнего ревью, подтверждено прецедентом Text-to-SQL)

Структура ниже — финальная, целевая. Но создавать репозиторий и начинать сохранять в него файлы — нужно **раньше**, не дожидаясь момента, когда вся структура будет готова:

1. **Сохранение (сейчас, без изысков).** Создать репозиторий `financial-rag-pipeline` на GitHub уже сегодня — пустой README, без сортировки по папкам — и залить туда как есть уже существующие ценные артефакты, которые физически живут только в Colab/Google Drive: рабочие скрипты `test_1_3_*`, `test_2_2_*`, `test_2_3_*`, `test_2_2_pool_*`; `eval_subset_250.parquet` (честная eval-выборка — если потерять, придётся заново стратифицированно пересобирать, не гарантируя точно те же 250 вопросов); `obzor_metodov_finqa_tatqa.md` и прочие рабочие документы. Причина срочности — реальный прецедент проекта Text-to-SQL: обрывы Colab-сессий (зависания, перебои с электричеством) без сохранённых на GitHub промежуточных файлов стоили повторной работы.
2. **Структурирование (неделя 3, по плану ниже).** Реорганизация в чистую структуру `/pipeline`, `/config`, `/notebooks`, `/docs`, `/tests` — после того как все 9 развилок закрыты (уже так и есть), не раньше. Переструктурировать репозиторий несколько раз по ходу экспериментов дороже, чем один раз позже.

**Это действие пользователя, не то, что может выполнить агент за него** — файлы (`test_1_3_*.ipynb`, `eval_subset_250.parquet` и т.д.) физически находятся в Google Drive/Colab, недоступны из этой среды, а создание GitHub-репозитория требует доступа к GitHub-аккаунту. Ниже — целевая структура, к которой нужно прийти после этапа сохранения.

## Имя репозитория

**`financial-rag-pipeline`**

Без привязки к T²-RAGBench в названии — это деталь академического бенчмарка, ничего не говорящая заказчику на Upwork, и риск, что репозиторий читается как прогон эксперимента, а не как рабочая система. Зафиксировано сейчас, не откладывается (урок из проекта Text-to-SQL — переименование на середине стоило времени).

## Структура папок

```
financial-rag-pipeline/
├── pipeline/                      # 9 модулей — см. specifikatsiya_moduley.md
│   ├── __init__.py
│   ├── ingestion.py                # модуль 1
│   ├── chunking.py                 # модуль 2
│   ├── enrichment.py               # модуль 4 (contextual enrichment)
│   ├── embedding.py                # модуль 3 — принимает готовый full_indexed_content,
│   │                                #   не строит его сам (guard на порядок вызовов, см. specifikatsiya_moduley.md)
│   ├── indexing.py                 # модуль 5 (MongoDB Atlas)
│   ├── retrieval.py                # модуль 6 (hybrid $rankFusion)
│   ├── reranking.py                # модуль 7 (Cohere Rerank v4.0 Pro)
│   ├── generation.py               # модуль 8 (Direct, Claude Sonnet 5)
│   ├── evaluation.py               # модуль 9 (LLM Judge Evaluation)
│   └── common/
│       ├── is_close_v2.py          # см. ТЗ п.7 — перенести логику из Colab-скрипта теста 2.4/2.5 буквально
│       ├── retry.py                 # обёртка tenacity, различает transient (429/5xx/timeout) vs non-transient (4xx) — см. ТЗ п.11
│       └── run_config.py            # сборка snapshot конфигурации для run_config.json — см. ТЗ п.11
│
├── config/
│   ├── config.yaml                  # см. схему ниже
│   └── config_schema.py             # Pydantic-модель для валидации config.yaml при старте (типы, допустимые значения pool_size>0 и т.д.)
│
├── data/                             # добавлено — пропущено в первой версии этого документа
│   └── t2-ragbench/
│       └── eval_subset_250.parquet   # честная eval-выборка (250 вопросов, стратифицированно) — источник baseline Recall@5=0.808,
│                                      #   критична для воспроизводимости; при потере пересборка не гарантирует те же вопросы
│
├── notebooks/                       # уже рабочие Colab-скрипты недель 1-2 — референс реализации, переносить логику, не переписывать с нуля.
│   │                                 # Список ниже — иллюстративный, сверить с реальными именами файлов в Google Drive перед заливкой
│   │                                 # (см. открытый вопрос: есть ли отдельные ноутбуки для тестов 2.1, 2.4, 2.5)
│   ├── test_1_3_embedding_comparison.ipynb
│   ├── test_2_2_reranker_v2.ipynb
│   ├── test_2_2_pool_size.ipynb
│   └── test_2_3_contextual_chunks_v2.ipynb
│
├── docs/
│   ├── tehnicheskoe_zadanie.md
│   ├── specifikatsiya_moduley.md
│   ├── struktura_repozitoriya.md     # этот файл
│   ├── plan_podgotovki_k_kodirovaniyu.md
│   ├── RAG_arkhitektura_i_tochki_vetvleniya.md
│   └── poshagovyi_plan_vypolneniya.md   # полная хронология тестов недель 1-2, источник всех чисел в ТЗ
│
├── tests/
│   ├── test_is_close_v2.py           # юнит-тесты с конкретными примерами — см. ТЗ п.7
│   │                                  #   ("100" vs "100.0" vs "1.00" как доля, "0.5" vs "50%" и т.п.)
│   ├── test_pipeline_modules.py      # локальные тесты модулей на срезе реальных данных
│   └── test_startup_validation.py    # тест на assert проверки индексов/полей при старте — см. ТЗ п.2
│
├── results/                          # артефакты прогонов, не в .gitignore для ключевых финальных результатов
│   └── <run_id>/
│       ├── run_config.json           # snapshot конфигурации этого прогона — см. ТЗ п.11
│       ├── predictions.jsonl
│       └── eval_report.md            # регрессионный отчёт + классификация ошибок — см. ТЗ п.10
│
├── .env.example                      # MONGODB_URI, VOYAGE_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY — без значений
├── requirements.txt
├── README.md                         # честная таблица метрик (baseline-relative, не абсолютная планка), архитектурная диаграмма, стратификация по source_dataset — см. ТЗ п.10
└── .gitignore                        # .env, __pycache__, крупные промежуточные файлы результатов
```

## Конфиг-файл (`config/config.yaml`)

```yaml
mongodb:
  uri: ${MONGODB_URI}                 # из .env, не хардкод
  db_name: "rag_project"
  collection_name: "t2_ragbench_full"
  vector_index_name: "vector_index_full"
  text_index_name: "text_index_full"

embedding:
  model: "voyage-4"
  batch_size: 32

enrichment:
  enabled: true
  model: "claude-haiku-4.5"
  temperature: 0.0
  prompt_version: "v1"                # часть ключа кэша судьи и run_config.json — см. ТЗ п.8/п.11

retrieval:
  pool_size: 50
  weights:
    vector: 0.5
    text: 0.5

reranker:
  enabled: true
  model: "rerank-v4.0-pro"
  pool_size: 50
  top_n: 5

generation:
  model: "claude-sonnet-5"
  temperature: 0.0

judge:
  model: "claude-sonnet-5"
  temperature: 0.0
  prompt_version: "v1"
  deterministic_check_enabled: true

retry:
  stop_after_attempt: 5
  wait_min_seconds: 1
  wait_max_seconds: 60
  # ретраить только transient-ошибки (429/5xx/timeout), не 4xx — см. ТЗ п.11
```

`config_schema.py` — Pydantic-модель поверх этого YAML: типизация полей, `pool_size: PositiveInt`, `enabled: bool`, assert на непустой `MONGODB_URI` — валидация конфига при старте пайплайна, до первого API-вызова, не постфактум на середине индексации всего корпуса.

## Обязательные некодовые файлы

- **`.env.example`** — перечисляет нужные переменные окружения без значений (`MONGODB_URI=`, `VOYAGE_API_KEY=`, `ANTHROPIC_API_KEY=`, `COHERE_API_KEY=`). Позволяет рецензенту понять, что нужно для запуска, не читая код.
- **`requirements.txt`** — с зафиксированными версиями (`==`, не `>=`) для критичных зависимостей (pymongo, voyageai, anthropic, cohere, tenacity, pydantic) — воспроизводимость окружения, не только кода.
- **README.md** — пишется в неделе 5 (портфолио-упаковка), но скелет создаётся сейчас: разделы «Архитектура», «Метрики» (с явной пометкой baseline-relative, не абсолютная планка), «Как запустить», «Известные ограничения» (открытые риски масштабирования из ТЗ п.5 — переносятся в README как есть, не замалчиваются).

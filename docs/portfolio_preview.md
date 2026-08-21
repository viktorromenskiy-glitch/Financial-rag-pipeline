# Portfolio short preview (Layer 1) — план доработки-2, пункт 6

Продолжение уже начатого в плане-1 пункте 8: там появилась двухуровневая структура README (Business summary → Deep Tech appendix). Этот файл — третий, самый короткий уровень: текст для мест, где даже "Business summary" (4 абзаца) — слишком длинно: открывающий абзац Upwork-заявки, описание репозитория в GitHub "About", социальная карточка при шаринге ссылки.

**Оговорка по источнику.** План-2 предполагал использовать как основу готовый ~250-словный черновик от ChatGPT — но точный текст этого черновика был в части переписки, которая к этому пункту уже сжалась (context compaction) и не сохранилась дословно. Пересказывать по памяти то, чего не вижу перед собой, значило бы нарушить тот же принцип "только проверенная информация", который применяется к остальному проекту — поэтому текст ниже написан заново, из проверенных цифр README (не из ChatGPT-черновика), а не реконструирован по памяти. Если у Виктора сохранился оригинальный черновик ChatGPT — его можно прислать, я сверю и при необходимости обновлю этот файл.

## GitHub "About" description (репозиторий, ограничение ~350 символов)

Установлено в настройках репозитория (Settings → General → About) 2026-08-21, взамен старого текста, оставшегося от исходного README до смены tagline в пункте 3:

> Financial-document RAG (text + tables) evaluated end-to-end on 7,300+ real filings: 76.8% answer accuracy, +11-13pp over published retrieval benchmarks, every decision backed by statistical testing (McNemar, power analysis), not intuition.

239 символов.

## Upwork proposal opener / short pitch (~180 слов)

Для использования как открывающий абзац заявки на Upwork или как отдельный "elevator pitch" текст.

> I built and rigorously evaluated a retrieval-augmented generation (RAG) pipeline for financial document Q&A — the kind of system that answers questions like "what was the change in revenue from 2018 to 2019?" by finding the right passage in a real financial filing (10-K style: mixed narrative text and tables) and returning a precise number, not a paraphrase.
>
> It's tested end-to-end on 7,318 real financial documents and over 23,000 questions — a harder benchmark than the clean-prose demos most RAG portfolios use, because financial answers have to be numerically exact and the source data mixes text with tables. Result: 76.8% answer accuracy (retrieval + reranking + generation + judging), and a retrieval stage that beats the closest published comparison by 11-13 percentage points.
>
> Every design decision — which reranker, how to route embeddings per document type, whether contextual enrichment actually helps — is backed by a real statistical test (paired McNemar, power analysis), not intuition or a single demo run. Full methodology, honest limitations, and reproducible results: [GitHub link].
>
> This is a research/evaluation-grade pipeline, not a packaged product — no UI or serving API yet — but the architecture, the measurement discipline, and the numbers are real and reproducible, and I'm glad to walk through any part of it or adapt the approach to your own document set.

**Честность масштаба.** Последний абзац сознательно оставляет то же ограничение, что и в README ("Scope"/"Known limitations") — research/evaluation-grade, не packaged product, нет UI/serving API — специально, чтобы короткий питч не создавал ожиданий, которые полный README потом опровергает. Это тот же принцип, что уже применялся в плане-1 пункте 8 (Business summary "what it is/isn't").

## Перевод на русский (для справки Виктору, не для публикации — целевая аудитория Upwork/GitHub англоязычная)

> Я построил и тщательно оценил RAG-пайплайн (retrieval-augmented generation) для ответов на вопросы по финансовым документам — систему, которая отвечает на вопросы вроде «как изменилась выручка с 2018 по 2019 год?», находя нужный фрагмент в реальном финансовом отчёте (в стиле 10-K: смешанный текст и таблицы) и возвращая точное число, а не пересказ.
>
> Протестирован end-to-end на 7318 реальных финансовых документах и более чем 23 000 вопросах — более сложный бенчмарк, чем чистый текст-проза, которую используют большинство RAG-портфолио, потому что финансовые ответы должны быть численно точными, а исходные данные смешивают текст с таблицами. Результат: 76.8% точности ответов (retrieval + reranking + generation + judging), и этап retrieval превосходит ближайшее опубликованное сравнение на 11-13 процентных пунктов.
>
> Каждое архитектурное решение — какой reranker, как маршрутизировать embedding-модели по типу документа, действительно ли помогает контекстное обогащение — подкреплено реальным статистическим тестом (парный McNemar, анализ мощности), а не интуицией или одним демо-прогоном. Полная методология, честные ограничения и воспроизводимые результаты: [ссылка на GitHub].
>
> Это пайплайн исследовательского/оценочного уровня, не готовый продукт — пока нет UI или serving API — но архитектура, дисциплина измерения и цифры реальны и воспроизводимы, и я готов пройтись по любой части или адаптировать подход под ваш набор документов.

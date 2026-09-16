"""Диагностика CUAD over-refusal, раунд 2 - продолжение
scripts/run_cuad_overrefusal_diagnostic.py (раунд 1, результат сохранён в
claude/status_agent_rezultaty_4_nahodki_kod.md и на Drive под run_id
"cuad_overrefusal_diagnostic").

*** ВАЖНО: как и раунд 1, это диагностический/качественный скрипт, НЕ
*** статистический эксперимент. n=5 вопросов x 2 варианта = 10 вызовов
*** assessor'а не дают статистической мощности. Цель - проверить гипотезу
*** 2 количественно и наконец получить текстовое рассуждение модели,
*** которое раунд 1 не мог получить структурно (см. ниже).

## Что показал раунд 1 (25 вызовов, 5 вопросов x 5 вариантов, все
## sufficient=no)

1. Гипотеза 1 (доменная формулировка "financial report" вместо "legal
   contract") - ОПРОВЕРГНУТА: варианты `neutral` и `explicit_legal` дали
   тот же результат (0/5), что и `original`. Замена одной фразы домена
   вердикт не меняет.
2. Гипотеза 3 (assessor не уверен в ПОЛНОТЕ, а не в релевантности) -
   ОПРОВЕРГНУТА: `original_with_gold_hint` (модели прямо названа ФОРМА
   верного ответа) тоже дал 0/5 - assessor не признаёт уже присутствующий
   факт нужной формы, даже когда прямо сказано, что искать.
3. Гипотеза 2 (структурная несовместимость с multi-document) - НЕ
   проверена количественно раундом 1: вариант `multi_document_instruction`
   только добавлял инструкцию быть ОСТОРОЖНЕЕ (не мог повысить долю "yes"),
   а не устранял сам источник путаницы - наличие 3 заведомо нерелевантных
   документов из 4 в контексте КАЖДОГО вопроса (реальный размер CUAD
   смок-корпуса - всего 4 документа, retrieval с pool_size=50 возвращает
   все 4 при любом запросе - см. claude/verifikaciya_context_ids_cuad_smoke.md).
4. Побочная находка, важнее самих счётчиков: raw_response во всех 25
   записях раунда 1 не содержит НИКАКОГО рассуждения - только
   "SUFFICIENT: no\\nREFORMULATED QUERY: ...". Это не баг логирования - сам
   _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py) прямо требует "Respond in
   exactly this format, nothing else". План раунда 1 ("прочитать
   raw_response, чтобы понять рассуждение") был невыполним при этом
   промпте. Зафиксировано в своде правил (svod_pravil_raboty.md,
   "Перед запуском НОВОГО платного/диагностического скрипта - проверка
   design-vs-обещание двумя независимыми внешними экспертами") как повод
   для нового обязательного шага.

## Два новых варианта этого раунда (каждый меняет РОВНО ОДНУ переменную
## относительно `original`, независимо друг от друга - как и в раунде 1)

- `single_relevant_document_only`: промпт `original` БЕЗ ИЗМЕНЕНИЙ (та же
  константа _ASSESSMENT_PROMPT_TEMPLATE, ни одного слова другой), но в
  {context} подаётся ТОЛЬКО тот один документ из 4, который реально
  содержит ответ на этот вопрос (определяется из
  pipeline.cuad_smoke.load_cuad_smoke_fixture() - каждому question_id там
  соответствует ровно один document_id). Три посторонних документа не
  включаются вовсе - не "помечаются как нерелевантные", а физически
  отсутствуют в контексте. Проверяет гипотезу 2 напрямую: если
  sufficient=yes появляется здесь, а в `original` (те же 4 документа) было
  no - причина в шуме от посторонних документов, а не в формулировке
  вопроса или домене.
- `with_reasoning`: тот же `original` (тот же контекст - все 4 документа,
  та же доменная формулировка "financial report" - домен здесь НЕ
  переменная, её уже проверил раунд 1), но с одной добавленной инструкцией
  дать одну строку рассуждения ПЕРЕД вердиктом. Единственная цель -
  получить наконец текстовую улику вместо голого "no", раз раунд 1 не мог
  её получить структурно. _parse_assessment() (agent/loop.py) уже
  устойчив к тексту до финальных SUFFICIENT/REFORMULATED QUERY строк
  (берёт ПОСЛЕДНЕЕ совпадение каждого маркера - см. его докстринг), так
  что добавление одной строки рассуждения перед этими строками не требует
  никаких изменений в парсинге.

## Почему отдельный скрипт, а не новые записи в VARIANTS раунда 1

run_cuad_overrefusal_diagnostic.py делает ОДИН retrieval-вызов на вопрос и
переиспользует один и тот же context_text для всех вариантов промпта
(нужно для честного ablation - "меняется ТОЛЬКО формулировка"). Вариант
`single_relevant_document_only` этого раунда, наоборот, специально меняет
САМ КОНТЕКСТ (не только формулировку) - технически несовместимо с этим
инвариантом раунда 1. Смешивать оба случая в одном VARIANTS-словаре с
общим циклом означало бы либо сломать инвариант раунда 1 для всех его
вариантов, либо городить ветвление, которое легко перепутать. Отдельный
скрипт с собственным (более коротким) циклом - меньше риска сломать уже
провалидированный раунд 1.

## Переиспользование того же индекса, что и раунд 1 - без исключений

Как и раунд 1: никакого нового ingestion/embedding/enrichment - тот же
принцип и тот же _check_already_indexed(), тот же search_fn через
agent.tools.search_documents против уже существующей коллекции
cuad_smoke_v1.

## Стоимость

5 вопросов x 2 варианта = 10 вызовов assessor'а (claude-sonnet-5, тот же
generation.model). При cost_per_llm_call_usd=0.02 - ориентировочно ~$0.20
+ 5 дополнительных retrieval/rerank вызовов (Voyage/Cohere, на порядок
дешевле; переиспользуются те же 5, что и раунд 1, поскольку тот же
question набор и тот же search_fn - НЕ 10 отдельных retrieval вызовов,
один на вопрос, как и в раунде 1).

## Перед запуском на реальные деньги

По новому правилу свода - этот скрипт должен быть показан ДВУМ
независимым внешним экспертам с вопросом "способен ли он реально доставить
заявленное, есть ли противоречие между целью и механизмом" ДО первого
платного запуска. Не запускать, пока это не сделано.

Использование (Colab, после git pull, после того как
scripts/run_cuad_smoke.py уже был прогнан хотя бы раз на этом MongoDB):
    !python scripts/run_cuad_overrefusal_diagnostic_round2.py
Прогон идемпотентен (resume by question_id+variant), как раунд 1.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import functools  # noqa: E402

from agent.loop import _ASSESSMENT_PROMPT_TEMPLATE, _parse_assessment  # noqa: E402
from agent.tools import search_documents  # noqa: E402
from config.config_schema import load_config  # noqa: E402
from pipeline.cli import ClaudeGenerator, build_clients  # noqa: E402
from pipeline.cuad_smoke import load_cuad_smoke_fixture  # noqa: E402
from pipeline.common.persist import find_canonical_root, verify_run_files  # noqa: E402
from pipeline.generation import build_context_block  # noqa: E402
from pipeline.indexing import is_indexed, validate_startup_indexes  # noqa: E402

RUN_ID = "cuad_overrefusal_diagnostic_round2"
CONFIG_PATH = REPO_ROOT / "config" / "config_cuad_smoke.yaml"

# Анкор для вставки инструкции про рассуждение - взят дословно из
# _ASSESSMENT_PROMPT_TEMPLATE (agent/loop.py), не перепечатан вручную, по
# той же причине, что в раунде 1: гарантированный assert-провал, если
# константа изменится без обновления этого скрипта, вместо тихого
# несовпадения.
_FORMAT_ANCHOR = (
    "Respond in exactly this format, nothing else:\n\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)
_FORMAT_WITH_REASONING = (
    "Before your verdict, on a line starting with \"REASONING:\", state in one sentence "
    "which single fact the question needs is present in, or missing from, the passages above. "
    "Then, on the following lines, respond in exactly this format:\n\n"
    "REASONING: <one sentence>\n"
    "SUFFICIENT: <yes or no>\n"
    "REFORMULATED QUERY: <a new search query, or NONE>"
)

_WITH_REASONING_TEMPLATE = _ASSESSMENT_PROMPT_TEMPLATE.replace(_FORMAT_ANCHOR, _FORMAT_WITH_REASONING)
if _WITH_REASONING_TEMPLATE == _ASSESSMENT_PROMPT_TEMPLATE:
    raise RuntimeError(
        "Ablation anchor _FORMAT_ANCHOR not found in current _ASSESSMENT_PROMPT_TEMPLATE - the "
        "prompt constant changed since this script was written; update _FORMAT_ANCHOR in "
        "scripts/run_cuad_overrefusal_diagnostic_round2.py to match before running."
    )

# ОГОВОРКА ПРИ ИНТЕРПРЕТАЦИИ (подтверждена двумя независимыми экспертами
# перед запуском - см. prompt_dlya_2_ekspertov_round2_script.md): просьба
# дать одну строку рассуждения ПЕРЕД вердиктом - это chain-of-thought
# elicitation, которая может изменить само решение модели, а не только
# сделать уже принятое решение видимым (широко задокументированный эффект).
# Если with_reasoning даст sufficient=yes там, где original/раунд 1 давал
# no - это НЕЛЬЗЯ автоматически трактовать как "original просто не видел
# факт". Возможная альтернатива: сама просьба рассуждать изменила процесс
# принятия решения. Единственная заявленная цель этого варианта - получить
# ТЕКСТ рассуждения (raw_response), а не чисто проверить гипотезу через
# факт наличия/отсутствия рассуждения - при анализе результатов читать
# именно raw_response/REASONING:, не только бинарный sufficient. Чистая
# проверка "меняет ли само рассуждение вердикт" - отдельный, более дорогой
# эксперимент (тот же контекст, original vs with_reasoning вперемешку,
# большее n), не то, что делает этот скрипт.

# Единственный вариант, который переиспользует original template БЕЗ
# ИЗМЕНЕНИЙ - только контекст, который он получает, отличается (собирается
# отдельно в main(), не через этот словарь). Оставлен здесь просто для
# единообразия итерации/отчёта, значение (template) не используется для
# single_relevant_document_only - см. main().
VARIANTS: dict[str, str] = {
    "single_relevant_document_only": _ASSESSMENT_PROMPT_TEMPLATE,
    "with_reasoning": _WITH_REASONING_TEMPLATE,
}


# Дополнительная, независимая от agent/loop.py._parse_assessment проверка
# (найдена вторым независимым экспертом при проверке этого скрипта перед
# запуском - см. prompt_dlya_2_ekspertov_round2_script.md): _parse_assessment
# берёт ПОСЛЕДНЕЕ совпадение маркеров в ответе - что верно, ЕСЛИ модель в
# конце концов действительно выдаёт настоящие финальные строки
# "SUFFICIENT: .../REFORMULATED QUERY: ...". with_reasoning просит модель
# сначала порассуждать - если модель случайно упомянёт "...is SUFFICIENT:
# yes, but..." ВНУТРИ рассуждения и НИКОГДА не выдаст отдельную финальную
# строку с настоящим вердиктом, _parse_assessment тихо примет это случайное
# упоминание за реальный вердикт - без краша, без предупреждения. Не меняет
# сам _parse_assessment (общий, используемый в проде код) - вместо этого
# независимо проверяет, что маркеры присутствуют как ОТДЕЛЬНЫЕ строки (не
# внутри предложения), и добавляет предупреждение в сохраняемую запись,
# если это не так - чтобы такая запись не была тихо доверена наравне с
# остальными при последующем чтении результатов.
_STRICT_SUFFICIENT_LINE_RE = re.compile(r"^\s*SUFFICIENT\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_STRICT_REFORMULATED_LINE_RE = re.compile(r"^\s*REFORMULATED QUERY\s*:", re.IGNORECASE | re.MULTILINE)


def _check_parse_reliability(raw_response: str) -> str | None:
    """Returns a human-readable warning if raw_response does not contain
    both markers as their own standalone lines (as the prompt instructs),
    or None if it looks like a clean, trustworthy final answer."""
    problems = []
    if not _STRICT_SUFFICIENT_LINE_RE.search(raw_response):
        problems.append("no standalone 'SUFFICIENT: yes/no' line found")
    if not _STRICT_REFORMULATED_LINE_RE.search(raw_response):
        problems.append("no standalone 'REFORMULATED QUERY: ...' line found")
    if not problems:
        return None
    return (
        "Parse reliability check failed (" + "; ".join(problems) + ") - the parsed "
        "'sufficient'/'reformulated_query' fields for this record may reflect a stray "
        "mention inside reasoning text rather than the model's real final verdict. "
        "Read raw_response manually before trusting this record."
    )


def _verify_relevant_doc_mapping(records, eval_items) -> None:
    """Belt-and-suspenders check (second independent expert's flagged
    concern, prompt_dlya_2_ekspertov_round2_script.md): relevant_doc_id_by_question
    is built by zipping records/eval_items positionally, which is correct
    BY CONSTRUCTION (pipeline/cuad_smoke.py's load_cuad_smoke_fixture()
    appends to both lists inside a single loop over the same question, no
    separate sort afterwards) - verified by reading that source directly,
    and empirically (manually, offline, gold_answer confirmed present in
    the assigned document's text for all 5 questions in the current
    fixture). This function turns that one-off manual check into a
    permanent runtime guard, so a FUTURE edit to the fixture loader (e.g.
    someone adding a sort) fails loudly here instead of silently feeding
    single_relevant_document_only the wrong document. Uses a short prefix
    of gold_answer (not the whole string) since some gold answers are full
    sentences that could have been retokenized/whitespace-normalized
    relative to the raw document text - a prefix match is enough to catch
    a genuine document_id mismatch without being fragile to that.
    """
    problems = []
    for rec, item in zip(records, eval_items):
        gold = item["gold_answer"].strip()
        probe = gold if len(gold) <= 40 else gold[:40]
        if probe.upper() not in rec.context.upper():
            problems.append(
                f"question_id={item['question_id']!r}: gold_answer prefix {probe!r} not found in "
                f"document {rec.context_id!r} - relevant_doc_id_by_question would be wrong for this "
                f"question"
            )
    if problems:
        raise RuntimeError(
            "relevant_doc_id_by_question mapping failed verification - "
            "single_relevant_document_only would test against the wrong document(s):\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def _check_already_indexed(collection, records) -> None:
    # Тот же баг, что был найден и исправлен в раунде 1 - DocumentRecord
    # это dataclass (атрибут .context_id), не dict. Не повторять r["context_id"].
    missing = sorted({r.context_id for r in records if not is_indexed(collection, r.context_id)})
    if missing:
        raise RuntimeError(
            "CUAD smoke collection is missing document(s) this diagnostic needs to reuse "
            f"as-is: {missing}. Run scripts/run_cuad_smoke.py once first, then re-run this script."
        )


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[f"{rec['question_id']}::{rec['variant']}"] = rec
    return done


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    clients = build_clients(config)
    collection = clients["collection"]

    records, eval_items = load_cuad_smoke_fixture()
    _check_already_indexed(collection, records)
    validate_startup_indexes(collection, check_source_dataset_filter=False)

    # records и eval_items построены в ОДНОМ цикле по одному и тому же
    # списку вопросов (pipeline/cuad_smoke.py) - один DocumentRecord на
    # вопрос, в том же порядке. relevant_doc_id_by_question сопоставляет
    # question_id -> id единственного документа, который реально отвечает
    # на этот вопрос (не через эвристику - это то же самое сопоставление,
    # что зашито в исходном фикстурном JSON).
    assert len(records) == len(eval_items), (
        "records и eval_items должны быть одной длины и в одном порядке - "
        "load_cuad_smoke_fixture() строит их в одном цикле по вопросам"
    )
    _verify_relevant_doc_mapping(records, eval_items)
    relevant_doc_id_by_question = {
        item["question_id"]: rec.context_id for rec, item in zip(records, eval_items)
    }

    print(
        f"Reusing already-indexed cuad_smoke_v1 collection - {len(eval_items)} question(s), "
        f"{len(VARIANTS)} new variant(s) each = {len(eval_items) * len(VARIANTS)} assessor call(s) total "
        f"(round 2, separate from round 1's 25 calls)."
    )

    drive_root = find_canonical_root(config.persistence.google_drive_results_dir)
    run_dir = drive_root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "diagnostic_results.jsonl"

    done = _load_checkpoint(out_path)
    if done:
        print(f"Resuming: {len(done)} (question, variant) pair(s) already done in {out_path}")

    generator = ClaudeGenerator(clients["anthropic"], config.generation.model, config.generation.temperature)
    search_fn = functools.partial(
        search_documents,
        clients["voyage"],
        collection,
        clients["cohere"],
        pool_size=config.retrieval.pool_size,
        vector_weight=config.retrieval.weights.vector,
        text_weight=config.retrieval.weights.text,
        reranker_enabled=config.reranker.enabled,
        reranker_top_n=config.reranker.top_n,
        embedding_model=config.embedding.model,
        source_dataset=None,
        exclude_source_datasets=None,
    )

    t_start = time.perf_counter()
    n_calls = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in eval_items:
            question_id = item["question_id"]
            question = item["question"]
            gold_answer = item["gold_answer"]
            relevant_doc_id = relevant_doc_id_by_question[question_id]

            # Один retrieval на вопрос, как в раунде 1 - переиспользуется
            # для построения ОБОИХ вариантов контекста ниже (полный и
            # отфильтрованный), не два отдельных вызова.
            call = search_fn(question)
            all_candidates = list(call.candidates)

            for variant, template in VARIANTS.items():
                key = f"{question_id}::{variant}"
                if key in done:
                    continue

                if variant == "single_relevant_document_only":
                    candidates = [c for c in all_candidates if c.context_id == relevant_doc_id]
                    if not candidates:
                        raise RuntimeError(
                            f"question_id={question_id!r}: relevant document {relevant_doc_id!r} "
                            f"not found among retrieved candidates "
                            f"({[c.context_id for c in all_candidates]!r}) - retrieval regression, "
                            f"not a prompt/context question this diagnostic can answer. Stopping "
                            f"rather than silently testing against the wrong document set."
                        )
                    context_text = build_context_block(candidates)
                else:
                    candidates = all_candidates
                    context_text = build_context_block(candidates)

                context_ids = sorted({c.context_id for c in candidates})
                calls_remaining = config.agent.max_additional_tool_calls
                prompt = template.format(question=question, context=context_text, calls_remaining=calls_remaining)

                raw_response = generator.generate(prompt)
                n_calls += 1
                parsed = _parse_assessment(raw_response)
                parse_warning = _check_parse_reliability(raw_response)

                result = {
                    "question_id": question_id,
                    "variant": variant,
                    "gold_answer": gold_answer,
                    "context_ids": context_ids,
                    "sufficient": parsed.sufficient,
                    "reformulated_query": parsed.reformulated_query,
                    "raw_response": raw_response,
                    "parse_warning": parse_warning,
                }
                if parse_warning:
                    print(f"      WARNING: {parse_warning}")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = result
                print(f"  [{n_calls}] {question_id} / {variant} -> sufficient={parsed.sufficient}")

    elapsed = time.perf_counter() - t_start
    print(f"\nГотово за {elapsed:.0f}с, {n_calls} новых вызовов assessor'а в этой сессии.")

    print("\nСводка sufficient=yes по вариантам (ориентир, НЕ статистический вывод при n=5):")
    for variant in VARIANTS:
        rows = [r for r in done.values() if r["variant"] == variant]
        n_yes = sum(1 for r in rows if r["sufficient"])
        print(f"  {variant:32s}: {n_yes}/{len(rows)} sufficient=yes")

    verify_run_files(run_dir, {"diagnostic_results.jsonl": len(eval_items) * len(VARIANTS)})
    print(f"\n{'=' * 70}\nSAVED TO PERSISTENT STORAGE: {run_dir.resolve()}\n{'=' * 70}\n")
    print(
        "single_relevant_document_only sufficient=yes здесь при sufficient=no в original раунда 1 "
        "(тот же вопрос, те же 4 документа в контексте) - прямая количественная улика за гипотезу 2 "
        "(шум от посторонних документов). with_reasoning's raw_response теперь содержит текст "
        "рассуждения - читать его перед любым выводом, не только счётчик sufficient=yes."
    )


if __name__ == "__main__":
    main()

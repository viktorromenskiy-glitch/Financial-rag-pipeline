"""Module 9 - LLM Judge Evaluation.

Renamed from "RAGAS evaluation" following the 2026-08-09 review - this
module does not use the RAGAS library or its metrics (faithfulness,
answer relevancy, context precision); it only uses a custom Claude judge
plus the deterministic is_close_v2 check. See
docs/specifikatsiya_moduley.md, module 9.

Judge model: Claude Sonnet 5, the sole judge - a second independent judge
(GPT-4o-mini) was evaluated and found unnecessary (93.3%, 28/30 agreement
with the deterministic check on 30 questions; both disagreements were the
same explainable edge case, not a scattered error source). See
docs/tehnicheskoe_zadanie.md, section 8.

JUDGE_PROMPT below is transplanted from the validated test 2.5 Colab cell
(experiments_weeks_1_2.ipynb) that produced the 93.3% agreement figure -
not rewritten from a description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from pipeline.common.is_close_v2 import is_close_v2
from pipeline.common.retry import retryable

MODEL = "claude-sonnet-5"
TEMPERATURE = 0.0
PROMPT_VERSION = "v1"
DETERMINISTIC_CHECK_ENABLED = True

JUDGE_PROMPT = """You are evaluating whether a generated answer to a financial question is correct, given the ground truth answer.

Question: {question}
Generated answer: {generated}
Ground truth answer: {gold}

Consider the answer correct if it matches the ground truth value, allowing for minor rounding, sign-convention differences (e.g. -60 vs 60 if direction is ambiguous), or equivalent expression as percentage vs fraction (e.g. 1.5 vs 0.015). Respond with exactly one word: CORRECT or INCORRECT."""


@dataclass(frozen=True)
class EvalResult:
    question_id: str
    judge_scores: dict
    deterministic_match: bool
    judge_agrees: bool


class JudgeProtocol(Protocol):
    """Abstraction over the Claude API - allows a fake judge to be
    substituted in tests without a real network call."""

    def judge(self, prompt: str) -> str: ...


def cache_key(question: str, context: str, answer: str, prompt_version: str = PROMPT_VERSION) -> str:
    """hash(question + context + answer + judge_prompt_version).

    The prompt version must be part of the key (tehnicheskoe_zadanie.md,
    section 8): otherwise changing the judge prompt silently returns
    stale scores computed under the old prompt, indistinguishable from
    "nothing changed".
    """
    raw = "\x1f".join([question, context, answer, prompt_version])
    return sha256(raw.encode("utf-8")).hexdigest()


class JudgeCache:
    """External cache of judge results, keyed by cache_key() (JSON Lines,
    one line per cached result) - a repeated run must not re-judge
    questions whose (question, context, answer, prompt_version) hash is
    unchanged (spec section 9, "Кэширование"; tehnicheskoe_zadanie.md,
    section 8), and a mid-run failure should not lose already-judged
    results. Mirrors module 4's EnrichmentCheckpoint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        cache: dict[str, dict] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["cache_key"]] = rec
        return cache

    def append(self, key: str, result: EvalResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "cache_key": key,
                        "question_id": result.question_id,
                        "judge_scores": result.judge_scores,
                        "deterministic_match": result.deterministic_match,
                        "judge_agrees": result.judge_agrees,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


@retryable()
def _judge_with_retry(judge: JudgeProtocol, prompt: str) -> str:
    return judge.judge(prompt)


def evaluate_answer(
    judge: JudgeProtocol,
    question_id: str,
    question: str,
    context: str,
    generated_answer: str,
    gold_answer: str,
    prompt_version: str = PROMPT_VERSION,
    deterministic_check_enabled: bool = DETERMINISTIC_CHECK_ENABLED,
) -> EvalResult:
    """Evaluate a single question - no caching (see evaluate_answers for
    the batch entry point with caching, mirroring module 4's
    enrich_document/enrich_documents split).

    context: the retrieved context actually used for generation (module
    6/7 candidates' full_indexed_content), part of the cache key so a
    retrieval/reranking change also invalidates stale judge results.
    """
    deterministic_match = (
        is_close_v2(generated_answer, gold_answer) if deterministic_check_enabled else False
    )

    prompt = JUDGE_PROMPT.format(question=question, generated=generated_answer, gold=gold_answer)
    raw_verdict = _judge_with_retry(judge, prompt)
    verdict = raw_verdict.strip().upper()
    judge_correct = "CORRECT" in verdict and "INCORRECT" not in verdict

    # Agreement is only a meaningful signal when the deterministic check
    # actually ran; with the check disabled there is nothing to disagree
    # with, so this is vacuously True rather than a measured comparison.
    judge_agrees = (judge_correct == deterministic_match) if deterministic_check_enabled else True

    judge_scores = {"verdict": verdict, "judge_correct": judge_correct}

    return EvalResult(
        question_id=question_id,
        judge_scores=judge_scores,
        deterministic_match=deterministic_match,
        judge_agrees=judge_agrees,
    )


def evaluate_answers(
    judge: JudgeProtocol,
    items: list[dict],
    cache: JudgeCache | None = None,
    prompt_version: str = PROMPT_VERSION,
    deterministic_check_enabled: bool = DETERMINISTIC_CHECK_ENABLED,
) -> list[EvalResult]:
    """items: list of {question_id, question, context, generated_answer,
    gold_answer} dicts (module 8 output joined with retrieved context and
    the dataset's gold answer).

    Loads the cache once up front (not per item - O(n), not O(n^2)) and
    skips re-judging any item whose cache key is already present.
    """
    cached = cache.load() if cache is not None else {}
    results: list[EvalResult] = []
    for item in items:
        key = cache_key(item["question"], item["context"], item["generated_answer"], prompt_version)
        hit = cached.get(key)
        if hit is not None:
            results.append(
                EvalResult(
                    question_id=item["question_id"],
                    judge_scores=hit["judge_scores"],
                    deterministic_match=hit["deterministic_match"],
                    judge_agrees=hit["judge_agrees"],
                )
            )
            continue

        result = evaluate_answer(
            judge,
            item["question_id"],
            item["question"],
            item["context"],
            item["generated_answer"],
            item["gold_answer"],
            prompt_version=prompt_version,
            deterministic_check_enabled=deterministic_check_enabled,
        )
        if cache is not None:
            cache.append(key, result)
        results.append(result)
    return results


def regression_report(previous: list[EvalResult], current: list[EvalResult]) -> dict[str, list[str]]:
    """Per-question improved/regressed/unchanged classification between
    two evaluation runs, keyed by question_id - the mandatory reporting
    output required by spec section 9 ("не только агрегированные
    метрики"). This is the same comparison that has repeatedly caught
    real bugs hidden behind a stable or improved aggregate metric in this
    project (tehnicheskoe_zadanie.md, section 10 - e.g. the bge-reranker
    case: 13 fixed but 44 broken, aggregate looked neutral).

    Questions present in `current` but not in `previous` are skipped (no
    prior result to compare against, not a regression signal).
    """
    prev_correct = {r.question_id: r.judge_scores["judge_correct"] for r in previous}
    report: dict[str, list[str]] = {
        "improved": [],
        "regressed": [],
        "unchanged_correct": [],
        "unchanged_incorrect": [],
    }
    for r in current:
        was_correct = prev_correct.get(r.question_id)
        if was_correct is None:
            continue
        is_correct = r.judge_scores["judge_correct"]
        if not was_correct and is_correct:
            report["improved"].append(r.question_id)
        elif was_correct and not is_correct:
            report["regressed"].append(r.question_id)
        elif is_correct:
            report["unchanged_correct"].append(r.question_id)
        else:
            report["unchanged_incorrect"].append(r.question_id)
    return report

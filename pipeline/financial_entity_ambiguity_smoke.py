"""Loader for the narrow financial entity-ambiguity verification fixture.

See data/financial_entity_ambiguity/build_fixture.py for how
data/financial_entity_ambiguity/fixture.json was built (selection criteria,
distractor logic, anonymization) and
claude/itog_ekspertizy_cuad_overrefusal_fix.md ("what's left to do", item 2)
for why this fixture exists at all - the converged 4-expert design for
checking whether removing the company name from a real financial question
(while document-level metadata_prefix stays in place) meaningfully reduces
`sufficient=yes`, using a manually-assembled (non-retrieval) context of the
gold document + 3 same-Sector distractor documents.

This module's only job is loading and shape-validating the fixture JSON -
same separation as pipeline.cuad_smoke.load_cuad_smoke_fixture(), which this
mirrors. It does not touch MongoDB/Voyage/Cohere/Anthropic and costs
nothing to call.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "financial_entity_ambiguity" / "fixture.json"
)

_REQUIRED_KEYS = {
    "question_id",
    "source_dataset",
    "question_original",
    "question_anonymized",
    "gold_answer",
    "gold_context_id",
    "gold_company_name",
    "gold_company_name_humanized",
    "gold_sector",
    "gold_report_year",
    "distractor_context_ids",
    "distractor_company_names",
    "distractor_report_years",
}


def load_financial_entity_ambiguity_fixture(path: str | Path = DEFAULT_FIXTURE_PATH) -> list[dict]:
    """Loads and shape-validates the fixture.

    Raises:
        FileNotFoundError: if `path` does not exist (run
            data/financial_entity_ambiguity/build_fixture.py first).
        ValueError: if an item is missing a required key, has other than
            3 distractors, or question_original == question_anonymized
            (anonymization silently did nothing - build_fixture.py already
            guards this at build time, but a hand-edited fixture file
            would not have been re-validated, so check again here).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Financial entity-ambiguity fixture not found: {path}. Run "
            f"data/financial_entity_ambiguity/build_fixture.py first."
        )
    items = json.loads(path.read_text(encoding="utf-8"))
    if not items:
        raise ValueError(f"Fixture at {path} is empty")

    for item in items:
        missing = _REQUIRED_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"Fixture item {item.get('question_id', '?')!r} missing keys: {sorted(missing)}")
        if len(item["distractor_context_ids"]) != 3:
            raise ValueError(
                f"Fixture item {item['question_id']!r} has "
                f"{len(item['distractor_context_ids'])} distractor(s), expected exactly 3"
            )
        if item["question_original"] == item["question_anonymized"]:
            raise ValueError(
                f"Fixture item {item['question_id']!r}: question_original == question_anonymized - "
                f"anonymization did nothing for this item"
            )
        if item["gold_context_id"] in item["distractor_context_ids"]:
            raise ValueError(
                f"Fixture item {item['question_id']!r}: gold_context_id appears in its own "
                f"distractor_context_ids"
            )

    question_ids = [item["question_id"] for item in items]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Fixture at {path} has duplicate question_id values")

    return items

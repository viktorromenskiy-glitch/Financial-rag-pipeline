"""Pydantic schema and loader for config/config.yaml.

Validates the pipeline configuration once, at startup, before the first
external API call - a typo or missing field should fail immediately with
a clear error, not surface as a confusing partial failure hours into a
full-corpus run. See docs/struktura_repozitoriya.md, "Конфиг-файл
(config/config.yaml)".
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveInt, model_validator

_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class MongoDBConfig(BaseModel):
    uri: str
    db_name: str
    collection_name: str
    vector_index_name: str
    text_index_name: str

    @model_validator(mode="after")
    def uri_not_empty(self) -> "MongoDBConfig":
        if not self.uri:
            raise ValueError("mongodb.uri is empty - check that MONGODB_URI is set in .env")
        return self


class EmbeddingRoutingConfig(BaseModel):
    """Per-dataset embedding routing (docs/tehnicheskoe_zadanie.md, п.3a,
    2026-08-15) - added after a direct A/B test (McNemar's exact test,
    n=2500/source_dataset, full 7318-document corpus) found voyage-finance-2
    measurably helps TAT-DQA retrieval but hurts ConvFinQA and gives no
    reliable benefit on FinQA, contradicting the uniform-improvement
    prediction from three independent AI-consultant reviews. enabled=false
    makes every document/query use embedding.model regardless of
    source_dataset (routed_sources is ignored) - the config-level on/off
    switch this project's convention requires for every architectural
    choice (docs/struktura_repozitoriya.md).
    """

    enabled: bool
    finance_model: str
    routed_sources: list[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: PositiveInt
    routing: EmbeddingRoutingConfig


class EnrichmentConfig(BaseModel):
    enabled: bool
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    prompt_version: str


class RetrievalWeights(BaseModel):
    vector: float = Field(ge=0.0, le=1.0)
    text: float = Field(ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    pool_size: PositiveInt
    weights: RetrievalWeights


class RerankerConfig(BaseModel):
    enabled: bool
    model: str
    pool_size: PositiveInt
    top_n: PositiveInt


class GenerationConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    # Фаза 5 (docs/tehnicheskoe_zadanie.md, section 28): selects a key of
    # pipeline.generation.PROMPT_TEMPLATE_VARIANTS. Defaults to
    # "baseline" (the production PROMPT_TEMPLATE, unchanged) so existing
    # config.yaml files and every prior saved run keep working without
    # this field being set. cmd_eval in pipeline/cli.py validates the
    # value against the actual variant dict at run time - kept as a
    # plain str here (not a Literal) so a new variant added to
    # generation.py doesn't require a schema change here too.
    prompt_variant: str = "baseline"


class JudgeConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    prompt_version: str
    deterministic_check_enabled: bool


class RetryConfig(BaseModel):
    stop_after_attempt: PositiveInt
    wait_min_seconds: PositiveInt
    wait_max_seconds: PositiveInt

    @model_validator(mode="after")
    def wait_max_not_below_min(self) -> "RetryConfig":
        if self.wait_max_seconds < self.wait_min_seconds:
            raise ValueError(
                f"retry.wait_max_seconds ({self.wait_max_seconds}) must be >= "
                f"retry.wait_min_seconds ({self.wait_min_seconds})"
            )
        return self


class PersistenceConfig(BaseModel):
    # "Правила сохранения долгих платных прогонов" (project doc,
    # 2026-08-24): THE canonical persistent-storage root, set once here,
    # not retyped in any Colab cell/script. pipeline/common/persist.py's
    # find_canonical_root()/save_run_to_drive() read it from here.
    # Exact casing matters - "RAG-project" (capital RAG, lowercase
    # "project") matches the folder already created on Drive
    # (notebooks/experiments_weeks_1_2.ipynb) - a mismatched case here is
    # precisely the bug that rule exists to prevent.
    google_drive_results_dir: str = "/content/drive/MyDrive/RAG-project/results"


class PipelineConfig(BaseModel):
    mongodb: MongoDBConfig
    embedding: EmbeddingConfig
    enrichment: EnrichmentConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    judge: JudgeConfig
    retry: RetryConfig
    persistence: PersistenceConfig = PersistenceConfig()


def _substitute_env_vars(value: Any) -> Any:
    """Recursively replaces "${VAR_NAME}" string values with the
    corresponding environment variable. Only whole-string matches are
    substituted (matching config.yaml's "uri: ${MONGODB_URI}" style) -
    partial/embedded ${...} inside a longer string is left as-is.

    Raises:
        ValueError: if a referenced environment variable is not set - a
            missing secret should fail loudly here, not surface later as
            an empty connection string deep in a MongoDB client error.
    """
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value)
        if match:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if not env_value:
                raise ValueError(
                    f"Config references ${{{var_name}}}, but the {var_name} environment "
                    f"variable is not set - see .env.example"
                )
            return env_value
    return value


def load_config(path: str | Path = "config/config.yaml") -> PipelineConfig:
    """Loads config.yaml, substitutes ${ENV_VAR} placeholders, and
    validates the result against PipelineConfig.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if a referenced environment variable is not set.
        pydantic.ValidationError: if the config does not match the schema
            (wrong type, missing field, pool_size <= 0, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = _substitute_env_vars(raw)
    return PipelineConfig.model_validate(resolved)

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
    """MongoDB Atlas connection and index configuration.

    Attributes:
        uri: MongoDB connection string (typically resolved from the
            MONGODB_URI environment variable via config.yaml's
            "${MONGODB_URI}" placeholder).
        db_name: Name of the MongoDB database holding the pipeline's collection.
        collection_name: Name of the collection storing indexed DocumentRecord chunks.
        vector_index_name: Name of the Atlas Vector Search index used for dense retrieval.
        text_index_name: Name of the Atlas Search (text) index used for lexical retrieval.
    """

    uri: str
    db_name: str
    collection_name: str
    vector_index_name: str
    text_index_name: str

    @model_validator(mode="after")
    def uri_not_empty(self) -> "MongoDBConfig":
        """Validates that `uri` was actually resolved to a non-empty value.

        Raises:
            ValueError: If `uri` is empty (e.g. MONGODB_URI was not set
                before ${MONGODB_URI} substitution).
        """
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

    Attributes:
        enabled: Turns per-dataset routing on/off - see above for the
            full on/off semantics.
        finance_model: Name of the finance-specialized embedding model
            (e.g. "voyage-finance-2") used for documents/queries from
            `routed_sources` when `enabled` is True.
        routed_sources: The source_dataset values that should use
            `finance_model` instead of `embedding.model`; ignored
            entirely when `enabled` is False.
    """

    enabled: bool
    finance_model: str
    routed_sources: list[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    """Embedding model and batching configuration for ingestion/retrieval.

    Attributes:
        model: Default embedding model used for all documents/queries.
        batch_size: Number of texts embedded per API call.
        routing: Optional per-dataset override that routes some documents
            to a different, finance-specialized embedding model instead
            of `model` - see EmbeddingRoutingConfig.
    """

    model: str
    batch_size: PositiveInt
    routing: EmbeddingRoutingConfig


class EnrichmentConfig(BaseModel):
    """LLM-based document enrichment configuration, applied during ingestion.

    Attributes:
        enabled: Whether enrichment runs at all; when False, ingestion
            skips it entirely.
        model: Name of the LLM used to enrich documents.
        temperature: Sampling temperature passed to the enrichment model.
        prompt_version: Identifier of the enrichment prompt template
            version in use, so a wording change can be traced against
            the results it produced.
    """

    enabled: bool
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    prompt_version: str


class RetrievalWeights(BaseModel):
    """Relative weighting between vector and text scores in hybrid retrieval.

    Attributes:
        vector: Weight given to the dense/vector search score.
        text: Weight given to the lexical/text search score.
    """

    vector: float = Field(ge=0.0, le=1.0)
    text: float = Field(ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    """Hybrid retrieval configuration.

    Attributes:
        pool_size: Number of candidate documents retrieved before
            reranking/truncation.
        weights: Relative weighting between vector and text search
            scores - see RetrievalWeights.
    """

    pool_size: PositiveInt
    weights: RetrievalWeights


class RerankerConfig(BaseModel):
    """Cross-encoder reranking configuration applied after initial retrieval.

    Attributes:
        enabled: Whether reranking runs at all; when False, the retrieval
            pool is used as-is (truncated to top_n) with no reranking pass.
        model: Name of the reranker model.
        pool_size: Number of retrieval candidates passed into the reranker.
        top_n: Number of top-ranked documents kept after reranking, for
            generation.
    """

    enabled: bool
    model: str
    pool_size: PositiveInt
    top_n: PositiveInt


class GenerationConfig(BaseModel):
    """Answer-generation model configuration.

    Attributes:
        model: Name of the LLM used to generate the final answer.
        temperature: Sampling temperature passed to the generation model.
        prompt_variant: Key selecting which generation prompt template is
            used - see the inline comment on this field for what it
            selects and the backward-compatibility rationale behind its
            default.
    """

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
    """LLM-judge configuration for scoring generated answers against gold answers.

    Attributes:
        model: Name of the LLM used as the judge.
        temperature: Sampling temperature passed to the judge model.
        prompt_version: Identifier of the judge prompt template version in use.
        deterministic_check_enabled: Whether the deterministic numeric
            check (is_close_v2) also runs alongside the LLM judge verdict.
    """

    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    prompt_version: str
    deterministic_check_enabled: bool


class RetryConfig(BaseModel):
    """Retry policy for external API calls (embedding/generation/judge/etc.).

    Attributes:
        stop_after_attempt: Maximum number of attempts before giving up.
        wait_min_seconds: Minimum backoff wait between attempts.
        wait_max_seconds: Maximum backoff wait between attempts; must be
            >= wait_min_seconds (enforced by wait_max_not_below_min below).
    """

    stop_after_attempt: PositiveInt
    wait_min_seconds: PositiveInt
    wait_max_seconds: PositiveInt

    @model_validator(mode="after")
    def wait_max_not_below_min(self) -> "RetryConfig":
        """Validates that the configured backoff bounds are internally consistent.

        Raises:
            ValueError: If `wait_max_seconds` is less than `wait_min_seconds`.
        """
        if self.wait_max_seconds < self.wait_min_seconds:
            raise ValueError(
                f"retry.wait_max_seconds ({self.wait_max_seconds}) must be >= "
                f"retry.wait_min_seconds ({self.wait_min_seconds})"
            )
        return self


class AgentConfig(BaseModel):
    """Bounded agentic tool-use loop for agent/ (see itog_ekspertizy_agent_profil.md,
    the 4-expert design review of this extension - consensus item 3).

    max_additional_tool_calls caps how many EXTRA search_documents calls
    the agent may make after its mandatory first search, before it must
    stop and answer (or give up - see agent/loop.py's forced-insufficient
    policy). Deliberately a config value, not a constant in agent/loop.py -
    all four independent experts explicitly rejected a hardcoded limit.
    0 is a valid value (disables re-querying entirely - the agent still
    runs its one mandatory search and evidence assessment, but never
    reformulates).

    max_wall_clock_seconds (День 2, plan_rabot_posle_ekspertizy_agent_profil.md,
    "Глобальные предохранители: максимум токенов / wall-clock / стоимость
    на прогон"): an optional per-QUESTION safety limit passed straight
    through to agent.loop.run_agent_query's max_wall_clock_seconds param -
    see that function's docstring for exactly when it is checked. None
    (the default) disables it entirely, so every config file written
    before this field existed keeps loading and behaving unchanged. This
    is distinct from AgentEvalConfig below, which caps a whole evaluation
    RUN (many questions), not one question.

    enable_deictic_entity_guard: gates agent.loop.run_agent_query's
    deictic/entity guard (see claude/itog_ekspertizy_cuad_overrefusal_fix.md,
    "Что осталось сделать", item 1, and agent/loop.py's
    _deictic_entity_guard_should_block docstring for the full rationale
    and the empirical false-positive check behind it). Default False -
    same backward-compatible pattern as max_wall_clock_seconds above:
    every config file written before this field existed keeps loading and
    behaving exactly as before. Set to True explicitly in a config file to
    turn the guard on for that pipeline (e.g. config_cuad_smoke.yaml, where
    it was validated).
    """

    max_additional_tool_calls: int = Field(ge=0)
    max_wall_clock_seconds: float | None = Field(default=None, ge=0)
    enable_deictic_entity_guard: bool = Field(default=False)


class AgentEvalConfig(BaseModel):
    """Global safety limits for a whole Day 2 agent-evaluation harness run
    (scripts/run_agent_eval.py) - plan_rabot_posle_ekspertizy_agent_profil.md,
    День 2: "Глобальные предохранители: максимум токенов / wall-clock /
    стоимость на прогон" plus "Зафиксировать заранее бюджет
    evaluation-прогона (число LLM-вызовов × стоимость)".

    Distinct from AgentConfig.max_wall_clock_seconds above, which caps a
    single question's loop, not the whole run - see agent/safety.py's
    RunBudgetTracker, which these fields configure.

    Every field is optional/None-disables-it and defaults are set here
    (not left unset) so `python scripts/run_agent_eval.py` has a sane,
    pre-committed budget out of the box rather than requiring the person
    running it to first go compute one - matching the "fix the budget in
    advance" requirement rather than leaving it to be decided ad hoc
    during a live paid run. Sizing rationale for the defaults below
    (n=20-30 questions, config.agent.max_additional_tool_calls=2):
    worst case per agent question is 1 mandatory + 2 additional
    assessments (3) + 1 final-answer generation = 4 agent LLM calls, plus
    1 baseline generation call, plus up to 2 judge calls (the normal
    correctness judge, and - only on a forced/INSUFFICIENT_CONTEXT answer -
    agent.success's insufficiency judge) = up to 7/question; at n=30 that
    is ~210 calls, so max_llm_calls=400 leaves comfortable headroom
    without being effectively unlimited. cost_per_llm_call_usd=0.02 is a
    deliberately conservative flat estimate (this pipeline's real
    per-question judged-eval cost has been documented elsewhere in this
    project at roughly $0.015-0.016/question for a full retrieve+rerank+
    generate+judge pass - $0.02/call is conservative per CALL, not per
    question) - real per-call cost varies by prompt/response length and
    is not tracked exactly here (see agent/safety.py's module docstring
    on why this counts calls, not tokens). max_estimated_cost_usd=10.0 is
    the corresponding hard budget ceiling (estimated as max_llm_calls *
    cost_per_llm_call_usd internally by RunBudgetTracker) - at the
    defaults above, 400 * $0.02 = $8, comfortably under the $10.0 ceiling,
    so max_llm_calls is expected to be the tighter of the two limits in
    practice.

    Attributes:
        max_llm_calls: Maximum total LLM calls (agent + baseline + judge)
            across the whole run before it is stopped; None disables the
            limit.
        max_wall_clock_seconds: Maximum wall-clock time for the whole run;
            None disables the limit.
        max_estimated_cost_usd: Maximum estimated total cost for the whole
            run (see sizing rationale above); None disables the limit.
        cost_per_llm_call_usd: Flat per-call cost estimate used to compute
            the running estimated cost against max_estimated_cost_usd.
    """

    max_llm_calls: int | None = Field(default=400, ge=1)
    max_wall_clock_seconds: float | None = Field(default=5400, ge=0)
    max_estimated_cost_usd: float | None = Field(default=10.0, ge=0)
    cost_per_llm_call_usd: float = Field(default=0.02, ge=0)


class PersistenceConfig(BaseModel):
    """Persistent-storage configuration for long, paid pipeline runs.

    Attributes:
        google_drive_results_dir: Canonical Google Drive results directory
            - see the comment on this field for why its exact casing
            matters and where it is consumed.
    """

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
    """Root pydantic schema for config/config.yaml - the full validated
    pipeline configuration returned by load_config().

    Attributes:
        mongodb: MongoDB Atlas connection and index settings - see
            MongoDBConfig.
        embedding: Embedding model, batching, and per-dataset routing
            settings - see EmbeddingConfig.
        enrichment: LLM-based document enrichment settings - see
            EnrichmentConfig.
        retrieval: Hybrid retrieval pool size and vector/text weighting -
            see RetrievalConfig.
        reranker: Cross-encoder reranking settings - see RerankerConfig.
        generation: Answer-generation model settings - see GenerationConfig.
        judge: LLM-judge scoring settings - see JudgeConfig.
        retry: Retry policy for external API calls - see RetryConfig.
        persistence: Persistent-storage settings for long paid runs;
            defaults to PersistenceConfig() so config files predating
            this field keep loading unchanged - see PersistenceConfig.
        agent: Bounded agentic tool-use loop settings for agent/; defaults
            to the 4-expert consensus value (max_additional_tool_calls=2)
            so config files predating agent/ keep loading unchanged - see
            AgentConfig.
        agent_eval: Global safety limits for a whole agent-evaluation
            harness run - see AgentEvalConfig.
    """

    mongodb: MongoDBConfig
    embedding: EmbeddingConfig
    enrichment: EnrichmentConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    judge: JudgeConfig
    retry: RetryConfig
    persistence: PersistenceConfig = PersistenceConfig()
    # Defaults to 2 (the 4-expert consensus value) so config_cite_and_check.yaml
    # and config_formula_base.yaml - written before agent/ existed - keep
    # loading unchanged, same convention as persistence's default above.
    agent: AgentConfig = Field(default_factory=lambda: AgentConfig(max_additional_tool_calls=2))
    # День 2 - see AgentEvalConfig's docstring for the default values'
    # rationale. Same "defaults so old config files keep loading
    # unchanged" convention as `agent` above.
    agent_eval: AgentEvalConfig = Field(default_factory=AgentEvalConfig)


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

    Args:
        path: Path to the YAML config file to load.

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

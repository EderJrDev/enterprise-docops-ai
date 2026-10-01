"""Configuração central carregada de variáveis de ambiente.

Todas as chaves sensíveis usam ``SecretStr``. Valores em branco no ``.env``
viram ``None``, para que um template copiado não seja tratado como segredo.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentName = Literal["development", "staging", "production"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LLMProviderName = Literal["openai", "anthropic", "ollama"]
EmbeddingProviderName = Literal["openai", "local", "hashing"]
RerankerProviderName = Literal["cohere", "local"]
FusionStrategyName = Literal["rrf", "dbsf"]
ChunkThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
MCPTransportName = Literal["stdio", "sse", "streamable-http"]

_SECRET_FIELDS: tuple[str, ...] = (
    "openai_api_key",
    "anthropic_api_key",
    "cohere_api_key",
    "qdrant_api_key",
    "langfuse_public_key",
    "langfuse_secret_key",
)


class Settings(BaseSettings):
    """Settings de runtime do Enterprise DocOps AI.

    O processo lê primeiro as variáveis de ambiente e, na ausência delas,
    o arquivo ``.env`` no diretório de trabalho. Variáveis desconhecidas
    são ignoradas.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Aplicação ---
    app_name: str = "Enterprise DocOps AI"
    app_env: EnvironmentName = "development"
    log_level: LogLevelName = "INFO"
    app_version: str = "0.1.0"
    data_dir: str = "data"
    sample_docs_dir: str = "data/sample_docs"
    eval_dataset_path: str = "data/eval_dataset.json"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=32)
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )
    api_rate_limit_requests: int = Field(default=60, ge=1)
    api_rate_limit_window_seconds: int = Field(default=60, ge=1)
    api_request_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- LLM e fallback ---
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_primary_provider: LLMProviderName = "openai"
    llm_fallback_provider: LLMProviderName = "anthropic"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Embeddings ---
    embedding_provider: EmbeddingProviderName = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=32, le=4096)
    local_embedding_model: str = "BAAI/bge-m3"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str | None = None
    qdrant_grpc_port: int = Field(default=6334, ge=1, le=65535)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "docops_documents"
    qdrant_prefer_grpc: bool = False
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0)
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "bm25"

    # --- Retrieval híbrido ---
    chunk_size: int = Field(default=800, ge=128, le=8000)
    chunk_overlap: int = Field(default=120, ge=0)
    semantic_breakpoint_type: ChunkThresholdType = "percentile"
    semantic_breakpoint_amount: float = Field(default=95.0, gt=0)
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    hybrid_fusion: FusionStrategyName = "rrf"
    hybrid_bm25_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    hybrid_dense_weight: float = Field(default=0.65, ge=0.0, le=1.0)

    # --- Rerank ---
    reranker_provider: RerankerProviderName = "cohere"
    cohere_api_key: SecretStr | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    local_reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Observabilidade ---
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # --- MCP ---
    mcp_host: str = "0.0.0.0"
    mcp_port: int = Field(default=8081, ge=1, le=65535)
    mcp_transport: MCPTransportName = "streamable-http"

    # --- Guardrails ---
    guardrails_enabled: bool = True
    pii_masking_enabled: bool = True
    prompt_injection_detection_enabled: bool = True
    max_input_chars: int = Field(default=8000, ge=256, le=100_000)
    spacy_model: str = "pt_core_news_sm"
    hitl_enabled: bool = True

    @field_validator(*_SECRET_FIELDS, mode="before")
    @classmethod
    def blank_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    @field_validator("openai_base_url", "qdrant_path", mode="before")
    @classmethod
    def blank_url_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")
        if self.rerank_top_n > self.retrieval_top_k:
            raise ValueError("RERANK_TOP_N não pode ser maior que RETRIEVAL_TOP_K")
        weight_total = self.hybrid_bm25_weight + self.hybrid_dense_weight
        if abs(weight_total - 1.0) > 0.001:
            raise ValueError(
                "HYBRID_BM25_WEIGHT e HYBRID_DENSE_WEIGHT devem somar 1.0"
            )
        if self.llm_primary_provider == self.llm_fallback_provider:
            raise ValueError(
                "LLM_FALLBACK_PROVIDER deve ser diferente de LLM_PRIMARY_PROVIDER"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def reveal(self, secret: SecretStr | None) -> str | None:
        """Devolve o valor de um segredo apenas no ponto de uso."""
        if secret is None:
            return None
        return secret.get_secret_value()

    def missing_runtime_secrets(self) -> list[str]:
        """Segredos ausentes para o provedor primário configurado.

        Não é chamado na importação: ambientes de teste e o serviço Qdrant
        sobem sem chave de LLM.
        """
        missing: list[str] = []
        if self.llm_primary_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        if self.llm_primary_provider == "anthropic" and self.anthropic_api_key is None:
            missing.append("ANTHROPIC_API_KEY")
        if self.llm_fallback_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        if self.llm_fallback_provider == "anthropic" and self.anthropic_api_key is None:
            missing.append("ANTHROPIC_API_KEY")
        if self.embedding_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        if self.reranker_provider == "cohere" and self.cohere_api_key is None:
            missing.append("COHERE_API_KEY")
        if self.langfuse_enabled and (
            self.langfuse_public_key is None or self.langfuse_secret_key is None
        ):
            missing.append("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY")
        return list(dict.fromkeys(missing))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância única de configuração do processo."""
    return Settings()

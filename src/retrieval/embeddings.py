"""Provedores de embedding e documento BM25 usados na ingestão e na busca."""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from qdrant_client.http import models as rest

from src.config import Settings

logger = structlog.get_logger(__name__)

_EMBED_BATCH_SIZE = 64
_BM25_MODEL = "Qdrant/bm25"


class EmbeddingProvider(Protocol):
    """Contrato assíncrono para vetores densos de documentos e consultas."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Retorna um vetor por texto, na mesma ordem."""

    async def embed_query(self, text: str) -> list[float]:
        """Retorna o vetor de uma consulta."""

    async def vector_size(self) -> int:
        """Dimensão do vetor denso gravado na coleção."""

    async def aclose(self) -> None:
        """Libera conexões do provedor."""


def bm25_document(text: str) -> rest.Document:
    """Monta o documento esparso com tokenização em português.

    O Qdrant calcula o BM25 no servidor (modelo ``Qdrant/bm25``) e aplica IDF
    na coleção. As opções precisam ser as mesmas na ingestão e na consulta.
    """
    return rest.Document(
        text=text,
        model=_BM25_MODEL,
        options=rest.Bm25Config(
            language="portuguese",
            tokenizer=rest.TokenizerType.WORD,
            lowercase=True,
            ascii_folding=True,
            stopwords=rest.Language.PORTUGUESE,
            stemmer=rest.SnowballParams(
                type=rest.Snowball.SNOWBALL,
                language=rest.SnowballLanguage.PORTUGUESE,
            ),
        ),
    )


def _retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


class OpenAIEmbeddingProvider:
    """Embeddings via API compatível com OpenAI, com lote e retentativa."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def vector_size(self) -> int:
        return self._settings.embedding_dimensions

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        api_key = self._settings.reveal(self._settings.openai_api_key)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY ausente para gerar embeddings")

        base_url = (self._settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        attempts = self._settings.llm_max_retries + 1
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=0.4, max=4),
            retry=retry_if_exception(_retryable_http_error),
            reraise=True,
        ):
            with attempt:
                response = await self._http.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": self._settings.embedding_model,
                        "input": texts,
                        "dimensions": self._settings.embedding_dimensions,
                    },
                )
                response.raise_for_status()
                payload = response.json()

        rows = sorted(payload["data"], key=lambda item: item["index"])
        if len(rows) != len(texts):
            raise RuntimeError("A API de embeddings devolveu uma quantidade inesperada de vetores")
        return [list(row["embedding"]) for row in rows]


class LocalEmbeddingProvider:
    """Fallback local com Sentence Transformers, carregado só no primeiro uso."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._encoder: object | None = None
        self._size: int | None = None
        self._lock = asyncio.Lock()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = await self._load()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            encoded = await asyncio.to_thread(self._encode, encoder, batch)
            vectors.extend(encoded)
        if self._size is None and vectors:
            self._size = len(vectors[0])
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def vector_size(self) -> int:
        if self._size is None:
            probe = await self.embed_query("dimensao")
            self._size = len(probe)
        if self._size != self._settings.embedding_dimensions:
            logger.warning(
                "local_embedding_dimension_differs",
                model=self._settings.local_embedding_model,
                actual=self._size,
                configured=self._settings.embedding_dimensions,
            )
        return self._size

    async def aclose(self) -> None:
        return None

    async def _load(self) -> object:
        if self._encoder is not None:
            return self._encoder
        async with self._lock:
            if self._encoder is None:
                model_name = self._settings.local_embedding_model

                def _build() -> object:
                    from sentence_transformers import SentenceTransformer

                    return SentenceTransformer(model_name)

                self._encoder = await asyncio.to_thread(_build)
        return self._encoder

    @staticmethod
    def _encode(encoder: object, texts: list[str]) -> list[list[float]]:
        encoded = encoder.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return encoded.tolist()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Escolhe o provedor denso conforme ``EMBEDDING_PROVIDER``."""
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings)
    return OpenAIEmbeddingProvider(settings)

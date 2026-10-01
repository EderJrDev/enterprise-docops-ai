"""Provedores de embedding e documento BM25 usados na ingestão e na busca."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
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
_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    {
        "apos",
        "ate",
        "com",
        "das",
        "dos",
        "esta",
        "este",
        "nao",
        "para",
        "pela",
        "pelo",
        "por",
        "que",
        "sao",
        "seu",
        "sua",
        "uma",
    }
)


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


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, sem stopwords curtas. A mesma função indexa e consulta."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in _TOKEN.findall(folded) if token not in _STOPWORDS]


def local_sparse_vector(text: str) -> rest.SparseVector:
    """TF saturado. O modificador IDF da coleção completa o BM25 no Qdrant local."""
    counts: dict[int, float] = {}
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "little")
        counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        counts[0] = 1.0
    indices = sorted(counts)
    values = [counts[index] * 2.2 / (counts[index] + 1.2) for index in indices]
    return rest.SparseVector(indices=indices, values=values)


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


class HashingEmbeddingProvider:
    """Encoder denso local, sem download e sem chave.

    Cada token cai num balde de uma tabela fixa. Frases com o mesmo vocabulário
    ficam próximas. Em produção o provedor troca para OpenAI ou ``BAAI/bge-m3``
    sem mudar a ingestão nem a busca.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("O encoder local precisa de pelo menos 32 dimensões")
        self._dimensions = dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self._dimensions) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _hash_embed(text, self._dimensions)

    async def vector_size(self) -> int:
        return self._dimensions

    async def aclose(self) -> None:
        return None


def _hash_embed(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        vector[0] = 1.0
        return vector
    for token in tokens:
        _accumulate(vector, token, 1.0)
    for left, right in zip(tokens, tokens[1:], strict=False):
        _accumulate(vector, f"{left}_{right}", 0.5)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _accumulate(vector: list[float], key: str, weight: float) -> None:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[bucket] += sign * weight


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Escolhe o provedor denso conforme ``EMBEDDING_PROVIDER``."""
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(settings.embedding_dimensions)
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings)
    return OpenAIEmbeddingProvider(settings)

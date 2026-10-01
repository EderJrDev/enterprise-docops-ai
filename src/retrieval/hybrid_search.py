"""Busca híbrida no Qdrant (denso + BM25) com fusão ponderada e rerank."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Protocol

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from src.config import FusionStrategyName, Settings, get_settings
from src.retrieval.embeddings import (
    EmbeddingProvider,
    bm25_document,
    build_embedding_provider,
)
from src.retrieval.ingester import _build_qdrant_client, _qdrant_timeout

logger = structlog.get_logger(__name__)

_RRF_K = 60
_PAYLOAD_KEYS = ("text", "source", "chunk_index")


class RetrievalError(RuntimeError):
    """Falha operacional na busca ou no rerank."""


@dataclass(frozen=True)
class RetrievedChunk:
    """Trecho recuperado, com o score da fusão e, quando houver, o do rerank."""

    id: str
    text: str
    score: float
    source: str
    chunk_index: int
    metadata: dict[str, str | int | float | bool]
    rerank_score: float | None = None


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Reordena os candidatos e devolve no máximo ``top_n``."""

    async def aclose(self) -> None:
        """Libera o cliente do reranker."""


def fuse_rankings(
    rankings: list[list[RetrievedChunk]],
    weights: list[float],
    strategy: FusionStrategyName,
    limit: int,
) -> list[RetrievedChunk]:
    """Combina listas ranqueadas com RRF ponderado ou DBSF ponderado."""
    if len(rankings) != len(weights):
        raise ValueError("A quantidade de rankings e de pesos precisa ser a mesma")
    if limit < 1:
        raise ValueError("limit precisa ser positivo")
    populated = [(ranking, weight) for ranking, weight in zip(rankings, weights, strict=True) if ranking]
    if not populated:
        return []
    if len(populated) == 1:
        return populated[0][0][:limit]
    active_rankings = [ranking for ranking, _ in populated]
    active_weights = [weight for _, weight in populated]
    if strategy == "dbsf":
        return _weighted_dbsf(active_rankings, active_weights, limit)
    return _weighted_rrf(active_rankings, active_weights, limit)


class CohereReranker:
    """Rerank via Cohere v2. O cliente é criado na primeira chamada."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: object | None = None

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        import cohere

        api_key = self._settings.reveal(self._settings.cohere_api_key)
        if not api_key:
            raise RetrievalError("COHERE_API_KEY ausente para o rerank")
        if self._client is None:
            self._client = cohere.AsyncClientV2(
                api_key=api_key,
                timeout=self._settings.llm_timeout_seconds,
                client_name="enterprise-docops",
            )
        response = await self._client.rerank(  # type: ignore[attr-defined]
            model=self._settings.cohere_rerank_model,
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=top_n,
        )
        ranked: list[RetrievedChunk] = []
        for item in response.results:
            base = chunks[item.index]
            ranked.append(replace(base, rerank_score=float(item.relevance_score)))
        return ranked

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)  # type: ignore[attr-defined]


class LocalReranker:
    """Cross-encoder local. O modelo só é baixado na primeira consulta."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._lock = asyncio.Lock()

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = await self._load()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = await asyncio.to_thread(model.predict, pairs)  # type: ignore[attr-defined]
        order = sorted(range(len(chunks)), key=lambda index: float(scores[index]), reverse=True)
        return [
            replace(chunks[index], rerank_score=float(scores[index]))
            for index in order[:top_n]
        ]

    async def aclose(self) -> None:
        return None

    async def _load(self) -> object:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                model_name = self._settings.local_reranker_model

                def _build() -> object:
                    from sentence_transformers import CrossEncoder

                    return CrossEncoder(model_name)

                self._model = await asyncio.to_thread(_build)
        return self._model


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "local":
        return LocalReranker(settings)
    return CohereReranker(settings)


class HybridSearcher:
    """Consulta o Qdrant em paralelo e reordena os candidatos.

    A fusão roda no processo, com os pesos ``HYBRID_DENSE_WEIGHT`` e
    ``HYBRID_BM25_WEIGHT``. O RRF nativo do Qdrant não recebe esses pesos.
    As duas buscas partem juntas, então a latência é a do ramo mais lento.
    Se o rerank falhar, a ordem da fusão é devolvida mesmo assim.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._owns_embedder = embedder is None
        self._owns_reranker = reranker is None
        self._client = client or _build_qdrant_client(self._settings)
        self._embedder = embedder or build_embedding_provider(self._settings)
        self._reranker = reranker

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        candidate_k: int | None = None,
        source: str | None = None,
    ) -> list[RetrievedChunk]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("A consulta não pode ser vazia")
        if len(cleaned) > self._settings.max_input_chars:
            raise ValueError(
                f"A consulta excede MAX_INPUT_CHARS ({self._settings.max_input_chars})"
            )

        final_limit = limit or self._settings.rerank_top_n
        candidates = candidate_k or self._settings.retrieval_top_k
        if final_limit < 1 or candidates < 1:
            raise ValueError("limit e candidate_k precisam ser positivos")
        candidates = max(candidates, final_limit)

        started = asyncio.get_running_loop().time()
        dense_vector = await self._embedder.embed_query(cleaned)
        query_filter = _source_filter(source)
        dense_result, sparse_result = await asyncio.gather(
            self._query_dense(dense_vector, candidates, query_filter),
            self._query_bm25(cleaned, candidates, query_filter),
            return_exceptions=True,
        )
        rankings, weights = _successful_branches(
            dense_result,
            sparse_result,
            self._settings.hybrid_dense_weight,
            self._settings.hybrid_bm25_weight,
        )
        fused = fuse_rankings(
            rankings,
            weights,
            self._settings.hybrid_fusion,
            candidates,
        )
        reranked = await self._rerank(cleaned, fused, final_limit)
        logger.info(
            "hybrid_search_completed",
            candidates=len(fused),
            returned=len(reranked),
            fusion=self._settings.hybrid_fusion,
            elapsed_ms=round((asyncio.get_running_loop().time() - started) * 1000, 1),
        )
        return reranked

    async def aclose(self) -> None:
        if self._owns_reranker and self._reranker is not None:
            await self._reranker.aclose()
        if self._owns_embedder:
            await self._embedder.aclose()
        if self._owns_client:
            await self._client.close()

    async def _query_dense(
        self,
        vector: list[float],
        limit: int,
        query_filter: rest.Filter | None,
    ) -> list[RetrievedChunk]:
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=vector,
            using=self._settings.qdrant_dense_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            timeout=_qdrant_timeout(self._settings),
        )
        return _chunks_from_points(response.points)

    async def _query_bm25(
        self,
        query: str,
        limit: int,
        query_filter: rest.Filter | None,
    ) -> list[RetrievedChunk]:
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=bm25_document(query),
            using=self._settings.qdrant_sparse_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            timeout=_qdrant_timeout(self._settings),
        )
        return _chunks_from_points(response.points)

    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        reranker = self._reranker
        if reranker is None:
            reranker = build_reranker(self._settings)
            self._reranker = reranker
        try:
            return await reranker.rerank(query, chunks, limit)
        except Exception as exc:
            logger.warning("rerank_failed", error=str(exc), provider=self._settings.reranker_provider)
            return chunks[:limit]


def _successful_branches(
    dense: list[RetrievedChunk] | BaseException,
    sparse: list[RetrievedChunk] | BaseException,
    dense_weight: float,
    sparse_weight: float,
) -> tuple[list[list[RetrievedChunk]], list[float]]:
    rankings: list[list[RetrievedChunk]] = []
    weights: list[float] = []
    failures: list[BaseException] = []
    for result, weight, branch in (
        (dense, dense_weight, "dense"),
        (sparse, sparse_weight, "bm25"),
    ):
        if isinstance(result, BaseException):
            failures.append(result)
            logger.warning("search_branch_failed", branch=branch, error=str(result))
            continue
        rankings.append(result)
        weights.append(weight)
    if not rankings:
        raise RetrievalError("A busca híbrida falhou no ramo denso e no BM25") from failures[0]
    return rankings, weights


def _chunks_from_points(points: list[rest.ScoredPoint]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        chunk_index = payload.get("chunk_index", 0)
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in _PAYLOAD_KEYS and isinstance(value, (str, int, float, bool))
        }
        chunks.append(
            RetrievedChunk(
                id=str(point.id),
                text=text,
                score=float(point.score),
                source=str(payload.get("source", "")),
                chunk_index=int(chunk_index) if isinstance(chunk_index, (int, float)) else 0,
                metadata=metadata,
            )
        )
    return chunks


def _source_filter(source: str | None) -> rest.Filter | None:
    if not source:
        return None
    return rest.Filter(
        must=[rest.FieldCondition(key="source", match=rest.MatchValue(value=source))]
    )


def _weighted_rrf(
    rankings: list[list[RetrievedChunk]],
    weights: list[float],
    limit: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chosen: dict[str, RetrievedChunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0:
            continue
        for position, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (_RRF_K + position + 1)
            chosen.setdefault(chunk.id, chunk)
    return _order_by_score(scores, chosen, limit)


def _weighted_dbsf(
    rankings: list[list[RetrievedChunk]],
    weights: list[float],
    limit: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chosen: dict[str, RetrievedChunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0 or not ranking:
            continue
        normalized = _normalize_scores([chunk.score for chunk in ranking])
        for chunk, score in zip(ranking, normalized, strict=True):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight * score
            chosen.setdefault(chunk.id, chunk)
    return _order_by_score(scores, chosen, limit)


def _normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 1:
        return [0.5]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    if variance == 0:
        return [0.5 for _ in scores]
    std = variance**0.5
    low = mean - 3 * std
    high = mean + 3 * std
    span = high - low
    if span == 0:
        return [0.5 for _ in scores]
    return [(score - low) / span for score in scores]


def _order_by_score(
    scores: dict[str, float],
    chosen: dict[str, RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [replace(chosen[point_id], score=score, rerank_score=None) for point_id, score in ordered[:limit]]

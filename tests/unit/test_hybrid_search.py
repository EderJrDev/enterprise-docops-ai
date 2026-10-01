from __future__ import annotations

from qdrant_client.http import models as rest

from src.config import Settings
from src.retrieval.hybrid_search import HybridSearcher, RetrievalError, RetrievedChunk


class _Embedder:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def vector_size(self) -> int:
        return 2

    async def aclose(self) -> None:
        return None


class _Reranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        reversed_chunks = list(reversed(chunks))
        return [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                score=chunk.score,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
                rerank_score=float(index + 1),
            )
            for index, chunk in enumerate(reversed_chunks[:top_n])
        ]

    async def aclose(self) -> None:
        return None


class _Qdrant:
    def __init__(self, *, fail_sparse: bool = False) -> None:
        self.fail_sparse = fail_sparse
        self.calls: list[dict[str, object]] = []

    async def query_points(self, **kwargs: object) -> rest.QueryResponse:
        self.calls.append(kwargs)
        using = kwargs["using"]
        if using == "bm25" and self.fail_sparse:
            raise RuntimeError("bm25 indisponível")
        if using == "dense":
            points = [
                _point("a", 0.9, "política de reembolso"),
                _point("b", 0.4, "sla de suporte"),
            ]
        else:
            points = [
                _point("b", 12.0, "sla de suporte"),
                _point("a", 3.0, "política de reembolso"),
            ]
        return rest.QueryResponse(points=points)


def _point(point_id: str, score: float, text: str) -> rest.ScoredPoint:
    return rest.ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={"text": text, "source": "docs.md", "chunk_index": 0, "title": "Doc"},
    )


def _settings() -> Settings:
    return Settings(
        reranker_provider="local",
        embedding_provider="local",
        retrieval_top_k=5,
        rerank_top_n=1,
        hybrid_bm25_weight=0.5,
        hybrid_dense_weight=0.5,
    )


async def test_search_reranks_fused_candidates() -> None:
    reranker = _Reranker()
    qdrant = _Qdrant()
    searcher = HybridSearcher(
        _settings(),
        client=qdrant,  # type: ignore[arg-type]
        embedder=_Embedder(),
        reranker=reranker,
    )
    results = await searcher.search("qual o prazo de reembolso?", limit=1, candidate_k=5)
    assert len(results) == 1
    assert results[0].rerank_score == 1.0
    assert reranker.queries == ["qual o prazo de reembolso?"]
    assert {call["using"] for call in qdrant.calls} == {"dense", "bm25"}
    assert results[0].metadata["title"] == "Doc"


async def test_search_survives_bm25_failure() -> None:
    searcher = HybridSearcher(
        _settings(),
        client=_Qdrant(fail_sparse=True),  # type: ignore[arg-type]
        embedder=_Embedder(),
        reranker=_Reranker(),
    )
    results = await searcher.search("reembolso", limit=1)
    assert len(results) == 1
    assert results[0].text


async def test_search_fails_when_both_branches_fail() -> None:
    class _Down:
        async def query_points(self, **kwargs: object) -> rest.QueryResponse:
            raise RuntimeError(str(kwargs.get("using")))

    searcher = HybridSearcher(
        _settings(),
        client=_Down(),  # type: ignore[arg-type]
        embedder=_Embedder(),
        reranker=_Reranker(),
    )
    try:
        await searcher.search("reembolso")
    except RetrievalError as exc:
        assert "falhou" in str(exc)
    else:
        raise AssertionError("os dois ramos indisponíveis deveriam falhar")


async def test_search_rejects_blank_query() -> None:
    searcher = HybridSearcher(
        _settings(),
        client=_Qdrant(),  # type: ignore[arg-type]
        embedder=_Embedder(),
        reranker=_Reranker(),
    )
    try:
        await searcher.search("   ")
    except ValueError as exc:
        assert "vazia" in str(exc)
    else:
        raise AssertionError("consulta em branco deveria falhar")

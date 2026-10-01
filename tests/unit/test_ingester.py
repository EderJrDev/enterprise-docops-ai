from __future__ import annotations

from pathlib import Path

from qdrant_client.http import models as rest

from src.config import Settings
from src.retrieval.ingester import DocumentIngester, chunk_point_id


class _Embedder:
    def __init__(self) -> None:
        self.documents: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.append(texts)
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def vector_size(self) -> int:
        return 2

    async def aclose(self) -> None:
        return None


class _Qdrant:
    def __init__(self) -> None:
        self.created = False
        self.deleted_sources: list[str] = []
        self.points: list[rest.PointStruct] = []

    async def collection_exists(self, name: str) -> bool:
        return self.created

    async def create_collection(self, **kwargs: object) -> bool:
        self.created = True
        self.collection = kwargs
        return True

    async def create_payload_index(self, **kwargs: object) -> rest.UpdateResult:
        return rest.UpdateResult(operation_id=1, status=rest.UpdateStatus.COMPLETED)

    async def delete(self, **kwargs: object) -> rest.UpdateResult:
        selector = kwargs["points_selector"]
        assert isinstance(selector, rest.Filter)
        condition = selector.must[0]
        assert isinstance(condition, rest.FieldCondition)
        assert isinstance(condition.match, rest.MatchValue)
        self.deleted_sources.append(str(condition.match.value))
        return rest.UpdateResult(operation_id=2, status=rest.UpdateStatus.COMPLETED)

    async def upsert(self, **kwargs: object) -> rest.UpdateResult:
        points = kwargs["points"]
        assert isinstance(points, list)
        self.points.extend(points)
        return rest.UpdateResult(operation_id=3, status=rest.UpdateStatus.COMPLETED)


def test_point_id_is_stable() -> None:
    first = chunk_point_id("doc.md", 0, "mesmo texto")
    second = chunk_point_id("doc.md", 0, "mesmo texto")
    changed = chunk_point_id("doc.md", 0, "outro texto")
    assert first == second
    assert first != changed


async def test_ingest_text_replaces_source_and_stores_both_vectors() -> None:
    qdrant = _Qdrant()
    settings = Settings(chunk_size=200, chunk_overlap=20)
    ingester = DocumentIngester(
        settings,
        client=qdrant,  # type: ignore[arg-type]
        embedder=_Embedder(),
    )
    result = await ingester.ingest_text(
        "O estorno ocorre em 7 dias. O crédito volta no cartão.",
        source="politica.md",
        metadata={"title": "Reembolso"},
    )
    assert result.chunk_count == 1
    assert result.point_ids
    assert qdrant.deleted_sources == ["politica.md"]
    point = qdrant.points[0]
    assert isinstance(point.vector, dict)
    assert "dense" in point.vector
    assert "bm25" in point.vector
    assert point.payload is not None
    assert point.payload["source"] == "politica.md"
    assert point.payload["title"] == "Reembolso"
    sparse = point.vector["bm25"]
    assert isinstance(sparse, rest.Document)
    assert sparse.model == "Qdrant/bm25"


async def test_ingest_path_reads_markdown(tmp_path: Path) -> None:
    document = tmp_path / "sla.md"
    document.write_text("# SLA\n\nA resposta de severidade 1 ocorre em 15 minutos.", encoding="utf-8")
    qdrant = _Qdrant()
    ingester = DocumentIngester(
        Settings(),
        client=qdrant,  # type: ignore[arg-type]
        embedder=_Embedder(),
    )
    result = await ingester.ingest_path(document)
    assert result.chunk_count >= 1
    assert result.source.endswith("sla.md")
    assert qdrant.points

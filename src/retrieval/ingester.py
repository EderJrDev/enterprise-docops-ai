"""Ingestão de documentos com chunking semântico e índice híbrido no Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import Settings, get_settings
from src.retrieval.embeddings import (
    EmbeddingProvider,
    bm25_document,
    build_embedding_provider,
    local_sparse_vector,
)

logger = structlog.get_logger(__name__)

_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_UPSERT_BATCH_SIZE = 64
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“(\[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")
_RESERVED_PAYLOAD_KEYS = {"text", "source", "chunk_index"}


class IngestionError(RuntimeError):
    """Falha operacional ao ler, fatiar ou indexar um documento."""


@dataclass(frozen=True)
class TextChunk:
    """Trecho pronto para indexação."""

    text: str
    index: int
    source: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestResult:
    """Resumo de uma ingestão idempotente por documento."""

    source: str
    chunk_count: int
    point_ids: tuple[str, ...]


def split_sentences(text: str) -> list[str]:
    """Quebra o texto em sentenças, preservando títulos Markdown como unidades."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", normalized):
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            joined = " ".join(part.strip() for part in buffer if part.strip())
            buffer.clear()
            if joined:
                sentences.extend(_split_punctuation(joined))

        for line in paragraph.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                flush()
                sentences.append(stripped)
                continue
            buffer.append(stripped)
        flush()
    return sentences


def character_windows(text: str, size: int, overlap: int) -> list[str]:
    """Corta um trecho longo em janelas, preferindo o último espaço da janela."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    step_floor = max(1, size - overlap)
    windows: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            space = cleaned.rfind(" ", start + step_floor // 2, end)
            if space > start:
                end = space
        piece = cleaned[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(cleaned):
            break
        next_start = max(end - overlap, start + 1)
        if next_start <= start:
            break
        start = next_start
    return windows


def chunk_point_id(source: str, index: int, text: str) -> str:
    """Identificador estável: a mesma fatia gera o mesmo ponto no Qdrant."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}:{digest}"))


class SemanticChunker:
    """Agrupa sentenças vizinhas enquanto a distância dos embeddings permanece baixa.

    O limiar segue ``SEMANTIC_BREAKPOINT_TYPE``:

    - ``percentile``: ``SEMANTIC_BREAKPOINT_AMOUNT`` é o percentil (0–100).
    - ``standard_deviation``: o limiar é média + quantidade × desvio.
    - ``interquartile``: o limiar é Q3 + quantidade × IQR.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        settings: Settings,
        *,
        buffer_size: int = 1,
    ) -> None:
        if buffer_size < 0:
            raise ValueError("buffer_size não pode ser negativo")
        self._embedder = embedder
        self._settings = settings
        self._buffer_size = buffer_size

    async def split(
        self,
        text: str,
        *,
        source: str,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[TextChunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        groups = await self._semantic_groups(sentences)
        pieces: list[str] = []
        for group in groups:
            joined = " ".join(group).strip()
            pieces.extend(
                character_windows(
                    joined,
                    self._settings.chunk_size,
                    self._settings.chunk_overlap,
                )
            )

        extra = dict(metadata or {})
        return [
            TextChunk(text=piece, index=index, source=source, metadata=extra)
            for index, piece in enumerate(pieces)
            if piece
        ]

    async def _semantic_groups(self, sentences: list[str]) -> list[list[str]]:
        if len(sentences) == 1:
            return [sentences]

        windows = [
            " ".join(
                sentences[max(0, index - self._buffer_size) : index + self._buffer_size + 1]
            )
            for index in range(len(sentences))
        ]
        vectors = await self._embedder.embed_documents(windows)
        if len(vectors) != len(sentences):
            raise IngestionError("O embedder devolveu um número de vetores diferente das sentenças")

        distances = [
            _cosine_distance(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]
        threshold = _breakpoint_threshold(
            distances,
            self._settings.semantic_breakpoint_type,
            self._settings.semantic_breakpoint_amount,
        )
        groups: list[list[str]] = [[sentences[0]]]
        for offset, sentence in enumerate(sentences[1:], start=1):
            if distances[offset - 1] > threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        return groups


class DocumentIngester:
    """Lê Markdown, texto ou PDF, fatia por semântica e grava denso + BM25."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._owns_embedder = embedder is None
        self._client = client or _build_qdrant_client(self._settings)
        self._embedder = embedder or build_embedding_provider(self._settings)
        self._chunker = SemanticChunker(self._embedder, self._settings)

    async def ingest_text(
        self,
        text: str,
        *,
        source: str,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> IngestResult:
        started = time.perf_counter()
        chunks = await self._chunker.split(text, source=source, metadata=metadata)
        await self.ensure_collection()
        await self._delete_source(source)
        point_ids = await self._upsert_chunks(chunks)
        logger.info(
            "document_ingested",
            source=source,
            chunks=len(chunks),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return IngestResult(source=source, chunk_count=len(chunks), point_ids=tuple(point_ids))

    async def ingest_path(self, path: Path) -> IngestResult:
        document = path.expanduser().resolve()
        if not document.is_file():
            raise IngestionError(f"Arquivo não encontrado: {document}")
        if document.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise IngestionError(f"Extensão não suportada: {document.suffix}")
        text = await asyncio.to_thread(_read_document, document)
        source = _source_name(document)
        return await self.ingest_text(
            text,
            source=source,
            metadata={"title": document.stem, "filename": document.name},
        )

    async def ingest_directory(self, directory: Path) -> list[IngestResult]:
        root = directory.expanduser().resolve()
        if not root.is_dir():
            raise IngestionError(f"Diretório não encontrado: {root}")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES and not path.name.startswith(".")
        )
        results: list[IngestResult] = []
        for document in files:
            results.append(await self.ingest_path(document))
        return results

    async def ensure_collection(self) -> None:
        name = self._settings.qdrant_collection
        dense_name = self._settings.qdrant_dense_vector_name
        sparse_name = self._settings.qdrant_sparse_vector_name
        size = await self._embedder.vector_size()

        if await self._client.collection_exists(name):
            info = await self._client.get_collection(name)
            _assert_compatible_collection(info, dense_name, sparse_name, size)
        else:
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    dense_name: rest.VectorParams(size=size, distance=rest.Distance.COSINE),
                },
                sparse_vectors_config={
                    sparse_name: rest.SparseVectorParams(modifier=rest.Modifier.IDF),
                },
            )
            logger.info("qdrant_collection_created", collection=name, dimensions=size)

        if self._settings.qdrant_path:
            return
        try:
            await self._client.create_payload_index(
                collection_name=name,
                field_name="source",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            if "already exists" not in str(exc).lower():
                raise

    async def aclose(self) -> None:
        if self._owns_embedder:
            await self._embedder.aclose()
        if self._owns_client:
            await self._client.close()

    async def _delete_source(self, source: str) -> None:
        await self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="source",
                        match=rest.MatchValue(value=source),
                    )
                ]
            ),
        )

    async def _upsert_chunks(self, chunks: list[TextChunk]) -> list[str]:
        if not chunks:
            return []
        vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise IngestionError("Falha ao vetorizar os chunks do documento")

        points: list[rest.PointStruct] = []
        point_ids: list[str] = []
        dense_name = self._settings.qdrant_dense_vector_name
        sparse_name = self._settings.qdrant_sparse_vector_name
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = chunk_point_id(chunk.source, chunk.index, chunk.text)
            point_ids.append(point_id)
            sparse_vector = _sparse_representation(chunk.text, self._settings)
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={dense_name: vector, sparse_name: sparse_vector},
                    payload=_payload(chunk),
                )
            )

        timeout = _qdrant_timeout(self._settings)
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            await self._client.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
                timeout=timeout,
            )
        return point_ids


def _build_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_path:
        return AsyncQdrantClient(path=settings.qdrant_path)
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.reveal(settings.qdrant_api_key),
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=_qdrant_timeout(settings),
    )


def _sparse_representation(text: str, settings: Settings) -> rest.Document | rest.SparseVector:
    if settings.qdrant_path:
        return local_sparse_vector(text)
    return bm25_document(text)


def _qdrant_timeout(settings: Settings) -> int:
    return max(1, int(settings.qdrant_timeout_seconds))


def _payload(chunk: TextChunk) -> dict[str, str | int | float | bool]:
    payload: dict[str, str | int | float | bool] = {
        key: value
        for key, value in chunk.metadata.items()
        if key not in _RESERVED_PAYLOAD_KEYS and isinstance(value, (str, int, float, bool))
    }
    payload["text"] = chunk.text
    payload["source"] = chunk.source
    payload["chunk_index"] = chunk.index
    return payload


def _assert_compatible_collection(
    info: rest.CollectionInfo,
    dense_name: str,
    sparse_name: str,
    size: int,
) -> None:
    vectors = info.config.params.vectors
    dense: rest.VectorParams | None
    if isinstance(vectors, rest.VectorParams):
        dense = vectors
    elif isinstance(vectors, dict):
        dense = vectors.get(dense_name)
    else:
        dense = None
    if dense is None or dense.size != size:
        found = dense.size if dense is not None else None
        raise IngestionError(
            f"A coleção existente não tem o vetor '{dense_name}' com dimensão {size} (encontrado: {found}). "
            "Recrie a coleção antes de ingerir de novo."
        )
    sparse_vectors = info.config.params.sparse_vectors or {}
    if sparse_name not in sparse_vectors:
        raise IngestionError(
            f"A coleção existente não tem o vetor esparso '{sparse_name}'. Recrie a coleção."
        )


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            pages.append(f"[página {number}]\n{content}")
    return "\n\n".join(pages)


def _source_name(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _split_punctuation(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return parts or [text.strip()]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 1.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _breakpoint_threshold(
    distances: list[float],
    kind: str,
    amount: float,
) -> float:
    if not distances:
        return 1.0
    if kind == "percentile":
        return _percentile(distances, min(max(amount, 0.0), 100.0))
    if kind == "standard_deviation":
        mean = sum(distances) / len(distances)
        variance = sum((value - mean) ** 2 for value in distances) / len(distances)
        return mean + amount * math.sqrt(variance)
    q1 = _percentile(distances, 25)
    q3 = _percentile(distances, 75)
    return q3 + amount * (q3 - q1)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


async def _run_cli(path: Path) -> None:
    ingester = DocumentIngester()
    try:
        results = (
            await ingester.ingest_directory(path)
            if path.is_dir()
            else [await ingester.ingest_path(path)]
        )
    finally:
        await ingester.aclose()
    for result in results:
        print(f"{result.source}\t{result.chunk_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere documentos no índice híbrido do Qdrant")
    parser.add_argument("path", type=Path, help="Arquivo ou diretório (.md, .txt, .pdf)")
    args = parser.parse_args()
    asyncio.run(_run_cli(args.path))


if __name__ == "__main__":
    main()

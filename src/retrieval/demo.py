"""Demonstração local: indexa as políticas e responde com o trecho recuperado.

Não chama OpenAI, Cohere nem um servidor Qdrant. O encoder denso é local.
O BM25 roda no Qdrant embutido, em disco.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from dataclasses import replace
from pathlib import Path

import structlog

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.retrieval.embeddings import HashingEmbeddingProvider, tokenize
from src.retrieval.hybrid_search import HybridSearcher, RetrievedChunk
from src.retrieval.ingester import DocumentIngester

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“(\[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QUESTIONS = (
    "Em quantos dias o cliente pode solicitar estorno?",
    "Qual a primeira resposta de um incidente de severidade 1?",
)


class LexicalReranker:
    """Segundo estágio local: quanto da pergunta aparece no trecho.

    É o substituto gratuito do cross-encoder. A ordem da fusão continua
    disponível no campo ``score``.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        query_terms = set(tokenize(query))
        scored = [(_coverage(query_terms, chunk.text), chunk) for chunk in chunks]
        if query_terms and any(score > 0 for score, _ in scored):
            scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [replace(chunk, rerank_score=score) for score, chunk in scored[:top_n]]

    async def aclose(self) -> None:
        return None


def grounded_passage(chunks: list[RetrievedChunk]) -> RetrievedChunk:
    """Prefere o trecho que contém a regra, não só o título."""
    for chunk in chunks:
        if len(chunk.text) > 80 and not chunk.text.startswith("#"):
            return chunk
    return chunks[0]


def grounded_sentence(query: str, passage: str) -> str:
    sentences = _passage_sentences(passage)
    if not sentences:
        return passage.strip()
    query_terms = set(tokenize(query))
    ranked = sorted(
        sentences,
        key=lambda sentence: (
            len(query_terms.intersection(tokenize(sentence))),
            len(sentence),
        ),
        reverse=True,
    )
    return ranked[0]


def _passage_sentences(passage: str) -> list[str]:
    text = passage.strip().lstrip("#").strip()
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


async def run_demo(questions: list[str]) -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    storage = _ROOT / "data" / "qdrant_local"
    settings = Settings(
        qdrant_path=str(storage),
        qdrant_collection="docops_demo",
        embedding_provider="hashing",
        embedding_dimensions=384,
    )
    client = AsyncQdrantClient(path=settings.qdrant_path)
    embedder = HashingEmbeddingProvider(settings.embedding_dimensions)
    ingester = DocumentIngester(settings, client=client, embedder=embedder)
    searcher = HybridSearcher(
        settings,
        client=client,
        embedder=embedder,
        reranker=LexicalReranker(),
    )
    try:
        results = await ingester.ingest_directory(_ROOT / "data" / "sample_docs")
        print("Enterprise DocOps AI — local hybrid retrieval demo")
        print("No API keys. No Docker. Qdrant is embedded on disk.")
        print("Dense encoder: local hashing. Sparse: BM25. Rerank: lexical coverage.")
        print("The answer is a sentence copied from the retrieved policy.")
        print()
        indexed = ", ".join(f"{item.source} ({item.chunk_count})" for item in results)
        print(f"Indexed: {indexed}")
        for question in questions:
            hits = await searcher.search(question, limit=3)
            _print_question(question, hits)
    finally:
        await searcher.aclose()
        await ingester.aclose()
        await client.close()


def _print_question(question: str, hits: list[RetrievedChunk]) -> None:
    print()
    print(f"Question: {question}")
    if not hits:
        print("No passage retrieved.")
        return
    passage = grounded_passage(hits)
    print()
    print("Grounded answer:")
    print(f"  {grounded_sentence(question, passage.text)}")
    print()
    print(f"Source: {passage.source}")
    print("Evidence:")
    for rank, hit in enumerate(hits, start=1):
        rerank = f"{hit.rerank_score:.2f}" if hit.rerank_score is not None else "-"
        preview = " ".join(hit.text.split())
        if len(preview) > 180:
            preview = preview[:177] + "..."
        print(f"  {rank}. fusion={hit.score:.4f}  rerank={rerank}  {hit.source}")
        print(f"     {preview}")


def _coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    found = query_terms.intersection(tokenize(text))
    return len(found) / len(query_terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca híbrida local nas políticas de exemplo")
    parser.add_argument("question", nargs="*", help="Pergunta. Sem argumentos, roda duas perguntas de exemplo.")
    args = parser.parse_args()
    questions = [" ".join(args.question)] if args.question else list(_DEFAULT_QUESTIONS)
    asyncio.run(run_demo(questions))


if __name__ == "__main__":
    main()

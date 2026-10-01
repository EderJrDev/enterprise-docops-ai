from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.retrieval.demo import LexicalReranker, grounded_sentence
from src.retrieval.embeddings import HashingEmbeddingProvider
from src.retrieval.hybrid_search import HybridSearcher, RetrievedChunk
from src.retrieval.ingester import DocumentIngester


def test_grounded_sentence_prefers_the_matching_rule() -> None:
    passage = (
        "# Reembolso\n\n"
        "O cliente pode solicitar estorno em até 7 dias corridos após a cobrança aprovada. "
        "O pedido entra pelo portal."
    )
    sentence = grounded_sentence("Em quantos dias o cliente pode solicitar estorno?", passage)
    assert "7 dias" in sentence


async def test_local_qdrant_answers_from_the_policy(tmp_path: Path) -> None:
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="docops_demo",
        embedding_provider="hashing",
        embedding_dimensions=384,
    )
    embedder = HashingEmbeddingProvider(384)
    client = AsyncQdrantClient(path=settings.qdrant_path)
    ingester = DocumentIngester(settings, client=client, embedder=embedder)
    searcher = HybridSearcher(
        settings,
        client=client,
        embedder=embedder,
        reranker=LexicalReranker(),
    )
    try:
        await ingester.ingest_text(
            "O cliente pode solicitar estorno em até 7 dias corridos após a cobrança aprovada.",
            source="politica-reembolso.md",
        )
        await ingester.ingest_text(
            "Incidentes de severidade 1 têm primeira resposta em 15 minutos.",
            source="sla-suporte.md",
        )
        hits = await searcher.search("Em quantos dias o cliente pode solicitar estorno?", limit=2)
    finally:
        await searcher.aclose()
        await ingester.aclose()
        await client.close()

    assert hits
    assert "7 dias" in hits[0].text
    assert hits[0].rerank_score is not None


async def test_lexical_reranker_keeps_fusion_order_without_overlap() -> None:
    chunks = [
        RetrievedChunk("a", "texto sem termo", 0.9, "a.md", 0, {}),
        RetrievedChunk("b", "outro texto", 0.2, "b.md", 0, {}),
    ]
    ranked = await LexicalReranker().rerank("zzzz", chunks, top_n=2)
    assert [item.id for item in ranked] == ["a", "b"]

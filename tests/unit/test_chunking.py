from __future__ import annotations

from src.config import Settings
from src.retrieval.ingester import SemanticChunker, character_windows, split_sentences


class _MappedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vectors[text] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def vector_size(self) -> int:
        return 2

    async def aclose(self) -> None:
        return None


def test_split_sentences_keeps_punctuation_and_headings() -> None:
    text = (
        "# Reembolso\n\n"
        "O estorno ocorre em 7 dias. O crédito volta no mesmo meio.\n\n"
        "A versão 1.2 do manual continua válida."
    )
    assert split_sentences(text) == [
        "# Reembolso",
        "O estorno ocorre em 7 dias.",
        "O crédito volta no mesmo meio.",
        "A versão 1.2 do manual continua válida.",
    ]


def test_character_windows_prefer_spaces_and_advance() -> None:
    text = "um dois tres quatro cinco seis sete oito"
    windows = character_windows(text, size=14, overlap=4)
    assert windows
    assert all(len(window) <= 14 for window in windows)
    assert " ".join(windows).count("tres") >= 1
    assert windows[0].startswith("um")


async def test_semantic_chunker_splits_on_topic_shift() -> None:
    refund_a = "O estorno ocorre em 7 dias."
    refund_b = "O crédito volta no cartão."
    infra_a = "O datacenter principal fica em São Paulo."
    infra_b = "O failover regional ocorre em Campinas."
    same = [1.0, 0.0]
    other = [0.0, 1.0]
    embedder = _MappedEmbedder(
        {
            refund_a: same,
            refund_b: same,
            infra_a: other,
            infra_b: other,
        }
    )
    chunker = SemanticChunker(
        embedder,
        Settings(semantic_breakpoint_amount=50),
        buffer_size=0,
    )
    chunks = await chunker.split(
        f"{refund_a} {refund_b} {infra_a} {infra_b}",
        source="politica.md",
    )
    assert len(chunks) == 2
    assert chunks[0].index == 0
    assert "estorno" in chunks[0].text
    assert "datacenter" in chunks[1].text
    assert chunks[1].source == "politica.md"


async def test_semantic_chunker_keeps_one_chunk_when_threshold_is_max() -> None:
    first = "Primeira frase do mesmo assunto."
    second = "Segunda frase do mesmo assunto."
    embedder = _MappedEmbedder({first: [1.0, 0.0], second: [0.0, 1.0]})
    chunker = SemanticChunker(
        embedder,
        Settings(semantic_breakpoint_amount=100),
        buffer_size=0,
    )
    chunks = await chunker.split(f"{first} {second}", source="doc.md")
    assert len(chunks) == 1


async def test_long_semantic_group_is_split_by_size() -> None:
    sentence = "palavra " * 40
    embedder = _MappedEmbedder({sentence.strip(): [1.0, 0.0]})
    chunker = SemanticChunker(
        embedder,
        Settings(chunk_size=128, chunk_overlap=20, semantic_breakpoint_amount=100),
        buffer_size=0,
    )
    chunks = await chunker.split(sentence, source="longo.md")
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 128 for chunk in chunks)


async def test_empty_text_does_not_call_embedder() -> None:
    embedder = _MappedEmbedder({})
    chunker = SemanticChunker(embedder, Settings())
    assert await chunker.split("   ", source="vazio.md") == []
    assert embedder.calls == []

from __future__ import annotations

from src.retrieval.hybrid_search import RetrievedChunk, fuse_rankings


def _chunk(point_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=point_id,
        text=f"texto {point_id}",
        score=score,
        source="doc.md",
        chunk_index=0,
        metadata={},
    )


def test_weighted_rrf_favors_the_heavier_branch() -> None:
    dense_only = _chunk("denso", 0.99)
    both = _chunk("ambos", 0.4)
    fused = fuse_rankings(
        rankings=[
            [dense_only, both],
            [both],
        ],
        weights=[0.2, 0.8],
        strategy="rrf",
        limit=2,
    )
    assert [item.id for item in fused] == ["ambos", "denso"]
    assert fused[0].score > fused[1].score
    assert fused[0].rerank_score is None


def test_single_branch_keeps_original_order() -> None:
    first = _chunk("a", 0.8)
    second = _chunk("b", 0.2)
    fused = fuse_rankings([[first, second]], [1.0], "rrf", limit=2)
    assert [item.id for item in fused] == ["a", "b"]
    assert fused[0].score == 0.8


def test_dbsf_combines_different_score_scales() -> None:
    dense = [_chunk("a", 0.91), _chunk("b", 0.2)]
    sparse = [_chunk("b", 18.0), _chunk("c", 1.0)]
    fused = fuse_rankings([dense, sparse], [0.5, 0.5], "dbsf", limit=3)
    assert fused[0].id == "b"
    assert {item.id for item in fused} == {"a", "b", "c"}

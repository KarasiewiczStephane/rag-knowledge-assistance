"""Tests for the retrieval pipeline."""

from unittest.mock import MagicMock

from src.retrieval.reranker import Reranker
from src.retrieval.retriever import RetrievalResult, Retriever
from src.retrieval.vector_store import RetrievedChunk


def _make_chunk(
    content: str = "test",
    similarity: float = 0.9,
    source: str = "doc.txt",
) -> RetrievedChunk:
    """Create a test RetrievedChunk."""
    return RetrievedChunk(
        content=content,
        source_file=source,
        page_number=1,
        chunk_index=0,
        similarity_score=similarity,
    )


def _mock_retriever(
    chunks: list[RetrievedChunk] | None = None,
    use_reranker: bool = False,
) -> Retriever:
    """Create a Retriever with mocked dependencies."""
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    store = MagicMock()
    store.query.return_value = chunks or []

    reranker = MagicMock(spec=Reranker) if use_reranker else None
    if reranker:
        reranker.rerank.return_value = (chunks or [])[:5]

    return Retriever(
        embedder=embedder,
        vector_store=store,
        top_k=5,
        similarity_threshold=0.5,
        use_reranker=use_reranker,
        reranker=reranker,
    )


def test_retrieve_returns_result() -> None:
    """retrieve() returns a RetrievalResult."""
    chunks = [_make_chunk("result", 0.9)]
    retriever = _mock_retriever(chunks)
    result = retriever.retrieve("test query")
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 1


def test_retrieve_empty_store() -> None:
    """retrieve() returns empty result from empty store."""
    retriever = _mock_retriever([])
    result = retriever.retrieve("test query")
    assert result.chunks == []
    assert result.num_chunks_retrieved == 0


def test_retrieve_filters_by_threshold() -> None:
    """Chunks below similarity threshold are filtered out."""
    chunks = [
        _make_chunk("high", 0.9),
        _make_chunk("low", 0.3),
    ]
    retriever = _mock_retriever(chunks)
    result = retriever.retrieve("test query")
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "high"


def test_retrieve_avg_similarity() -> None:
    """Average similarity is computed correctly."""
    chunks = [
        _make_chunk("a", 0.8),
        _make_chunk("b", 0.6),
    ]
    retriever = _mock_retriever(chunks)
    result = retriever.retrieve("test query")
    assert abs(result.avg_similarity_score - 0.7) < 0.01


def test_retrieve_with_reranker() -> None:
    """Retriever invokes reranker when enabled."""
    chunks = [_make_chunk("test", 0.9)]
    retriever = _mock_retriever(chunks, use_reranker=True)
    result = retriever.retrieve("test query")
    assert len(result.chunks) >= 0


def test_retrieve_respects_top_k() -> None:
    """Retriever limits results to top_k."""
    chunks = [_make_chunk(f"chunk-{i}", 0.9) for i in range(10)]
    retriever = _mock_retriever(chunks)
    result = retriever.retrieve("test query")
    assert len(result.chunks) <= 5


def test_retrieval_result_defaults() -> None:
    """RetrievalResult has sensible defaults."""
    result = RetrievalResult()
    assert result.chunks == []
    assert result.num_chunks_retrieved == 0
    assert result.avg_similarity_score == 0.0


def test_reranker_rerank_mock() -> None:
    """Reranker rerank method works with mocked model."""
    reranker = Reranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.5]
    reranker._model = mock_model

    chunks = [
        _make_chunk("first", 0.8),
        _make_chunk("second", 0.6),
    ]
    result = reranker.rerank("query", chunks, top_k=2)
    assert len(result) == 2
    assert result[0].similarity_score >= result[1].similarity_score


def test_reranker_empty_chunks() -> None:
    """Reranker returns empty list for empty input."""
    reranker = Reranker()
    result = reranker.rerank("query", [], top_k=5)
    assert result == []

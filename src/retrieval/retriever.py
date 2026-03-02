"""Retrieval pipeline: embed query, search vector store, optionally re-rank."""

import logging
from dataclasses import dataclass, field

from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval query.

    Attributes:
        chunks: List of retrieved and optionally re-ranked chunks.
        num_chunks_retrieved: Total number of chunks from initial retrieval.
        avg_similarity_score: Average similarity across returned chunks.
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)
    num_chunks_retrieved: int = 0
    avg_similarity_score: float = 0.0


class Retriever:
    """Orchestrates query embedding, vector search, and optional re-ranking.

    Args:
        embedder: EmbeddingGenerator for query vectorization.
        vector_store: VectorStore for similarity search.
        top_k: Number of chunks to retrieve.
        similarity_threshold: Minimum similarity score to include.
        use_reranker: Whether to apply cross-encoder re-ranking.
        reranker: Optional pre-configured Reranker instance.
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        use_reranker: bool = False,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._use_reranker = use_reranker
        self._reranker = reranker

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant chunks for a user query.

        Steps:
            1. Embed the query text.
            2. Search the vector store for top-K similar chunks.
            3. Filter by similarity threshold.
            4. Optionally re-rank with a cross-encoder.

        Args:
            query: The user's question or search text.

        Returns:
            RetrievalResult with matched chunks and stats.
        """
        query_embedding = self._embedder.embed_single(query)

        fetch_k = self._top_k * 3 if self._use_reranker else self._top_k
        raw_chunks = self._vector_store.query(
            query_embedding=query_embedding,
            n_results=fetch_k,
        )

        filtered = [
            c for c in raw_chunks if c.similarity_score >= self._similarity_threshold
        ]

        if self._use_reranker and self._reranker and filtered:
            filtered = self._reranker.rerank(
                query=query,
                chunks=filtered,
                top_k=self._top_k,
            )
        else:
            filtered = filtered[: self._top_k]

        avg_score = 0.0
        if filtered:
            avg_score = sum(c.similarity_score for c in filtered) / len(filtered)

        logger.info(
            "Retrieved %d chunks for query (avg similarity: %.3f)",
            len(filtered),
            avg_score,
        )

        return RetrievalResult(
            chunks=filtered,
            num_chunks_retrieved=len(filtered),
            avg_similarity_score=avg_score,
        )

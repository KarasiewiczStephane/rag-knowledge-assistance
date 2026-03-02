"""Optional cross-encoder re-ranking for retrieved chunks."""

import logging
from typing import Any

from src.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    Args:
        model_name: Name of the cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy-load the cross-encoder model.

        Returns:
            Loaded CrossEncoder model.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-rank chunks by relevance to the query.

        Args:
            query: The user query string.
            chunks: List of retrieved chunks to re-rank.
            top_k: Maximum number of chunks to return.

        Returns:
            Re-ranked list of chunks sorted by relevance.
        """
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = model.predict(pairs)

        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        result = []
        for chunk, score in scored_chunks[:top_k]:
            result.append(
                RetrievedChunk(
                    content=chunk.content,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    similarity_score=float(score),
                    metadata=chunk.metadata,
                )
            )

        logger.info(
            "Re-ranked %d chunks, returning top %d",
            len(chunks),
            len(result),
        )
        return result

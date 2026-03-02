"""Embedding generation using sentence-transformers."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings using sentence-transformers models.

    Args:
        model_name: Name of the sentence-transformers model to use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        results = self.embed([text])
        return results[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model.

        Returns:
            Integer dimension of the embedding vectors.
        """
        model = self._get_model()
        dim: int = model.get_sentence_embedding_dimension()
        return dim

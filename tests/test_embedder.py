"""Tests for the embedding generator."""

from unittest.mock import MagicMock, patch

from src.ingestion.embedder import EmbeddingGenerator


def _mock_generator() -> EmbeddingGenerator:
    """Create an EmbeddingGenerator with a mocked model."""
    gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    mock_model = MagicMock()
    import numpy as np

    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_model.get_sentence_embedding_dimension.return_value = 3
    gen._model = mock_model
    return gen


def test_embed_returns_list_of_vectors() -> None:
    """embed() returns a list of float lists."""
    gen = _mock_generator()
    result = gen.embed(["hello", "world"])
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)


def test_embed_empty_list() -> None:
    """embed() returns empty list for empty input."""
    gen = _mock_generator()
    result = gen.embed([])
    assert result == []


def test_embed_single() -> None:
    """embed_single() returns a single vector."""
    gen = EmbeddingGenerator()
    mock_model = MagicMock()
    import numpy as np

    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    gen._model = mock_model

    result = gen.embed_single("hello")
    assert isinstance(result, list)
    assert len(result) == 3


def test_model_name_property() -> None:
    """model_name property returns the configured model."""
    gen = EmbeddingGenerator(model_name="test-model")
    assert gen.model_name == "test-model"


def test_dimension_property() -> None:
    """dimension property returns the embedding dimension."""
    gen = _mock_generator()
    assert gen.dimension == 3


def test_lazy_loading() -> None:
    """Model is not loaded until first use."""
    gen = EmbeddingGenerator(model_name="test-model")
    assert gen._model is None


@patch("src.ingestion.embedder.SentenceTransformer")
def test_model_loads_on_first_embed(
    mock_st_class: MagicMock,
) -> None:
    """Model is loaded on first embed call."""
    import numpy as np

    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([[0.1, 0.2]])
    mock_st_class.return_value = mock_instance

    gen = EmbeddingGenerator(model_name="test-model")
    gen.embed(["hello"])
    mock_st_class.assert_called_once_with("test-model")

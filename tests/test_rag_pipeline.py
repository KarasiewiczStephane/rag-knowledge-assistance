"""Tests for the RAG pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.generation.citation_tracker import AnswerWithCitations, Citation
from src.generation.llm_client import LLMResponse
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.retrieval.vector_store import RetrievedChunk


@pytest.fixture()
def mock_pipeline(tmp_path: Path) -> RAGPipeline:
    """Create a RAGPipeline with mocked components."""
    config = {
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50},
        "retrieval": {
            "top_k": 5,
            "similarity_threshold": 0.5,
            "use_reranker": False,
        },
        "llm": {"provider": "anthropic", "model": "test"},
        "embeddings": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {
            "persist_directory": str(tmp_path / "chromadb"),
            "collection_name": "test",
        },
        "memory": {"window_size": 5},
    }

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._config = config

    pipeline._embedder = MagicMock()
    pipeline._embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    pipeline._embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    pipeline._chunker = MagicMock()
    pipeline._vector_store = MagicMock()
    pipeline._vector_store.list_documents.return_value = []
    pipeline._vector_store.count.return_value = 0

    pipeline._retriever = MagicMock()
    pipeline._retriever.retrieve.return_value = MagicMock(
        chunks=[
            RetrievedChunk(
                content="Context",
                source_file="doc.txt",
                page_number=1,
                chunk_index=0,
                similarity_score=0.9,
            )
        ]
    )

    pipeline._llm_client = MagicMock()
    pipeline._llm_client.generate.return_value = LLMResponse(
        content="The answer is 42.",
        model="test",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    pipeline._prompt_builder = MagicMock()
    pipeline._prompt_builder.build_prompt.return_value = "prompt"
    pipeline._prompt_builder.system_prompt = "system"

    pipeline._citation_tracker = MagicMock()
    pipeline._citation_tracker.extract_citations.return_value = AnswerWithCitations(
        answer="The answer is 42.",
        citations=[
            Citation(
                source_file="doc.txt",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                excerpt="Context...",
            )
        ],
        overall_confidence=0.9,
        low_confidence=False,
    )
    pipeline._citation_tracker.generate_source_markdown.return_value = (
        "### Sources\n..."
    )

    from src.memory.conversation import ConversationMemory

    pipeline._memory = ConversationMemory(sessions_dir=str(tmp_path / "sessions"))

    return pipeline


def test_process_query(mock_pipeline: RAGPipeline) -> None:
    """process_query returns a RAGResponse."""
    result = mock_pipeline.process_query("What is 6*7?")
    assert isinstance(result, RAGResponse)
    assert result.answer == "The answer is 42."
    assert result.confidence == 0.9
    assert result.low_confidence is False


def test_process_query_with_session(
    mock_pipeline: RAGPipeline,
) -> None:
    """process_query with session updates memory."""
    sid = mock_pipeline.start_new_session()
    result = mock_pipeline.process_query("Question?", session_id=sid)
    assert result.answer == "The answer is 42."

    history = mock_pipeline.get_session_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_start_new_session(mock_pipeline: RAGPipeline) -> None:
    """start_new_session returns a session ID."""
    sid = mock_pipeline.start_new_session()
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_get_session_history_empty(
    mock_pipeline: RAGPipeline,
) -> None:
    """get_session_history returns empty for unknown session."""
    assert mock_pipeline.get_session_history("bad") == []


def test_clear_vector_store(mock_pipeline: RAGPipeline) -> None:
    """clear_vector_store calls clear on the store."""
    mock_pipeline.clear_vector_store()
    mock_pipeline._vector_store.clear.assert_called_once()


def test_ingest_document(mock_pipeline: RAGPipeline, tmp_path: Path) -> None:
    """ingest_document parses and stores chunks."""
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Test content for ingestion.")

    from src.ingestion.chunker import Chunk
    from src.ingestion.parser import DocumentMetadata, ParsedDocument

    mock_doc = ParsedDocument(
        content="Test content",
        metadata=DocumentMetadata(
            source_file=str(doc_file),
            title="test",
            created_date="2024-01-01",
            page_count=1,
            file_hash="unique_hash",
            file_type="txt",
        ),
        pages=["Test content"],
    )

    mock_chunks = [
        Chunk(
            content="Test content",
            chunk_index=0,
            source_file=str(doc_file),
            page_number=1,
            start_char=0,
            end_char=12,
            metadata={
                "title": "test",
                "file_hash": "unique_hash",
                "file_type": "txt",
            },
        )
    ]

    mock_pipeline._chunker.chunk_document.return_value = mock_chunks

    with patch("src.rag_pipeline.ParserFactory") as mock_factory:
        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_doc
        mock_factory.get_parser.return_value = mock_parser

        count = mock_pipeline.ingest_document(str(doc_file))
        assert count == 1
        mock_pipeline._vector_store.add_chunks.assert_called_once()


def test_ingest_duplicate_document(mock_pipeline: RAGPipeline, tmp_path: Path) -> None:
    """ingest_document skips duplicates."""
    doc_file = tmp_path / "dup.txt"
    doc_file.write_text("Duplicate content.")

    mock_pipeline._vector_store.list_documents.return_value = [
        {"file_hash": "dup_hash", "source_file": "other.txt"}
    ]

    from src.ingestion.parser import DocumentMetadata, ParsedDocument

    mock_doc = ParsedDocument(
        content="Dup",
        metadata=DocumentMetadata(
            source_file=str(doc_file),
            title="dup",
            created_date="2024-01-01",
            page_count=1,
            file_hash="dup_hash",
            file_type="txt",
        ),
        pages=["Dup"],
    )

    with patch("src.rag_pipeline.ParserFactory") as mock_factory:
        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_doc
        mock_factory.get_parser.return_value = mock_parser

        count = mock_pipeline.ingest_document(str(doc_file))
        assert count == 0


def test_vector_store_property(
    mock_pipeline: RAGPipeline,
) -> None:
    """vector_store property provides access."""
    assert mock_pipeline.vector_store is not None


def test_memory_property(mock_pipeline: RAGPipeline) -> None:
    """memory property provides access."""
    assert mock_pipeline.memory is not None


def test_rag_response_fields() -> None:
    """RAGResponse dataclass has all fields."""
    resp = RAGResponse(
        answer="test",
        citations=AnswerWithCitations(answer="test"),
        confidence=0.8,
        low_confidence=False,
        sources_markdown="md",
    )
    assert resp.answer == "test"
    assert resp.confidence == 0.8

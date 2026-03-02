"""End-to-end integration tests for the RAG pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.generation.llm_client import LLMResponse
from src.ingestion.chunker import DocumentChunker
from src.ingestion.parser import ParserFactory, TextParser
from src.memory.conversation import ConversationMemory
from src.rag_pipeline import RAGPipeline
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def test_full_ingestion_to_query_flow(tmp_path: Path) -> None:
    """Full workflow: ingest a document, query it, get cited answer."""
    doc_file = tmp_path / "test_doc.txt"
    doc_file.write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It uses algorithms to learn patterns from data. "
        "Common ML methods include neural networks, decision trees, "
        "and support vector machines."
    )

    config = {
        "ingestion": {"chunk_size": 200, "chunk_overlap": 20},
        "retrieval": {
            "top_k": 3,
            "similarity_threshold": 0.0,
            "use_reranker": False,
        },
        "llm": {"provider": "anthropic", "model": "test"},
        "embeddings": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {
            "persist_directory": str(tmp_path / "chromadb"),
            "collection_name": "integration_test",
        },
        "memory": {"window_size": 5},
    }

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._config = config

    from src.ingestion.embedder import EmbeddingGenerator

    pipeline._embedder = MagicMock(spec=EmbeddingGenerator)
    pipeline._embedder.embed.side_effect = lambda texts: [
        [0.1 * (i + 1)] * 10 for i in range(len(texts))
    ]
    pipeline._embedder.embed_single.return_value = [0.15] * 10

    pipeline._chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

    pipeline._vector_store = VectorStore(
        persist_directory=str(tmp_path / "chromadb"),
        collection_name="integration_test",
    )

    from src.retrieval.retriever import Retriever

    pipeline._retriever = Retriever(
        embedder=pipeline._embedder,
        vector_store=pipeline._vector_store,
        top_k=3,
        similarity_threshold=0.0,
    )

    pipeline._llm_client = MagicMock()
    pipeline._llm_client.generate.return_value = LLMResponse(
        content="ML is a subset of AI that learns from data.",
        model="test",
        usage={"input_tokens": 50, "output_tokens": 20},
    )

    from src.generation.prompt_builder import PromptBuilder

    pipeline._prompt_builder = PromptBuilder()

    from src.generation.citation_tracker import CitationTracker

    pipeline._citation_tracker = CitationTracker()
    pipeline._memory = ConversationMemory(sessions_dir=str(tmp_path / "sessions"))

    count = pipeline.ingest_document(str(doc_file))
    assert count > 0

    docs = pipeline.vector_store.list_documents()
    assert len(docs) == 1

    response = pipeline.process_query("What is machine learning?")
    assert response.answer != ""
    assert response.confidence >= 0
    assert response.sources_markdown is not None


def test_multi_document_ingestion(tmp_path: Path) -> None:
    """Multiple documents can be ingested and queried."""
    for i in range(3):
        doc = tmp_path / f"doc_{i}.txt"
        doc.write_text(f"Document {i} contains information about topic {i}.")

    parser = TextParser()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

    for i in range(3):
        result = parser.parse(tmp_path / f"doc_{i}.txt")
        chunks = chunker.chunk_document(result)
        assert len(chunks) > 0


def test_conversation_memory_multi_turn(tmp_path: Path) -> None:
    """Conversation memory tracks multiple turns correctly."""
    memory = ConversationMemory(window_size=3, sessions_dir=str(tmp_path / "sessions"))
    sid = memory.create_session()

    for i in range(5):
        memory.add_message(sid, "user", f"Question {i}")
        memory.add_message(sid, "assistant", f"Answer {i}")

    window = memory.get_window(sid)
    assert len(window) == 6

    text = memory.get_window_text(sid)
    assert "Question 4" in text
    assert "Answer 4" in text


def test_session_save_load_roundtrip(tmp_path: Path) -> None:
    """Sessions survive save/load roundtrip."""
    memory = ConversationMemory(sessions_dir=str(tmp_path / "sessions"))
    sid = memory.create_session()
    memory.add_message(sid, "user", "Hello")
    memory.add_message(
        sid,
        "assistant",
        "Hi",
        citations=[{"source": "test.txt"}],
    )

    memory.save_session(sid)

    new_memory = ConversationMemory(sessions_dir=str(tmp_path / "sessions"))
    loaded_sid = new_memory.load_session(sid)
    session = new_memory.get_session(loaded_sid)

    assert len(session.messages) == 2
    assert session.messages[0].content == "Hello"
    assert session.messages[1].citations == [{"source": "test.txt"}]


def test_parser_chunker_integration() -> None:
    """Parser output feeds cleanly into chunker."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("First paragraph about machine learning.\n\n")
        f.write("Second paragraph about data science.\n\n")
        f.write("Third paragraph about deep learning.\n")
        f.flush()

        parser = ParserFactory.get_parser(f.name)
        doc = parser.parse(Path(f.name))

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.source_file == f.name
        assert chunk.page_number >= 1

    Path(f.name).unlink()


def test_vector_store_query_roundtrip(tmp_path: Path) -> None:
    """Chunks stored in ChromaDB can be queried back."""
    store = VectorStore(
        persist_directory=str(tmp_path / "chromadb"),
        collection_name="roundtrip_test",
    )

    store.add_chunks(
        ids=["c1", "c2"],
        documents=["ML is great", "Python is versatile"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[
            {
                "source_file": "ml.txt",
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": "h1",
                "title": "ml",
            },
            {
                "source_file": "py.txt",
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": "h2",
                "title": "py",
            },
        ],
    )

    results = store.query(query_embedding=[0.1, 0.2, 0.3], n_results=2)
    assert len(results) == 2
    assert any("ML" in r.content for r in results)


def test_config_loads_correctly() -> None:
    """Config loads with all expected sections."""
    config = load_config()
    assert "ingestion" in config
    assert "retrieval" in config
    assert "llm" in config
    assert "embeddings" in config
    assert "memory" in config
    assert "vector_store" in config


def test_evaluation_with_test_data() -> None:
    """Evaluator loads and processes the test Q&A file."""
    from src.evaluation.ragas_eval import RAGASEvaluator

    cases = RAGASEvaluator.load_test_cases("data/test_qa_pairs.json")
    assert len(cases) == 20

    evaluator = RAGASEvaluator(pipeline=None)
    result = evaluator.evaluate(cases[:3])
    assert result.num_cases == 3

    report = evaluator.generate_report(result)
    assert "RAG Evaluation Report" in report


def test_error_handling_missing_document(
    tmp_path: Path,
) -> None:
    """Pipeline handles missing documents gracefully."""
    config = {
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50},
        "retrieval": {
            "top_k": 5,
            "similarity_threshold": 0.5,
            "use_reranker": False,
        },
        "llm": {"provider": "anthropic"},
        "embeddings": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {
            "persist_directory": str(tmp_path / "chromadb"),
            "collection_name": "error_test",
        },
        "memory": {"window_size": 5},
    }

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._config = config
    pipeline._embedder = MagicMock()
    pipeline._chunker = MagicMock()
    pipeline._vector_store = MagicMock()
    pipeline._retriever = MagicMock()
    pipeline._llm_client = MagicMock()
    pipeline._prompt_builder = MagicMock()
    pipeline._citation_tracker = MagicMock()
    pipeline._memory = ConversationMemory(sessions_dir=str(tmp_path / "sessions"))

    with pytest.raises(FileNotFoundError):
        pipeline.ingest_document("/nonexistent/file.txt")

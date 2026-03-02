"""Tests for the ChromaDB vector store."""

from pathlib import Path

import pytest

from src.retrieval.vector_store import RetrievedChunk, VectorStore


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    """Create a VectorStore using a temporary directory."""
    return VectorStore(
        persist_directory=str(tmp_path / "chromadb"),
        collection_name="test_docs",
    )


def _add_sample_chunks(store: VectorStore) -> None:
    """Add sample chunks to the store."""
    store.add_chunks(
        ids=["chunk-0", "chunk-1"],
        documents=["First chunk text.", "Second chunk text."],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[
            {
                "source_file": "doc1.txt",
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": "abc123",
                "title": "doc1",
            },
            {
                "source_file": "doc1.txt",
                "page_number": 1,
                "chunk_index": 1,
                "file_hash": "abc123",
                "title": "doc1",
            },
        ],
    )


def test_add_and_count(store: VectorStore) -> None:
    """Adding chunks increases the count."""
    assert store.count() == 0
    _add_sample_chunks(store)
    assert store.count() == 2


def test_query_returns_chunks(store: VectorStore) -> None:
    """Querying returns RetrievedChunk objects."""
    _add_sample_chunks(store)
    results = store.query(query_embedding=[0.1, 0.2, 0.3], n_results=2)
    assert len(results) > 0
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].content in [
        "First chunk text.",
        "Second chunk text.",
    ]


def test_query_empty_store(store: VectorStore) -> None:
    """Querying an empty store returns empty list."""
    results = store.query(query_embedding=[0.1, 0.2, 0.3], n_results=5)
    assert results == []


def test_query_similarity_score(store: VectorStore) -> None:
    """Retrieved chunks have a similarity score."""
    _add_sample_chunks(store)
    results = store.query(query_embedding=[0.1, 0.2, 0.3], n_results=1)
    assert results[0].similarity_score is not None


def test_list_documents(store: VectorStore) -> None:
    """list_documents returns unique document entries."""
    _add_sample_chunks(store)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["source_file"] == "doc1.txt"
    assert docs[0]["chunk_count"] == 2


def test_delete_document(store: VectorStore) -> None:
    """delete_document removes all chunks for a source file."""
    _add_sample_chunks(store)
    deleted = store.delete_document("doc1.txt")
    assert deleted == 2
    assert store.count() == 0


def test_delete_nonexistent_document(
    store: VectorStore,
) -> None:
    """Deleting a nonexistent document returns 0."""
    _add_sample_chunks(store)
    deleted = store.delete_document("nonexistent.txt")
    assert deleted == 0
    assert store.count() == 2


def test_clear(store: VectorStore) -> None:
    """clear() removes all chunks from the store."""
    _add_sample_chunks(store)
    store.clear()
    assert store.count() == 0


def test_collection_name_property(store: VectorStore) -> None:
    """collection_name property returns the configured name."""
    assert store.collection_name == "test_docs"


def test_get_duplicate_hashes(store: VectorStore) -> None:
    """get_duplicate_hashes returns empty when no duplicates."""
    _add_sample_chunks(store)
    dupes = store.get_duplicate_hashes()
    assert dupes == {}


def test_get_duplicate_hashes_with_dupes(
    store: VectorStore,
) -> None:
    """get_duplicate_hashes detects duplicate file hashes."""
    _add_sample_chunks(store)
    store.add_chunks(
        ids=["chunk-2"],
        documents=["Duplicate content."],
        embeddings=[[0.7, 0.8, 0.9]],
        metadatas=[
            {
                "source_file": "doc2.txt",
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": "abc123",
                "title": "doc2",
            }
        ],
    )
    dupes = store.get_duplicate_hashes()
    assert "abc123" in dupes
    assert len(dupes["abc123"]) == 2


def test_retrieved_chunk_dataclass() -> None:
    """RetrievedChunk has all expected fields."""
    chunk = RetrievedChunk(
        content="test",
        source_file="f.txt",
        page_number=1,
        chunk_index=0,
        similarity_score=0.95,
    )
    assert chunk.content == "test"
    assert chunk.similarity_score == 0.95
    assert chunk.metadata == {}

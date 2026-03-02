"""ChromaDB vector store for document chunk storage and retrieval."""

import logging
from dataclasses import dataclass, field
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with similarity score.

    Attributes:
        content: The chunk text.
        source_file: Original source file path.
        page_number: Page number in the source document.
        chunk_index: Chunk index within the document.
        similarity_score: Cosine similarity score (0-1).
        metadata: Additional metadata from the stored chunk.
    """

    content: str
    source_file: str
    page_number: int
    chunk_index: int
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """ChromaDB-backed vector store for document chunks.

    Args:
        persist_directory: Path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "documents",
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Initialized vector store at %s (collection: %s)",
            persist_directory,
            collection_name,
        )

    @property
    def collection_name(self) -> str:
        """Return the collection name."""
        return self._collection_name

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add document chunks with embeddings to the vector store.

        Args:
            ids: Unique identifiers for each chunk.
            documents: Text content of each chunk.
            embeddings: Embedding vectors for each chunk.
            metadatas: Metadata dictionaries for each chunk.
        """
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to vector store", len(ids))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[RetrievedChunk]:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: Embedding vector of the query.
            n_results: Maximum number of results to return.

        Returns:
            List of RetrievedChunk sorted by similarity.
        """
        count = self._collection.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - dist
            chunks.append(
                RetrievedChunk(
                    content=doc,
                    source_file=meta.get("source_file", ""),
                    page_number=meta.get("page_number", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    similarity_score=similarity,
                    metadata=meta,
                )
            )

        return chunks

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents in the vector store.

        Returns:
            List of dicts with source_file, chunk_count, file_hash.
        """
        all_meta = self._collection.get(include=["metadatas"])
        metadatas = all_meta.get("metadatas", [])

        doc_map: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            source = meta.get("source_file", "unknown")
            if source not in doc_map:
                doc_map[source] = {
                    "source_file": source,
                    "chunk_count": 0,
                    "file_hash": meta.get("file_hash", ""),
                    "title": meta.get("title", ""),
                }
            doc_map[source]["chunk_count"] += 1

        return list(doc_map.values())

    def delete_document(self, source_file: str) -> int:
        """Delete all chunks for a given source file.

        Args:
            source_file: Path of the document to delete.

        Returns:
            Number of chunks deleted.
        """
        all_data = self._collection.get(
            include=["metadatas"],
            where={"source_file": source_file},
        )
        ids_to_delete = all_data.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks for %s",
                len(ids_to_delete),
                source_file,
            )
        return len(ids_to_delete)

    def get_duplicate_hashes(self) -> dict[str, list[str]]:
        """Find documents with duplicate file hashes.

        Returns:
            Dict mapping file_hash to list of source_files.
        """
        docs = self.list_documents()
        hash_map: dict[str, list[str]] = {}
        for doc in docs:
            h = doc.get("file_hash", "")
            if h:
                hash_map.setdefault(h, []).append(doc["source_file"])
        return {h: files for h, files in hash_map.items() if len(files) > 1}

    def count(self) -> int:
        """Return the total number of chunks in the store.

        Returns:
            Integer count of stored chunks.
        """
        return self._collection.count()

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared vector store collection")

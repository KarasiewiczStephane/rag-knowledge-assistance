"""Complete RAG pipeline orchestrator integrating all components."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.generation.citation_tracker import (
    AnswerWithCitations,
    CitationTracker,
)
from src.generation.llm_client import BaseLLMClient, get_llm_client
from src.generation.prompt_builder import PromptBuilder
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.parser import ParserFactory
from src.memory.conversation import ConversationMemory
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline.

    Attributes:
        answer: Generated answer text.
        citations: Answer with linked citations.
        confidence: Overall confidence score.
        low_confidence: Whether confidence is below threshold.
        sources_markdown: Formatted source references.
    """

    answer: str
    citations: AnswerWithCitations
    confidence: float
    low_confidence: bool
    sources_markdown: str


class RAGPipeline:
    """Orchestrates document ingestion, retrieval, and generation.

    Integrates: parser, chunker, embedder, vector store, retriever,
    LLM client, prompt builder, citation tracker, and conversation
    memory into a unified pipeline.

    Args:
        config: Application config dict. Loads defaults if None.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or load_config()

        embedding_cfg = self._config.get("embeddings", {})
        self._embedder = EmbeddingGenerator(
            model_name=embedding_cfg.get("model", "all-MiniLM-L6-v2")
        )

        ingestion_cfg = self._config.get("ingestion", {})
        self._chunker = DocumentChunker(
            chunk_size=ingestion_cfg.get("chunk_size", 500),
            chunk_overlap=ingestion_cfg.get("chunk_overlap", 50),
        )

        vs_cfg = self._config.get("vector_store", {})
        self._vector_store = VectorStore(
            persist_directory=vs_cfg.get("persist_directory", "./data/chromadb"),
            collection_name=vs_cfg.get("collection_name", "documents"),
        )

        retrieval_cfg = self._config.get("retrieval", {})
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            top_k=retrieval_cfg.get("top_k", 5),
            similarity_threshold=retrieval_cfg.get("similarity_threshold", 0.7),
            use_reranker=retrieval_cfg.get("use_reranker", False),
        )

        self._llm_client: BaseLLMClient = get_llm_client(self._config)
        self._prompt_builder = PromptBuilder()
        self._citation_tracker = CitationTracker()

        memory_cfg = self._config.get("memory", {})
        self._memory = ConversationMemory(
            window_size=memory_cfg.get("window_size", 5),
        )

        logger.info("RAG pipeline initialized")

    @property
    def vector_store(self) -> VectorStore:
        """Access the underlying vector store."""
        return self._vector_store

    @property
    def memory(self) -> ConversationMemory:
        """Access the conversation memory manager."""
        return self._memory

    def ingest_document(self, file_path: str) -> int:
        """Ingest a document: parse, chunk, embed, store.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks stored.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported.
        """
        path = Path(file_path)
        logger.info("Ingesting document: %s", path.name)

        parser = ParserFactory.get_parser(path)
        document = parser.parse(path)

        existing = self._vector_store.list_documents()
        for doc in existing:
            if doc.get("file_hash") == document.metadata.file_hash:
                logger.warning("Duplicate document detected: %s", path.name)
                return 0

        chunks = self._chunker.chunk_document(document)
        if not chunks:
            logger.warning("No chunks produced for %s", path.name)
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed(texts)

        ids = [f"{document.metadata.file_hash}_{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "title": c.metadata.get("title", ""),
                "file_hash": c.metadata.get("file_hash", ""),
                "file_type": c.metadata.get("file_type", ""),
            }
            for c in chunks
        ]

        self._vector_store.add_chunks(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Ingested %s: %d chunks stored", path.name, len(chunks))
        return len(chunks)

    def process_query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> RAGResponse:
        """Process a user question through the RAG pipeline.

        Steps: embed query -> retrieve -> build prompt -> generate
        -> extract citations -> update memory.

        Args:
            question: The user's question.
            session_id: Optional conversation session for context.

        Returns:
            RAGResponse with answer, citations, and confidence.
        """
        conversation_history = None
        if session_id:
            self._memory.add_message(session_id, "user", question)
            conversation_history = self._memory.get_window_text(session_id)

        retrieval_result = self._retriever.retrieve(question)

        prompt = self._prompt_builder.build_prompt(
            question=question,
            chunks=retrieval_result.chunks,
            conversation_history=conversation_history,
        )

        llm_response = self._llm_client.generate(
            prompt=prompt,
            system_prompt=self._prompt_builder.system_prompt,
        )

        cited_answer = self._citation_tracker.extract_citations(
            answer=llm_response.content,
            chunks=retrieval_result.chunks,
        )

        sources_md = self._citation_tracker.generate_source_markdown(cited_answer)

        if session_id:
            self._memory.add_message(
                session_id,
                "assistant",
                llm_response.content,
                citations=[
                    {
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "relevance_score": c.relevance_score,
                    }
                    for c in cited_answer.citations
                ],
            )

        return RAGResponse(
            answer=llm_response.content,
            citations=cited_answer,
            confidence=cited_answer.overall_confidence,
            low_confidence=cited_answer.low_confidence,
            sources_markdown=sources_md,
        )

    def start_new_session(self) -> str:
        """Create a new conversation session.

        Returns:
            New session ID.
        """
        return self._memory.create_session()

    def load_session(self, session_id: str) -> str:
        """Load a previous conversation session.

        Args:
            session_id: Session ID to load.

        Returns:
            The loaded session ID.
        """
        return self._memory.load_session(session_id)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Target session ID.

        Returns:
            List of message dicts with role, content, timestamp.
        """
        session = self._memory.get_session(session_id)
        if session is None:
            return []
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in session.messages
        ]

    def clear_vector_store(self) -> None:
        """Clear all documents from the vector store."""
        self._vector_store.clear()
        logger.info("Vector store cleared")

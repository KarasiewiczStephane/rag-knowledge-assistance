"""Text chunking pipeline using LangChain's RecursiveCharacterTextSplitter."""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk produced by the chunking pipeline.

    Attributes:
        content: The chunk text.
        chunk_index: Zero-based index within the document.
        source_file: Path to the original source file.
        page_number: Page number the chunk originates from (1-based).
        start_char: Starting character offset in the page text.
        end_char: Ending character offset in the page text.
        metadata: Additional metadata carried from the document.
    """

    content: str
    chunk_index: int
    source_file: str
    page_number: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Splits parsed documents into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter with a hierarchy
    of separators to preserve paragraph boundaries where possible.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    @property
    def chunk_size(self) -> int:
        """Return the configured chunk size."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return the configured chunk overlap."""
        return self._chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks with position tracking.

        Processes each page independently and tracks the original page
        number for each resulting chunk.

        Args:
            document: A ParsedDocument to split.

        Returns:
            List of Chunk objects with position and metadata.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page_num, page_text in enumerate(document.pages, start=1):
            if not page_text.strip():
                continue

            splits = self._splitter.split_text(page_text)
            search_start = 0

            for split_text in splits:
                start = page_text.find(split_text, search_start)
                if start == -1:
                    start = search_start
                end = start + len(split_text)

                chunks.append(
                    Chunk(
                        content=split_text,
                        chunk_index=chunk_index,
                        source_file=document.metadata.source_file,
                        page_number=page_num,
                        start_char=start,
                        end_char=end,
                        metadata={
                            "title": document.metadata.title,
                            "file_type": document.metadata.file_type,
                            "file_hash": document.metadata.file_hash,
                        },
                    )
                )
                chunk_index += 1
                search_start = max(start + 1, end - self._chunk_overlap)

        logger.info(
            "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
            document.metadata.title,
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    def chunk_text(self, text: str, source_file: str = "unknown") -> list[Chunk]:
        """Split raw text into chunks.

        Args:
            text: Raw text to chunk.
            source_file: Source file identifier for metadata.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        splits = self._splitter.split_text(text)
        chunks = []
        search_start = 0

        for idx, split_text in enumerate(splits):
            start = text.find(split_text, search_start)
            if start == -1:
                start = search_start
            end = start + len(split_text)

            chunks.append(
                Chunk(
                    content=split_text,
                    chunk_index=idx,
                    source_file=source_file,
                    page_number=1,
                    start_char=start,
                    end_char=end,
                )
            )
            search_start = max(start + 1, end - self._chunk_overlap)

        return chunks

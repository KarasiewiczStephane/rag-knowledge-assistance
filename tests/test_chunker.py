"""Tests for the text chunking pipeline."""

from src.ingestion.chunker import Chunk, DocumentChunker
from src.ingestion.parser import DocumentMetadata, ParsedDocument


def _make_document(content: str, source: str = "test.txt") -> ParsedDocument:
    """Create a test ParsedDocument."""
    return ParsedDocument(
        content=content,
        metadata=DocumentMetadata(
            source_file=source,
            title="test",
            created_date="2024-01-01",
            page_count=1,
            file_hash="abc123",
            file_type="txt",
        ),
        pages=[content],
    )


def test_chunk_short_text() -> None:
    """Short text that fits in one chunk produces a single chunk."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Short text.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunk_long_text_produces_multiple() -> None:
    """Text longer than chunk_size produces multiple chunks."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "Word " * 200
    doc = _make_document(text.strip())
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1


def test_chunk_preserves_source_file() -> None:
    """Chunks carry the source file from the document."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Some text.", source="/path/to/file.txt")
    chunks = chunker.chunk_document(doc)
    assert chunks[0].source_file == "/path/to/file.txt"


def test_chunk_indexes_are_sequential() -> None:
    """Chunk indexes are sequential starting from 0."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "Paragraph one. " * 20
    doc = _make_document(text.strip())
    chunks = chunker.chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_metadata_carried() -> None:
    """Chunks carry title, file_type, file_hash from document."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Test content.")
    chunks = chunker.chunk_document(doc)
    assert chunks[0].metadata["title"] == "test"
    assert chunks[0].metadata["file_type"] == "txt"
    assert chunks[0].metadata["file_hash"] == "abc123"


def test_chunk_page_number() -> None:
    """Chunks track the correct page number."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Page content.")
    chunks = chunker.chunk_document(doc)
    assert chunks[0].page_number == 1


def test_chunk_multipage_document() -> None:
    """Multi-page documents produce chunks from each page."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = ParsedDocument(
        content="Page 1 content.\n\nPage 2 content.",
        metadata=DocumentMetadata(
            source_file="multi.pdf",
            title="multi",
            created_date="2024-01-01",
            page_count=2,
            file_hash="xyz",
            file_type="pdf",
        ),
        pages=["Page 1 content.", "Page 2 content."],
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_empty_page_skipped() -> None:
    """Empty pages are skipped during chunking."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = ParsedDocument(
        content="Content.",
        metadata=DocumentMetadata(
            source_file="test.txt",
            title="test",
            created_date="2024-01-01",
            page_count=2,
            file_hash="xyz",
            file_type="txt",
        ),
        pages=["Content.", "   "],
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1


def test_chunk_text_direct() -> None:
    """chunk_text splits raw text directly."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "Hello world. " * 50
    chunks = chunker.chunk_text(text.strip(), source_file="raw.txt")
    assert len(chunks) > 1
    assert chunks[0].source_file == "raw.txt"


def test_chunk_text_empty() -> None:
    """chunk_text returns empty list for empty text."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text("   ")
    assert chunks == []


def test_chunk_properties() -> None:
    """Chunker exposes chunk_size and chunk_overlap properties."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    assert chunker.chunk_size == 300
    assert chunker.chunk_overlap == 30


def test_chunk_has_position_info() -> None:
    """Chunks include start_char and end_char positions."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Some test content here.")
    chunks = chunker.chunk_document(doc)
    assert chunks[0].start_char >= 0
    assert chunks[0].end_char > chunks[0].start_char


def test_chunk_dataclass_fields() -> None:
    """Chunk dataclass has all expected fields."""
    c = Chunk(
        content="test",
        chunk_index=0,
        source_file="f.txt",
        page_number=1,
        start_char=0,
        end_char=4,
    )
    assert c.content == "test"
    assert c.chunk_index == 0
    assert c.metadata == {}

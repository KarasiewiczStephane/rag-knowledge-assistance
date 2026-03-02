"""Tests for the citation tracking system."""

from src.generation.citation_tracker import (
    AnswerWithCitations,
    Citation,
    CitationTracker,
)
from src.retrieval.vector_store import RetrievedChunk


def _make_chunk(
    content: str = "Some context text here",
    source: str = "/docs/file.txt",
    page: int = 1,
    similarity: float = 0.9,
) -> RetrievedChunk:
    """Create a test RetrievedChunk."""
    return RetrievedChunk(
        content=content,
        source_file=source,
        page_number=page,
        chunk_index=0,
        similarity_score=similarity,
    )


def test_extract_citations_basic() -> None:
    """extract_citations creates citations from chunks."""
    tracker = CitationTracker()
    chunks = [_make_chunk("ML is powerful.", "/docs/ml.txt")]
    result = tracker.extract_citations("Answer here", chunks)
    assert result.answer == "Answer here"
    assert len(result.citations) == 1
    assert result.citations[0].source_file == "/docs/ml.txt"


def test_extract_citations_empty() -> None:
    """extract_citations with no chunks produces empty citations."""
    tracker = CitationTracker()
    result = tracker.extract_citations("No context", [])
    assert result.citations == []
    assert result.overall_confidence == 0.0


def test_overall_confidence() -> None:
    """Overall confidence is the average of relevance scores."""
    tracker = CitationTracker()
    chunks = [
        _make_chunk(similarity=0.8),
        _make_chunk(similarity=0.6),
    ]
    result = tracker.extract_citations("Answer", chunks)
    assert abs(result.overall_confidence - 0.7) < 0.01


def test_low_confidence_flag() -> None:
    """Low confidence flag is set when below threshold."""
    tracker = CitationTracker(confidence_threshold=0.7)
    chunks = [_make_chunk(similarity=0.3)]
    result = tracker.extract_citations("Answer", chunks)
    assert result.low_confidence is True


def test_high_confidence_flag() -> None:
    """Low confidence flag is False when above threshold."""
    tracker = CitationTracker(confidence_threshold=0.5)
    chunks = [_make_chunk(similarity=0.9)]
    result = tracker.extract_citations("Answer", chunks)
    assert result.low_confidence is False


def test_excerpt_truncation() -> None:
    """Long content is truncated in the excerpt."""
    long_text = "A" * 200
    tracker = CitationTracker()
    chunks = [_make_chunk(content=long_text)]
    result = tracker.extract_citations("Answer", chunks)
    assert result.citations[0].excerpt.endswith("...")
    assert len(result.citations[0].excerpt) == 103


def test_excerpt_short_text() -> None:
    """Short content is not truncated."""
    tracker = CitationTracker()
    chunks = [_make_chunk(content="Short")]
    result = tracker.extract_citations("Answer", chunks)
    assert result.citations[0].excerpt == "Short"


def test_generate_source_markdown_with_citations() -> None:
    """Markdown output includes source details."""
    tracker = CitationTracker()
    result = AnswerWithCitations(
        answer="The answer",
        citations=[
            Citation(
                source_file="/docs/file.txt",
                page_number=2,
                chunk_index=0,
                relevance_score=0.85,
                excerpt="Some text...",
            )
        ],
        overall_confidence=0.85,
        low_confidence=False,
    )
    md = tracker.generate_source_markdown(result)
    assert "file.txt" in md
    assert "Page 2" in md
    assert "0.85" in md
    assert "Good" in md


def test_generate_source_markdown_no_citations() -> None:
    """Markdown output shows 'No sources' when empty."""
    tracker = CitationTracker()
    result = AnswerWithCitations(answer="The answer")
    md = tracker.generate_source_markdown(result)
    assert "No sources" in md


def test_generate_source_markdown_low_confidence() -> None:
    """Markdown output shows 'Low' for low confidence."""
    tracker = CitationTracker()
    result = AnswerWithCitations(
        answer="The answer",
        citations=[
            Citation(
                source_file="f.txt",
                page_number=1,
                chunk_index=0,
                relevance_score=0.3,
                excerpt="text",
            )
        ],
        overall_confidence=0.3,
        low_confidence=True,
    )
    md = tracker.generate_source_markdown(result)
    assert "Low" in md


def test_citation_dataclass() -> None:
    """Citation dataclass fields are accessible."""
    c = Citation(
        source_file="doc.pdf",
        page_number=5,
        chunk_index=2,
        relevance_score=0.88,
        excerpt="sample",
    )
    assert c.source_file == "doc.pdf"
    assert c.page_number == 5
    assert c.chunk_index == 2

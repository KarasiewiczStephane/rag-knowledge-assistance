"""Citation tracking linking answers to retrieved source chunks."""

import logging
from dataclasses import dataclass, field

from src.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_EXCERPT_LENGTH = 100
_LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Citation:
    """A reference to a specific source chunk.

    Attributes:
        source_file: Path to the source document.
        page_number: Page number in the document.
        chunk_index: Chunk index within the document.
        relevance_score: Similarity/relevance score (0-1).
        excerpt: Short excerpt from the source text.
    """

    source_file: str
    page_number: int
    chunk_index: int
    relevance_score: float
    excerpt: str


@dataclass
class AnswerWithCitations:
    """An LLM answer with attached source citations.

    Attributes:
        answer: The generated answer text.
        citations: List of source citations.
        overall_confidence: Average relevance across citations.
        low_confidence: Whether confidence is below threshold.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    overall_confidence: float = 0.0
    low_confidence: bool = False


class CitationTracker:
    """Extracts citations from retrieved chunks and formats them.

    Args:
        confidence_threshold: Threshold below which answers are
            flagged as low confidence.
    """

    def __init__(
        self,
        confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._threshold = confidence_threshold

    def extract_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerWithCitations:
        """Create an AnswerWithCitations from retrieved chunks.

        Args:
            answer: The LLM-generated answer.
            chunks: Retrieved chunks used as context.

        Returns:
            AnswerWithCitations with extracted citations.
        """
        citations = []
        for chunk in chunks:
            excerpt = chunk.content[:_EXCERPT_LENGTH]
            if len(chunk.content) > _EXCERPT_LENGTH:
                excerpt += "..."

            citations.append(
                Citation(
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    relevance_score=chunk.similarity_score,
                    excerpt=excerpt,
                )
            )

        confidence = 0.0
        if citations:
            confidence = sum(c.relevance_score for c in citations) / len(citations)

        low_confidence = confidence < self._threshold

        if low_confidence:
            logger.warning(
                "Low confidence answer (%.3f < %.3f)",
                confidence,
                self._threshold,
            )

        return AnswerWithCitations(
            answer=answer,
            citations=citations,
            overall_confidence=confidence,
            low_confidence=low_confidence,
        )

    def generate_source_markdown(self, result: AnswerWithCitations) -> str:
        """Format citations as markdown for display.

        Args:
            result: An AnswerWithCitations to format.

        Returns:
            Markdown-formatted string of source references.
        """
        if not result.citations:
            return "_No sources available._"

        lines = ["### Sources\n"]
        for i, citation in enumerate(result.citations, start=1):
            filename = citation.source_file.split("/")[-1]
            lines.append(
                f"**{i}. {filename}** "
                f"(Page {citation.page_number}, "
                f"Relevance: {citation.relevance_score:.2f})"
            )
            lines.append(f"> {citation.excerpt}")
            lines.append("")

        conf_label = "Low" if result.low_confidence else "Good"
        lines.append(
            f"**Overall Confidence:** {result.overall_confidence:.2f} ({conf_label})"
        )

        return "\n".join(lines)

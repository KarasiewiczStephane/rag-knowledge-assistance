"""Tests for the prompt builder."""

from src.generation.prompt_builder import PromptBuilder
from src.retrieval.vector_store import RetrievedChunk


def _make_chunk(
    content: str = "Context text",
    source: str = "/docs/file.txt",
    page: int = 1,
) -> RetrievedChunk:
    """Create a test chunk."""
    return RetrievedChunk(
        content=content,
        source_file=source,
        page_number=page,
        chunk_index=0,
        similarity_score=0.9,
    )


def test_build_prompt_with_context() -> None:
    """Prompt includes retrieved context and question."""
    builder = PromptBuilder()
    chunks = [_make_chunk("ML is a field of study.")]
    prompt = builder.build_prompt("What is ML?", chunks)
    assert "ML is a field of study." in prompt
    assert "What is ML?" in prompt
    assert "Retrieved Context" in prompt


def test_build_prompt_no_context() -> None:
    """Prompt without context still includes the question."""
    builder = PromptBuilder()
    prompt = builder.build_prompt("What is ML?", [])
    assert "What is ML?" in prompt
    assert "Retrieved Context" not in prompt


def test_build_prompt_with_history() -> None:
    """Prompt includes conversation history when provided."""
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        "Follow-up question?",
        [],
        conversation_history="User: Hi\nAssistant: Hello",
    )
    assert "Conversation History" in prompt
    assert "User: Hi" in prompt


def test_build_prompt_source_formatting() -> None:
    """Source references include filename and page number."""
    builder = PromptBuilder()
    chunks = [_make_chunk(source="/path/to/doc.pdf", page=3)]
    prompt = builder.build_prompt("Q?", chunks)
    assert "doc.pdf" in prompt
    assert "Page 3" in prompt


def test_build_prompt_multiple_chunks() -> None:
    """Multiple chunks are numbered sequentially."""
    builder = PromptBuilder()
    chunks = [
        _make_chunk("First context"),
        _make_chunk("Second context"),
    ]
    prompt = builder.build_prompt("Q?", chunks)
    assert "Source 1" in prompt
    assert "Source 2" in prompt


def test_system_prompt_property() -> None:
    """system_prompt property returns the configured value."""
    builder = PromptBuilder(system_prompt="Custom prompt")
    assert builder.system_prompt == "Custom prompt"


def test_default_system_prompt() -> None:
    """Default system prompt includes RAG instructions."""
    builder = PromptBuilder()
    assert "context" in builder.system_prompt.lower()


def test_build_summarization_prompt() -> None:
    """Summarization prompt includes conversation text."""
    builder = PromptBuilder()
    prompt = builder.build_summarization_prompt("User: Hello\nBot: Hi there")
    assert "Summarize" in prompt
    assert "User: Hello" in prompt

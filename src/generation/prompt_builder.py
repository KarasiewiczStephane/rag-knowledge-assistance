"""Prompt construction for RAG-based question answering."""

import logging

from src.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions based on "
    "the provided context. Follow these rules:\n"
    "1. Only use information from the provided context to answer.\n"
    "2. If the context doesn't contain enough information, say so.\n"
    "3. Cite your sources by referring to the document names.\n"
    "4. Be concise and accurate."
)


class PromptBuilder:
    """Builds prompts combining user questions with retrieved context.

    Args:
        system_prompt: Custom system prompt. Uses default if None.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        """Return the configured system prompt."""
        return self._system_prompt

    def build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        conversation_history: str | None = None,
    ) -> str:
        """Build a prompt incorporating retrieved context.

        Args:
            question: The user's question.
            chunks: Retrieved context chunks.
            conversation_history: Optional prior conversation context.

        Returns:
            Formatted prompt string for the LLM.
        """
        parts: list[str] = []

        if conversation_history:
            parts.append("## Conversation History")
            parts.append(conversation_history)
            parts.append("")

        if chunks:
            parts.append("## Retrieved Context")
            for i, chunk in enumerate(chunks, start=1):
                source = chunk.source_file.split("/")[-1]
                header = f"[Source {i}: {source}, Page {chunk.page_number}]"
                parts.append(header)
                parts.append(chunk.content)
                parts.append("")

        parts.append("## Question")
        parts.append(question)

        prompt = "\n".join(parts)
        logger.debug("Built prompt with %d context chunks", len(chunks))
        return prompt

    def build_summarization_prompt(self, conversation_text: str) -> str:
        """Build a prompt for summarizing a conversation.

        Args:
            conversation_text: The conversation to summarize.

        Returns:
            Formatted summarization prompt.
        """
        return (
            "Summarize the following conversation concisely, "
            "capturing the key topics discussed and any conclusions "
            "reached:\n\n"
            f"{conversation_text}"
        )

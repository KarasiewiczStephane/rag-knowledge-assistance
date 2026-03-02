"""Tests for LLM client abstraction (mocked API calls)."""

from unittest.mock import MagicMock

import pytest

from src.generation.llm_client import (
    AnthropicClient,
    LLMResponse,
    OpenAIClient,
    get_llm_client,
)


def test_llm_response_dataclass() -> None:
    """LLMResponse has content, model, and usage."""
    resp = LLMResponse(
        content="Hello",
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert resp.content == "Hello"
    assert resp.model == "test-model"
    assert resp.usage["input_tokens"] == 10


def test_anthropic_client_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicClient generates a response via mocked API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = AnthropicClient(model="test-model")

    mock_anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer text")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_anthropic.messages.create.return_value = mock_response
    client._client = mock_anthropic

    result = client.generate("What is Python?")
    assert result.content == "Answer text"
    assert result.usage["input_tokens"] == 100


def test_anthropic_client_with_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicClient passes system prompt correctly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = AnthropicClient()
    mock_anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_anthropic.messages.create.return_value = mock_response
    client._client = mock_anthropic

    client.generate("Q", system_prompt="Be helpful")
    call_kwargs = mock_anthropic.messages.create.call_args[1]
    assert call_kwargs["system"] == "Be helpful"


def test_openai_client_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIClient generates a response via mocked API."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAIClient(model="gpt-3.5-turbo")

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40
    mock_openai.chat.completions.create.return_value = mock_response
    client._client = mock_openai

    result = client.generate("What is ML?")
    assert result.content == "OpenAI answer"
    assert result.usage["output_tokens"] == 40


def test_openai_client_with_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIClient includes system message when provided."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAIClient()
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_openai.chat.completions.create.return_value = mock_response
    client._client = mock_openai

    client.generate("Q", system_prompt="Be concise")
    call_kwargs = mock_openai.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be concise"


def test_get_llm_client_anthropic() -> None:
    """get_llm_client returns AnthropicClient for anthropic."""
    config = {"llm": {"provider": "anthropic"}}
    client = get_llm_client(config)
    assert isinstance(client, AnthropicClient)


def test_get_llm_client_openai() -> None:
    """get_llm_client returns OpenAIClient for openai."""
    config = {"llm": {"provider": "openai"}}
    client = get_llm_client(config)
    assert isinstance(client, OpenAIClient)


def test_get_llm_client_unsupported() -> None:
    """get_llm_client raises ValueError for unknown provider."""
    config = {"llm": {"provider": "unknown"}}
    with pytest.raises(ValueError, match="Unsupported"):
        get_llm_client(config)


def test_get_llm_client_defaults() -> None:
    """get_llm_client uses defaults when config is minimal."""
    config = {}
    client = get_llm_client(config)
    assert isinstance(client, AnthropicClient)

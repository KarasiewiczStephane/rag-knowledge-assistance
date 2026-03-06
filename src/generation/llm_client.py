"""LLM client abstraction supporting Anthropic and OpenAI providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.utils.config import get_api_key

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: Generated text content.
        model: Model identifier used.
        usage: Token usage stats (input_tokens, output_tokens).
    """

    content: str
    model: str
    usage: dict[str, int]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt / question.
            system_prompt: Optional system-level instructions.

        Returns:
            LLMResponse with generated content.
        """


class AnthropicClient(BaseLLMClient):
    """Client for the Anthropic Claude API.

    Args:
        model: Model identifier to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the Anthropic client.

        Returns:
            Initialized Anthropic client.
        """
        if self._client is None:
            import anthropic

            api_key = get_api_key("anthropic")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using the Anthropic API.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with the generated answer.
        """
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        logger.info(
            "Anthropic response: model=%s, tokens=%s",
            self._model,
            usage,
        )
        return LLMResponse(content=content, model=self._model, usage=usage)


class OllamaClient(BaseLLMClient):
    """Client for locally-hosted Ollama models (free, no API key needed).

    Args:
        model: Ollama model name (e.g. "llama3", "mistral", "phi3").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        base_url: Ollama server URL.
    """

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using a local Ollama model.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with the generated answer.
        """
        import requests

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        response = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"]
        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }

        logger.info(
            "Ollama response: model=%s, tokens=%s",
            self._model,
            usage,
        )
        return LLMResponse(content=content, model=self._model, usage=usage)


class OpenAIClient(BaseLLMClient):
    """Client for the OpenAI API.

    Args:
        model: Model identifier to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client.

        Returns:
            Initialized OpenAI client.
        """
        if self._client is None:
            import openai

            api_key = get_api_key("openai")
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using the OpenAI API.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with the generated answer.
        """
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        logger.info(
            "OpenAI response: model=%s, tokens=%s",
            self._model,
            usage,
        )
        return LLMResponse(content=content, model=self._model, usage=usage)


def get_llm_client(config: dict[str, Any]) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.

    Args:
        config: Application config dictionary with 'llm' section.

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "anthropic")
    model = llm_config.get("model", "")
    temperature = llm_config.get("temperature", 0.1)
    max_tokens = llm_config.get("max_tokens", 2048)

    if provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "ollama":
        return OllamaClient(
            model=model or "llama3",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=llm_config.get("base_url", "http://localhost:11434"),
        )
    elif provider == "openai":
        return OpenAIClient(
            model=model or "gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

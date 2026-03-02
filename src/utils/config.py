"""Configuration loader with YAML parsing and environment variable overrides."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file with environment variable overrides.

    Args:
        config_path: Path to the YAML config file. Defaults to configs/config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply environment variable overrides to the configuration.

    Supported env vars:
        - LLM_PROVIDER -> config["llm"]["provider"]
        - LLM_MODEL -> config["llm"]["model"]
        - LLM_TEMPERATURE -> config["llm"]["temperature"]
        - LLM_MAX_TOKENS -> config["llm"]["max_tokens"]
        - CHUNK_SIZE -> config["ingestion"]["chunk_size"]
        - CHUNK_OVERLAP -> config["ingestion"]["chunk_overlap"]
        - RETRIEVAL_TOP_K -> config["retrieval"]["top_k"]
        - SIMILARITY_THRESHOLD -> config["retrieval"]["similarity_threshold"]
        - EMBEDDING_MODEL -> config["embeddings"]["model"]
        - VECTOR_STORE_DIR -> config["vector_store"]["persist_directory"]
        - MEMORY_WINDOW_SIZE -> config["memory"]["window_size"]

    Args:
        config: Configuration dictionary to modify in place.
    """
    env_map: list[tuple[str, list[str], type]] = [
        ("LLM_PROVIDER", ["llm", "provider"], str),
        ("LLM_MODEL", ["llm", "model"], str),
        ("LLM_TEMPERATURE", ["llm", "temperature"], float),
        ("LLM_MAX_TOKENS", ["llm", "max_tokens"], int),
        ("CHUNK_SIZE", ["ingestion", "chunk_size"], int),
        ("CHUNK_OVERLAP", ["ingestion", "chunk_overlap"], int),
        ("RETRIEVAL_TOP_K", ["retrieval", "top_k"], int),
        ("SIMILARITY_THRESHOLD", ["retrieval", "similarity_threshold"], float),
        ("EMBEDDING_MODEL", ["embeddings", "model"], str),
        ("VECTOR_STORE_DIR", ["vector_store", "persist_directory"], str),
        ("MEMORY_WINDOW_SIZE", ["memory", "window_size"], int),
    ]

    for env_var, keys, cast in env_map:
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(config, keys, cast(value))


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dictionary by key path.

    Args:
        d: Dictionary to modify.
        keys: List of keys forming the path.
        value: Value to set.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_api_key(provider: str) -> str:
    """Retrieve an API key from environment variables.

    Args:
        provider: Provider name (e.g. 'anthropic', 'openai').

    Returns:
        The API key string.

    Raises:
        ValueError: If the API key is not set.
    """
    env_var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"API key not set. Please set the {env_var} environment variable."
        )
    return key

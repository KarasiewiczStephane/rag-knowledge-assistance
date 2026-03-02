"""Tests for configuration loading and environment variable overrides."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import get_api_key, load_config


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config = {
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "supported_formats": ["pdf", "docx", "md", "txt"],
        },
        "retrieval": {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "use_reranker": False,
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "embeddings": {"model": "all-MiniLM-L6-v2"},
        "memory": {"window_size": 5, "enable_summarization": True},
        "vector_store": {
            "persist_directory": "./data/chromadb",
            "collection_name": "documents",
        },
        "logging": {"level": "INFO"},
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


def test_load_config_valid(config_dir: Path) -> None:
    """Loading a valid YAML config returns the expected dictionary."""
    config = load_config(str(config_dir))
    assert config["ingestion"]["chunk_size"] == 500
    assert config["llm"]["provider"] == "anthropic"
    assert config["retrieval"]["top_k"] == 5
    assert config["embeddings"]["model"] == "all-MiniLM-L6-v2"


def test_load_config_missing_file() -> None:
    """Loading a nonexistent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_default_path() -> None:
    """Loading config with default path succeeds with project config."""
    config = load_config()
    assert "ingestion" in config
    assert "llm" in config
    assert "retrieval" in config


def test_env_override_llm_provider(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variable LLM_PROVIDER overrides config value."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    config = load_config(str(config_dir))
    assert config["llm"]["provider"] == "openai"


def test_env_override_chunk_size(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variable CHUNK_SIZE overrides config value."""
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    config = load_config(str(config_dir))
    assert config["ingestion"]["chunk_size"] == 1000


def test_env_override_temperature(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variable LLM_TEMPERATURE overrides config value."""
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    config = load_config(str(config_dir))
    assert config["llm"]["temperature"] == 0.5


def test_env_override_top_k(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variable RETRIEVAL_TOP_K overrides config value."""
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    config = load_config(str(config_dir))
    assert config["retrieval"]["top_k"] == 10


def test_get_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_api_key returns value when environment variable is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    assert get_api_key("anthropic") == "test-key-123"


def test_get_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_api_key raises ValueError when environment variable is not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key not set"):
        get_api_key("anthropic")


def test_load_config_malformed(tmp_path: Path) -> None:
    """Loading a malformed YAML config handles gracefully."""
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("valid_yaml: true\nnested:\n  key: value\n")
    config = load_config(str(config_file))
    assert config["valid_yaml"] is True


def test_all_directories_exist() -> None:
    """All required source directories exist."""
    base = Path(__file__).resolve().parent.parent / "src"
    required_dirs = [
        "ingestion",
        "retrieval",
        "generation",
        "memory",
        "evaluation",
        "dashboard",
        "utils",
    ]
    for d in required_dirs:
        assert (base / d).is_dir(), f"Missing directory: src/{d}"
        assert (base / d / "__init__.py").exists(), f"Missing __init__.py in src/{d}"


def test_imports_work() -> None:
    """All package imports resolve without errors."""
    import importlib

    modules = [
        "src.ingestion",
        "src.retrieval",
        "src.generation",
        "src.memory",
        "src.evaluation",
        "src.dashboard",
        "src.utils",
        "src.utils.config",
        "src.utils.logger",
    ]
    for mod in modules:
        importlib.import_module(mod)

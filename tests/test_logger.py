"""Tests for the logging utility module."""

import logging

from src.utils.logger import configure_logging, get_logger


def test_get_logger_returns_logger() -> None:
    """get_logger returns a logging.Logger with the given name."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_default_level() -> None:
    """get_logger defaults to INFO level."""
    logger = get_logger("test_info")
    assert logger.level == logging.INFO


def test_get_logger_custom_level() -> None:
    """get_logger respects custom level argument."""
    logger = get_logger("test_debug", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_get_logger_has_handler() -> None:
    """get_logger attaches a StreamHandler."""
    logger = get_logger("test_handler")
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_configure_logging_sets_root_level() -> None:
    """configure_logging sets the root logger level from config."""
    config = {"logging": {"level": "WARNING"}}
    configure_logging(config)
    root = logging.getLogger()
    assert root.level == logging.WARNING
    # reset
    config = {"logging": {"level": "INFO"}}
    configure_logging(config)


def test_configure_logging_default_level() -> None:
    """configure_logging defaults to INFO when no level specified."""
    configure_logging({})
    root = logging.getLogger()
    assert root.level == logging.INFO

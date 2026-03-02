"""Structured logging setup for the application."""

import logging
import sys
from typing import Any


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create and configure a logger with structured formatting.

    Args:
        name: Logger name, typically __name__ from the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def configure_logging(config: dict[str, Any]) -> None:
    """Configure root logging from application config.

    Args:
        config: Application configuration dictionary with optional 'logging' section.
    """
    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    fmt = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

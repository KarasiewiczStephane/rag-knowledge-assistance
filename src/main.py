"""Main entry point for the RAG Knowledge Assistant."""

import logging

from src.utils.config import load_config
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize and run the RAG Knowledge Assistant."""
    config = load_config()
    configure_logging(config)
    logger.info("RAG Knowledge Assistant starting up")
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    logger.info("Provider: %s | Model: %s", provider, model)
    logger.info("Use 'streamlit run src/dashboard/app.py' to launch the UI")


if __name__ == "__main__":
    main()

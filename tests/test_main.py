"""Tests for the main entry point."""

from src.main import main


def test_main_runs_without_error() -> None:
    """main() initializes the application without raising."""
    main()

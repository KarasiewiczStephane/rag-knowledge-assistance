"""Tests for the Streamlit dashboard module (import and structure)."""

import importlib


def test_dashboard_module_imports() -> None:
    """Dashboard app module can be imported."""
    mod = importlib.import_module("src.dashboard.app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_init_pipeline")
    assert hasattr(mod, "_init_session")
    assert hasattr(mod, "_render_sidebar")
    assert hasattr(mod, "_render_chat")


def test_dashboard_main_is_callable() -> None:
    """main function is callable."""
    from src.dashboard.app import main

    assert callable(main)

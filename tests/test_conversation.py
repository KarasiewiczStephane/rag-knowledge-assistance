"""Tests for conversation memory management."""

from pathlib import Path

import pytest

from src.memory.conversation import (
    ConversationMemory,
    ConversationSession,
    Message,
)


@pytest.fixture()
def memory(tmp_path: Path) -> ConversationMemory:
    """Create a ConversationMemory with temp sessions directory."""
    return ConversationMemory(window_size=3, sessions_dir=str(tmp_path / "sessions"))


def test_create_session(memory: ConversationMemory) -> None:
    """create_session returns a valid session ID."""
    sid = memory.create_session()
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_get_session(memory: ConversationMemory) -> None:
    """get_session retrieves a created session."""
    sid = memory.create_session()
    session = memory.get_session(sid)
    assert session is not None
    assert session.session_id == sid


def test_get_session_missing(memory: ConversationMemory) -> None:
    """get_session returns None for unknown ID."""
    assert memory.get_session("nonexistent") is None


def test_add_message(memory: ConversationMemory) -> None:
    """add_message appends to session history."""
    sid = memory.create_session()
    memory.add_message(sid, "user", "Hello")
    memory.add_message(sid, "assistant", "Hi there")
    session = memory.get_session(sid)
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"


def test_add_message_invalid_session(
    memory: ConversationMemory,
) -> None:
    """add_message raises KeyError for unknown session."""
    with pytest.raises(KeyError):
        memory.add_message("bad-id", "user", "Hello")


def test_get_window(memory: ConversationMemory) -> None:
    """get_window returns last N exchanges."""
    sid = memory.create_session()
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        memory.add_message(sid, role, f"Message {i}")

    window = memory.get_window(sid)
    assert len(window) == 6  # window_size=3 -> 3*2 = 6 messages


def test_get_window_empty(memory: ConversationMemory) -> None:
    """get_window returns empty for unknown session."""
    assert memory.get_window("nonexistent") == []


def test_get_window_text(memory: ConversationMemory) -> None:
    """get_window_text returns formatted conversation text."""
    sid = memory.create_session()
    memory.add_message(sid, "user", "What is ML?")
    memory.add_message(sid, "assistant", "ML is...")
    text = memory.get_window_text(sid)
    assert "User: What is ML?" in text
    assert "Assistant: ML is..." in text


def test_get_window_text_empty(memory: ConversationMemory) -> None:
    """get_window_text returns empty string for no messages."""
    assert memory.get_window_text("nonexistent") == ""


def test_clear_history(memory: ConversationMemory) -> None:
    """clear_history removes all messages."""
    sid = memory.create_session()
    memory.add_message(sid, "user", "Hello")
    memory.clear_history(sid)
    session = memory.get_session(sid)
    assert len(session.messages) == 0


def test_to_dict(memory: ConversationMemory) -> None:
    """to_dict serializes session to dict."""
    sid = memory.create_session()
    memory.add_message(sid, "user", "Hi")
    data = memory.to_dict(sid)
    assert data["session_id"] == sid
    assert len(data["messages"]) == 1


def test_to_dict_missing_session(
    memory: ConversationMemory,
) -> None:
    """to_dict raises KeyError for unknown session."""
    with pytest.raises(KeyError):
        memory.to_dict("bad-id")


def test_save_and_load_session(
    memory: ConversationMemory,
) -> None:
    """Sessions can be saved and loaded from disk."""
    sid = memory.create_session()
    memory.add_message(sid, "user", "Question")
    memory.add_message(sid, "assistant", "Answer")
    memory.save_session(sid)

    new_memory = ConversationMemory(sessions_dir=str(memory._sessions_dir))
    loaded_sid = new_memory.load_session(sid)
    session = new_memory.get_session(loaded_sid)
    assert len(session.messages) == 2
    assert session.messages[0].content == "Question"


def test_load_session_missing_file(
    memory: ConversationMemory,
) -> None:
    """load_session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        memory.load_session("nonexistent-id")


def test_list_sessions(memory: ConversationMemory) -> None:
    """list_sessions returns all session IDs."""
    sid1 = memory.create_session()
    sid2 = memory.create_session()
    sessions = memory.list_sessions()
    assert sid1 in sessions
    assert sid2 in sessions


def test_message_timestamp() -> None:
    """Message gets a timestamp on creation."""
    msg = Message(role="user", content="Hello")
    assert msg.timestamp != ""


def test_conversation_session_defaults() -> None:
    """ConversationSession gets defaults on creation."""
    session = ConversationSession()
    assert session.session_id != ""
    assert session.created_at != ""
    assert session.messages == []


def test_add_message_with_citations(
    memory: ConversationMemory,
) -> None:
    """Messages can include citation data."""
    sid = memory.create_session()
    citations = [{"source": "doc.txt", "page": 1}]
    memory.add_message(sid, "assistant", "Answer", citations=citations)
    session = memory.get_session(sid)
    assert session.messages[0].citations == citations

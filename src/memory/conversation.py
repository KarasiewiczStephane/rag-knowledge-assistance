"""Conversation memory with sliding window and session management."""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Either 'user' or 'assistant'.
        content: Message text content.
        timestamp: ISO timestamp of when the message was created.
        citations: Optional list of citation dicts.
    """

    role: str
    content: str
    timestamp: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ConversationSession:
    """A conversation session with message history.

    Attributes:
        session_id: Unique session identifier.
        messages: Ordered list of messages.
        summary: Optional summary of the conversation so far.
        created_at: ISO timestamp of session creation.
    """

    session_id: str = ""
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        """Set defaults if not provided."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ConversationMemory:
    """Manages conversation history with sliding window.

    Args:
        window_size: Number of recent exchanges to keep in window.
        sessions_dir: Directory for persisting session files.
    """

    def __init__(
        self,
        window_size: int = 5,
        sessions_dir: str = "data/sessions",
    ) -> None:
        self._window_size = window_size
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self) -> str:
        """Create a new conversation session.

        Returns:
            Session ID for the new session.
        """
        session = ConversationSession()
        self._sessions[session.session_id] = session
        logger.info("Created session: %s", session.session_id)
        return session.session_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Get a session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            ConversationSession if found, None otherwise.
        """
        return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a message to a session's history.

        Args:
            session_id: Target session ID.
            role: 'user' or 'assistant'.
            content: Message text.
            citations: Optional citation data.

        Raises:
            KeyError: If the session does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        message = Message(
            role=role,
            content=content,
            citations=citations or [],
        )
        session.messages.append(message)
        logger.debug(
            "Added %s message to session %s",
            role,
            session_id,
        )

    def get_window(self, session_id: str) -> list[Message]:
        """Get the most recent messages within the window.

        Returns the last N exchanges (user + assistant pairs).

        Args:
            session_id: Target session ID.

        Returns:
            List of recent Message objects.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []

        max_messages = self._window_size * 2
        return session.messages[-max_messages:]

    def get_window_text(self, session_id: str) -> str:
        """Get windowed history formatted as text.

        Args:
            session_id: Target session ID.

        Returns:
            Formatted conversation history string.
        """
        messages = self.get_window(session_id)
        if not messages:
            return ""

        parts = []
        for msg in messages:
            label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{label}: {msg.content}")

        return "\n".join(parts)

    def clear_history(self, session_id: str) -> None:
        """Clear all messages from a session.

        Args:
            session_id: Target session ID.
        """
        session = self._sessions.get(session_id)
        if session:
            session.messages.clear()
            session.summary = ""
            logger.info("Cleared history for session %s", session_id)

    def to_dict(self, session_id: str) -> dict[str, Any]:
        """Serialize a session to a dictionary.

        Args:
            session_id: Target session ID.

        Returns:
            Dictionary representation of the session.

        Raises:
            KeyError: If the session does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return asdict(session)

    def save_session(self, session_id: str) -> Path:
        """Persist a session to a JSON file.

        Args:
            session_id: Target session ID.

        Returns:
            Path to the saved file.

        Raises:
            KeyError: If the session does not exist.
        """
        data = self.to_dict(session_id)
        file_path = self._sessions_dir / f"{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved session %s to %s", session_id, file_path)
        return file_path

    def load_session(self, session_id: str) -> str:
        """Load a session from a JSON file.

        Args:
            session_id: Session ID to load.

        Returns:
            The loaded session ID.

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        file_path = self._sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", ""),
                citations=m.get("citations", []),
            )
            for m in data.get("messages", [])
        ]

        session = ConversationSession(
            session_id=data["session_id"],
            messages=messages,
            summary=data.get("summary", ""),
            created_at=data.get("created_at", ""),
        )
        self._sessions[session.session_id] = session
        logger.info("Loaded session %s", session_id)
        return session.session_id

    def list_sessions(self) -> list[str]:
        """List all session IDs (in-memory and on disk).

        Returns:
            List of session ID strings.
        """
        disk_ids = {f.stem for f in self._sessions_dir.glob("*.json")}
        memory_ids = set(self._sessions.keys())
        return sorted(disk_ids | memory_ids)

import uuid
from datetime import datetime
from typing import Dict, List


class SessionManager:
    """
    Manages per-exam proctoring sessions in memory.
    Each session tracks its metadata and a log of gaze events.

    Note: For production, replace the in-memory store with a
    persistent backend (e.g. Redis, PostgreSQL).
    """

    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    def start_session(self, exam_id: str, student_id: str) -> str:
        """Create a new proctoring session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "events": [],
        }
        return session_id

    def end_session(self, session_id: str) -> dict:
        """Mark a session as ended and return a summary."""
        session = self._get_session(session_id)
        session["ended_at"] = datetime.utcnow().isoformat()

        events: List[dict] = session["events"]
        suspicious_count = sum(1 for e in events if e.get("is_suspicious"))

        return {
            "total_frames": len(events),
            "suspicious_frames": suspicious_count,
            "started_at": session["started_at"],
            "ended_at": session["ended_at"],
        }

    def record_event(self, session_id: str, event: dict) -> None:
        """Append a gaze analysis event to the session log."""
        session = self._get_session(session_id)
        session["events"].append(event)

    def get_session(self, session_id: str) -> dict:
        """Return the full session object."""
        return self._get_session(session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        return self._sessions[session_id]

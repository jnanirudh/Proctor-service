import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

@dataclass
class ViolationEvent:
    timestamp: float
    zone: str
    frame_index: int

@dataclass
class ExamSession:
    session_id: str
    user_id: str
    quiz_id: str
    started_at: float = field(default_factory=time.time)
    total_frames: int = 0
    violation_frames: int = 0
    violations: List[ViolationEvent] = field(default_factory=list)
    consecutive_violation_count: int = 0
    cooldown_remaining: int = 0

    @property
    def violation_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.violation_frames / self.total_frames

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "quiz_id": self.quiz_id,
            "duration_seconds": time.time() - self.started_at,
            "total_frames_analyzed": self.total_frames,
            "violation_frames": self.violation_frames,
            "violation_rate": round(self.violation_rate, 4),
            "violation_count": len(self.violations),
            "violations": [
                {"timestamp": v.timestamp, "zone": v.zone, "frame": v.frame_index}
                for v in self.violations[-50:]  # Last 50 violations max
            ]
        }

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ExamSession] = {}

    def create_session(self, session_id: str, user_id: str, quiz_id: str) -> ExamSession:
        session = ExamSession(session_id=session_id, user_id=user_id, quiz_id=quiz_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ExamSession | None:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> dict | None:
        session = self._sessions.pop(session_id, None)
        if session:
            return session.to_summary()
        return None

    def record_frame(self, session_id: str, gaze_result, config) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        session.total_frames += 1

        if session.cooldown_remaining > 0:
            session.cooldown_remaining -= 1

        if gaze_result.is_violation:
            session.violation_frames += 1
            session.consecutive_violation_count += 1

            should_alert = (
                session.consecutive_violation_count >= config.VIOLATION_FRAME_THRESHOLD
                and session.cooldown_remaining == 0
            )
            if should_alert:
                event = ViolationEvent(
                    timestamp=time.time(),
                    zone=gaze_result.zone.value,
                    frame_index=session.total_frames
                )
                session.violations.append(event)
                session.cooldown_remaining = config.VIOLATION_COOLDOWN_FRAMES
                session.consecutive_violation_count = 0

            return {
                "status": "VIOLATION",
                "zone": gaze_result.zone.value,
                "alert": should_alert,
                "message": gaze_result.message,
                "violation_rate": session.violation_rate
            }
        else:
            session.consecutive_violation_count = 0
            return {
                "status": "OK",
                "zone": gaze_result.zone.value,
                "alert": False,
                "message": gaze_result.message,
                "violation_rate": session.violation_rate
            }
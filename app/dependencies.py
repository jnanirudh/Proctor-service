# app/dependencies.py
"""
FastAPI dependency providers.
These are injected into route handlers via Depends().
Centralising them here means you swap implementations (e.g. real DB)
in one place without touching any router.
"""
from app.gaze_detector import GazeDetector
from app.session_manager import SessionManager
from config import gaze_config, GazeConfig

# These are set once at startup via the lifespan in main.py
_detector: GazeDetector | None = None
_session_manager: SessionManager | None = None


def set_detector(d: GazeDetector):
    global _detector
    _detector = d


def set_session_manager(sm: SessionManager):
    global _session_manager
    _session_manager = sm


def get_detector() -> GazeDetector:
    if _detector is None:
        raise RuntimeError("GazeDetector not initialised")
    return _detector


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialised")
    return _session_manager


def get_config() -> GazeConfig:
    return gaze_config

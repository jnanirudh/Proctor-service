# config.py
from dataclasses import dataclass

@dataclass
class GazeConfig:
    # Horizontal: iris ratio < this = looking LEFT (violation)
    HORIZONTAL_LEFT_THRESHOLD: float = 0.40
    # Horizontal: iris ratio > this = looking RIGHT (violation)
    HORIZONTAL_RIGHT_THRESHOLD: float = 0.60
    # Vertical: iris ratio < this = looking UP (violation)
    VERTICAL_UP_THRESHOLD: float = 0.35
    # Vertical: iris ratio > this = looking DOWN (violation)
    VERTICAL_DOWN_THRESHOLD: float = 0.65

    # How many consecutive violations before flagging
    VIOLATION_FRAME_THRESHOLD: int = 15  # ~0.5 sec at 30fps

    # Cooldown frames after a violation is reported
    VIOLATION_COOLDOWN_FRAMES: int = 30

gaze_config = GazeConfig()
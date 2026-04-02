# app/gaze_detector.py
# Uses mediapipe Tasks API (mediapipe >= 0.10.x)
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks

# Landmark indices (same 478-point mesh as legacy FaceMesh)
_LEFT_IRIS        = [474, 475, 476, 477]
_RIGHT_IRIS       = [469, 470, 471, 472]
_LEFT_EYE_CORNERS = [33, 133]
_RIGHT_EYE_CORNERS= [362, 263]
_LEFT_EYE_TB      = [159, 145]
_RIGHT_EYE_TB     = [386, 374]

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"


class GazeZone(str, Enum):
    SCREEN         = "SCREEN"
    LEFT           = "LEFT"
    RIGHT          = "RIGHT"
    UP             = "UP"
    DOWN           = "DOWN"
    NO_FACE        = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"


@dataclass
class GazeResult:
    zone:             GazeZone
    confidence:       float
    horizontal_ratio: float   # 0=far left, 1=far right, 0.5=center
    vertical_ratio:   float   # 0=far up,   1=far down,  0.5=center
    is_violation:     bool
    message:          str


class GazeDetector:

    def __init__(self, config):
        self.config = config

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {MODEL_PATH}.\n"
                "Download it with:\n"
                "  curl -L -o models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_faces=2,            # detect up to 2 to catch multiple faces
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_frame(self, frame_bytes: bytes) -> GazeResult:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return GazeResult(GazeZone.NO_FACE, 0.0, 0.5, 0.5, True, "Invalid frame")

        img_h, img_w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return GazeResult(GazeZone.NO_FACE, 0.0, 0.5, 0.5, True, "No face detected")

        if len(result.face_landmarks) > 1:
            return GazeResult(GazeZone.MULTIPLE_FACES, 1.0, 0.5, 0.5, True, "Multiple faces detected")

        landmarks = result.face_landmarks[0]   # list of NormalizedLandmark
        return self._process_landmarks(landmarks, img_w, img_h)

    def close(self):
        self._landmarker.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_landmarks(self, landmarks, img_w, img_h) -> GazeResult:
        def pts(indices):
            return np.array([
                (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
                for i in indices
            ])

        left_iris_center  = pts(_LEFT_IRIS).mean(axis=0).astype(int)
        right_iris_center = pts(_RIGHT_IRIS).mean(axis=0).astype(int)
        left_corners      = pts(_LEFT_EYE_CORNERS)
        right_corners     = pts(_RIGHT_EYE_CORNERS)
        left_tb           = pts(_LEFT_EYE_TB)
        right_tb          = pts(_RIGHT_EYE_TB)

        h_ratio = (
            self._h_ratio(left_iris_center, left_corners) +
            self._h_ratio(right_iris_center, right_corners)
        ) / 2

        v_ratio = (
            self._v_ratio(left_iris_center, left_tb) +
            self._v_ratio(right_iris_center, right_tb)
        ) / 2

        cfg = self.config
        zone, message = self._classify(h_ratio, v_ratio, cfg)
        is_violation  = zone != GazeZone.SCREEN
        confidence    = self._confidence(h_ratio, v_ratio)

        return GazeResult(zone, confidence, h_ratio, v_ratio, is_violation, message)

    @staticmethod
    def _h_ratio(iris_center, corners) -> float:
        left_x, right_x = corners[0][0], corners[1][0]
        width = right_x - left_x
        if width == 0:
            return 0.5
        return float(np.clip((iris_center[0] - left_x) / width, 0.0, 1.0))

    @staticmethod
    def _v_ratio(iris_center, top_bottom) -> float:
        top_y, bot_y = top_bottom[0][1], top_bottom[1][1]
        height = bot_y - top_y
        if height == 0:
            return 0.5
        return float(np.clip((iris_center[1] - top_y) / height, 0.0, 1.0))

    @staticmethod
    def _classify(h: float, v: float, cfg) -> tuple[GazeZone, str]:
        if v < cfg.VERTICAL_UP_THRESHOLD:
            return GazeZone.UP,    "Looking up"
        if v > cfg.VERTICAL_DOWN_THRESHOLD:
            return GazeZone.DOWN,  "Looking down"
        if h < cfg.HORIZONTAL_LEFT_THRESHOLD:
            return GazeZone.LEFT,  "Looking left"
        if h > cfg.HORIZONTAL_RIGHT_THRESHOLD:
            return GazeZone.RIGHT, "Looking right"
        return GazeZone.SCREEN, "Looking at screen"

    @staticmethod
    def _confidence(h: float, v: float) -> float:
        return float(np.clip(1.0 - abs(h - 0.5) - abs(v - 0.5), 0.0, 1.0))
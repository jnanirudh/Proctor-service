# app/gaze_detector.py
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum

class GazeZone(str, Enum):
    SCREEN = "SCREEN"       # Looking at monitor — ALLOWED
    LEFT = "LEFT"           # Looking left — VIOLATION
    RIGHT = "RIGHT"         # Looking right — VIOLATION
    UP = "UP"               # Looking up — VIOLATION
    DOWN = "DOWN"           # Looking down — VIOLATION
    NO_FACE = "NO_FACE"     # Face not detected — VIOLATION
    MULTIPLE_FACES = "MULTIPLE_FACES"  # Cheating risk

@dataclass
class GazeResult:
    zone: GazeZone
    confidence: float
    horizontal_ratio: float   # 0=far left, 1=far right, 0.5=center
    vertical_ratio: float     # 0=far up, 1=far down, 0.5=center
    is_violation: bool
    message: str

class GazeDetector:
    # Iris landmark indices from MediaPipe Face Mesh
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    
    # Eye corner landmarks
    LEFT_EYE_CORNERS = [33, 133]    # [left_corner, right_corner]
    RIGHT_EYE_CORNERS = [362, 263]
    
    # Eye top/bottom for vertical gaze
    LEFT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_TOP_BOTTOM = [386, 374]

    def __init__(self, config):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,          # Detect up to 2 to catch multiple faces
            refine_landmarks=True,    # Enables iris landmarks
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def _get_landmark_coords(self, landmarks, indices, img_w, img_h):
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append((int(lm.x * img_w), int(lm.y * img_h)))
        return np.array(coords)

    def _compute_horizontal_ratio(self, iris_center, eye_corners):
        """
        Ratio of iris position within the eye horizontally.
        0.0 = iris at far left corner
        1.0 = iris at far right corner
        ~0.5 = looking straight (at screen)
        """
        left_corner_x = eye_corners[0][0]
        right_corner_x = eye_corners[1][0]
        eye_width = right_corner_x - left_corner_x
        if eye_width == 0:
            return 0.5
        ratio = (iris_center[0] - left_corner_x) / eye_width
        return float(np.clip(ratio, 0.0, 1.0))

    def _compute_vertical_ratio(self, iris_center, top_bottom):
        """
        Ratio of iris position within the eye vertically.
        0.0 = iris at top
        1.0 = iris at bottom
        ~0.5 = looking straight
        """
        top_y = top_bottom[0][1]
        bottom_y = top_bottom[1][1]
        eye_height = bottom_y - top_y
        if eye_height == 0:
            return 0.5
        ratio = (iris_center[1] - top_y) / eye_height
        return float(np.clip(ratio, 0.0, 1.0))

    def analyze_frame(self, frame_bytes: bytes) -> GazeResult:
        # Decode frame
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return GazeResult(GazeZone.NO_FACE, 0.0, 0.5, 0.5, True, "Invalid frame")

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return GazeResult(GazeZone.NO_FACE, 0.0, 0.5, 0.5, True, "No face detected")

        if len(results.multi_face_landmarks) > 1:
            return GazeResult(GazeZone.MULTIPLE_FACES, 1.0, 0.5, 0.5, True, "Multiple faces detected")

        landmarks = results.multi_face_landmarks[0].landmark

        # Get iris centers (average of 4 iris landmarks)
        left_iris_pts = self._get_landmark_coords(landmarks, self.LEFT_IRIS, img_w, img_h)
        right_iris_pts = self._get_landmark_coords(landmarks, self.RIGHT_IRIS, img_w, img_h)
        left_iris_center = left_iris_pts.mean(axis=0).astype(int)
        right_iris_center = right_iris_pts.mean(axis=0).astype(int)

        # Get eye corners
        left_corners = self._get_landmark_coords(landmarks, self.LEFT_EYE_CORNERS, img_w, img_h)
        right_corners = self._get_landmark_coords(landmarks, self.RIGHT_EYE_CORNERS, img_w, img_h)

        # Get eye top/bottom
        left_tb = self._get_landmark_coords(landmarks, self.LEFT_EYE_TOP_BOTTOM, img_w, img_h)
        right_tb = self._get_landmark_coords(landmarks, self.RIGHT_EYE_TOP_BOTTOM, img_w, img_h)

        # Compute ratios for both eyes, then average
        h_ratio_left = self._compute_horizontal_ratio(left_iris_center, left_corners)
        h_ratio_right = self._compute_horizontal_ratio(right_iris_center, right_corners)
        h_ratio = (h_ratio_left + h_ratio_right) / 2

        v_ratio_left = self._compute_vertical_ratio(left_iris_center, left_tb)
        v_ratio_right = self._compute_vertical_ratio(right_iris_center, right_tb)
        v_ratio = (v_ratio_left + v_ratio_right) / 2

        # Classify gaze zone using configurable thresholds
        cfg = self.config
        zone, message = self._classify_gaze(h_ratio, v_ratio, cfg)
        is_violation = zone != GazeZone.SCREEN
        confidence = self._compute_confidence(h_ratio, v_ratio, zone, cfg)

        return GazeResult(zone, confidence, h_ratio, v_ratio, is_violation, message)

    def _classify_gaze(self, h_ratio, v_ratio, cfg):
        # Vertical check first (looking up/down is more obvious)
        if v_ratio < cfg.VERTICAL_UP_THRESHOLD:
            return GazeZone.UP, "Looking up"
        if v_ratio > cfg.VERTICAL_DOWN_THRESHOLD:
            return GazeZone.DOWN, "Looking down"
        
        # Horizontal check
        if h_ratio < cfg.HORIZONTAL_LEFT_THRESHOLD:
            return GazeZone.LEFT, "Looking left"
        if h_ratio > cfg.HORIZONTAL_RIGHT_THRESHOLD:
            return GazeZone.RIGHT, "Looking right"
        
        return GazeZone.SCREEN, "Looking at screen"

    def _compute_confidence(self, h_ratio, v_ratio, zone, cfg):
        """Distance from threshold as a confidence proxy."""
        center_h = abs(h_ratio - 0.5)
        center_v = abs(v_ratio - 0.5)
        # Max possible deviation from center is 0.5
        conf = 1.0 - (center_h + center_v)
        return float(np.clip(conf, 0.0, 1.0))

    def close(self):
        self.face_mesh.close()
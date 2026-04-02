"""
demo_client.py — Live webcam demo for the Eye Proctoring Service.

Run AFTER the FastAPI server is up:
    uvicorn app.main:app --reload

Then in a second terminal:
    python demo_client.py
"""

import cv2
import requests
import numpy as np
import time
import uuid
import sys

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL    = "http://localhost:8000"
SESSION_ID  = str(uuid.uuid4())
USER_ID     = "demo_user"
QUIZ_ID     = "demo_quiz"
CAMERA_INDEX = 0          # Change to 1 if your built-in cam isn't index 0
FPS_LIMIT   = 10          # Send at most N frames/sec to the API
# ───────────────────────────────────────────────────────────────────────────────

# Gaze zone → colour (BGR)
ZONE_COLORS = {
    "SCREEN":         (0, 200, 80),    # green
    "LEFT":           (0, 80, 255),    # red-orange
    "RIGHT":          (0, 80, 255),
    "UP":             (0, 80, 255),
    "DOWN":           (0, 80, 255),
    "NO_FACE":        (100, 100, 100), # grey
    "MULTIPLE_FACES": (0, 0, 255),     # red
}


def start_session():
    resp = requests.post(f"{BASE_URL}/sessions/start", json={
        "session_id": SESSION_ID,
        "user_id": USER_ID,
        "quiz_id": QUIZ_ID,
    })
    resp.raise_for_status()
    print(f"[demo] Session started: {SESSION_ID}")


def end_session():
    resp = requests.post(f"{BASE_URL}/sessions/{SESSION_ID}/end")
    if resp.ok:
        summary = resp.json()
        print("\n[demo] Session summary:")
        print(f"  Total frames  : {summary.get('total_frames_analyzed', 0)}")
        print(f"  Violations    : {summary.get('violation_frames', 0)}")
        print(f"  Violation rate: {summary.get('violation_rate', 0):.1%}")
        print(f"  Alerts fired  : {summary.get('violation_count', 0)}")


def analyze_frame(frame_bgr: np.ndarray) -> dict | None:
    """Encode frame as JPEG and POST to the analyze endpoint."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/sessions/{SESSION_ID}/analyze", files=files)
    if resp.ok:
        return resp.json()
    return None


def draw_overlay(frame: np.ndarray, result: dict):
    """Draw gaze info HUD on the frame."""
    zone   = result.get("zone", "?")
    status = result.get("status", "?")
    alert  = result.get("alert", False)
    rate   = result.get("violation_rate", 0.0)
    h      = result.get("h_ratio", 0.5)
    v      = result.get("v_ratio", 0.5)

    color  = ZONE_COLORS.get(zone, (255, 255, 255))
    h_frame, w_frame = frame.shape[:2]

    # Semi-transparent banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Zone text
    cv2.putText(frame, f"Zone: {zone}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # Violation rate
    cv2.putText(frame, f"Violation rate: {rate:.1%}", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    # Ratios
    cv2.putText(frame, f"H:{h:.2f}  V:{v:.2f}", (w_frame - 175, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Alert banner
    if alert:
        cv2.rectangle(frame, (0, h_frame - 45), (w_frame, h_frame), (0, 0, 200), -1)
        cv2.putText(frame, "⚠  GAZE VIOLATION ALERT", (12, h_frame - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Status border
    border_color = (0, 200, 80) if status == "OK" else (0, 60, 220)
    cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), border_color, 3)

    return frame


def main():
    print("[demo] Connecting to server…")
    try:
        requests.get(f"{BASE_URL}/health", timeout=3).raise_for_status()
    except Exception:
        print(f"[demo] ERROR: Cannot reach {BASE_URL}. Is the server running?")
        sys.exit(1)

    start_session()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[demo] ERROR: Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[demo] Camera open. Press Q to quit.")

    last_result: dict = {}
    last_api_call  = 0.0
    min_interval   = 1.0 / FPS_LIMIT

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_api_call >= min_interval:
                result = analyze_frame(frame)
                if result:
                    last_result = result
                last_api_call = now

            if last_result:
                frame = draw_overlay(frame, last_result)

            cv2.imshow("Eye Proctoring — Live Demo  [Q to quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        end_session()
        print("[demo] Done.")


if __name__ == "__main__":
    main()

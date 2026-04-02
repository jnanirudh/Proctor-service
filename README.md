# 👁️ Eye Proctoring Service

A real-time gaze tracking microservice built with **FastAPI** and **MediaPipe**. It detects where a student is looking during an online exam and flags violations when they look away from the screen.

---

## How It Works

```
Webcam frame (JPEG)
        │
        ▼
POST /sessions/{id}/analyze
        │
        ▼
 GazeDetector
  └── MediaPipe FaceLandmarker
       └── Iris position → H/V ratios
              │
              ▼
       GazeConfig thresholds
              │
        ┌─────┴─────┐
      SCREEN     VIOLATION
      (OK)    (LEFT/RIGHT/UP/DOWN/NO_FACE)
              │
              ▼
       SessionManager
        └── tracks consecutive violations
             └── fires alert after N frames
```

---

## Project Structure

```
eye-proctoring-service/
├── app/
│   ├── main.py              # FastAPI app — startup, middleware, router registration
│   ├── gaze_detector.py     # Core gaze logic using MediaPipe Tasks API
│   ├── session_manager.py   # Per-exam session state (in-memory)
│   ├── models.py            # Pydantic request/response schemas
│   ├── dependencies.py      # FastAPI dependency injection providers
│   └── routers/
│       └── sessions.py      # All /sessions/* API routes
├── models/
│   └── face_landmarker.task # MediaPipe face landmark model
├── config.py                # Gaze threshold configuration
├── demo_client.py           # Live webcam demo script
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## Requirements

- Python **3.11** (3.12+ not yet supported by all dependencies)
- macOS / Linux
- A webcam (for the demo client)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/jnanirudh/Proctor-service.git
cd Proctor-service
```

### 2. Download the MediaPipe face model

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 3. Create a virtual environment with Python 3.11

```bash
python3.11 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install opencv-python requests   # extra deps for demo client only
```

---

## Running the Service

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

Server starts at **http://127.0.0.1:8000**

Interactive API docs available at **http://127.0.0.1:8000/docs**

---

## Live Demo (Webcam)

With the server running, open a second terminal and run:

```bash
source venv/bin/activate
python demo_client.py
```

A camera window opens with a live HUD overlay:

| HUD Element | Meaning |
|---|---|
| Zone label (top-left) | `SCREEN` ✅ or `LEFT/RIGHT/UP/DOWN/NO_FACE` 🚫 |
| Violation rate | % of frames flagged in this session |
| H / V ratios | Raw iris position (0.5 = dead centre) |
| Green border | Gaze is OK |
| Red border | Gaze violation |
| Red bottom banner | Alert fired (15 consecutive violation frames) |
| `Q` key | Quit — prints session summary to terminal |

> **macOS note:** On first run, macOS will ask for camera permission. Grant it in  
> **System Settings → Privacy & Security → Camera**.

---

## API Reference

### `GET /health`
Returns service status.

```json
{ "status": "ok", "service": "eye-proctoring-service" }
```

---

### `POST /sessions/start`
Start a new proctoring session.

**Request body:**
```json
{
  "session_id": "uuid-or-custom-id",
  "user_id": "student_42",
  "quiz_id": "exam_2025_final"
}
```

**Response:**
```json
{
  "session_id": "...",
  "status": "STARTED",
  "message": "Proctoring session started"
}
```

---

### `POST /sessions/{session_id}/analyze`
Submit a webcam frame for gaze analysis.

**Request:** `multipart/form-data` with a `frame` field (JPEG image file).

**Response:**
```json
{
  "session_id": "...",
  "status": "OK",
  "zone": "SCREEN",
  "alert": false,
  "message": "Looking at screen",
  "violation_rate": 0.02,
  "h_ratio": 0.51,
  "v_ratio": 0.49
}
```

| `zone` value | Meaning |
|---|---|
| `SCREEN` | Looking at the monitor — allowed |
| `LEFT` / `RIGHT` | Looking sideways — violation |
| `UP` / `DOWN` | Looking away vertically — violation |
| `NO_FACE` | Face not detected — violation |
| `MULTIPLE_FACES` | More than one face — cheat risk |

`alert: true` — fires when **15 consecutive violation frames** are detected (configurable).

---

### `GET /sessions/{session_id}/status`
Get live session stats without ending it.

---

### `POST /sessions/{session_id}/end`
End the session and get a full summary.

**Response:**
```json
{
  "session_id": "...",
  "user_id": "student_42",
  "quiz_id": "exam_2025_final",
  "duration_seconds": 120.4,
  "total_frames_analyzed": 1200,
  "violation_frames": 48,
  "violation_rate": 0.04,
  "violation_count": 2,
  "violations": [
    { "timestamp": 1712044800.0, "zone": "RIGHT", "frame": 300 }
  ]
}
```

---

## Configuration (`config.py`)

All gaze thresholds are in `config.py`. Iris ratios range from `0.0` (far left/top) to `1.0` (far right/bottom) — `0.5` is dead centre.

```python
@dataclass
class GazeConfig:
    HORIZONTAL_LEFT_THRESHOLD:  float = 0.40   # iris ratio below this → LEFT violation
    HORIZONTAL_RIGHT_THRESHOLD: float = 0.60   # iris ratio above this → RIGHT violation
    VERTICAL_UP_THRESHOLD:      float = 0.35   # iris ratio below this → UP violation
    VERTICAL_DOWN_THRESHOLD:    float = 0.65   # iris ratio above this → DOWN violation

    VIOLATION_FRAME_THRESHOLD:  int = 15       # consecutive frames before alert fires (~0.5s @ 30fps)
    VIOLATION_COOLDOWN_FRAMES:  int = 30       # cooldown frames after an alert
```

Tighten thresholds (e.g. `0.45` / `0.55`) to make detection stricter.

---

## Docker

```bash
docker build -t eye-proctoring-service .
docker run -p 8000:8000 eye-proctoring-service
```

---

## Known Limitations

| Limitation | Notes |
|---|---|
| **In-memory sessions** | All session data is lost on restart. Add Redis/PostgreSQL for production persistence. |
| **Single worker only** | Multiple uvicorn workers won't share session state. Requires a shared store. |
| **Python 3.11 required** | pydantic-core has no pre-built wheel for Python 3.12+ yet. |
| **30fps cap** | Demo client sends at most 10 frames/sec to the API by default (configurable). |

---

## Tech Stack

| | |
|---|---|
| **API** | FastAPI + Uvicorn |
| **Gaze tracking** | MediaPipe FaceLandmarker (Tasks API) |
| **Vision** | OpenCV |
| **Schemas** | Pydantic v2 |
| **Containerisation** | Docker |

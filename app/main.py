from fastapi import FastAPI
from app.models import SessionStartRequest, SessionEndRequest, GazeFrameRequest
from app.session_manager import SessionManager
from app.gaze_detector import GazeDetector
import config

app = FastAPI(
    title="Eye Proctoring Service",
    description="Real-time gaze tracking and proctoring API",
    version="1.0.0",
)

session_manager = SessionManager()
gaze_detector = GazeDetector()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "eye-proctoring-service"}


@app.post("/session/start")
def start_session(request: SessionStartRequest):
    session_id = session_manager.start_session(request.exam_id, request.student_id)
    return {"session_id": session_id, "status": "started"}


@app.post("/session/end")
def end_session(request: SessionEndRequest):
    result = session_manager.end_session(request.session_id)
    return {"session_id": request.session_id, "summary": result}


@app.post("/gaze/analyze")
def analyze_gaze(request: GazeFrameRequest):
    result = gaze_detector.analyze(request.frame_b64, request.session_id)
    session_manager.record_event(request.session_id, result)
    return result

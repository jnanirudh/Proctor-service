# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.models import StartSessionRequest, SessionResponse, FrameAnalysisResponse
from app.gaze_detector import GazeDetector
from app.session_manager import SessionManager
from config import gaze_config

detector: GazeDetector = None
session_manager: SessionManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, session_manager
    detector = GazeDetector(gaze_config)
    session_manager = SessionManager()
    yield
    detector.close()

app = FastAPI(title="Eye Proctoring Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/sessions/start", response_model=SessionResponse)
def start_session(req: StartSessionRequest):
    existing = session_manager.get_session(req.session_id)
    if existing:
        raise HTTPException(400, "Session already exists")
    session_manager.create_session(req.session_id, req.user_id, req.quiz_id)
    return SessionResponse(
        session_id=req.session_id,
        status="STARTED",
        message="Proctoring session started"
    )

@app.post("/sessions/{session_id}/analyze", response_model=FrameAnalysisResponse)
async def analyze_frame(
    session_id: str,
    frame: UploadFile = File(...)
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    frame_bytes = await frame.read()
    result = detector.analyze_frame(frame_bytes)
    record = session_manager.record_frame(session_id, result, gaze_config)

    return FrameAnalysisResponse(
        session_id=session_id,
        status=record["status"],
        zone=record["zone"],
        alert=record["alert"],
        message=record["message"],
        violation_rate=record["violation_rate"],
        h_ratio=result.horizontal_ratio,
        v_ratio=result.vertical_ratio
    )

@app.post("/sessions/{session_id}/end")
def end_session(session_id: str):
    summary = session_manager.end_session(session_id)
    if not summary:
        raise HTTPException(404, "Session not found")
    return summary

@app.get("/sessions/{session_id}/status")
def session_status(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_summary()

@app.get("/health")
def health():
    return {"status": "ok"}
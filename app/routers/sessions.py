# app/routers/sessions.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.models import StartSessionRequest, SessionResponse, FrameAnalysisResponse
from app.dependencies import get_detector, get_session_manager, get_config

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("/start", response_model=SessionResponse)
def start_session(
    req: StartSessionRequest,
    session_manager=Depends(get_session_manager),
):
    if session_manager.get_session(req.session_id):
        raise HTTPException(status_code=400, detail="Session already exists")

    session_manager.create_session(req.session_id, req.user_id, req.quiz_id)
    return SessionResponse(
        session_id=req.session_id,
        status="STARTED",
        message="Proctoring session started",
    )


@router.post("/{session_id}/analyze", response_model=FrameAnalysisResponse)
async def analyze_frame(
    session_id: str,
    frame: UploadFile = File(...),
    detector=Depends(get_detector),
    session_manager=Depends(get_session_manager),
    config=Depends(get_config),
):
    if not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    frame_bytes = await frame.read()
    result = detector.analyze_frame(frame_bytes)
    record = session_manager.record_frame(session_id, result, config)

    return FrameAnalysisResponse(
        session_id=session_id,
        status=record["status"],
        zone=record["zone"],
        alert=record["alert"],
        message=record["message"],
        violation_rate=record["violation_rate"],
        h_ratio=result.horizontal_ratio,
        v_ratio=result.vertical_ratio,
    )


@router.get("/{session_id}/status")
def session_status(
    session_id: str,
    session_manager=Depends(get_session_manager),
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_summary()


@router.post("/{session_id}/end")
def end_session(
    session_id: str,
    session_manager=Depends(get_session_manager),
):
    summary = session_manager.end_session(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary

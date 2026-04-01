from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    exam_id: str = Field(..., description="Unique identifier for the exam")
    student_id: str = Field(..., description="Unique identifier for the student")


class SessionEndRequest(BaseModel):
    session_id: str = Field(..., description="Active proctoring session ID")


class GazeFrameRequest(BaseModel):
    session_id: str = Field(..., description="Active proctoring session ID")
    frame_b64: str = Field(..., description="Base64-encoded image frame from the webcam")


class GazeAnalysisResult(BaseModel):
    session_id: str
    timestamp: str
    gaze_direction: str = Field(..., description="Detected gaze direction: center | left | right | up | down")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    is_suspicious: bool = Field(..., description="True if the gaze is directed off-screen")


class SessionSummary(BaseModel):
    total_frames: int
    suspicious_frames: int
    started_at: str
    ended_at: str

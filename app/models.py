from pydantic import BaseModel

class StartSessionRequest(BaseModel):
    session_id: str
    user_id: str
    quiz_id: str

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

class FrameAnalysisResponse(BaseModel):
    session_id: str
    status: str        # "OK" or "VIOLATION"
    zone: str
    alert: bool        # True when a reportable violation fires
    message: str
    violation_rate: float
    h_ratio: float
    v_ratio: float
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import gaze_config
from app.gaze_detector import GazeDetector
from app.session_manager import SessionManager
from app.dependencies import set_detector, set_session_manager
from app.routers import sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise heavy objects once. Shutdown: release resources."""
    detector = GazeDetector(gaze_config)
    sm = SessionManager()
    set_detector(detector)
    set_session_manager(sm)
    yield
    detector.close()


app = FastAPI(
    title="Eye Proctoring Service",
    description="Real-time gaze tracking and proctoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(sessions.router)


# ── Utility routes (no router needed — app-level) ──────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "eye-proctoring-service"}
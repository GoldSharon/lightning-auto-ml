"""
Lightning AutoML — Main FastAPI Application
==========================================
Run:  uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.core.config import settings
from app.api.upload_routes import router as upload_router
from app.api.routes import router as pipeline_router

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.APP_TITLE,
    version     = settings.APP_VERSION,
    description = "Automated ML Pipeline — Upload, Analyse, Train, Predict.",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(upload_router)
app.include_router(pipeline_router)

# ── Static files ──────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── SPA catch-all ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_TITLE, "version": settings.APP_VERSION}

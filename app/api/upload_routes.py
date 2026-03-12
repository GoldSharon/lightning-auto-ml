"""
Upload Routes
=============
POST /api/upload        — upload dataset file
POST /api/configure     — set pipeline config (target, task, etc.)
GET  /api/session       — get current session status
DELETE /api/session     — clear session
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.models import (
    DatasetInfoResponse,
    PipelineConfigRequest,
    PipelineConfigResponse,
    SessionStatusResponse,
    ClearSessionResponse,
)
from app.core.config import settings
from app.services.session_service import session_service
from app.services.pipeline_service import pipeline_service

router = APIRouter(prefix="/api", tags=["upload"])


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DatasetInfoResponse)
async def upload_dataset(file: UploadFile = File(...)):
    # Validate extension
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not supported. Allowed: {settings.allowed_extensions_list}",
        )

    # Validate size
    contents = await file.read()
    size_mb  = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # Clear previous session and save file
    session_service.clear()
    save_path = settings.uploads_path / file.filename
    save_path.write_bytes(contents)

    # Load and inspect
    try:
        df = pipeline_service.load_dataset(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    detected_task, suggested_target = pipeline_service.detect_task_and_target(df)

    # Build dtype map
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Init session
    session_service.init_session(
        project_name = Path(file.filename).stem,
        filename     = file.filename,
        rows         = len(df),
        columns      = len(df.columns),
        column_names = df.columns.tolist(),
    )

    return DatasetInfoResponse(
        filename         = file.filename,
        rows             = len(df),
        columns          = len(df.columns),
        column_names     = df.columns.tolist(),
        dtypes           = dtypes,
        sample_data      = df.head(5).fillna("").to_dict(orient="records"),
        missing_counts   = df.isnull().sum().to_dict(),
        detected_task    = detected_task,
        suggested_target = suggested_target,
        message          = "Dataset uploaded successfully",
    )


# ── Configure ──────────────────────────────────────────────────────────────────

@router.post("/configure", response_model=PipelineConfigResponse)
async def configure_pipeline(body: PipelineConfigRequest):
    if not session_service.exists():
        raise HTTPException(status_code=400, detail="No dataset uploaded. Upload first.")

    config = {
        "project_name":  body.project_name,
        "target_column": body.target_column,
        "index_column":  body.index_column,
        "ml_type":       body.ml_type.value,
        "learning_type": body.learning_type.value,
        "test_size":     body.test_size,
    }
    session_service.set_config(config)

    return PipelineConfigResponse(
        status  = "ok",
        config  = config,
        message = f"Pipeline configured: {body.learning_type.value} / target={body.target_column}",
    )


# ── Session status ─────────────────────────────────────────────────────────────

@router.get("/session", response_model=SessionStatusResponse)
async def get_session():
    if not session_service.exists():
        return SessionStatusResponse(has_session=False, stage="empty")

    s = session_service.full()
    return SessionStatusResponse(
        has_session      = True,
        stage            = s.get("stage", "empty"),
        project_name     = s.get("project_name"),
        learning_type    = s.get("config", {}).get("learning_type"),
        target_column    = s.get("config", {}).get("target_column"),
        rows             = s.get("rows"),
        columns          = s.get("columns"),
        best_model       = s.get("best_model_name"),
        prediction_count = session_service.prediction_count(),
    )


# ── Clear session ──────────────────────────────────────────────────────────────

@router.delete("/session", response_model=ClearSessionResponse)
async def clear_session():
    session_service.clear()
    return ClearSessionResponse(status="ok", message="Session cleared successfully.")

"""
Pipeline Routes
===============
POST /api/analyze           — run data analysis + get preprocessing plan
GET  /api/insights          — get structured data insights for UI
GET  /api/preprocessing     — get preprocessing plan details
POST /api/train             — run full preprocessing + training
GET  /api/results           — get training results
POST /api/predict           — run single prediction
GET  /api/predictions       — get prediction history
GET  /api/downloads         — list available download files
GET  /api/download/{key}    — download a specific file
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.api.models import (
    DataInsightsResponse,
    PreprocessingPlanResponse,
    TrainingResultsResponse,
    ModelResultItem,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    DownloadManifestResponse,
    AnalysisStatusResponse,
)
from app.services.session_service import session_service
from app.services.pipeline_service import pipeline_service, _sanitize_for_json

router = APIRouter(prefix="/api", tags=["pipeline"])


# ── Analyze ───────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def run_analysis():
    if session_service.get_stage() not in ("configured", "analyzed"):
        raise HTTPException(
            status_code=400,
            detail="Configure pipeline first (/api/configure).",
        )
    try:
        report, prep_config = pipeline_service.run_analysis()
        insights            = pipeline_service.extract_insights(report, prep_config)
        return {"status": "complete", "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Insights ──────────────────────────────────────────────────────────────────

@router.get("/insights")
async def get_insights():
    stage = session_service.get_stage()
    if stage not in ("analyzed", "trained"):
        raise HTTPException(status_code=400, detail="Analysis not run yet.")

    report      = session_service.get("analysis_report", {})
    prep_config = session_service.get("prep_config", {})
    insights    = pipeline_service.extract_insights(report, prep_config)
    return {"status": "ok", "insights": insights}


# ── Preprocessing plan ────────────────────────────────────────────────────────

@router.get("/preprocessing")
async def get_preprocessing_plan():
    stage = session_service.get_stage()
    if stage not in ("analyzed", "trained"):
        raise HTTPException(status_code=400, detail="Analysis not run yet.")

    prep_config = session_service.get("prep_config", {})
    pipeline    = prep_config.get("pipeline_recommendation", {})
    fe          = prep_config.get("feature_engineering_suggestions", [])
    summary     = prep_config.get("summary", {})

    missing_cfg  = pipeline.get("step_1_missing_values", {})
    outlier_cfg  = pipeline.get("step_2_outliers", {})
    encoding_cfg = pipeline.get("step_3_encoding", {})
    scaling_cfg  = pipeline.get("step_5_scaling", {})

    return {
        "status": "ok",
        "plan": {
            "missing_strategy":     missing_cfg.get("numeric_strategy", "median"),
            "outlier_method":       outlier_cfg.get("method", "winsorize"),
            "scaling_method":       scaling_cfg.get("method", "standard"),
            "onehot_columns":       encoding_cfg.get("onehot_columns", []),
            "frequency_columns":    encoding_cfg.get("frequency_columns", []),
            "feature_engineering":  fe,
            "columns_to_drop":      prep_config.get("columns_to_drop", {}).get("column_list", []),
            "columns_to_keep":      prep_config.get("columns_to_keep", {}).get("column_list", []),
            "key_insights":         summary.get("key_insights", []),
            "expected_improvement": summary.get("expected_model_improvement", ""),
            "original_features":    summary.get("original_features", 0),
            "recommended_features": summary.get("recommended_features", 0),
        },
    }


# ── Train ─────────────────────────────────────────────────────────────────────

@router.post("/train")
async def run_training():
    stage = session_service.get_stage()
    if stage not in ("analyzed", "trained"):
        raise HTTPException(status_code=400, detail="Run analysis first (/api/analyze).")
    try:
        results = pipeline_service.run_training()
        payload = _sanitize_for_json({"status": "complete", "results": results})
        return JSONResponse(content=payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Results ───────────────────────────────────────────────────────────────────

@router.get("/results")
async def get_results():
    if session_service.get_stage() != "trained":
        raise HTTPException(status_code=400, detail="No training results yet.")
    results = session_service.get("training_results", {})
    return {"status": "ok", "results": results}


# ── Predict ───────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(body: PredictionRequest):
    if session_service.get_stage() != "trained":
        raise HTTPException(status_code=400, detail="No trained model. Run training first.")
    try:
        result  = pipeline_service.predict(body.features)
        history = session_service.get_prediction_history()
        return {
            "status":  "ok",
            "result":  result,
            "history": history[-10:],   # last 10
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction history ────────────────────────────────────────────────────────

@router.get("/predictions")
async def get_predictions():
    if session_service.get_stage() != "trained":
        raise HTTPException(status_code=400, detail="No trained model.")
    return {
        "status":  "ok",
        "history": session_service.get_prediction_history(),
    }


# ── Downloads ─────────────────────────────────────────────────────────────────

@router.get("/downloads")
async def list_downloads():
    files = session_service.available_downloads()
    return {
        "status": "ok",
        "files":  [{"key": f["key"], "label": f["label"], "filename": f["filename"]}
                   for f in files],
    }


@router.get("/download/{key}")
async def download_file(key: str):
    files = {f["key"]: f for f in session_service.available_downloads()}
    if key not in files:
        raise HTTPException(status_code=404, detail=f"File '{key}' not available.")
    entry = files[key]
    return FileResponse(
        path      = str(entry["path"]),
        filename  = entry["filename"],
        media_type = "application/octet-stream",
    )


# ── Feature names for prediction form ─────────────────────────────────────────

@router.get("/feature-names")
async def get_feature_names():
    """Return feature column names needed for the prediction form."""
    if session_service.get_stage() != "trained":
        raise HTTPException(status_code=400, detail="Not trained yet.")

    feat_path = session_service.features_path()
    if not feat_path.exists():
        raise HTTPException(status_code=404, detail="Processed features file not found.")

    import pandas as pd
    df = pd.read_csv(feat_path, nrows=0)   # headers only
    return {"status": "ok", "features": df.columns.tolist()}

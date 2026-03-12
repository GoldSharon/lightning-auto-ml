"""
Pipeline Service
================
Wraps IntelligentDataAnalyzer + DataPrepAgent + AutoMLTrainer.
All ML logic lives here. Routes stay thin.
"""

import json
import math
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.session_service import session_service

# ── Set env vars before importing ML modules ───────────────────────────────────
os.environ.setdefault("LLM_API_KEY",    settings.LLM_API_KEY)
os.environ.setdefault("LLM_MODEL_NAME", settings.LLM_MODEL_NAME)

from app.core.ml_types import (
    InputConfiguration,
    MLType,
    LearningType,
    ProcessingType,
    Hardware,
)
from app.services.data_analyzer import IntelligentDataAnalyzer
from app.services.data_prep_agent import DataPrepAgent
from automl_trainer import AutoMLTrainer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_ml_type(val: str) -> MLType:
    mapping = {"Supervised": MLType.SUPERVISED, "Unsupervised": MLType.UNSUPERVISED}
    return mapping.get(val, MLType.SUPERVISED)


def _to_learning_type(val: str) -> LearningType:
    mapping = {
        "Regression":     LearningType.REGRESSION,
        "Classification": LearningType.CLASSIFICATION,
        "Clustering":     LearningType.CLUSTERING,
    }
    return mapping.get(val, LearningType.CLASSIFICATION)


def _safe_float(v) -> Optional[float]:
    """
    Convert v to a JSON-safe float.
    Returns None for nan/inf so the JSON encoder doesn't choke.
    """
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except Exception:
        return None


def _serialize(obj):
    """JSON-safe serialiser for numpy / pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        # nan / inf are not JSON-compliant — convert to None
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return str(obj)

def _sanitize_for_json(obj):
    """
    Recursively walk any structure (dict, list, scalar) and replace
    nan / inf / -inf floats with None so json.dumps never chokes.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return _sanitize_for_json(obj.tolist())
    return obj


# ─────────────────────────────────────────────────────────────────────────────

class PipelineService:

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        ext = filepath.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        elif ext == ".json":
            return pd.read_json(filepath)
        raise ValueError(f"Unsupported file type: {ext}")

    def detect_task_and_target(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Heuristic: guess learning type and target column."""
        last_col = df.columns[-1]
        n_unique  = df[last_col].nunique()
        n_rows    = len(df)

        if n_unique == 1:
            return "Clustering", None
        elif n_unique <= 20 and n_unique / n_rows < 0.05:
            return "Classification", last_col
        elif pd.api.types.is_numeric_dtype(df[last_col]):
            return "Regression", last_col
        else:
            return "Classification", last_col

    # ── Analysis ──────────────────────────────────────────────────────────────

    def run_analysis(self) -> Tuple[Dict, Dict]:
        filepath = session_service.upload_path()
        if not filepath or not filepath.exists():
            raise FileNotFoundError("No uploaded dataset found in session.")

        df     = self.load_dataset(filepath)
        config = session_service.get("config", {})

        ipc = InputConfiguration(
            project_name    = config.get("project_name", "automl_project"),
            file_name       = filepath.name,
            ml_type         = _to_ml_type(config.get("ml_type", "Supervised")),
            learning_type   = _to_learning_type(config.get("learning_type", "Classification")),
            processing_type = ProcessingType.PREPROCESS_TRAIN,
            llm_name        = settings.LLM_MODEL_NAME,
            target_column   = config.get("target_column"),
            index_column    = config.get("index_column"),
        )

        analyzer             = IntelligentDataAnalyzer(ipc)
        report, prep_config  = analyzer.analyze_dataframe(df)

        report_dict = json.loads(json.dumps(report.to_dict(), default=_serialize))
        prep_dict   = json.loads(json.dumps(prep_config,      default=_serialize))

        session_service.set_analysis(report_dict, prep_dict)

        session_service.report_path().write_text(
            json.dumps(prep_dict, indent=2, default=_serialize)
        )

        return report_dict, prep_dict

    # ── Training ──────────────────────────────────────────────────────────────

    def run_training(self) -> Dict[str, Any]:
        filepath = session_service.upload_path()
        if not filepath or not filepath.exists():
            raise FileNotFoundError("No uploaded dataset found.")

        config      = session_service.get("config", {})
        prep_config = session_service.get("prep_config")
        if not prep_config:
            raise ValueError("Run analysis before training.")

        df = self.load_dataset(filepath)

        ipc = InputConfiguration(
            project_name    = config.get("project_name", "automl_project"),
            file_name       = filepath.name,
            ml_type         = _to_ml_type(config.get("ml_type", "Supervised")),
            learning_type   = _to_learning_type(config.get("learning_type", "Classification")),
            processing_type = ProcessingType.PREPROCESS_TRAIN,
            llm_name        = settings.LLM_MODEL_NAME,
            target_column   = config.get("target_column"),
            index_column    = config.get("index_column"),
            test_size       = float(config.get("test_size", 0.2)),
        )

        # ── Preprocessing ──────────────────────────────────────────────────
        agent             = DataPrepAgent(ipc, df)
        features, target  = agent.fit_transform(prep_config)

        features.to_csv(session_service.features_path(), index=False)
        if target is not None:
            pd.DataFrame({"target": target}).to_csv(session_service.target_path(), index=False)

        session_service.save_agent(agent)

        # ── Training ───────────────────────────────────────────────────────
        trainer          = AutoMLTrainer(ipc, features, target, verbose=False)
        training_results = trainer.train()

        if trainer.X_test is not None:
            trainer.X_test.to_csv(session_service.test_features_path(), index=False)
        if trainer.y_test is not None:
            pd.DataFrame({"target": trainer.y_test}).to_csv(
                session_service.test_target_path(), index=False
            )

        if training_results.get("best_model"):
            session_service.save_model(training_results["best_model"])

        results_df = trainer.get_results_df()
        results_df.to_csv(session_service.metrics_path(), index=False)

        # Serialise — replace nan/inf in all metric values
        safe_results = {
            "best_model_name": training_results.get("best_model_name", ""),
            "best_score":      _safe_float(training_results.get("best_score")),
            "training_time":   _safe_float(training_results.get("training_time")),
            "llm_analysis":    training_results.get("llm_analysis", {}),
            "all_results": [
                {
                    "model_name": r["model_name"],
                    "metrics": {
                        k: _safe_float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                        for k, v in r["metrics"].items()
                    },
                    "hyperparameters": r.get("hyperparameters", {}),
                    "fit_time":        _safe_float(r["metrics"].get("fit_time", 0)),
                }
                for r in training_results.get("all_results", [])
            ],
        }

        safe_results = _sanitize_for_json(safe_results)
        session_service.set_trained(safe_results)
        return safe_results

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, feature_values: Dict[str, Any]) -> Dict[str, Any]:
        agent = session_service.load_agent()
        model = session_service.load_model()
        if agent is None or model is None:
            raise ValueError("No trained model found. Run training first.")

        config        = session_service.get("config", {})
        learning_type = _to_learning_type(config.get("learning_type", "Classification"))

        input_df      = pd.DataFrame([feature_values])
        X_transformed = agent.transform(input_df)

        raw_pred   = model.predict(X_transformed)
        prediction = raw_pred[0]

        label = None
        if hasattr(agent, "inverse_transform_target"):
            inv   = agent.inverse_transform_target(np.array([prediction]))
            label = str(inv[0]) if inv is not None else None

        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba      = model.predict_proba(X_transformed)[0]
                confidence = round(float(proba.max()) * 100, 1)
            except Exception:
                pass

        result = {
            "prediction":  float(prediction) if isinstance(prediction, (np.floating, np.integer)) else prediction,
            "confidence":  confidence,
            "label":       label,
            "input_data":  feature_values,
        }

        session_service.append_prediction(result)
        return result

    # ── Insights ──────────────────────────────────────────────────────────────

    def extract_insights(self, report: Dict, prep_config: Dict) -> Dict[str, Any]:
        quality_score = report.get("data_quality_score", 0)

        drop_info    = prep_config.get("columns_to_drop", {})
        keep_info    = prep_config.get("columns_to_keep", {})
        drop_list    = drop_info.get("column_list", [])
        drop_reasons = drop_info.get("reasons", {})

        keep_details = keep_info.get("details", {})
        priority_map  = {"high": 3, "medium": 2, "low": 1}
        key_features  = sorted(
            keep_details.keys(),
            key=lambda c: priority_map.get(keep_details[c].get("priority", "low"), 0),
            reverse=True,
        )[:6]

        pipeline     = prep_config.get("pipeline_recommendation", {})
        missing_cfg  = pipeline.get("step_1_missing_values", {})
        outlier_cfg  = pipeline.get("step_2_outliers", {})
        encoding_cfg = pipeline.get("step_3_encoding", {})
        scaling_cfg  = pipeline.get("step_5_scaling", {})

        fe_suggestions = prep_config.get("feature_engineering_suggestions", [])

        return {
            "quality_score":          quality_score,
            "key_features":           key_features,
            "dropped_columns":        drop_list,
            "drop_reasons":           drop_reasons,
            "missing_values":         report.get("missing_percentage", {}),
            "outliers_detected":      report.get("outliers_detected", {}),
            "correlations":           report.get("correlations", {}),
            "numerical_stats":        report.get("numerical_stats", {}),
            "categorical_stats":      report.get("categorical_stats", {}),
            "feature_count_original": report.get("shape", [0, 0])[1] if isinstance(report.get("shape"), (list, tuple)) else 0,
            "feature_count_final":    len(keep_info.get("column_list", [])),
            "preprocessing_plan": {
                "missing_strategy":    missing_cfg.get("numeric_strategy", "median"),
                "outlier_method":      outlier_cfg.get("method", "winsorize"),
                "scaling_method":      scaling_cfg.get("method", "standard"),
                "onehot_columns":      encoding_cfg.get("onehot_columns", []),
                "frequency_columns":   encoding_cfg.get("frequency_columns", []),
                "feature_engineering": fe_suggestions,
            },
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
pipeline_service = PipelineService()
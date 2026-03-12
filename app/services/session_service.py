"""
Session Service
===============
Manages a single active pipeline session.
Persists state to session_data/ directory.
Thread-safe for single-user HuggingFace deployment.
"""

import json
import shutil
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.core.config import settings


_SESSION_FILE = "session.json"
_FEATURES_FILE = "processed_features.csv"
_TARGET_FILE = "processed_target.csv"
_TEST_FEATURES_FILE = "test_features.csv"
_TEST_TARGET_FILE = "test_target.csv"
_MODEL_FILE = "best_model.joblib"
_METRICS_FILE = "model_metrics.csv"
_REPORT_FILE = "preprocessing_report.json"
_PREDICTIONS_FILE = "prediction_history.json"


class SessionService:
    """Single-session state manager."""

    def __init__(self):
        self._base = settings.session_data_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _session_path(self) -> Path:
        return self._base / _SESSION_FILE

    def _load(self) -> Dict[str, Any]:
        p = self._session_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}

    def _save(self, data: Dict[str, Any]):
        data["updated_at"] = datetime.utcnow().isoformat()
        self._session_path().write_text(json.dumps(data, indent=2, default=str))

    # ── Public state API ──────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self._session_path().exists()

    def get_stage(self) -> str:
        s = self._load()
        return s.get("stage", "empty")

    def get(self, key: str, default=None):
        return self._load().get(key, default)

    def set(self, key: str, value: Any):
        s = self._load()
        s[key] = value
        self._save(s)

    def update(self, data: Dict[str, Any]):
        s = self._load()
        s.update(data)
        self._save(s)

    def full(self) -> Dict[str, Any]:
        return self._load()

    # ── Stage transitions ─────────────────────────────────────────────────────

    def init_session(self, project_name: str, filename: str, rows: int, columns: int, column_names: list):
        self._save({
            "stage":          "uploaded",
            "project_name":   project_name,
            "filename":       filename,
            "rows":           rows,
            "columns":        columns,
            "column_names":   column_names,
            "created_at":     datetime.utcnow().isoformat(),
        })

    def set_config(self, config: Dict[str, Any]):
        s = self._load()
        s["config"]  = config
        s["stage"]   = "configured"
        self._save(s)

    def set_analysis(self, report: Dict, prep_config: Dict):
        s = self._load()
        s["analysis_report"]   = report
        s["prep_config"]       = prep_config
        s["stage"]             = "analyzed"
        self._save(s)

    def set_trained(self, results: Dict[str, Any]):
        s = self._load()
        s["training_results"]  = results
        s["best_model_name"]   = results.get("best_model_name", "")
        s["best_score"]        = results.get("best_score")
        s["stage"]             = "trained"
        self._save(s)

    # ── File paths ────────────────────────────────────────────────────────────

    def features_path(self) -> Path:
        return self._base / _FEATURES_FILE

    def target_path(self) -> Path:
        return self._base / _TARGET_FILE

    def test_features_path(self) -> Path:
        return self._base / _TEST_FEATURES_FILE

    def test_target_path(self) -> Path:
        return self._base / _TEST_TARGET_FILE

    def model_path(self) -> Path:
        return self._base / _MODEL_FILE

    def metrics_path(self) -> Path:
        return self._base / _METRICS_FILE

    def report_path(self) -> Path:
        return self._base / _REPORT_FILE

    def predictions_path(self) -> Path:
        return self._base / _PREDICTIONS_FILE

    def upload_path(self) -> Optional[Path]:
        filename = self.get("filename")
        if not filename:
            return None
        return settings.uploads_path / filename

    # ── Model persistence ─────────────────────────────────────────────────────

    def save_model(self, model: Any):
        joblib.dump(model, self.model_path())

    def load_model(self) -> Optional[Any]:
        p = self.model_path()
        if p.exists():
            return joblib.load(p)
        return None

    def save_agent(self, agent: Any):
        """Persist the fitted DataPrepAgent for inference."""
        joblib.dump(agent, self._base / "data_prep_agent.joblib")

    def load_agent(self) -> Optional[Any]:
        p = self._base / "data_prep_agent.joblib"
        if p.exists():
            return joblib.load(p)
        return None

    # ── Prediction history ────────────────────────────────────────────────────

    def append_prediction(self, record: Dict[str, Any]):
        p = self.predictions_path()
        history: List = []
        if p.exists():
            try:
                history = json.loads(p.read_text())
            except Exception:
                history = []
        record["timestamp"] = datetime.utcnow().isoformat()
        history.append(record)
        p.write_text(json.dumps(history, indent=2, default=str))

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        p = self.predictions_path()
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text())
        except Exception:
            return []

    def prediction_count(self) -> int:
        return len(self.get_prediction_history())

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self):
        """Wipe all session data and uploaded files."""
        # Remove session data files
        for item in self._base.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception:
                pass

        # Remove uploaded datasets
        up = settings.uploads_path
        for item in up.iterdir():
            try:
                if item.is_file():
                    item.unlink()
            except Exception:
                pass

    # ── Download manifest ─────────────────────────────────────────────────────

    def available_downloads(self) -> List[Dict[str, str]]:
        files = [
            {"key": "features",      "label": "Processed Features (CSV)",   "path": self.features_path(),      "filename": "processed_features.csv"},
            {"key": "model",         "label": "Trained Model (.joblib)",     "path": self.model_path(),         "filename": "best_model.joblib"},
            {"key": "report",        "label": "Preprocessing Report (JSON)", "path": self.report_path(),        "filename": "preprocessing_report.json"},
            {"key": "metrics",       "label": "Model Metrics (CSV)",         "path": self.metrics_path(),       "filename": "model_metrics.csv"},
            {"key": "test_features", "label": "Test Features (CSV)",         "path": self.test_features_path(), "filename": "test_features.csv"},
            {"key": "test_target",   "label": "Test Target (CSV)",           "path": self.test_target_path(),   "filename": "test_target.csv"},
        ]
        return [f for f in files if Path(f["path"]).exists()]


# ── Singleton ──────────────────────────────────────────────────────────────────
session_service = SessionService()

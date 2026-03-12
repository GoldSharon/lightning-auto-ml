"""
AutoML Trainer Module
=====================
Supports: Regression, Classification, Clustering
Hardware: Optimised for 16GB RAM, 2-core CPU

Design decisions:
    - n_jobs=1 everywhere  — concurrency handled externally
    - n_estimators capped at 100 for all tree models
    - Sequential training (one model at a time)
    - No stacking / no heavy ensembles
    - LLM evaluation via AUTOML_EVALUATION_SYSTEM_PROMPT after all models trained

Usage (in ml_pipeline.py __main__):
    from automl_trainer import AutoMLTrainer
    trainer = AutoMLTrainer(ipc, features, target)
    results = trainer.train()
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import logging
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    # Regression
    r2_score, mean_squared_error, mean_absolute_error,
    # Classification
    accuracy_score, f1_score, precision_score, recall_score,
    # Clustering
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# ml_trainer enums and dataclasses
from ml_pipeline import InputConfiguration, LearningType, MLType, chat_llm, AUTOML_EVALUATION_SYSTEM_PROMPT

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ==============================================================================
# MODEL REGISTRY
# ==============================================================================
REGRESSION_MODELS: Dict[str, Any] = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=1000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=42, verbosity=0, eval_metric="rmse"),
}

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=42, verbosity=0, eval_metric="logloss", use_label_encoder=False),
}

CLUSTERING_MODELS: Dict[str, Any] = {
    "KMeans_3": KMeans(n_clusters=3, n_init=10, random_state=42),
    "KMeans_4": KMeans(n_clusters=4, n_init=10, random_state=42),
    "KMeans_5": KMeans(n_clusters=5, n_init=10, random_state=42),
    "Agglomerative_3": AgglomerativeClustering(n_clusters=3),
    "Agglomerative_4": AgglomerativeClustering(n_clusters=4),
    "Agglomerative_5": AgglomerativeClustering(n_clusters=5),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
}


# ==============================================================================
# AUTOML TRAINER
# ==============================================================================
class AutoMLTrainer:
    """Sequential AutoML trainer for Regression, Classification, Clustering."""

    def __init__(self, ipc: InputConfiguration, features: pd.DataFrame, target: Optional[pd.Series] = None, test_size: float = 0.2, verbose: bool = True):
        self.ipc = ipc
        self.features = features
        self.target = target
        self.test_size = ipc.test_size if ipc.test_size else test_size
        self.verbose = verbose
        self.learning_type = ipc.learning_type
        self.ml_type = ipc.ml_type
        self.chat_llm = chat_llm
        self.eval_prompt = AUTOML_EVALUATION_SYSTEM_PROMPT

        # Train/Test splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_model_name: str = ""
        self.best_model: Optional[Any] = None
        self.llm_analysis: Dict[str, Any] = {}

        if self.verbose:
            print(f"\nAutoMLTrainer initialized:")
            print(f"  • Task: {self.learning_type.value}")
            print(f"  • Features: {features.shape}")
            print(f"  • Has target: {target is not None}")
            print(f"  • Test size: {self.test_size}")

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """Run full AutoML pipeline sequentially."""
        start_time = time.time()

        if self.learning_type.value in (LearningType.REGRESSION.value, LearningType.CLASSIFICATION.value):
            self._prepare_supervised_split()
            self._train_supervised()
        elif self.learning_type.value == LearningType.CLUSTERING.value:
            self._train_clustering()
        else:
            raise ValueError(f"Unsupported learning_type: {self.learning_type}")

        # Sort results by primary metric
        primary = self._primary_metric_key()
        self.results.sort(key=lambda x: x["metrics"].get(primary, float('-inf')), reverse=True)

        # Best model
        if self.results:
            best = self.results[0]
            self.best_model_name = best["model_name"]
            self.best_model = best["model_object"]

        # LLM analysis
        self.llm_analysis = self._get_llm_analysis()

        total_time = round(time.time() - start_time, 2)
        if self.verbose:
            self._print_summary(total_time)

        return {
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "best_score": self.results[0]["metrics"].get(primary) if self.results else None,
            "all_results": self.results,
            "llm_analysis": self.llm_analysis,
            "training_time": total_time,
            "X_test": self.X_test,
            "y_test": self.y_test,
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def _prepare_supervised_split(self):
        if self.target is None:
            raise ValueError("Target is required for supervised learning")

        stratify = self.target if self.learning_type == LearningType.CLASSIFICATION else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=self.test_size, random_state=42, stratify=stratify
        )

        if self.verbose:
            print(f"\nTrain/Test Split: {self.X_train.shape[0]} train / {self.X_test.shape[0]} test")

    # ------------------------------------------------------------------
    # Supervised training
    # ------------------------------------------------------------------
    def _train_supervised(self):
        models = REGRESSION_MODELS if self.learning_type.value == LearningType.REGRESSION.value else CLASSIFICATION_MODELS
        if self.verbose:
            print(f"\nTraining {len(models)} models sequentially...\n")

        for model_name, model in models.items():
            self._train_single_supervised(model_name, model)

    def _train_single_supervised(self, model_name: str, model: Any):
        if self.verbose:
            print(f"  [{model_name}]", end=" ", flush=True)

        t0 = time.time()
        try:
            model.fit(self.X_train, self.y_train)
            fit_time = round(time.time() - t0, 3)

            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            train_metrics = self._compute_supervised_metrics(self.y_train, y_train_pred, prefix="train")
            test_metrics = self._compute_supervised_metrics(self.y_test, y_test_pred, prefix="val")
            metrics = {**train_metrics, **test_metrics, "fit_time": fit_time}

            # Safe formatting for printing
            primary = self._primary_metric_key()
            val_metric = metrics.get(primary, None)
            train_metric = metrics.get("train_" + primary.split("val_")[-1], None)

            val_str = f"{val_metric:.4f}" if isinstance(val_metric, (int, float)) else str(val_metric)
            train_str = f"{train_metric:.4f}" if isinstance(train_metric, (int, float)) else str(train_metric)

            if self.verbose:
                print(f"{primary}={val_str}  (train={train_str})  fit={fit_time}s")

            self.results.append({
                "model_name": model_name,
                "model_object": model,
                "metrics": metrics,
                "hyperparameters": self._get_hyperparams(model),
            })

        except Exception as e:
            logger.warning(f"[{model_name}] FAILED: {e}")
            if self.verbose:
                print(f"FAILED — {e}")

    # ------------------------------------------------------------------
    # Clustering training
    # ------------------------------------------------------------------
    def _train_clustering(self):
        X = self.features.values
        if self.verbose:
            print(f"\nTraining {len(CLUSTERING_MODELS)} clustering models sequentially...\n")
        for model_name, model in CLUSTERING_MODELS.items():
            self._train_single_clustering(model_name, model, X)

    def _train_single_clustering(self, model_name: str, model: Any, X: np.ndarray):
        if self.verbose:
            print(f"  [{model_name}]", end=" ", flush=True)

        t0 = time.time()
        try:
            labels = model.fit_predict(X)
            fit_time = round(time.time() - t0, 3)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int((labels == -1).sum())
            if n_clusters < 2:
                if self.verbose:
                    print(f"SKIPPED — only {n_clusters} cluster(s) found")
                return

            metrics = {
                "n_clusters": n_clusters,
                "noise_points": noise_count,
                "silhouette_score": round(float(silhouette_score(X, labels)), 4),
                "davies_bouldin_score": round(float(davies_bouldin_score(X, labels)), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 4),
                "fit_time": fit_time,
            }

            if self.verbose:
                print(f"silhouette={metrics['silhouette_score']:.4f}  "
                      f"davies_bouldin={metrics['davies_bouldin_score']:.4f}  "
                      f"n_clusters={n_clusters}  fit={fit_time}s")

            self.results.append({
                "model_name": model_name,
                "model_object": model,
                "labels": labels,
                "metrics": metrics,
                "hyperparameters": self._get_hyperparams(model),
            })

        except Exception as e:
            logger.warning(f"[{model_name}] FAILED: {e}")
            if self.verbose:
                print(f"FAILED — {e}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_supervised_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        metrics = {}
        if self.learning_type.value == LearningType.REGRESSION.value:
            metrics[f"{prefix}_r2"] = round(float(r2_score(y_true, y_pred)), 4)
            metrics[f"{prefix}_rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
            metrics[f"{prefix}_mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)
        elif self.learning_type.value == LearningType.CLASSIFICATION.value:
            avg = "weighted"
            metrics[f"{prefix}_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            metrics[f"{prefix}_f1"] = round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4)
            metrics[f"{prefix}_precision"] = round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4)
            metrics[f"{prefix}_recall"] = round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        return metrics

    def _primary_metric_key(self) -> str:
        if self.learning_type.value == LearningType.REGRESSION.value:
            return "val_r2"
        elif self.learning_type.value == LearningType.CLASSIFICATION.value:
            return "val_f1"
        elif self.learning_type.value == LearningType.CLUSTERING.value:
            return "silhouette_score"
        return "val_r2"

    # ------------------------------------------------------------------
    # LLM Analysis
    # ------------------------------------------------------------------
    def _get_llm_analysis(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        summary = []
        for r in self.results:
            summary.append({
                "model_name": r["model_name"],
                "hyperparameters": r["hyperparameters"],
                "metrics": r["metrics"],
            })

        input_data = {"task": self.learning_type.value, "models": summary}
        if self.verbose:
            print(f"\n  Requesting LLM analysis for {len(summary)} model(s)...")

        result = self.chat_llm(self.eval_prompt, json.dumps(input_data, indent=2))
        if not result:
            logger.warning("LLM analysis failed — returning empty.")
            return {}
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_hyperparams(self, model: Any) -> Dict[str, Any]:
        try:
            params = model.get_params()
            return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
        except Exception:
            return {}

    def _print_summary(self, total_time: float):
        primary = self._primary_metric_key()
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE  ({total_time}s)")
        print("=" * 70)
        print(f"{'Rank':<5} {'Model':<25} {primary:<20} {'Fit Time':<10}")
        print("-" * 70)

        for i, r in enumerate(self.results, 1):
            score = r["metrics"].get(primary, "N/A")
            fit_time = r["metrics"].get("fit_time", "N/A")
            marker = " ← BEST" if i == 1 else ""
            print(f"{i:<5} {r['model_name']:<25} {score:<20} {fit_time}{marker}")

        print("=" * 70)
        if self.llm_analysis:
            print("\nLLM ANALYSIS:")
            print(f"  Best Model:      {self.llm_analysis.get('best_model', 'N/A')}")
            print(f"  Analysis:        {self.llm_analysis.get('analysis', 'N/A')}")
            print(f"  Recommendations: {self.llm_analysis.get('recommendations', 'N/A')}")
            print("=" * 70)

    # ------------------------------------------------------------------
    def get_best_model(self) -> Tuple[str, Any]:
        if not self.best_model:
            raise ValueError("Call train() first.")
        return self.best_model_name, self.best_model

    def get_results_df(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"model_name": r["model_name"]}
            row.update(r["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.best_model:
            raise ValueError("Call train() first.")
        return self.best_model.predict(X)
"""
app/services/data_analyzer.py
==============================
IntelligentDataAnalyzer — LLM-powered data analysis and preprocessing recommendations.
Supports: Regression, Classification, Clustering.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.ml_types import (
    DataAnalysisReport,
    InputConfiguration,
    LearningType,
    MLType,
)
from app.core.llm_engine import (
    chat_llm,
    DATATYPE_SELECTION_SYSTEM_PROMPT,
    SUPERVISED_COLUMN_SELECTION_SYSTEM_PROMPT,
    CLUSTERING_COLUMN_SELECTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class IntelligentDataAnalyzer:
    """
    LLM-powered data analyzer for automated preprocessing recommendations.
    Supports: Regression, Classification, Clustering.
    """

    def __init__(self, ipc: InputConfiguration):
        self.ml_type       = ipc.ml_type
        self.learning_type = ipc.learning_type
        self.target_column = ipc.target_column
        self.index_column  = ipc.index_column

    # ------------------------------------------------------------------
    def analyze_dataframe(self, df: pd.DataFrame) -> Tuple[DataAnalysisReport, Dict[str, Any]]:
        if df.empty:
            raise ValueError("Input dataframe is empty")

        logger.info(
            f"Starting analysis — ML: {self.ml_type.value}, Task: {self.learning_type.value}"
        )

        logger.info("Step 1: Generating dataset summary…")
        summary = self.dataset_summary(df)

        logger.info("Step 2: Assigning data types via LLM…")
        df = self._data_type_assignment(df, summary)

        logger.info("Step 3: Preparing feature analysis…")
        if self.ml_type == MLType.SUPERVISED and self.target_column:
            if self.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in dataframe"
                )
            feature_df = df.drop(columns=[self.target_column])
        else:
            feature_df = df

        logger.info("Step 4: Analysing features…")
        report = self._build_report(feature_df, self.target_column)

        logger.info("Step 5: Getting preprocessing recommendations…")
        preprocessing_config = self._get_llm_preprocessing_suggestion(df, report)

        return report, preprocessing_config

    # ------------------------------------------------------------------
    def dataset_summary(self, df: pd.DataFrame, sample_values: int = 5) -> Dict[str, Any]:
        stats_df = df.describe(include="all").transpose()
        stats    = stats_df.fillna("").to_dict()

        columns_info = {}
        for col in df.columns:
            col_data = df[col]
            col_info: Dict[str, Any] = {
                "dtype":              str(col_data.dtype),
                "missing_values":     int(col_data.isnull().sum()),
                "missing_percentage": float((col_data.isnull().sum() / len(df)) * 100),
                "unique_values":      int(col_data.nunique()),
                "sample_values":      col_data.dropna().unique()[:sample_values].tolist(),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update({
                    "min":               float(col_data.min())    if not col_data.isnull().all() else None,
                    "max":               float(col_data.max())    if not col_data.isnull().all() else None,
                    "mean":              float(col_data.mean())   if not col_data.isnull().all() else None,
                    "std":               float(col_data.std())    if not col_data.isnull().all() else None,
                    "median":            float(col_data.median()) if not col_data.isnull().all() else None,
                    "variance":          float(col_data.var())    if not col_data.isnull().all() else None,
                    "likely_categorical": col_data.nunique() < 20,
                })
            else:
                mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else None
                col_info["most_frequent"] = str(mode_val) if mode_val is not None else None
                col_info["value_counts"]  = {
                    str(k): int(v) for k, v in col_data.value_counts().head(10).items()
                }

            columns_info[col] = col_info

        return {
            "stats":           stats,
            "columns":         columns_info,
            "rows":            len(df),
            "columns_count":   len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2),
        }

    # ------------------------------------------------------------------
    def _data_type_assignment(self, df: pd.DataFrame, summary: Dict[str, Any]) -> pd.DataFrame:
        """Use LLM to assign correct dtypes. Direct column mapping only."""
        input_data = {
            "columns_list":  df.columns.tolist(),
            "data_preview":  df.head(5).to_dict(orient="records"),
            "summary_stats": summary["columns"],
        }
        result = chat_llm(DATATYPE_SELECTION_SYSTEM_PROMPT, json.dumps(input_data, indent=2))

        if not result or "column_dtypes" not in result:
            logger.warning("Datatype assignment failed — keeping original dtypes.")
            return df

        valid_dtypes = {
            col: dtype
            for col, dtype in result["column_dtypes"].items()
            if col in df.columns
        }
        return df.astype(valid_dtypes, errors="ignore")

    # ------------------------------------------------------------------
    def _build_report(self, feature_df: pd.DataFrame,
                      target_column: Optional[str]) -> DataAnalysisReport:
        shape          = feature_df.shape
        dtypes         = {col: str(dtype) for col, dtype in feature_df.dtypes.items()}
        missing_values = feature_df.isnull().sum().to_dict()
        missing_pct    = {col: (v / len(feature_df)) * 100 for col, v in missing_values.items()}

        numerical_cols   = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        numerical_stats   = {}
        categorical_stats = {}
        outliers_detected = {}
        correlations      = {}

        for col in numerical_cols:
            stats             = feature_df[col].describe().to_dict()
            stats["outliers"] = self._detect_outliers(feature_df[col], method="iqr")
            stats["variance"] = (
                float(feature_df[col].var()) if not feature_df[col].isnull().all() else 0.0
            )
            if self.learning_type == LearningType.CLUSTERING and stats.get("mean", 0) != 0:
                stats["cv_coefficient"] = stats["std"] / stats["mean"]

            numerical_stats[col] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in stats.items()
            }
            outliers_detected[col] = stats["outliers"]

        for col in categorical_cols:
            categorical_stats[col] = {
                "unique_values": int(feature_df[col].nunique()),
                "most_frequent": (
                    str(feature_df[col].mode().iloc[0])
                    if not feature_df[col].mode().empty else None
                ),
                "value_counts": {
                    str(k): int(v)
                    for k, v in feature_df[col].value_counts().head(10).items()
                },
            }

        if len(numerical_cols) > 1:
            corr_matrix = feature_df[numerical_cols].corr()
            threshold   = 0.9 if self.learning_type == LearningType.CLUSTERING else 0.7
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > threshold:
                        key = f"{corr_matrix.columns[i]}_vs_{corr_matrix.columns[j]}"
                        correlations[key] = float(val)

        quality_metrics = self._calculate_quality_metrics(
            feature_df, missing_pct, outliers_detected
        )

        return DataAnalysisReport(
            shape=shape,
            dtypes=dtypes,
            missing_values=missing_values,
            missing_percentage=missing_pct,
            numerical_stats=numerical_stats,
            categorical_stats=categorical_stats,
            outliers_detected=outliers_detected,
            correlations=correlations,
            data_quality_score=quality_metrics["overall_score"],
            target_column=target_column,
            data_quality_metrics=quality_metrics,
            ml_type=self.ml_type,
            learning_type=self.learning_type,
        )

    # ------------------------------------------------------------------
    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> int:
        if not pd.api.types.is_numeric_dtype(series) or series.isnull().all():
            return 0
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR    = Q3 - Q1
        return int(((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum())

    # ------------------------------------------------------------------
    def _calculate_quality_metrics(
        self,
        df:          pd.DataFrame,
        missing_pct: Dict[str, float],
        outliers:    Dict[str, int],
    ) -> Dict[str, Any]:
        total_missing   = sum(missing_pct.values()) / len(missing_pct) if missing_pct else 0
        total_outliers  = sum(outliers.values())
        completeness    = max(0, 100 - total_missing)
        outlier_penalty = min(50, (total_outliers / len(df)) * 100)
        overall         = max(0, min(100, completeness - outlier_penalty))
        return {
            "overall_score":            round(overall, 2),
            "completeness_score":       round(completeness, 2),
            "total_missing_percentage": round(total_missing, 2),
            "total_outliers":           total_outliers,
            "outlier_percentage":       round((total_outliers / len(df)) * 100, 2),
        }

    # ------------------------------------------------------------------
    def _get_llm_preprocessing_suggestion(
        self,
        df:     pd.DataFrame,
        report: DataAnalysisReport,
    ) -> Dict[str, Any]:
        if self.learning_type in (LearningType.REGRESSION, LearningType.CLASSIFICATION):
            system_prompt = SUPERVISED_COLUMN_SELECTION_SYSTEM_PROMPT
        elif self.learning_type == LearningType.CLUSTERING:
            system_prompt = CLUSTERING_COLUMN_SELECTION_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unsupported learning_type: {self.learning_type}")

        feature_columns = {
            col: dtype
            for col, dtype in report.dtypes.items()
            if col != report.target_column
        }
        feature_df = (
            df.drop(columns=[report.target_column], errors="ignore")
            if report.target_column else df
        )

        input_data = {
            "ml_type":              self.ml_type.value,
            "learning_type":        self.learning_type.value,
            "features":             feature_columns,
            "first_5_rows":         feature_df.head(3).to_dict(orient="records"),
            "column_names":         list(feature_df.columns),
            "shape":                [feature_df.shape[0], feature_df.shape[1]],
            "missing_values":       report.missing_values,
            "missing_percentage":   report.missing_percentage,
            "outliers_detected":    report.outliers_detected,
            "correlations":         report.correlations,
            "numerical_stats":      report.numerical_stats,
            "categorical_stats":    report.categorical_stats,
            "target_column":        report.target_column,
            "data_quality_metrics": report.data_quality_metrics,
        }

        logger.info(
            f"Requesting preprocessing suggestions for "
            f"{self.ml_type.value}/{self.learning_type.value}…"
        )
        result = chat_llm(system_prompt, json.dumps(input_data, indent=2))

        if not result:
            logger.warning("LLM preprocessing suggestion failed — using default config.")
            return self._get_default_preprocessing_config(report)

        # Safety: strip target column from feature lists
        if report.target_column:
            for section in ["columns_to_drop", "columns_to_keep"]:
                if section in result and "column_list" in result[section]:
                    result[section]["column_list"] = [
                        col for col in result[section]["column_list"]
                        if col != report.target_column
                    ]
        return result

    # ------------------------------------------------------------------
    def _get_default_preprocessing_config(
        self, report: Optional[DataAnalysisReport] = None
    ) -> Dict[str, Any]:
        return {
            "columns_to_drop":                {"column_list": [], "reasons": {}},
            "columns_to_keep":                {
                "column_list": list(report.dtypes.keys()) if report else [],
                "details":     {},
            },
            "feature_engineering_suggestions": [],
            "pipeline_recommendation":         {},
            "summary": {
                "original_features":          len(report.dtypes) if report else 0,
                "recommended_features":       len(report.dtypes) if report else 0,
                "features_dropped":           0,
                "new_features_created":       0,
                "expected_model_improvement": "Default config — LLM call failed",
                "key_insights":               [],
            },
        }

    # ------------------------------------------------------------------
    def get_preprocessing_report(
        self,
        report:               DataAnalysisReport,
        preprocessing_config: Dict[str, Any],
    ) -> str:
        lines = [
            "=" * 80,
            f"DATA ANALYSIS REPORT — {self.ml_type.value} / {self.learning_type.value}",
            "=" * 80,
            f"\nDataset Shape: {report.shape[0]} rows × {report.shape[1]} columns",
            f"ML Task Type:  {self.ml_type.value}",
            f"Method:        {self.learning_type.value}",
            f"Target Column: {report.target_column or 'None (Unsupervised)'}",
            f"Quality Score: {report.data_quality_score}/100",
            f"\n{'='*80}\nCOLUMNS TO DROP:\n{'-'*80}",
        ]

        for col in preprocessing_config.get("columns_to_drop", {}).get("column_list", []):
            reasons = preprocessing_config["columns_to_drop"]["reasons"].get(col, [])
            lines.append(f"  • {col}")
            for r in reasons:
                lines.append(f"      - {r}")

        lines += [f"\n{'='*80}\nCOLUMNS TO KEEP (top 5 by priority):\n{'-'*80}"]
        keep_details = preprocessing_config.get("columns_to_keep", {}).get("details", {})
        priority_map = {"high": 3, "medium": 2, "low": 1}
        sorted_cols  = sorted(
            keep_details.items(),
            key=lambda x: priority_map.get(x[1].get("priority", "low"), 0),
            reverse=True,
        )[:5]

        for col, details in sorted_cols:
            lines.append(
                f"  • {col} (Priority: {details.get('priority')}, "
                f"Confidence: {details.get('confidence')})"
            )
            lines.append(f"      Reason: {details.get('reason')}")
            lines.append(
                f"      Preprocessing: "
                f"{', '.join(details.get('recommended_preprocessing', []))}"
            )

        s = preprocessing_config.get("summary", {})
        lines += [
            f"\n{'='*80}\nSUMMARY:\n{'-'*80}",
            f"  Original Features:    {s.get('original_features', 'N/A')}",
            f"  Recommended Features: {s.get('recommended_features', 'N/A')}",
            f"  Features Dropped:     {s.get('features_dropped', 'N/A')}",
            f"  New Features Created: {s.get('new_features_created', 'N/A')}",
            f"\n  Expected Impact: {s.get('expected_model_improvement', 'N/A')}",
            f"\n{'='*80}\n",
        ]
        return "\n".join(lines)
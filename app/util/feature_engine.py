"""
app/util/feature_engine.py
==========================
FeatureEngineeringEngine  — safe, no-exec feature creation
TargetProcessor           — target variable preprocessing (encode, winsorize)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder

from app.core.ml_types import LearningType

logger = logging.getLogger(__name__)


# ==============================================================================
# FEATURE ENGINEERING ENGINE
# ==============================================================================

class FeatureEngineeringEngine:
    """Safe feature engineering — no exec()."""

    ALLOWED_OPERATIONS = {
        'divide':   lambda df, a, b: df[a] / df[b].replace(0, np.nan),
        'multiply': lambda df, a, b: df[a] * df[b],
        'add':      lambda df, a, b: df[a] + df[b],
        'subtract': lambda df, a, b: df[a] - df[b],
    }

    @staticmethod
    def create_feature(df: pd.DataFrame, config: Dict) -> pd.Series:
        operation   = config.get('operation')
        source_cols = config['source_columns']

        if operation not in FeatureEngineeringEngine.ALLOWED_OPERATIONS:
            raise ValueError(
                f"Operation '{operation}' not allowed. "
                f"Use: {list(FeatureEngineeringEngine.ALLOWED_OPERATIONS.keys())}"
            )
        if len(source_cols) != 2:
            raise ValueError(f"Operations require exactly 2 columns, got {len(source_cols)}")
        missing = [c for c in source_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        result = FeatureEngineeringEngine.ALLOWED_OPERATIONS[operation](
            df, source_cols[0], source_cols[1]
        )
        return result.replace([np.inf, -np.inf], np.nan)


# ==============================================================================
# TARGET PROCESSOR
# ==============================================================================

class TargetProcessor:
    """Handles all target variable preprocessing."""

    def __init__(self, learning_type: LearningType, verbose: bool = True):
        self.learning_type = learning_type
        self.verbose       = verbose
        self.encoder       = None
        self.winsorizer    = None
        self.label_mapping = {}
        self._is_fitted    = False

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        target:          pd.Series,
        feature_indices: pd.Index,
    ) -> Tuple[pd.Series, pd.Index]:
        """Fit and transform target (TRAINING ONLY). May drop rows with missing target."""
        if self.verbose:
            print("\n[Target Preprocessing] Fitting and transforming target…")

        target       = target.copy()
        original_len = len(target)

        # use .values so boolean mask aligns positionally, not by index label
        valid_mask    = target.notna().values
        target        = target.iloc[valid_mask]
        valid_indices = feature_indices[valid_mask]

        dropped = original_len - len(target)
        if dropped > 0 and self.verbose:
            print(f"  • Removed {dropped} rows with missing target")

        # Classification: encode categorical target
        if self.learning_type == LearningType.CLASSIFICATION:
            if target.dtype == 'object' or target.dtype.name == 'category':
                self.encoder = LabelEncoder()
                target = pd.Series(
                    self.encoder.fit_transform(target),
                    index=target.index,
                    name=target.name,
                )
                self.label_mapping = dict(enumerate(self.encoder.classes_))
                if self.verbose:
                    print(f"  • Encoded categorical target: {len(self.label_mapping)} classes")

        # Regression: handle outliers in target
        elif self.learning_type == LearningType.REGRESSION:
            if target.dtype == 'object':
                target      = pd.to_numeric(target, errors='coerce')
                valid_mask2 = target.notna().values
                invalid     = (~valid_mask2).sum()
                if invalid > 0:
                    target        = target.iloc[valid_mask2]
                    valid_indices = valid_indices[valid_mask2]
                    if self.verbose:
                        print(f"  • Removed {invalid} non-numeric target values")

            Q1, Q3  = target.quantile(0.25), target.quantile(0.75)
            IQR     = Q3 - Q1
            extreme = ((target < Q1 - 3 * IQR) | (target > Q3 + 3 * IQR)).sum()
            if extreme > 0:
                if self.verbose:
                    print(f"  ⚠ Found {extreme} extreme outliers in target — winsorizing…")
                tname     = target.name or 'target'
                target_df = pd.DataFrame({tname: target})
                self.winsorizer = Winsorizer(
                    capping_method='iqr', tail='both', fold=3.0, variables=[tname]
                )
                target = self.winsorizer.fit_transform(target_df)[tname]
                if self.verbose:
                    print(
                        f"  • Target range after winsorization: "
                        f"[{target.min():.2f}, {target.max():.2f}]"
                    )

        self._is_fitted = True
        if self.verbose:
            print(f"  ✓ Target shape: {target.shape}, dtype: {target.dtype}")

        return target, valid_indices

    # ------------------------------------------------------------------
    def transform_target(self, target: pd.Series) -> pd.Series:
        """Transform target for validation/test — NEVER drops rows."""
        if not self._is_fitted:
            raise ValueError("TargetProcessor must be fitted first")

        target = target.copy()

        if self.learning_type == LearningType.CLASSIFICATION and self.encoder is not None:
            known_mask = target.isin(self.encoder.classes_)
            encoded    = np.full(len(target), -1, dtype=int)
            encoded[known_mask.values] = self.encoder.transform(target[known_mask])
            target = pd.Series(encoded, index=target.index, name=target.name)
            unknown = (~known_mask).sum()
            if unknown > 0 and self.verbose:
                print(f"  ⚠ {unknown} unknown labels encoded as -1")

        elif self.learning_type == LearningType.REGRESSION:
            if target.dtype == 'object':
                target = pd.to_numeric(target, errors='coerce')
            if self.winsorizer is not None:
                tname     = target.name or 'target'
                target_df = pd.DataFrame({tname: target})
                target    = self.winsorizer.transform(target_df)[tname]

        return target

    # ------------------------------------------------------------------
    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray:
        if self.learning_type == LearningType.CLASSIFICATION and self.encoder is not None:
            mask   = y_pred != -1
            result = np.full(len(y_pred), 'UNKNOWN', dtype=object)
            result[mask] = self.encoder.inverse_transform(y_pred[mask])
            return result
        return y_pred
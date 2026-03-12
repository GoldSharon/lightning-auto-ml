"""
app/services/data_prep_agent.py
================================
DataPrepAgent  — automated data preprocessing pipeline
_DF_STORE      — in-memory dataset registry
load_df()      — file loader helper

Execution order (fit_transform):
    -1. Target preprocessing (may drop rows)
     0. Drop columns
     1. Missing value imputation
     2. Feature engineering        ← BEFORE outliers
     3. Outlier handling           ← AFTER feature engineering
     4. Categorical encoding
     5. Feature transformation (log, yeo-johnson)
     6. Scaling
     7. Feature selection
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from feature_engine.encoding import CountFrequencyEncoder, OneHotEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.core.ml_types import InputConfiguration, LearningType, MLType
from app.util.feature_engine import FeatureEngineeringEngine, TargetProcessor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ==============================================================================
# DATAFRAME STORE
# ==============================================================================

_DF_STORE: Dict[str, Optional[pd.DataFrame]] = {
    "boston":     None,
    "california": None,
    "diabetes":   None,
    "iris":       None,
    "wine":       None,
    "file":       None,
    "mall":       None,
}


def load_df(key: str, filepath: str, **read_kwargs) -> pd.DataFrame:
    """Load a file into _DF_STORE and return the DataFrame."""
    ext = Path(filepath).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(filepath, **read_kwargs)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(filepath, **read_kwargs)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    _DF_STORE[key] = df
    logger.info(f"Loaded '{key}': {df.shape}")
    return df


# ==============================================================================
# DATA PREP AGENT
# ==============================================================================

class DataPrepAgent:
    """Automated data preprocessing agent."""

    def __init__(self, ipc: InputConfiguration, df: pd.DataFrame):
        self.ml_type       = ipc.ml_type
        self.learning_type = ipc.learning_type
        self.verbose       = True

        self.feature_df = df.drop(columns=[ipc.index_column], errors="ignore").copy()

        self.target_column_name = ipc.target_column
        if ipc.target_column and ipc.target_column in df.columns:
            self.original_target = df[ipc.target_column].copy()
            self.feature_df      = self.feature_df.drop(columns=[ipc.target_column])
        else:
            self.original_target = None

        self.original_df = self.feature_df.copy()

        self.transformers                = {}
        self.preprocessing_report        = {}
        self.feature_engineering_configs = []

        self.target_processor = None
        if self.learning_type in [LearningType.REGRESSION, LearningType.CLASSIFICATION]:
            self.target_processor = TargetProcessor(self.learning_type, self.verbose)

        self._is_fitted   = False
        self._last_target: Optional[pd.Series] = None

        if self.verbose:
            print(f"\nDataPrepAgent initialised:")
            print(f"  • Features shape: {self.feature_df.shape}")
            print(f"  • ML type:        {self.ml_type.value}")
            print(f"  • Task type:      {self.learning_type.value}")
            print(f"  • Has target:     {self.original_target is not None}")

    # ------------------------------------------------------------------
    def fit_transform(
        self, preprocessing_config: Dict
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit on training data and transform. MAY drop rows (missing target)."""
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"FIT_TRANSFORM — {self.learning_type.value}")
            print("=" * 70)

        self._validate_config(preprocessing_config)

        # Step -1: Target
        target        = None
        valid_indices = self.feature_df.index

        if self.target_processor and self.original_target is not None:
            target, valid_indices = self.target_processor.fit_transform(
                self.original_target, self.feature_df.index
            )
            self.feature_df = self.feature_df.loc[valid_indices]

        # Step 0: Drop columns
        self._step_0_drop_columns(preprocessing_config)

        # Step 1: Imputation
        self._step_1_fit_transform_missing(preprocessing_config)

        # Step 2: Feature engineering  ← CRITICAL: before outliers
        self._step_2_feature_engineering(preprocessing_config)

        # Fill NaN introduced by feature engineering before outlier handling
        self._fill_post_engineering_nans(self.feature_df)

        # Step 3: Outlier handling     ← CRITICAL: after feature engineering
        self._step_3_fit_transform_outliers(preprocessing_config)

        # Step 4: Categorical encoding
        self._step_4_fit_transform_categorical(preprocessing_config)

        # Step 5: Feature transformation
        if self.learning_type in [LearningType.REGRESSION, LearningType.CLASSIFICATION]:
            self._step_5_fit_transform_features(preprocessing_config)

        # Step 6: Scaling
        self._step_6_fit_transform_scale(preprocessing_config)

        # Step 7: Feature selection
        self._step_7_fit_transform_feature_selection(preprocessing_config)

        self.feature_df   = self._sanitize_column_names(self.feature_df)
        self._is_fitted   = True
        self._last_target = target
        self._generate_report(preprocessing_config, target)

        if self.verbose:
            print("\n" + "=" * 70)
            print("FIT_TRANSFORM COMPLETE")
            print(f"Final features shape: {self.feature_df.shape}")
            if target is not None:
                print(f"Target shape:         {target.shape}")
            print("=" * 70)

        return self.feature_df, target

    # ------------------------------------------------------------------
    def transform(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data. NEVER drops rows."""
        if not self._is_fitted:
            raise ValueError("Call fit_transform() on training data first.")

        if self.verbose:
            print("\n" + "=" * 70)
            print("TRANSFORM (PREDICTION MODE)")
            print(f"Input shape: {new_df.shape}")
            print("=" * 70)

        df           = new_df.copy()
        original_len = len(df)

        if self.target_column_name and self.target_column_name in df.columns:
            df = df.drop(columns=[self.target_column_name])

        # Step 0
        if 'dropped_columns' in self.transformers:
            cols = [c for c in self.transformers['dropped_columns'] if c in df.columns]
            if cols:
                df = df.drop(columns=cols)

        # Step 1
        for key in ['numeric_imputer', 'categorical_imputer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 2
        if self.verbose and self.feature_engineering_configs:
            print(f"  • Applying {len(self.feature_engineering_configs)} feature engineering rule(s)…")

        for config in self.feature_engineering_configs:
            new_feature = config['new_feature']
            try:
                if all(c in df.columns for c in config['source_columns']):
                    df[new_feature] = FeatureEngineeringEngine.create_feature(df, config)
                    if self.verbose:
                        print(f"    → Created: {new_feature}")
                else:
                    missing = [c for c in config['source_columns'] if c not in df.columns]
                    if self.verbose:
                        print(f"    ⚠ Skipped {new_feature}: missing source cols {missing}")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Feature engineering failed for {new_feature}: {e}")

        self._fill_post_engineering_nans(df)

        # Step 3
        if 'winsorizer' in self.transformers:
            try:
                df = self.transformers['winsorizer'].transform(df)
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Winsorizer failed: {e}")

        # Step 4
        for key in ['onehot_encoder', 'frequency_encoder']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 5
        for key in ['log_transformer', 'yeo_johnson_transformer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 6
        if 'scaler' in self.transformers:
            try:
                df = self.transformers['scaler'].transform(df)
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Scaler failed: {e}")

        # Step 7
        for key in ['constant_dropper', 'duplicate_dropper']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Emergency NaN fill
        if df.isnull().any().any():
            for col in df.columns[df.isnull().any()]:
                fill = 0 if pd.api.types.is_numeric_dtype(df[col]) else 'missing'
                df[col] = df[col].fillna(fill)

        if len(df) != original_len:
            raise RuntimeError(
                f"FATAL: transform() changed row count! "
                f"Input: {original_len}, Output: {len(df)}"
            )

        if df.isnull().any().any():
            raise RuntimeError(
                f"FATAL: NaN still present after all transforms! "
                f"Cols: {df.columns[df.isnull().any()].tolist()}"
            )

        if self.verbose:
            print(f"  ✓ Output shape: {df.shape} (rows preserved)")

        return df

    # ------------------------------------------------------------------
    def transform_target(self, target: pd.Series) -> pd.Series:
        """Transform target for validation/test. NEVER drops rows."""
        if not self._is_fitted:
            raise ValueError("Call fit_transform() first.")
        if self.target_processor is None:
            raise ValueError("No target processor (unsupervised task?)")
        original_len = len(target)
        result       = self.target_processor.transform_target(target)
        if len(result) != original_len:
            raise RuntimeError(
                f"FATAL: transform_target() changed row count! "
                f"Input: {original_len}, Output: {len(result)}"
            )
        return result

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        if self.target_processor is None:
            return y_pred
        return self.target_processor.inverse_transform(y_pred)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _validate_config(self, config: Dict):
        for key in ['columns_to_drop', 'columns_to_keep', 'pipeline_recommendation']:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

    def _step_0_drop_columns(self, config: Dict):
        cols = [
            c for c in config['columns_to_drop']['column_list']
            if c in self.feature_df.columns
        ]
        if cols:
            if self.verbose:
                print(f"\n[Step 0] Dropping {len(cols)} column(s): {cols}")
            self.feature_df = self.feature_df.drop(columns=cols)
            self.transformers['dropped_columns'] = cols

    def _step_1_fit_transform_missing(self, config: Dict):
        if self.verbose:
            print(f"\n[Step 1] Imputing missing values…")

        pipeline    = config['pipeline_recommendation']
        missing_cfg = pipeline.get('step_1_missing_values', {})
        strategy    = missing_cfg.get(
            'numeric_strategy',
            'median' if self.learning_type == LearningType.CLUSTERING else 'mean'
        )

        numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_miss = [c for c in numeric_cols if self.feature_df[c].isnull().any()]
        if numeric_miss:
            imputer = MeanMedianImputer(imputation_method=strategy, variables=numeric_miss)
            self.feature_df[numeric_miss] = imputer.fit_transform(self.feature_df[numeric_miss])
            self.transformers['numeric_imputer'] = imputer
            if self.verbose:
                print(f"  • Imputed {len(numeric_miss)} numeric column(s) ({strategy})")

        cat_cols = self.feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_miss = [c for c in cat_cols if self.feature_df[c].isnull().any()]
        if cat_miss:
            imputer = CategoricalImputer(imputation_method='frequent', variables=cat_miss)
            self.feature_df[cat_miss] = imputer.fit_transform(self.feature_df[cat_miss])
            self.transformers['categorical_imputer'] = imputer
            if self.verbose:
                print(f"  • Imputed {len(cat_miss)} categorical column(s)")

    def _step_2_feature_engineering(self, config: Dict):
        suggestions = config.get('feature_engineering_suggestions', [])
        if not suggestions:
            return

        if self.verbose:
            print(f"\n[Step 2] Feature engineering ({len(suggestions)} suggestion(s))…")

        for feat_cfg in suggestions:
            new_feature    = feat_cfg['new_feature']
            source_cols    = feat_cfg['source_columns']
            transformation = feat_cfg.get('transformation', [])

            if not all(c in self.feature_df.columns for c in source_cols):
                if self.verbose:
                    print(f"  ⚠ Skipped '{new_feature}': source cols not found")
                continue

            operation = self._parse_operation(transformation)
            if not operation:
                if self.verbose:
                    print(f"  ⚠ Skipped '{new_feature}': unrecognised operation")
                continue

            safe_cfg = {
                'new_feature':    new_feature,
                'source_columns': source_cols,
                'operation':      operation,
            }
            try:
                self.feature_df[new_feature] = FeatureEngineeringEngine.create_feature(
                    self.feature_df, safe_cfg
                )
                self.feature_engineering_configs.append(safe_cfg)
                if self.verbose:
                    print(f"  • Created: {new_feature} ({operation} of {source_cols})")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Failed '{new_feature}': {e}")

    def _step_3_fit_transform_outliers(self, config: Dict):
        pipeline    = config["pipeline_recommendation"]
        outlier_cfg = pipeline.get("step_2_outliers", {})
        method      = outlier_cfg.get("method", "winsorize")

        config_columns  = outlier_cfg.get("columns_affected", [])
        engineered_cols = [
            c for c in self.feature_df.select_dtypes(include=[np.number]).columns
            if c not in self.original_df.columns
        ]
        candidate_cols = list(dict.fromkeys(
            [c for c in config_columns if c in self.feature_df.columns] + engineered_cols
        ))

        def _is_winsorizable(col: str) -> bool:
            series       = self.feature_df[col]
            std          = series.std()
            n_unique     = series.nunique()
            unique_ratio = n_unique / len(series)
            if std == 0:
                return False
            if n_unique <= 10 and unique_ratio < 0.05:
                return False
            return True

        all_numeric  = [c for c in candidate_cols if _is_winsorizable(c)]
        skipped_cols = [c for c in candidate_cols if not _is_winsorizable(c)]

        if skipped_cols and self.verbose:
            print(f"  [Step 3] Skipping low-variance/categorical-like cols: {skipped_cols}")

        if not all_numeric:
            logger.warning("[Step 3] No suitable numeric columns for outlier handling — skipped.")
            return

        if self.verbose:
            print(f"  [Step 3] Outlier handling ({method}) on {len(all_numeric)} numeric column(s)...")

        if method == "winsorize":
            params         = outlier_cfg.get("parameters", {})
            capping_method = params.get("capping_method", "iqr")
            fold           = params.get("fold", 1.5)
            if self.learning_type == LearningType.CLUSTERING:
                fold = max(fold, 2.0)

            winsorizer = Winsorizer(
                capping_method=capping_method,
                tail="both",
                fold=fold,
                variables=all_numeric,
            )
            self.feature_df[all_numeric] = winsorizer.fit_transform(self.feature_df[all_numeric])
            self.transformers["winsorizer"] = winsorizer

            if self.verbose:
                original_cols = [c for c in all_numeric if c in self.original_df.columns]
                new_eng_cols  = [c for c in all_numeric if c not in self.original_df.columns]
                print(
                    f"  Winsorized {len(all_numeric)} column(s) "
                    f"({len(original_cols)} original + {len(new_eng_cols)} engineered)"
                )

    def _step_4_fit_transform_categorical(self, config: Dict):
        if self.verbose:
            print(f"\n[Step 4] Categorical encoding…")

        pipeline     = config['pipeline_recommendation']
        encoding_cfg = pipeline.get('step_3_encoding', {})

        onehot_cols = [
            c for c in encoding_cfg.get('onehot_columns', [])
            if c in self.feature_df.columns
        ]
        if onehot_cols:
            for col in onehot_cols:
                self.feature_df[col] = self.feature_df[col].astype(str).replace('nan', 'missing')
            drop_last = self.learning_type == LearningType.CLUSTERING
            encoder   = OneHotEncoder(variables=onehot_cols, drop_last=drop_last)
            self.feature_df = encoder.fit_transform(self.feature_df)
            self.transformers['onehot_encoder'] = encoder
            if self.verbose:
                print(f"  • One-hot encoded {len(onehot_cols)} column(s)")

        freq_cols = [
            c for c in encoding_cfg.get('frequency_columns', [])
            if c in self.feature_df.columns
        ]
        if freq_cols:
            for col in freq_cols:
                self.feature_df[col] = self.feature_df[col].astype(str)
            encoder = CountFrequencyEncoder(variables=freq_cols, encoding_method='frequency')
            self.feature_df = encoder.fit_transform(self.feature_df)
            self.transformers['frequency_encoder'] = encoder
            if self.verbose:
                print(f"  • Frequency encoded {len(freq_cols)} column(s)")

    def _step_5_fit_transform_features(self, config: Dict):
        if self.verbose:
            print(f"\n[Step 5] Feature transformation…")

        pipeline      = config['pipeline_recommendation']
        transform_cfg = pipeline.get('step_4_transformation', {})

        log_cols = [
            c for c in transform_cfg.get('log_transform', [])
            if c in self.feature_df.columns
            and pd.api.types.is_numeric_dtype(self.feature_df[c])
        ]
        if log_cols:
            try:
                for col in log_cols:
                    if (self.feature_df[col] <= 0).any():
                        self.feature_df[col] -= self.feature_df[col].min() - 1
                transformer = LogTransformer(variables=log_cols)
                self.feature_df = transformer.fit_transform(self.feature_df)
                self.transformers['log_transformer'] = transformer
                if self.verbose:
                    print(f"  • Log-transformed {len(log_cols)} column(s)")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Log transformation failed: {e}")

        yeo_cols = [
            c for c in transform_cfg.get('yeo_johnson', [])
            if c in self.feature_df.columns
            and pd.api.types.is_numeric_dtype(self.feature_df[c])
        ]
        if yeo_cols:
            try:
                transformer = YeoJohnsonTransformer(variables=yeo_cols)
                self.feature_df = transformer.fit_transform(self.feature_df)
                self.transformers['yeo_johnson_transformer'] = transformer
                if self.verbose:
                    print(f"  • Yeo-Johnson transformed {len(yeo_cols)} column(s)")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Yeo-Johnson transformation failed: {e}")

    def _step_6_fit_transform_scale(self, config: Dict):
        pipeline    = config['pipeline_recommendation']
        scaling_cfg = pipeline.get('step_5_scaling', {})
        method      = scaling_cfg.get('method', 'standard')

        columns = [c for c in scaling_cfg.get('columns', []) if c in self.feature_df.columns]
        if not columns:
            columns = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            return

        if self.verbose:
            print(f"\n[Step 6] Scaling {len(columns)} column(s) ({method})…")

        scaler_map = {
            'standard': StandardScaler(),
            'minmax':   MinMaxScaler(),
            'robust':   RobustScaler(),
        }
        scaler  = scaler_map.get(method, StandardScaler())
        wrapper = SklearnTransformerWrapper(transformer=scaler, variables=columns)
        self.feature_df = wrapper.fit_transform(self.feature_df)
        self.transformers['scaler'] = wrapper

    def _step_7_fit_transform_feature_selection(self, config: Dict):
        if self.verbose:
            print(f"\n[Step 7] Feature selection…")

        pipeline      = config['pipeline_recommendation']
        selection_cfg = pipeline.get('step_6_feature_selection', {})
        n_features    = len(self.feature_df.columns)

        if n_features < 2:
            if self.verbose:
                print(f"  ⚠ Skipping — only {n_features} feature(s)")
            return

        if selection_cfg.get('drop_constant', True):
            tol = 0.98 if self.learning_type == LearningType.CLUSTERING else 0.99
            try:
                selector = DropConstantFeatures(tol=tol)
                self.feature_df = selector.fit_transform(self.feature_df)
                self.transformers['constant_dropper'] = selector
                if hasattr(selector, 'features_to_drop_') and selector.features_to_drop_:
                    if self.verbose:
                        print(f"  • Dropped {len(selector.features_to_drop_)} constant feature(s)")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Constant dropper failed: {e}")

        if len(self.feature_df.columns) < 2:
            return

        try:
            selector = DropDuplicateFeatures()
            self.feature_df = selector.fit_transform(self.feature_df)
            self.transformers['duplicate_dropper'] = selector
            if hasattr(selector, 'features_to_drop_') and selector.features_to_drop_:
                if self.verbose:
                    print(f"  • Dropped {len(selector.features_to_drop_)} duplicate feature(s)")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Duplicate dropper failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_operation(self, transformation: list) -> Optional[str]:
        if not transformation:
            return None
        code = ' '.join(transformation).lower()
        if '/' in code: return 'divide'
        if '*' in code: return 'multiply'
        if '+' in code: return 'add'
        if '-' in code: return 'subtract'
        return None

    def _fill_post_engineering_nans(self, df: pd.DataFrame):
        if not df.isnull().any().any():
            return
        nan_cols = df.columns[df.isnull().any()].tolist()
        if self.verbose:
            print(f"  • Filling NaN in {len(nan_cols)} column(s) after feature engineering…")
        for col in nan_cols:
            fill_value = None
            if 'numeric_imputer' in self.transformers:
                imp = self.transformers['numeric_imputer']
                if hasattr(imp, 'imputer_dict_') and col in imp.imputer_dict_:
                    fill_value = imp.imputer_dict_[col]
            if fill_value is None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                else:
                    fill_value = 'missing'
            df[col] = df[col].fillna(fill_value)

    def _sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [
            str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            for c in df.columns
        ]
        return df

    def _generate_report(self, config: Dict, target: Optional[pd.Series]):
        self.preprocessing_report = {
            'learning_type':         self.learning_type.value,
            'original_shape':        self.original_df.shape,
            'final_shape':           self.feature_df.shape,
            'original_columns':      self.original_df.columns.tolist(),
            'final_columns':         self.feature_df.columns.tolist(),
            'columns_added':         list(set(self.feature_df.columns) - set(self.original_df.columns)),
            'columns_removed':       list(set(self.original_df.columns) - set(self.feature_df.columns)),
            'transformers_applied':  list(self.transformers.keys()),
            'missing_values_before': int(self.original_df.isnull().sum().sum()),
            'missing_values_after':  int(self.feature_df.isnull().sum().sum()),
        }
        if self.target_processor and target is not None:
            self.preprocessing_report['target_info'] = {
                'original_dtype':  str(self.original_target.dtype),
                'final_dtype':     str(target.dtype),
                'original_shape':  self.original_target.shape,
                'final_shape':     target.shape,
                'was_encoded':     self.target_processor.encoder    is not None,
                'was_winsorized':  self.target_processor.winsorizer is not None,
                'label_mapping':   self.target_processor.label_mapping or None,
                'rows_dropped':    len(self.original_target) - len(target),
            }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if not self._is_fitted:
            raise ValueError("Call fit_transform() first.")
        return self.feature_df, self._last_target

    def get_report(self) -> Dict:
        return self.preprocessing_report

    def print_report(self):
        r = self.preprocessing_report
        print("\n" + "=" * 70)
        print("PREPROCESSING REPORT")
        print("=" * 70)
        print(f"Task:          {r['learning_type']}")
        print(f"Original:      {r['original_shape']}")
        print(f"Final:         {r['final_shape']}")
        print(f"Transformers:  {', '.join(r['transformers_applied'])}")
        if 'target_info' in r:
            ti = r['target_info']
            print(f"\nTarget:")
            print(f"  • Encoded:      {ti['was_encoded']}")
            print(f"  • Winsorized:   {ti['was_winsorized']}")
            print(f"  • Rows dropped: {ti['rows_dropped']}")
        print("=" * 70)

    def is_fitted(self) -> bool:
        return self._is_fitted

    def save_processed_data(
        self,
        features:          pd.DataFrame,
        target:            Optional[pd.Series] = None,
        features_filepath: Optional[str]       = None,
        target_filepath:   Optional[str]       = None,
        combined_filepath: Optional[str]       = None,
        save_combined:     bool                = False,
    ) -> Dict[str, str]:
        saved_paths = {}

        def _ensure_dir(filepath: str):
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

        if features_filepath:
            _ensure_dir(features_filepath)
            features.to_csv(features_filepath, index=False)
            saved_paths['features'] = features_filepath

        if target is not None and target_filepath:
            _ensure_dir(target_filepath)
            pd.DataFrame(target).to_csv(target_filepath, index=False)
            saved_paths['target'] = target_filepath

        if save_combined and combined_filepath:
            _ensure_dir(combined_filepath)
            combined = features.copy()
            if target is not None:
                col_name = target.name if hasattr(target, 'name') and target.name else 'target'
                combined[col_name] = target.values
            combined.to_csv(combined_filepath, index=False)
            saved_paths['combined'] = combined_filepath

        return saved_paths
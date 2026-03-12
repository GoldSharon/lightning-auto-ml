"""
ML Pipeline - Fully Integrated Module
Supports: Supervised (Regression, Classification) + Unsupervised (Clustering)

FIXES APPLIED:
    1. Winsorizer fitted AFTER feature engineering (includes engineered cols)
    2. get_data() returns actual target
    3. TargetProcessor boolean mask uses .values to prevent index misalignment
    4. _detect_outliers parameter renamed from learning_type → method
    5. Silent outlier skip now logs a warning
    6. save_processed_data guards against empty dirname
    7. Removed _normalize_column_mapping entirely
    8. Removed all Apriori/Association logic

EXECUTION ORDER (fit_transform):
    -1. Target preprocessing (may drop rows)
     0. Drop columns
     1. Missing value imputation
     2. Feature engineering        ← BEFORE outliers
     3. Outlier handling           ← AFTER feature engineering (fitted on all current numeric cols)
     4. Categorical encoding
     5. Feature transformation (log, yeo-johnson)
     6. Scaling
     7. Feature selection
"""

# ==============================================================================
# SECTION 0: IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import json
import logging
import time
import os
import warnings

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv, set_key
from groq import Groq


from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

warnings.filterwarnings('ignore')
load_dotenv()


# ==============================================================================
# SECTION 1: LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 2: ENUMS & CORE MODELS
# ==============================================================================

class MLType(Enum):
    SUPERVISED   = "Supervised"
    UNSUPERVISED = "Unsupervised"

class LearningType(Enum):
    REGRESSION     = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING     = "Clustering"

class ProcessingType(Enum):
    TRAINING_ONLY    = "Training-Alone"
    PREPROCESS_TRAIN = "Pre-Processes+Training"

class Hardware(Enum):
    CPU    = "CPU"
    GPU    = "GPU"
    HYBRID = "Hybrid"


@dataclass
class InputConfiguration:
    project_name:           str
    file_name:              str
    ml_type:                MLType
    learning_type:          LearningType
    processing_type:        ProcessingType
    llm_name:               str
    target_column:          Optional[str]   = None
    index_column:           Optional[str]   = None
    output_folder:          Optional[str]   = None
    acceleration_hardware:  Hardware        = Hardware.CPU
    test_size:              Optional[float] = 0.2
    hyper_parameter_tuning: bool            = False

    def __post_init__(self):
        if self.ml_type == MLType.SUPERVISED and not self.target_column:
            raise ValueError("target_column is required for supervised learning")

    def to_dict(self):
        return {
            'project_name':           self.project_name,
            'file_name':              self.file_name,
            'ml_type':                self.ml_type.value,
            'learning_type':          self.learning_type.value,
            'processing_type':        self.processing_type.value,
            'llm_name':               self.llm_name,
            'target_column':          self.target_column,
            'index_column':           self.index_column,
            'output_folder':          self.output_folder,
            'acceleration_hardware':  self.acceleration_hardware.value,
            'test_size':              self.test_size,
            'hyper_parameter_tuning': self.hyper_parameter_tuning,
        }


@dataclass
class DataAnalysisReport:
    """Comprehensive data analysis report."""
    shape:                Tuple[int, int]
    dtypes:               Dict[str, str]
    missing_values:       Dict[str, int]
    missing_percentage:   Dict[str, float]
    numerical_stats:      Dict[str, Dict]
    categorical_stats:    Dict[str, Dict]
    outliers_detected:    Dict[str, int]
    correlations:         Dict[str, float]
    data_quality_score:   float
    target_column:        Optional[str]
    data_quality_metrics: Dict[str, Any]
    ml_type:              MLType
    learning_type:        LearningType

    def __iter__(self):
        return iter(self.__dict__.items())

    def to_dict(self):
        d = asdict(self)
        d["ml_type"]       = self.ml_type.name
        d["learning_type"] = self.learning_type.name
        return d


class LLMConfiguration:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        load_dotenv()
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME")
        self.api_key    = api_key    or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and not found in .env")
        self._save_to_env()

    def _save_to_env(self):
        set_key(".env", "LLM_MODEL_NAME", self.model_name)
        set_key(".env", "LLM_API_KEY",    self.api_key)


class ProjectMetadata:
    def __init__(self, project_name: str, output_folder: str):
        self.project_name  = project_name
        self.output_folder = output_folder
        self.metadata_file = Path(output_folder) / "metadata.json"

    def save(self, config: InputConfiguration, status: str = "training"):
        metadata = {
            "project_name": self.project_name,
            "created_at":   str(Path(self.output_folder).stat().st_ctime),
            "status":       status,
            "config":       config.to_dict(),
            "models":       [],
            "evaluation_scores": None,
            "best_model":   None,
        }
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def load(self):
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return None

    def update(self, **kwargs):
        metadata = self.load() or {}
        metadata.update(kwargs)
        self.metadata_file.write_text(json.dumps(metadata, indent=2))


# ==============================================================================
# SECTION 3: COGNITIVE ENGINE (LLM caller)
# ==============================================================================


client = Groq(
    api_key=os.getenv("LLM_API_KEY"),
    max_retries=0   # disable Groq automatic retry
)

def chat_llm(system_instructions: str,
             user_input: str,
             max_retries: int = 2) -> dict:
    """
    Fast LLM call with minimal retry delay.
    """

    for attempt in range(max_retries):

        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME"),
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                stream=False,
                timeout=10   # fail fast
            )

            content = response.choices[0].message.content

            if not content:
                return {}

            return json.loads(content)

        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            return {}

        except Exception as e:

            err = str(e).lower()

            if "429" in err or "rate" in err:
                wait = 1.5  # short wait
                logger.warning(f"Rate limit. Retrying in {wait}s")
                time.sleep(wait)

            else:
                logger.error(f"LLM error: {e}")
                return {}

    return {}


# ==============================================================================
# SECTION 4: PROMPTS  — paste your actual prompts inside the triple-quotes
# ==============================================================================

DATATYPE_SELECTION_SYSTEM_PROMPT = """You are an AI programming assistant,  only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
You are an expert pandas data analyst. Your task is to infer the most appropriate pandas data type (dtype) for each column in a given dataset preview.

**CRITICAL RULES:**
1. **PRESERVE EXACT COLUMN NAMES** - The keys in "column_dtypes" MUST match the keys in "summary_stats" EXACTLY, character by character. DO NOT add/remove spaces, change parentheses/hyphens, modify capitalization, or alter formatting.
2. Only choose dtypes from: ["int8","int16","int32","int64","uint8","uint16","uint32","uint64","float16","float32","float64","bool","boolean","object","string","category","datetime64[ns]","timedelta64[ns]","Int8","Int16","Int32","Int64"].
3. Selection criteria: Integers use smallest sufficient type (e.g., int8 for -128 to 127); nullable for missings (Int8+). Floats: float64 default. Booleans: bool/boolean. Strings: string (text), category (low cardinality <50% unique). Dates: datetime64[ns].
4. **OUTPUT ONLY VALID JSON** in this exact format: {"column_dtypes": {"exact_column_name_1": "dtype", "exact_column_name_2": "dtype"}}

**ANALYZE THE INPUT DATA:** Input has columns_list (exact names), data_preview (first 10 rows), summary_stats (uniques, missings, ranges, sample_values).

**EXAMPLE 1 - Customer Data with Special Characters:**
INPUT:
{
  "columns_list": ["customer_id", "full name", "age (years)", "is_active", "gender"],
  "data_preview": [
    {"customer_id": 1, "full name": "Alice", "age (years)": 23, "is_active": true, "gender": "Female"},
    {"customer_id": 2, "full name": "Bob", "age (years)": 30, "is_active": false, "gender": "Male"}
  ],
  "summary_stats": {
    "customer_id": {"dtype": "int64", "missing_values": 0, "unique_values": 1000},
    "full name": {"dtype": "object", "missing_values": 0, "unique_values": 950},
    "age (years)": {"dtype": "int64", "missing_values": 0, "unique_values": 60, "min": 18, "max": 75},
    "is_active": {"dtype": "bool", "missing_values": 0, "unique_values": 2},
    "gender": {"dtype": "object", "missing_values": 0, "unique_values": 3}
  }
}

OUTPUT:
{"column_dtypes": {"customer_id": "int64", "full name": "string", "age (years)": "int8", "is_active": "bool", "gender": "category"}}

**EXAMPLE 2 - Iris Dataset with Measurements:**
INPUT:
{
  "columns_list": ["Unnamed: 0", "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"],
  "data_preview": [
    {"Unnamed: 0": 0, "sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2, "target": 0},
    {"Unnamed: 0": 1, "sepal length (cm)": 4.9, "sepal width (cm)": 3.0, "petal length (cm)": 1.4, "petal width (cm)": 0.2, "target": 0}
  ],
  "summary_stats": {
    "Unnamed: 0": {"dtype": "int64", "missing_values": 0, "unique_values": 150, "min": 0, "max": 149},
    "sepal length (cm)": {"dtype": "float64", "missing_values": 0, "unique_values": 35, "min": 4.3, "max": 7.9},
    "sepal width (cm)": {"dtype": "float64", "missing_values": 0, "unique_values": 23, "min": 2.0, "max": 4.4},
    "petal length (cm)": {"dtype": "float64", "missing_values": 0, "unique_values": 43, "min": 1.0, "max": 6.9},
    "petal width (cm)": {"dtype": "float64", "missing_values": 0, "unique_values": 22, "min": 0.1, "max": 2.5},
    "target": {"dtype": "int64", "missing_values": 0, "unique_values": 3, "min": 0, "max": 2, "likely_categorical": true}
  }
}

OUTPUT:
{"column_dtypes": {"Unnamed: 0": "int64", "sepal length (cm)": "float64", "sepal width (cm)": "float64", "petal length (cm)": "float64", "petal width (cm)": "float64", "target": "int8"}}

**EXAMPLE 3 - Sales Data with Mixed Types:**
INPUT:
{
  "columns_list": ["order-id", "product_name", "price ($)", "quantity", "order date", "shipped?"],
  "data_preview": [
    {"order-id": 1001, "product_name": "Laptop", "price ($)": 899.99, "quantity": 2, "order date": "2024-01-15", "shipped?": true},
    {"order-id": 1002, "product_name": "Mouse", "price ($)": 24.99, "quantity": 5, "order date": "2024-01-16", "shipped?": false}
  ],
  "summary_stats": {
    "order-id": {"dtype": "int64", "missing_values": 0, "unique_values": 500, "min": 1001, "max": 1500},
    "product_name": {"dtype": "object", "missing_values": 0, "unique_values": 50},
    "price ($)": {"dtype": "float64", "missing_values": 0, "unique_values": 200, "min": 9.99, "max": 2999.99},
    "quantity": {"dtype": "int64", "missing_values": 0, "unique_values": 20, "min": 1, "max": 100},
    "order date": {"dtype": "object", "missing_values": 0, "unique_values": 365, "sample_values": ["2024-01-15", "2024-01-16", "2024-01-17"]},
    "shipped?": {"dtype": "bool", "missing_values": 0, "unique_values": 2}
  }
}

OUTPUT:
{"column_dtypes": {"order-id": "int64", "product_name": "category", "price ($)": "float64", "quantity": "int8", "order date": "datetime64[ns]", "shipped?": "bool"}}

**VERIFICATION CHECKLIST:** Before output: ✓ Keys match summary_stats exactly. ✓ No extra spaces/special char changes. ✓ Dtypes from allowed list.

Now analyze the provided input and output ONLY the JSON with column_dtypes.

### Response:
"""

SUPERVISED_COLUMN_SELECTION_SYSTEM_PROMPT = """You are an AI programming assistant,  only answer questions related to computer science and supervised machine learning preprocessing. ### Instruction: You are an expert Feature Engineer and Data Scientist. Receive dataset info and produce a preprocessing/feature engineering plan specifically for **supervised ML tasks** (classification or regression). **CRITICAL RULES:** 
1. OUTPUT **ONLY VALID JSON**. No extra text. 
2. Reference only existing columns. 
3. Use stats: missing_percentage, outliers_detected, correlations, numerical_stats, categorical_stats, target_column. 
4. Drop IDs/unique indices always. 
5. Allowed options: Missing ["drop","median","mode","forward_fill","backward_fill"]; 
Outliers ["winsorize","clip","remove"]; 
Categorical ["onehot","frequency","label","target"]; 
Scaling ["standard","minmax","robust"]; 
Transform ["log","sqrt","boxcox","yeo-johnson"]. 
6. No repeated keys; balanced JSON. 
**INPUT STRUCTURE:** columns (name->dtype), 
first_5_rows (samples), column_names, shape, missing_values/percentage, outliers_detected, correlations, numerical_stats, categorical_stats, target_column, data_quality_metrics. 
**REQUIRED OUTPUT JSON SCHEMA:** 
{ "columns_to_drop": { "column_list": ["col1", "col2"], "reasons": { "col1": ["reason1", "reason2"], "col2": ["reason1"] } }, "columns_to_keep": { "column_list": ["col3", "col4"], "details": { "col3": { "reason": "why keep this", "pipeline_recommendation": ["median", "standard"], "priority": "high|medium|low", "confidence": 0.9, "evidence": { "missing_pct": 5.2, "unique_count": 45, "dtype": "float64", "outliers": 3, "corr_with_target": 0.45 } } } }, "feature_engineering_suggestions": [ { "new_feature": "feature_name", "description": "what it represents", "source_columns": ["col1", "col2"], "transformation": ["df['new_feature'] = df['col1'] + df['col2']"], "expected_type": "numeric|categorical", "expected_impact": "high|medium|low", "confidence": 0.8, "risk_assessment": "leakage risk if any" } ], "pipeline_recommendation": { "step_1_missing_values": { "numeric_strategy": "median", "categorical_strategy": "mode", "columns_affected": ["col1", "col2"] }, "step_2_outliers": { "method": "winsorize", "columns_affected": ["col3"], "parameters": {"capping_method": "iqr", "fold": 1.5} }, "step_3_encoding": { "onehot_columns": ["col4"], "frequency_columns": ["col5"], "parameters": {"drop": "first"} }, "step_4_transformation": { "log_transform": ["col6"], "yeo_johnson": ["col7"] }, "step_5_scaling": { "method": "standard", "columns": ["col6", "col7", "col8"] }, "step_6_feature_selection": { "drop_correlated": true, "correlation_threshold": 0.95, "drop_constant": true } }, "summary": { "original_features": 12, "recommended_features": 15, "features_dropped": 3, "new_features_created": 6, "expected_model_improvement": "Brief explanation", "key_insights": ["insight1", "insight2"] } } **EXAMPLE - Diabetes Dataset (Supervised Classification):** INPUT: { "columns": { "Pregnancies": "int64", "Glucose": "int64", "BloodPressure": "int64", "SkinThickness": "int64", "Insulin": "int64", "BMI": "float64", "DiabetesPedigreeFunction": "float64", "Age": "int64", "Outcome": "int64" }, "shape": [768, 9], "missing_percentage": {"Glucose": 0.0, "BloodPressure": 0.0, "SkinThickness": 29.0, "Insulin": 49.0, "BMI": 0.0}, "numerical_stats": {"Glucose": {"mean": 120.9, "std": 31.9}, "BMI": {"mean": 32.0, "std": 7.9}}, "categorical_stats": {"Outcome": {"unique_values": 2}}, "correlations": {"Glucose_vs_Outcome": 0.47, "BMI_vs_Outcome": 0.17}, "outliers_detected": {"Glucose": 5, "BMI": 3}, "target_column": "Outcome" } OUTPUT: { "columns_to_drop": { "column_list": ["Insulin", "SkinThickness"], "reasons": { "Insulin": ["49% missing values, unreliable feature"], "SkinThickness": ["29% missing values, noisy measurements"] } }, "columns_to_keep": { "column_list": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"], "details": { "Glucose": { "reason": "strong correlation with outcome", "recommended_preprocessing": ["median", "standard"], "priority": "high", "confidence": 0.95, "evidence": { "missing_pct": 0.0, "outliers": 5, "corr_with_target": 0.47 } }, "BMI": { "reason": "moderate correlation with outcome", "recommended_preprocessing": ["median", "standard"], "priority": "medium", "confidence": 0.8, "evidence": { "missing_pct": 0.0, "outliers": 3, "corr_with_target": 0.17 } }, "Pregnancies": { "reason": "relevant demographic factor", "recommended_preprocessing": ["median", "standard"], "priority": "medium", "confidence": 0.75, "evidence": { "missing_pct": 0.0, "unique_count": 17, "dtype": "int64", "outliers": 0, "corr_with_target": 0.22 } }, "BloodPressure": { "reason": "physiological indicator", "recommended_preprocessing": ["median", "standard"], "priority": "low", "confidence": 0.7, "evidence": { "missing_pct": 0.0, "outliers": 2, "corr_with_target": 0.07 } }, "DiabetesPedigreeFunction": { "reason": "genetic risk factor", "recommended_preprocessing": ["median", "standard"], "priority": "medium", "confidence": 0.8, "evidence": { "missing_pct": 0.0, "outliers": 4, "corr_with_target": 0.17 } }, "Age": { "reason": "age-related risk", "recommended_preprocessing": ["median", "standard"], "priority": "medium", "confidence": 0.85, "evidence": { "missing_pct": 0.0, "outliers": 1, "corr_with_target": 0.24 } } } }, "feature_engineering_suggestions": [ { "new_feature": "Age_BMI_Ratio", "description": "BMI normalized by Age", "source_columns": ["Age", "BMI"], "transformation": ["df['Age_BMI_Ratio'] = df['BMI']/df['Age']"], "expected_type": "numeric", "expected_impact": "medium", "confidence": 0.8, "risk_assessment": "no leakage risk" }, { "new_feature": "Glucose_Age_Interaction", "description": "Interaction between Glucose and Age", "source_columns": ["Glucose", "Age"], "transformation": ["df['Glucose_Age_Interaction'] = df['Glucose'] * df['Age']"], "expected_type": "numeric", "expected_impact": "high", "confidence": 0.85, "risk_assessment": "low risk of overfitting" } ], "pipeline_recommendation": { "step_1_missing_values": { "numeric_strategy": "median", "categorical_strategy": "mode", "columns_affected": ["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"] }, "step_2_outliers": { "method": "winsorize", "columns_affected": ["Glucose", "BMI"], "parameters": {"capping_method": "iqr", "fold": 1.5} }, "step_3_encoding": { "onehot_columns": [], "frequency_columns": [], "parameters": {"drop": "first"} }, "step_4_transformation": { "log_transform": ["DiabetesPedigreeFunction"], "yeo_johnson": ["BMI"] }, "step_5_scaling": { "method": "standard", "columns": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age", "Age_BMI_Ratio", "Glucose_Age_Interaction"] }, "step_6_feature_selection": { "drop_correlated": true, "correlation_threshold": 0.95, "drop_constant": true } }, "summary": { "original_features": 9, "recommended_features": 8, "features_dropped": 2, "new_features_created": 2, "expected_model_improvement": "Improved model accuracy by removing sparse features and adding interaction terms for better prediction of diabetes outcome.", "key_insights": ["Glucose is the strongest predictor", "BMI and Age show moderate correlations", "Insulin and SkinThickness are unreliable due to high missing values", "New features like Age_BMI_Ratio may capture non-linear relationships"] } } **IMPORTANT:** Analyze the ACTUAL input dataset provided, not the Diabetes example. Output ONLY valid JSON. ### Response:
"""

CLUSTERING_COLUMN_SELECTION_SYSTEM_PROMPT = """
You are an AI programming assistant, utilizing the  only answer questions related to unsupervised learning (clustering).

### Instruction:
You are an expert Data Scientist. Receive dataset info and produce a preprocessing plan for **clustering tasks**.

**CRITICAL RULES:**
1. OUTPUT **ONLY VALID JSON**. No extra explanatory text.
2. Reference only existing columns from the dataset.
3. Use dataset stats: missing_percentage, outliers_detected, correlations, numerical_stats, categorical_stats.
4. Always drop IDs, unique identifiers, or constant features.
5. Features should NOT be dropped unless:
   - They have >60% missing values, OR
   - They are irrelevant identifiers, OR
   - They provide no variance (constant values).
6. Outliers should be **treated** (winsorize, clip, robust scaling) instead of dropping features.
7. Scaling is **mandatory** for clustering, since distance metrics are sensitive to feature magnitudes.
8. Skewed features should be log/yeo-johnson transformed before scaling.
9. Allowed preprocessing options:
   - Missing: ["drop","median","mode","forward_fill","backward_fill"]
   - Outliers: ["winsorize","clip","remove"]
   - Categorical encoding: ["onehot","frequency","label"]
   - Scaling: ["standard","minmax","robust"]
   - Transformations: ["log","sqrt","boxcox","yeo-johnson"]
10. Do not repeat keys in JSON. Maintain consistency across sections.
11. If a column is dropped, it must not appear in pipeline steps.

**INPUT STRUCTURE:**
{
  "columns": { "col1": "dtype", "col2": "dtype" },
  "first_5_rows": {...},
  "column_names": [...],
  "shape": [rows, cols],
  "missing_values": {...},
  "missing_percentage": {...},
  "outliers_detected": {...},
  "correlations": {...},
  "numerical_stats": {...},
  "categorical_stats": {...},
  "data_quality_metrics": {...}
}

**REQUIRED OUTPUT JSON SCHEMA:**
{
  "columns_to_drop": {
    "column_list": ["col1"],
    "reasons": { "col1": ["reason1","reason2"] }
  },
  "columns_to_keep": {
    "column_list": ["col2","col3"],
    "details": {
      "col2": {
        "reason": "why keep this",
        "recommended_preprocessing": ["standard"],
        "priority": "high|medium|low",
        "confidence": 0.9,
        "evidence": {
          "missing_pct": 0,
          "unique_count": 45,
          "dtype": "float64",
          "outliers": 3
        }
      }
    }
  },
  "feature_engineering_suggestions": [
    {
      "new_feature": "feature_name",
      "description": "what it represents",
      "source_columns": ["col2","col3"],
      "transformation": ["df['feature_name'] = df['col2']/df['col3']"],
      "expected_type": "numeric",
      "expected_impact": "medium",
      "confidence": 0.8,
      "risk_assessment": "any risks"
    }
  ],
  "pipeline_recommendation": {
    "step_1_missing_values": {
      "numeric_strategy": "median",
      "categorical_strategy": "mode",
      "columns_affected": ["col2","col3"]
    },
    "step_2_outliers": {
      "method": "winsorize",
      "columns_affected": ["col4"],
      "parameters": {"capping_method": "iqr","fold":1.5}
    },
    "step_3_encoding": {
      "onehot_columns": ["col5"],
      "frequency_columns": [],
      "parameters": {"drop":"first"}
    },
    "step_4_transformation": {
      "log_transform": ["col6"],
      "yeo_johnson": ["col7"]
    },
    "step_5_scaling": {
      "method": "standard",
      "columns": ["col2","col3","col6","col7"]
    },
    "step_6_feature_selection": {
      "drop_correlated": true,
      "correlation_threshold": 0.95,
      "drop_constant": true
    }
  },
  "summary": {
    "original_features": 12,
    "recommended_features": 14,
    "features_dropped": 2,
    "new_features_created": 4,
    "expected_model_improvement": "Better cluster separation through normalization and ratio features",
    "key_insights": ["insight1","insight2"]
  }
}


**IMPORTANT:** Analyze the ACTUAL input dataset provided, not the Mall Customers example. Output ONLY valid JSON.

### Response:
"""

AUTOML_EVALUATION_SYSTEM_PROMPT = """You are an AI programming assistant,  only answer questions related to machine learning and AutoML model evaluation. For politically sensitive questions, security and privacy issues, and other non-ML questions, you will refuse to answer.

### Instruction:
You are an expert in AutoML and machine learning model evaluation. Your task is to analyze the summary of multiple trained models and provide actionable insights.

**CRITICAL RULES:**
1. Only analyze the models provided in the input JSON. Do not assume or fabricate other models.
2. Focus on evaluation metrics (accuracy, F1-score, RMSE, etc.), training vs validation performance, and hyperparameters.
3. Identify overfitting or underfitting based on train/validation metrics.
4. Suggest improvements in model choice, hyperparameters, or training strategy if needed.
5. **OUTPUT ONLY VALID JSON** in this exact format:
{
  "best_model": "model_name_or_id",
  "analysis": "brief explanation of model performance, overfitting/underfitting",
  "recommendations": "suggested improvements or hyperparameter tuning advice"
}

**ANALYZE THE INPUT DATA:** Input will include a JSON summary of trained models with fields like: model_name, train_score, val_score, hyperparameters, fit_time, etc.

**EXAMPLE 1 - Classification Models Summary:**
INPUT:
{
    "models": [
        {"model_name": "RandomForest_1", "train_score": 0.98, "val_score": 0.88, "hyperparameters": {"n_estimators": 100, "max_depth": 10}},
        {"model_name": "XGBoost_1", "train_score": 0.92, "val_score": 0.90, "hyperparameters": {"learning_rate": 0.1, "n_estimators": 200}}
    ]
}

OUTPUT:
{
    "best_model": "XGBoost_1",
    "analysis": "RandomForest_1 shows overfitting (train_score much higher than val_score). XGBoost_1 has balanced train and val scores.",
    "recommendations": "Consider tuning RandomForest_1 max_depth or using regularization. XGBoost_1 is well-configured, but slight learning_rate adjustment could improve further."
}

**EXAMPLE 2 - Regression Models Summary:**
INPUT:
{
    "models": [
        {"model_name": "LinearRegression_1", "train_score": 0.75, "val_score": 0.73, "hyperparameters": {}},
        {"model_name": "GradientBoosting_1", "train_score": 0.95, "val_score": 0.78, "hyperparameters": {"n_estimators": 150, "learning_rate": 0.05}}
    ]
}

OUTPUT:
{
    "best_model": "LinearRegression_1",
    "analysis": "GradientBoosting_1 is overfitting heavily (train_score >> val_score). LinearRegression_1 has consistent train and val scores.",
    "recommendations": "For GradientBoosting_1, reduce n_estimators or increase regularization. LinearRegression_1 is stable, but consider feature engineering for improvement."
}

**VERIFICATION CHECKLIST:** Before output: ✓ JSON keys must be exactly 'best_model', 'analysis', 'recommendations'. ✓ Do not add extra fields. ✓ Base analysis only on input models.

Now analyze the provided input and output ONLY the JSON with the evaluation.
"""


# ==============================================================================
# SECTION 5: DATAFRAME PLACEHOLDER STORE
# ==============================================================================
# Drop any DataFrame you want to test into this dict.
# Keys are human-readable labels; values start as None — fill them in.
#
# Usage:
#   df = load_df("boston", "Datasets/boston_house.csv")
#   df = load_df("iris",   "Datasets/iris.parquet")     # parquet auto-detected
#
# Then pass _DF_STORE["boston"] wherever a df argument is needed.
# ==============================================================================

_DF_STORE: Dict[str, Optional[pd.DataFrame]] = {
    # ── Regression ────────────────────────────────────────────────────────────
    "boston":     None,   # load_df("boston",     "Datasets/boston_house.csv")
    "california": None,   # load_df("california", "Datasets/california_housing.csv")
    "diabetes":   None,   # load_df("diabetes",   "Datasets/diabetes.csv")

    # ── Classification ────────────────────────────────────────────────────────
    "iris":       None,   # load_df("iris",  "Datasets/iris.csv")
    "wine":       None,   # load_df("wine",  "Datasets/wine_balanced.csv")
    "file":       None,   # load_df("file",  "Datasets/file.csv")

    # ── Clustering ────────────────────────────────────────────────────────────
    "mall":       None,   # load_df("mall",  "Datasets/mall_customers.csv")
}


def load_df(key: str, filepath: str, **read_kwargs) -> pd.DataFrame:
    """
    Load a file into _DF_STORE and return the DataFrame.
    Supports .csv, .parquet, .xlsx / .xls.
    """
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
# SECTION 6: INTELLIGENT DATA ANALYZER
# ==============================================================================

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

        # Only map columns that actually exist (normalization handled externally)
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
    # FIX #4: parameter renamed from `learning_type` to `method`
    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> int:
        """Detect outliers using IQR. Parameter renamed to avoid shadowing self.learning_type."""
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


# ==============================================================================
# SECTION 7: FEATURE ENGINEERING ENGINE
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
# SECTION 8: TARGET PROCESSOR
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

        # FIX #3: use .values so boolean mask aligns positionally, not by index label
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


# ==============================================================================
# SECTION 9: DATA PREP AGENT
# ==============================================================================

class DataPrepAgent:
    """
    Automated data preprocessing agent.

    Execution order (fit_transform):
        -1. Target preprocessing
         0. Drop columns
         1. Missing value imputation
         2. Feature engineering        ← BEFORE outliers
         3. Outlier handling           ← AFTER feature engineering,
                                         fitted on ALL current numeric cols
         4. Categorical encoding
         5. Feature transformation
         6. Scaling
         7. Feature selection
    """

    def __init__(self, ipc: InputConfiguration, df: pd.DataFrame):
        self.ml_type       = ipc.ml_type
        self.learning_type = ipc.learning_type
        self.verbose       = True

        self.feature_df = df.drop(columns=[ipc.index_column], errors="ignore").copy()

        # Separate target from features
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

        # Fill NaN introduced by feature engineering (e.g. division by zero)
        # MUST happen before outlier handling — winsorizer crashes on NaN
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
        """
        Transform new data (validation / test / production).
        NEVER drops rows. Steps mirror fit_transform() order exactly.
        """
        if not self._is_fitted:
            raise ValueError("Call fit_transform() on training data first.")

        if self.verbose:
            print("\n" + "=" * 70)
            print("TRANSFORM (PREDICTION MODE)")
            print(f"Input shape: {new_df.shape}")
            print("=" * 70)

        df          = new_df.copy()
        original_len = len(df)

        if self.target_column_name and self.target_column_name in df.columns:
            df = df.drop(columns=[self.target_column_name])

        # Step 0: Drop columns
        if 'dropped_columns' in self.transformers:
            cols = [c for c in self.transformers['dropped_columns'] if c in df.columns]
            if cols:
                df = df.drop(columns=cols)

        # Step 1: Imputation
        for key in ['numeric_imputer', 'categorical_imputer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 2: Feature engineering
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

        # Fill NaN from feature engineering before further transforms
        self._fill_post_engineering_nans(df)

        # Step 3: Outlier handling
        if 'winsorizer' in self.transformers:
            try:
                df = self.transformers['winsorizer'].transform(df)
                if self.verbose:
                    print("  ✓ Winsorizer applied")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Winsorizer failed: {e}")

        # Step 4: Encoding
        for key in ['onehot_encoder', 'frequency_encoder']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 5: Transformation
        for key in ['log_transformer', 'yeo_johnson_transformer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ {key} failed: {e}")

        # Step 6: Scaling
        if 'scaler' in self.transformers:
            try:
                df = self.transformers['scaler'].transform(df)
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Scaler failed: {e}")

        # Step 7: Feature selection
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
                if self.verbose:
                    print(f"    → Emergency fill '{col}' with {fill}")

        # Row count contract
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

    # ------------------------------------------------------------------
    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        if self.target_processor is None:
            return y_pred
        return self.target_processor.inverse_transform(y_pred)

    # ------------------------------------------------------------------
    # Preprocessing Steps
    # ------------------------------------------------------------------

    def _validate_config(self, config: Dict):
        for key in ['columns_to_drop', 'columns_to_keep', 'pipeline_recommendation']:
            # print(config)
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _step_3_fit_transform_outliers(self, config: Dict):
        """
        FIX #1 — Winsorizer is fitted AFTER feature engineering.
        FIX #NEW — Skip cols unsuitable for IQR winsorization:
          - Zero variance (std == 0)
          - Low cardinality + low unique ratio: binary or categorical-like
            (handles binary 0/1 AND multi-class 0/1/2/3 columns)
        Uses LLM columns_affected as base + adds engineered cols.
        """
        pipeline    = config["pipeline_recommendation"]
        outlier_cfg = pipeline.get("step_2_outliers", {})
        method      = outlier_cfg.get("method", "winsorize")

        # Use LLM columns_affected as base set
        config_columns = outlier_cfg.get("columns_affected", [])

        # Engineered cols (not in original df) — always include if numeric
        engineered_cols = [
            c for c in self.feature_df.select_dtypes(include=[np.number]).columns
            if c not in self.original_df.columns
        ]

        # Combine: LLM specified cols + engineered cols (preserve order, no duplicates)
        candidate_cols = list(dict.fromkeys(
            [c for c in config_columns if c in self.feature_df.columns] + engineered_cols
        ))

        # Filter out cols unsuitable for IQR winsorization
        def _is_winsorizable(col: str) -> bool:
            series       = self.feature_df[col]
            std          = series.std()
            n_unique     = series.nunique()
            unique_ratio = n_unique / len(series)
            if std == 0:
                return False  # zero variance
            if n_unique <= 10 and unique_ratio < 0.05:
                return False  # binary or categorical-like (0/1 or 0/1/2/3)
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
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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
        """Fill NaN values introduced by feature engineering before further transforms."""
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

    # FIX #2: returns actual stored target instead of None
    def get_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Return (features, target) after fit_transform."""
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

    # FIX #6: guards against empty dirname
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
            if dirpath:  # guard: dirname returns '' for flat filenames
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



# ==============================================================================
# SECTION 10: QUICK-TEST BLOCK
# ==============================================================================
# 1. Uncomment ONE dataset block.
# 2. Fill in the correct target_column name.
# 3. Run:  python ml_pipeline.py
# ==============================================================================

if __name__ == "__main__":

    # ── Regression ────────────────────────────────────────────────────────────
    # df = load_df("boston", "Datasets/boston_house.csv")
    # ipc = InputConfiguration(
    #     project_name="test_regression",
    #     file_name="boston_house.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.REGRESSION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="MEDV",        # ← adjust to your actual target col
    # )

    # df = load_df("california", "Datasets/california_housing.csv")
    # ipc = InputConfiguration(
    #     project_name="test_california",
    #     file_name="california_housing.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.REGRESSION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="median_house_value",   # ← adjust
    # )



    # ── Classification ────────────────────────────────────────────────────────

    # df = load_df("diabetes", "Datasets/diabetes.csv")
    # ipc = InputConfiguration(
    #     project_name="test_diabetes",
    #     file_name="diabetes.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.CLASSIFICATION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="Outcome",       # ← adjust
    # )

    # df = load_df("iris", "Datasets/iris.csv")
    # ipc = InputConfiguration(
    #     project_name="test_iris",
    #     file_name="iris.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.CLASSIFICATION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="Species",      # ← adjust
    # )

    # df = load_df("wine", "Datasets/wine_balanced.csv")
    # ipc = InputConfiguration(
    #     project_name="test_wine",
    #     file_name="wine_balanced.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.CLASSIFICATION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="quality",      # ← adjust
    # )

    # df = load_df("file", "Datasets/file.csv")
    # ipc = InputConfiguration(
    #     project_name="test_file",
    #     file_name="file.csv",
    #     ml_type=MLType.SUPERVISED,
    #     learning_type=LearningType.CLASSIFICATION,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    #     target_column="<YOUR_TARGET>",  # ← adjust
    # )

    # ── Clustering ────────────────────────────────────────────────────────────
    # df = load_df("mall", "Datasets/mall_customers.csv")
    # ipc = InputConfiguration(
    #     project_name="test_clustering",
    #     file_name="mall_customers.csv",
    #     ml_type=MLType.UNSUPERVISED,
    #     learning_type=LearningType.CLUSTERING,
    #     processing_type=ProcessingType.PREPROCESS_TRAIN,
    #     llm_name=os.getenv("LLM_MODEL_NAME"),
    # )

    # ── Common pipeline (uncomment one block above first) ─────────────────────
    analyzer            = IntelligentDataAnalyzer(ipc)
    report, prep_config = analyzer.analyze_dataframe(df)
    print(analyzer.get_preprocessing_report(report, prep_config))
    
    agent           = DataPrepAgent(ipc, df)
    features, target = agent.fit_transform(prep_config)
    agent.print_report()
    
    print("\nFeatures shape:", features.shape)
    if target is not None:
        print("Target shape:  ", target.shape)
    #
    # # Access store
    # print(_DF_STORE.keys())


    # ── AutoML Training (uncomment after pipeline runs successfully) ─────────
    from automl_trainer import AutoMLTrainer
    
    trainer          = AutoMLTrainer(ipc, features, target)
    training_results = trainer.train()
    
    print("Best model:", training_results["best_model_name"])
    print("Best score:", training_results["best_score"])
    print("All results:")
    print(trainer.get_results_df())

    print("ml_pipeline.py loaded. Uncomment a dataset block in __main__ to test.")
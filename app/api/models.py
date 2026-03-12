from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class MLTypeEnum(str, Enum):
    SUPERVISED   = "Supervised"
    UNSUPERVISED = "Unsupervised"


class LearningTypeEnum(str, Enum):
    REGRESSION     = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING     = "Clustering"


# ─── Upload & Config ──────────────────────────────────────────────────────────

class DatasetInfoResponse(BaseModel):
    filename:       str
    rows:           int
    columns:        int
    column_names:   List[str]
    dtypes:         Dict[str, str]
    sample_data:    List[Dict[str, Any]]
    missing_counts: Dict[str, int]
    detected_task:  Optional[LearningTypeEnum] = None
    suggested_target: Optional[str] = None
    message:        str = "Dataset uploaded successfully"


class PipelineConfigRequest(BaseModel):
    target_column:  Optional[str]        = None
    index_column:   Optional[str]        = None
    ml_type:        MLTypeEnum           = MLTypeEnum.SUPERVISED
    learning_type:  LearningTypeEnum     = LearningTypeEnum.CLASSIFICATION
    test_size:      float                = Field(0.2, ge=0.05, le=0.5)
    project_name:   str                  = "automl_project"


class PipelineConfigResponse(BaseModel):
    status:        str
    config:        Dict[str, Any]
    message:       str


# ─── Analysis ─────────────────────────────────────────────────────────────────

class AnalysisStatusResponse(BaseModel):
    status:      str   # "running" | "complete" | "error"
    step:        int
    step_name:   str
    progress:    int   # 0-100
    message:     str


class DataInsightsResponse(BaseModel):
    quality_score:         float
    key_features:          List[str]
    dropped_columns:       List[str]
    drop_reasons:          Dict[str, List[str]]
    missing_values:        Dict[str, float]
    outliers_detected:     Dict[str, int]
    correlations:          Dict[str, float]
    numerical_stats:       Dict[str, Any]
    categorical_stats:     Dict[str, Any]
    feature_count_original: int
    feature_count_final:   int


# ─── Preprocessing ────────────────────────────────────────────────────────────

class PreprocessingPlanResponse(BaseModel):
    missing_strategy:       str
    outlier_method:         str
    scaling_method:         str
    encoding_methods:       Dict[str, List[str]]
    feature_engineering:    List[Dict[str, Any]]
    columns_to_drop:        List[str]
    columns_to_keep:        List[str]
    expected_improvement:   str
    key_insights:           List[str]


# ─── Training ─────────────────────────────────────────────────────────────────

class ModelResultItem(BaseModel):
    model_name:      str
    metrics:         Dict[str, Any]
    hyperparameters: Dict[str, Any]
    fit_time:        float


class TrainingResultsResponse(BaseModel):
    status:          str
    best_model_name: str
    best_score:      float
    primary_metric:  str
    models:          List[ModelResultItem]
    llm_analysis:    Dict[str, Any]
    training_time:   float


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResult(BaseModel):
    prediction:  Any
    confidence:  Optional[float] = None
    label:       Optional[str]   = None
    input_data:  Dict[str, Any]


class PredictionResponse(BaseModel):
    status:  str
    result:  PredictionResult
    history: List[PredictionResult]


# ─── Session ──────────────────────────────────────────────────────────────────

class SessionStatusResponse(BaseModel):
    has_session:     bool
    stage:           str   # "empty" | "uploaded" | "analyzed" | "trained"
    project_name:    Optional[str]   = None
    learning_type:   Optional[str]   = None
    target_column:   Optional[str]   = None
    rows:            Optional[int]   = None
    columns:         Optional[int]   = None
    best_model:      Optional[str]   = None
    prediction_count: int            = 0


class ClearSessionResponse(BaseModel):
    status:  str
    message: str


# ─── Download ─────────────────────────────────────────────────────────────────

class DownloadManifestResponse(BaseModel):
    available_files: List[Dict[str, str]]

import itertools
import json
import logging
import os
import time
from typing import Optional
 
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
 
load_dotenv()
logger = logging.getLogger(__name__)
 
 
# ==============================================================================
# PROVIDER CLIENTS
# ==============================================================================
 
def _init_groq() -> Optional[Groq]:
    api_key = os.getenv("GR_API_KEY")
    if not api_key:
        logger.warning("GR_API_KEY not set — Groq disabled")
        return None
    return Groq(api_key=api_key, max_retries=0)
 
 
def _init_gemini() -> bool:
    api_key = os.getenv("GO_API_KEY")
    if not api_key:
        logger.warning("GO_API_KEY not set — Gemini disabled")
        return False
    genai.configure(api_key=api_key)
    return True
 
 
groq_client  = _init_groq()
gemini_ready = _init_gemini()
 
 
# ==============================================================================
# PROVIDER CALLERS
# ==============================================================================
 
def _call_groq(system_instructions: str, user_input: str) -> dict:
    if not groq_client:
        raise RuntimeError("Groq client not initialised")
 
    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL_NAME", "qwen/qwen3-32b"),
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user",   "content": user_input},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        stream=False,
        timeout=10,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned empty content")
    return json.loads(content)
 
 
def _call_gemini(system_instructions: str, user_input: str) -> dict:
    if not gemini_ready:
        raise RuntimeError("Gemini not initialised")
 
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
        system_instruction=system_instructions,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(user_input)
    content  = response.text
    if not content:
        raise ValueError("Gemini returned empty content")
    return json.loads(content)
 
 
# ==============================================================================
# ROUND-ROBIN PROVIDER REGISTRY
# ==============================================================================
 
_PROVIDERS = [
    ("Groq",   _call_groq),
    ("Gemini", _call_gemini),
]
 
# Infinite round-robin iterator — each chat_llm() call pops the next provider
_provider_cycle = itertools.cycle(_PROVIDERS)
 
 
def _get_next_provider() -> tuple:
    """Returns the next (name, fn) in round-robin order."""
    return next(_provider_cycle)
 
 
# ==============================================================================
# MAIN CALLER — round-robin across providers, fallback on failure
# ==============================================================================
 
def chat_llm(system_instructions: str,
             user_input: str,
             num_tries: int = 3,
             retry_delay: float = 5.0) -> dict:
    """
    Calls LLM providers in round-robin order: Groq → Gemini → Groq → …
    Each successive call to chat_llm() uses a different provider automatically,
    spreading the load evenly across both.
 
    On failure, falls back to the other provider before giving up.
    Returns {} only if every available provider fails all attempts.
    """
    # Start from wherever the round-robin is currently pointing
    primary_name, primary_fn = _get_next_provider()
 
    # Build the attempt order: primary first, then the other as fallback
    all_providers = _PROVIDERS.copy()
    ordered = [(primary_name, primary_fn)] + [
        (name, fn) for name, fn in all_providers if name != primary_name
    ]
 
    logger.info(f"[LLM] Round-robin selected primary provider: {primary_name}")
 
    for provider_name, provider_fn in ordered:
        logger.info(f"[LLM] Trying provider: {provider_name}")
        last_error = None
 
        for attempt in range(1, num_tries + 1):
            try:
                logger.info(f"  [{provider_name}] Attempt {attempt}/{num_tries}…")
                result = provider_fn(system_instructions, user_input)
                logger.info(f"  [{provider_name}] ✓ Success on attempt {attempt}")
                return result
 
            except json.JSONDecodeError as e:
                # Malformed JSON won't improve with retries — skip to next provider
                logger.error(f"  [{provider_name}] Invalid JSON: {e} — switching provider")
                last_error = f"invalid JSON: {e}"
                break
 
            except RuntimeError as e:
                # Provider not configured — skip immediately, no point retrying
                logger.warning(f"  [{provider_name}] Not available: {e} — switching provider")
                last_error = str(e)
                break
 
            except Exception as e:
                err        = str(e).lower()
                last_error = str(e)
 
                if "429" in err or "rate" in err or "quota" in err:
                    logger.warning(f"  [{provider_name}] Attempt {attempt}: rate limited")
                elif "timeout" in err:
                    logger.warning(f"  [{provider_name}] Attempt {attempt}: timeout")
                else:
                    logger.error(f"  [{provider_name}] Attempt {attempt}: {e}")
 
            # Wait before retry (skip on final attempt of this provider)
            if attempt < num_tries:
                logger.info(f"  [{provider_name}] Waiting {retry_delay}s before retry…")
                time.sleep(retry_delay)
 
        logger.warning(
            f"[LLM] {provider_name} exhausted all {num_tries} attempts. "
            f"Last error: {last_error} — trying next provider…"
        )
 
    logger.error("[LLM] All providers failed. Returning empty dict.")
    return {}



# ==============================================================================
# SYSTEM PROMPTS
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
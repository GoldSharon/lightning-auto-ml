# ⚡ Lightning AutoML

> An end-to-end automated machine learning web application — upload a dataset, get insights, train models, and run predictions. No code required.

---

## 🌐 Live Demo

> 🔗 **[Try it on Hugging Face Spaces](<YOUR_HUGGINGFACE_SPACE_URL_HERE>)**

---

## 📸 Overview

Lightning AutoML is a fully automated ML pipeline wrapped in a clean web UI. Upload a CSV or Excel file, configure your task, and the system handles everything: data analysis, LLM-powered preprocessing recommendations, model training, evaluation, and predictions — all through a browser.

---

## ✨ Features

- **Smart Data Analysis** — LLM-powered dtype inference, outlier detection, correlation analysis, and data quality scoring
- **Auto Preprocessing** — Missing value imputation, outlier winsorization, categorical encoding, feature scaling, and feature engineering suggestions — all guided by an LLM
- **Multi-Task Support** — Classification, Regression, and Clustering (supervised & unsupervised)
- **AutoML Training** — Trains up to 7 models sequentially and ranks them by primary metric
- **LLM Model Evaluation** — Post-training analysis with overfitting detection and recommendations
- **Interactive Predictions** — Run single predictions via a form UI after training
- **Downloadable Artifacts** — Processed features, trained model (`.joblib`), metrics CSV, preprocessing report

---

## 🧠 How It Works

```
Upload Dataset
     ↓
Configure Task (target column, task type, test size)
     ↓
Run Analysis (LLM assigns dtypes + recommends preprocessing plan)
     ↓
Train Models (DataPrepAgent preprocesses → AutoMLTrainer trains)
     ↓
View Results (ranked models + LLM insights)
     ↓
Predict (enter feature values → get prediction + confidence)
     ↓
Download (model, features, metrics, report)
```

---

## 🗂️ Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── routes.py            # Pipeline routes (analyze, train, predict, download)
│   │   └── upload_routes.py     # Upload & session routes
│   ├── core/
│   │   ├── config.py            # App settings (Pydantic BaseSettings)
│   │   ├── llm_engine.py        # Round-robin LLM caller (Groq + Gemini)
│   │   └── ml_types.py          # Enums & dataclasses (MLType, LearningType, etc.)
│   ├── services/
│   │   ├── data_analyzer.py     # IntelligentDataAnalyzer — LLM-powered analysis
│   │   ├── data_prep_agent.py   # DataPrepAgent — 7-step preprocessing pipeline
│   │   ├── pipeline_service.py  # Orchestrates analysis + training + prediction
│   │   └── session_service.py   # Single-session state manager (file-backed)
│   ├── static/                  # Frontend (HTML + CSS + JS)
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/app.js
│   ├── util/
│   │   ├── feature_engine.py    # Safe feature engineering (no exec())
│   │   └── file_handler.py
│   └── main.py                  # FastAPI app entry point
├── automl_trainer.py            # AutoMLTrainer — sequential model training
├── ml_pipeline.py               # Standalone pipeline (local testing)
├── Dockerfile
├── requirements.txt
├── uploads/                     # Uploaded dataset files
└── session_data/                # Session state, processed data, saved model
```

---

## 🤖 Supported Models

**Classification**
- Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, XGBoost

**Regression**
- Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost

**Clustering**
- KMeans (3/4/5 clusters), Agglomerative (3/4/5 clusters), DBSCAN

---

## 📊 Preprocessing Pipeline

The `DataPrepAgent` runs in a fixed, safe order:

| Step | Operation |
|------|-----------|
| -1 | Target preprocessing (encode / winsorize / drop nulls) |
| 0 | Drop columns recommended by LLM |
| 1 | Missing value imputation (median / mode) |
| 2 | Feature engineering (ratio, product, sum, difference) |
| 3 | Outlier handling (Winsorizer — **after** engineering) |
| 4 | Categorical encoding (one-hot / frequency) |
| 5 | Feature transformation (log, Yeo-Johnson) |
| 6 | Scaling (Standard / MinMax / Robust) |
| 7 | Feature selection (drop constant & duplicate features) |

---

## 🔌 LLM Integration

The system uses two LLM providers in a **round-robin** pattern with automatic failover:

- **Groq** (`qwen/qwen3-32b` by default) — primary
- **Gemini** (`gemini-1.5-flash` by default) — fallback

LLMs are used for three tasks:
1. **Dtype inference** — assigns correct pandas dtypes per column
2. **Preprocessing recommendation** — generates the full preprocessing config as JSON
3. **Model evaluation** — post-training analysis with overfitting detection

---

## 🚀 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/upload` | Upload dataset file |
| `POST` | `/api/configure` | Set target column, task type, test size |
| `GET` | `/api/session` | Get current session state |
| `DELETE` | `/api/session` | Clear session |
| `POST` | `/api/analyze` | Run LLM data analysis |
| `GET` | `/api/insights` | Fetch structured data insights |
| `GET` | `/api/preprocessing` | Fetch preprocessing plan |
| `POST` | `/api/train` | Run preprocessing + model training |
| `GET` | `/api/results` | Get training results |
| `POST` | `/api/predict` | Run a single prediction |
| `GET` | `/api/predictions` | Get prediction history |
| `GET` | `/api/downloads` | List available download files |
| `GET` | `/api/download/{key}` | Download a specific artifact |

Interactive API docs available at `/docs` (Swagger UI) and `/redoc`.

---

## 🛠️ Local Setup

### Prerequisites

- Python 3.10+
- A Groq API key (get one free at [console.groq.com](https://console.groq.com))
- Optionally a Google Gemini API key

### Installation

```bash
git clone <your-repo-url>
cd lightning-automl

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GR_API_KEY=your_groq_api_key_here
GR_MODEL_NAME=qwen/qwen3-32b

GO_API_KEY=your_gemini_api_key_here      # optional
GEMINI_MODEL_NAME=gemini-1.5-flash       # optional
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🐳 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads session_data

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

Build and run locally:

```bash
docker build -t lightning-automl .
docker run -p 7860:7860 --env-file .env lightning-automl
```

---

## ☁️ Deployment

This app is deployed on **Hugging Face Spaces** with:
- **Hardware:** 2 vCPU, 16 GB RAM (CPU Basic)
- **Runtime:** Docker (FastAPI + Uvicorn)
- **Port:** 7860

All models are trained sequentially (`n_jobs=1`) and capped at 100 estimators to stay within the memory budget.

---

## 📋 Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |

Max upload size: **50 MB**

---

## ⚠️ Limitations

- Single-user session (one active session at a time — designed for HuggingFace Spaces)
- No hyperparameter tuning (sequential training only, tuning flag reserved for future)
- Clustering predictions not supported (clustering is fit-only, no `predict` for new rows)
- Session data is not persisted across Space restarts

---

## 📄 License

MIT License 

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [Feature-engine](https://feature-engine.traindatascience.com/) — preprocessing transformers
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [Groq](https://groq.com/) — LLM inference
- [Google Gemini](https://deepmind.google/technologies/gemini/) — LLM fallback
- [Hugging Face Spaces](https://huggingface.co/spaces) — hosting
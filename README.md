# ⚡ Lightning AutoML

Automated ML Pipeline Builder with LLM-powered data analysis.

## Stack
- **Backend**: FastAPI + Pydantic v2
- **ML**: scikit-learn, XGBoost, feature-engine
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Frontend**: Vanilla HTML/CSS/JS

## Project Structure

```
lightning_automl/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic request/response models
│   │   ├── routes.py          # Pipeline routes (analyze, train, predict, download)
│   │   └── upload_routes.py   # Upload + session routes
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Settings from .env
│   ├── services/
│   │   ├── __init__.py
│   │   ├── session_service.py # Single-session state manager
│   │   └── pipeline_service.py# ML pipeline wrapper
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/app.js
│   │   └── index.html
│   ├── __init__.py
│   └── main.py                # FastAPI app entry point
├── session_data/              # Active session artefacts
├── uploads/                   # Uploaded datasets
├── ml_pipeline.py             # ← Your existing ML pipeline (unchanged)
├── automl_trainer.py          # ← Your existing trainer (unchanged)
├── .env                       # Config (fill in API key)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone / copy your ml_pipeline.py and automl_trainer.py into root
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key in .env
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Open http://localhost:7860

## Pipeline Flow

```
Upload CSV/Excel/JSON
        ↓
  Auto-detect task & target
        ↓
  Configure (target col, task type, test split)
        ↓
  LLM Analysis (dtype assignment + preprocessing plan)
        ↓
  Data Insights (quality score, missing values, correlations)
        ↓
  Preprocessing Plan (review cleaning steps)
        ↓
  Training (DataPrepAgent + AutoMLTrainer — 7 models)
        ↓
  Results (best model, metrics, LLM evaluation)
        ↓
  Predictions (live inference + history)
        ↓
  Downloads (model, features, metrics, report)
```

## Session Management

- One active session at a time
- Stored in `session_data/` directory
- Clear via UI button or `DELETE /api/session`
- Survives server restart (JSON-backed)

## HuggingFace Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## API Docs

Visit `/docs` for interactive Swagger UI after starting the server.

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # LLM
    LLM_API_KEY: str = os.getenv("GR_API_KEY")
    LLM_MODEL_NAME: str = os.getenv("GR_MODEL_NAME")

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 7860
    APP_TITLE: str = "Lightning AutoML"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Storage
    UPLOADS_DIR: str = "uploads"
    SESSION_DATA_DIR: str = "session_data"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = "csv,xlsx,xls,json"

    # ML
    DEFAULT_TEST_SIZE: float = 0.2
    MAX_MODELS_TO_TRAIN: int = 7

    class Config:
        env_file = str(".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def uploads_path(self) -> Path:
        p = BASE_DIR / self.UPLOADS_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def session_data_path(self) -> Path:
        p = BASE_DIR / self.SESSION_DATA_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

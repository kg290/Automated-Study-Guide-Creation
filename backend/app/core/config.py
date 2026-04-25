from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Generative AI Agent for Automated Study Guide Creation"
    api_prefix: str = "/api/v1"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    google_application_credentials: str = ""

    upload_dir: str = Field(default=str(BASE_DIR / "uploads"))
    vector_db_dir: str = Field(default=str(BASE_DIR / "vector_store"))
    sqlite_db_path: str = Field(default=str(BASE_DIR / "study_guides.db"))

    max_upload_size_mb: int = 30
    low_text_char_threshold: int = 350
    chunk_size: int = 1200
    chunk_overlap: int = 180
    retrieval_k: int = 10
    context_max_chars: int = 32000

    cors_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_db_path(self) -> Path:
        path = Path(self.vector_db_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sqlite_path(self) -> Path:
        path = Path(self.sqlite_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

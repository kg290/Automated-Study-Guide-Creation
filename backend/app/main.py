from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.environment import clear_broken_local_proxy_settings
from app.core.logging import configure_logging, logger

configure_logging()
clear_broken_local_proxy_settings()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI agent backend for automated study guide generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Generative AI Agent for Automated Study Guide Creation",
        "status": "running",
    }


@app.on_event("startup")
def startup_event() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)

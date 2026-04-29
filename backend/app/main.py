import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, logger


def _clear_broken_local_proxy_settings() -> None:
    proxy_env_names = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    broken_targets = {"127.0.0.1:9", "localhost:9"}
    cleared: list[str] = []

    for env_name in proxy_env_names:
        value = os.environ.get(env_name, "").strip()
        if not value:
            continue
        normalized = value.lower()
        if any(target in normalized for target in broken_targets):
            os.environ.pop(env_name, None)
            cleared.append(env_name)

    if cleared:
        logger.warning(
            "Cleared broken proxy environment settings: %s",
            ", ".join(cleared),
        )

configure_logging()
_clear_broken_local_proxy_settings()
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

import os

from app.core.logging import logger


def clear_broken_local_proxy_settings() -> None:
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

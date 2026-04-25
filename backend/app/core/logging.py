import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


logger = logging.getLogger("study-guide-agent")

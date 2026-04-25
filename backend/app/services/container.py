from functools import lru_cache

from app.core.config import get_settings
from app.services.generation_service import GeminiGenerationService
from app.services.history_service import HistoryService
from app.services.rag_service import RAGService
from app.services.study_pipeline import StudyGuidePipeline


class ServiceContainer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.rag_service = RAGService(
            vector_root=self.settings.vector_db_path,
            retrieval_k=self.settings.retrieval_k,
        )
        self.history_service = HistoryService(self.settings.sqlite_path)
        self._generation_service: GeminiGenerationService | None = None
        self._pipeline: StudyGuidePipeline | None = None

    def get_generation_service(self) -> GeminiGenerationService:
        if self._generation_service is None:
            self._generation_service = GeminiGenerationService(
                api_key=self.settings.gemini_api_key,
                model_name=self.settings.gemini_model,
            )
        return self._generation_service

    def get_pipeline(self) -> StudyGuidePipeline:
        if self._pipeline is None:
            self._pipeline = StudyGuidePipeline(
                settings=self.settings,
                rag_service=self.rag_service,
                generation_service=self.get_generation_service(),
                history_service=self.history_service,
            )
        return self._pipeline


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "completed", "failed"]
BloomLevel = Literal["remember", "understand", "apply", "analyze"]


class Flashcard(BaseModel):
    question: str
    answer: str
    bloom_level: BloomLevel = "remember"


class QAItem(BaseModel):
    question: str
    answer: str
    bloom_level: BloomLevel = "remember"


class TopicNote(BaseModel):
    topic: str
    notes: str


class UnitSummary(BaseModel):
    unit_title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)


class DifficultyExplanation(BaseModel):
    mode: Literal["beginner", "intermediate", "advanced"]
    explanation: str


class StudyGuideResult(BaseModel):
    summary_notes: str = ""
    unit_wise_summaries: list[UnitSummary] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    topic_wise_notes: list[TopicNote] = Field(default_factory=list)
    flashcards: list[Flashcard] = Field(default_factory=list)
    qa_sets: list[QAItem] = Field(default_factory=list)
    viva_questions: list[str] = Field(default_factory=list)
    difficulty_explanations: list[DifficultyExplanation] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class GenerationOptions(BaseModel):
    output_types: list[str] = Field(
        default_factory=lambda: [
            "summary",
            "unit_summaries",
            "key_concepts",
            "topic_notes",
            "flashcards",
            "qa",
            "viva",
            "difficulty",
        ]
    )
    difficulty_modes: list[str] = Field(
        default_factory=lambda: ["beginner", "intermediate", "advanced"]
    )
    bloom_levels: list[BloomLevel] = Field(
        default_factory=lambda: ["remember", "understand", "apply", "analyze"]
    )
    custom_prompt: str = ""


class JobCreateResponse(BaseModel):
    job_id: str
    session_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: JobStatus
    progress: int
    stage: str
    message: str = ""
    error: str | None = None
    result: StudyGuideResult | None = None
    created_at: datetime
    updated_at: datetime


class HistoryItem(BaseModel):
    session_id: str
    job_id: str
    file_names: list[str]
    created_at: datetime


class HistoryDetail(BaseModel):
    session_id: str
    job_id: str
    file_names: list[str]
    created_at: datetime
    options: GenerationOptions
    result: StudyGuideResult


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]

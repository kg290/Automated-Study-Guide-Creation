from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from app.models.schemas import GenerationOptions, JobCreateResponse, JobStatusResponse
from app.services import jobs
from app.services.container import ServiceContainer, get_container
from app.services.file_service import save_upload_file, validate_upload_file

router = APIRouter(prefix="/study-guides", tags=["study-guides"])


class RegenerateRequest(BaseModel):
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
    bloom_levels: list[str] = Field(
        default_factory=lambda: ["remember", "understand", "apply", "analyze"]
    )
    custom_prompt: str = ""


def _parse_list_field(value: str, fallback: list[str]) -> list[str]:
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    return parsed or fallback


def _build_queries(options: GenerationOptions) -> list[str]:
    intents: list[str] = []

    if "summary" in options.output_types:
        intents.append("summary and overview")
    if "unit_summaries" in options.output_types:
        intents.append("unit-wise chapter summaries that cover the full document sequence")
    if "key_concepts" in options.output_types:
        intents.append("key concepts and definitions")
    if "topic_notes" in options.output_types:
        intents.append("topic-wise notes")
    if "flashcards" in options.output_types:
        intents.append("flashcard style revision points")
    if "qa" in options.output_types:
        intents.append("question and answer preparation")
    if "viva" in options.output_types:
        intents.append("viva and oral exam prompts")
    if "difficulty" in options.output_types:
        intents.append("beginner intermediate advanced explanations")
    if "flashcards" in options.output_types or "qa" in options.output_types:
        intents.append("Bloom level framing using " + ", ".join(options.bloom_levels))

    queries = [
        (
            "Retrieve text from uploaded session documents relevant to: "
            + "; ".join(intents)
        ).strip()
    ]

    if options.custom_prompt.strip():
        queries.append(options.custom_prompt.strip())
    queries.append("Definitions, formulas, examples, and exam-relevant passages")

    return queries or ["Comprehensive understanding of uploaded material"]


@router.post("/generate", response_model=JobCreateResponse)
async def generate_study_guides(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    output_types: str = Form(
        default="summary,unit_summaries,key_concepts,topic_notes,flashcards,qa,viva,difficulty"
    ),
    difficulty_modes: str = Form(default="beginner,intermediate,advanced"),
    bloom_levels: str = Form(default="remember,understand,apply,analyze"),
    custom_prompt: str = Form(default=""),
    container: ServiceContainer = Depends(get_container),
) -> JobCreateResponse:
    if not container.settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is missing in backend environment.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    options = GenerationOptions(
        output_types=_parse_list_field(
            output_types,
            fallback=[
                "summary",
                "unit_summaries",
                "key_concepts",
                "topic_notes",
                "flashcards",
                "qa",
                "viva",
                "difficulty",
            ],
        ),
        difficulty_modes=_parse_list_field(
            difficulty_modes,
            fallback=["beginner", "intermediate", "advanced"],
        ),
        bloom_levels=_parse_list_field(
            bloom_levels,
            fallback=["remember", "understand", "apply", "analyze"],
        ),
        custom_prompt=custom_prompt.strip(),
    )

    session_id = uuid4().hex
    job_id = uuid4().hex

    session_directory = container.settings.upload_path / session_id
    saved_paths = []

    for upload_file in files:
        validate_upload_file(upload_file, container.settings.max_upload_size_mb)
        saved_path = await save_upload_file(upload_file, session_directory)
        saved_paths.append(saved_path)

    jobs.create_job(job_id=job_id, session_id=session_id)

    try:
        pipeline = container.get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(
        pipeline.run_job,
        job_id,
        session_id,
        saved_paths,
        options,
    )

    return JobCreateResponse(job_id=job_id, session_id=session_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    return job


def _run_regeneration_job(
    job_id: str,
    session_id: str,
    options: GenerationOptions,
    container: ServiceContainer,
) -> None:
    try:
        history_item = container.history_service.get_session(session_id)
        source_documents = history_item.file_names if history_item else []

        session_directory = container.settings.upload_path / session_id
        saved_file_paths = [
            session_directory / file_name
            for file_name in source_documents
            if (session_directory / file_name).exists()
        ]

        if not saved_file_paths:
            raise ValueError("No uploaded files found for this session.")

        pipeline = container.get_pipeline()
        pipeline.run_job(job_id, session_id, saved_file_paths, options)
    except Exception as exc:  # pragma: no cover
        jobs.fail_job(job_id, str(exc))


@router.post("/sessions/{session_id}/regenerate", response_model=JobCreateResponse)
def regenerate_outputs(
    session_id: str,
    payload: RegenerateRequest,
    background_tasks: BackgroundTasks,
    container: ServiceContainer = Depends(get_container),
) -> JobCreateResponse:
    if not container.settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is missing in backend environment.",
        )

    history_session = container.history_service.get_session(session_id)
    if history_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History session not found.",
        )

    options = GenerationOptions(
        output_types=[item.strip().lower() for item in payload.output_types if item.strip()],
        difficulty_modes=[item.strip().lower() for item in payload.difficulty_modes if item.strip()],
        bloom_levels=[item.strip().lower() for item in payload.bloom_levels if item.strip()],
        custom_prompt=payload.custom_prompt,
    )

    job_id = uuid4().hex
    jobs.create_job(job_id=job_id, session_id=session_id)

    background_tasks.add_task(
        _run_regeneration_job,
        job_id,
        session_id,
        options,
        container,
    )

    return JobCreateResponse(job_id=job_id, session_id=session_id, status="queued")

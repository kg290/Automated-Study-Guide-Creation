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
        jobs.update_job(
            job_id,
            status="processing",
            progress=50,
            stage="retrieving",
            message="Retrieving existing vector context.",
        )

        history_item = container.history_service.get_session(session_id)
        source_documents = history_item.file_names if history_item else []

        retrieval_k = container.settings.retrieval_k
        if "unit_summaries" in options.output_types:
            retrieval_k = max(retrieval_k, 10)

        context = container.rag_service.get_context_bundle(
            session_id=session_id,
            queries=_build_queries(options),
            k=retrieval_k,
            max_chars=container.settings.context_max_chars,
            allowed_sources=set(source_documents),
        )
        if not context.strip():
            raise ValueError("No context found in the stored vector index for this session.")

        jobs.update_job(
            job_id,
            progress=80,
            stage="generating",
            message="Regenerating study material with Gemini.",
        )

        result = container.get_generation_service().generate_from_context(
            context=context,
            options=options,
            source_documents=source_documents,
        )

        container.history_service.save_session(
            session_id=session_id,
            job_id=job_id,
            file_names=source_documents,
            options=options,
            result=result,
        )

        jobs.complete_job(job_id, result)
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

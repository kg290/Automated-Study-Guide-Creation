from copy import deepcopy
from datetime import datetime
from threading import Lock

from app.models.schemas import JobStatusResponse, StudyGuideResult

_jobs: dict[str, JobStatusResponse] = {}
_lock = Lock()


def create_job(job_id: str, session_id: str) -> JobStatusResponse:
    now = datetime.utcnow()
    job = JobStatusResponse(
        job_id=job_id,
        session_id=session_id,
        status="queued",
        progress=0,
        stage="queued",
        message="Job created.",
        created_at=now,
        updated_at=now,
        result=None,
    )
    with _lock:
        _jobs[job_id] = job
    return deepcopy(job)


def get_job(job_id: str) -> JobStatusResponse | None:
    with _lock:
        job = _jobs.get(job_id)
        return deepcopy(job) if job else None


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    stage: str | None = None,
    message: str | None = None,
) -> JobStatusResponse | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None

        if status is not None:
            job.status = status  # type: ignore[assignment]
        if progress is not None:
            job.progress = max(0, min(100, progress))
        if stage is not None:
            job.stage = stage
        if message is not None:
            job.message = message
        job.updated_at = datetime.utcnow()
        _jobs[job_id] = job
        return deepcopy(job)


def complete_job(job_id: str, result: StudyGuideResult) -> JobStatusResponse | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None

        job.status = "completed"
        job.progress = 100
        job.stage = "completed"
        job.message = "Study guide generation finished."
        job.error = None
        job.result = result
        job.updated_at = datetime.utcnow()
        _jobs[job_id] = job
        return deepcopy(job)


def fail_job(job_id: str, error: str) -> JobStatusResponse | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None

        job.status = "failed"
        job.stage = "failed"
        job.message = "Generation failed."
        job.error = error
        job.updated_at = datetime.utcnow()
        _jobs[job_id] = job
        return deepcopy(job)

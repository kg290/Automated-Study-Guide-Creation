from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def validate_upload_file(upload_file: UploadFile, max_size_mb: int) -> None:
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One of the uploaded files has no filename.",
        )

    extension = Path(upload_file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension}. Allowed: PDF, DOCX.",
        )

    upload_file.file.seek(0, 2)
    size_bytes = upload_file.file.tell()
    upload_file.file.seek(0)

    max_size_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File {upload_file.filename} exceeds {max_size_mb}MB.",
        )


async def save_upload_file(upload_file: UploadFile, session_directory: Path) -> Path:
    session_directory.mkdir(parents=True, exist_ok=True)

    original_name = Path(upload_file.filename or "upload.bin")
    safe_stem = "".join(ch for ch in original_name.stem if ch.isalnum() or ch in {"-", "_"})
    safe_stem = safe_stem[:70] if safe_stem else f"file-{uuid4().hex[:8]}"

    file_name = f"{safe_stem}{original_name.suffix.lower()}"
    destination = session_directory / file_name

    content = await upload_file.read()
    destination.write_bytes(content)
    await upload_file.close()

    return destination

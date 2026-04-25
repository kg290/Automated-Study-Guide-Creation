from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.schemas import HistoryDetail, HistoryListResponse
from app.services.container import ServiceContainer, get_container

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryListResponse)
def list_history(
    limit: int = Query(default=30, ge=1, le=200),
    container: ServiceContainer = Depends(get_container),
) -> HistoryListResponse:
    items = container.history_service.list_sessions(limit=limit)
    return HistoryListResponse(items=items)


@router.get("/{session_id}", response_model=HistoryDetail)
def get_history_detail(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
) -> HistoryDetail:
    session = container.history_service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History session not found.",
        )
    return session

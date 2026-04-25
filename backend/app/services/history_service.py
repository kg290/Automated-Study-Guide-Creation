import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.models.schemas import (
    GenerationOptions,
    HistoryDetail,
    HistoryItem,
    StudyGuideResult,
)


class HistoryService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_names TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_session(
        self,
        session_id: str,
        job_id: str,
        file_names: list[str],
        options: GenerationOptions,
        result: StudyGuideResult,
    ) -> None:
        created_at = datetime.utcnow().isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id,
                    job_id,
                    created_at,
                    file_names,
                    options_json,
                    result_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    job_id,
                    created_at,
                    json.dumps(file_names),
                    options.model_dump_json(),
                    result.model_dump_json(),
                ),
            )
            connection.commit()

    def list_sessions(self, limit: int = 50) -> list[HistoryItem]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, job_id, created_at, file_names
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items: list[HistoryItem] = []
        for row in rows:
            items.append(
                HistoryItem(
                    session_id=row["session_id"],
                    job_id=row["job_id"],
                    file_names=json.loads(row["file_names"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )

        return items

    def get_session(self, session_id: str) -> HistoryDetail | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, job_id, created_at, file_names, options_json, result_json
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        options_data = json.loads(row["options_json"])
        result_data = json.loads(row["result_json"])

        return HistoryDetail(
            session_id=row["session_id"],
            job_id=row["job_id"],
            file_names=json.loads(row["file_names"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            options=GenerationOptions(**options_data),
            result=StudyGuideResult(**result_data),
        )

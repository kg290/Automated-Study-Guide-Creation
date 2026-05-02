import shutil
import re
import os
from pathlib import Path
from typing import Any
from collections import defaultdict

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class RAGService:
    def __init__(self, vector_root: Path, retrieval_k: int) -> None:
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        os.environ.setdefault("CHROMA_TELEMETRY", "False")
        self.vector_root = vector_root
        self.retrieval_k = retrieval_k
        self.vector_root.mkdir(parents=True, exist_ok=True)
        self.chroma_settings = ChromaSettings(anonymized_telemetry=False)
        self._embeddings: HuggingFaceEmbeddings | None = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def _collection_directory(self, session_id: str) -> Path:
        return self.vector_root / session_id

    def build_vector_store(
        self,
        session_id: str,
        chunks: list[str],
        metadata: list[dict],
    ) -> None:
        if not chunks:
            raise ValueError("No text chunks available for vector store creation.")

        persist_directory = self._collection_directory(session_id)
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self._get_embeddings(),
            metadatas=metadata,
            persist_directory=str(persist_directory),
            collection_name="study_material",
            client_settings=self.chroma_settings,
        )

        if hasattr(vector_store, "persist"):
            vector_store.persist()

    def load_vector_store(self, session_id: str) -> Chroma:
        persist_directory = self._collection_directory(session_id)
        if not persist_directory.exists():
            raise FileNotFoundError("Vector store not found for this session.")

        return Chroma(
            embedding_function=self._get_embeddings(),
            persist_directory=str(persist_directory),
            collection_name="study_material",
            client_settings=self.chroma_settings,
        )

    def retrieve(
        self,
        session_id: str,
        query: str,
        k: int | None = None,
        *,
        vector_store: Chroma | None = None,
    ) -> list[Document]:
        active_store = vector_store if vector_store is not None else self.load_vector_store(session_id)
        return active_store.similarity_search(query, k=k or self.retrieval_k)

    def _normalize_queries(self, queries: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for query in queries:
            cleaned = " ".join(query.split()).strip()
            if not cleaned:
                continue

            key = cleaned.lower()
            if key in seen:
                continue

            seen.add(key)
            normalized.append(cleaned)

        return normalized

    def _search_with_scores(
        self,
        vector_store: Chroma,
        query: str,
        k: int,
    ) -> list[tuple[Document, float | None]]:
        search_with_distances = getattr(vector_store, "similarity_search_with_score", None)
        if callable(search_with_distances):
            try:
                raw_results = search_with_distances(query, k=k)
                return [
                    (
                        document,
                        (
                            max(0.0, min(1.0, 1.0 / (1.0 + max(float(distance), 0.0))))
                            if isinstance(distance, (float, int))
                            else None
                        ),
                    )
                    for document, distance in raw_results
                ]
            except Exception:
                pass

        search_with_scores = getattr(vector_store, "similarity_search_with_relevance_scores", None)
        if callable(search_with_scores):
            try:
                raw_results = search_with_scores(query, k=k)
                normalized: list[tuple[Document, float | None]] = []
                for document, score in raw_results:
                    if isinstance(score, (float, int)):
                        normalized.append((document, max(0.0, min(1.0, float(score)))))
                    else:
                        normalized.append((document, None))
                return normalized
            except Exception:
                pass

        documents = vector_store.similarity_search(query, k=k)
        return [(document, None) for document in documents]

    def _build_context_block(self, document: Document) -> str:
        content = document.page_content.strip()
        if not content:
            return ""

        metadata: dict[str, Any] = document.metadata or {}
        source = str(metadata.get("source", "uploaded_document")).strip() or "uploaded_document"
        chunk_index = metadata.get("chunk_index", "na")
        chunk_label = str(chunk_index).strip() or "na"

        return f"[source:{source} | chunk:{chunk_label}]\n{content}"

    def _is_informative_content(self, content: str) -> bool:
        compact = " ".join(content.split()).strip()
        if not compact:
            return False

        if re.search(r"(?:\b[a-zA-Z]\b\s+){6,}", compact):
            return False

        tokens = re.findall(r"[A-Za-z0-9\-]+", compact)
        if len(tokens) < 10:
            return False

        alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
        if not alpha_tokens:
            return False

        if (len(alpha_tokens) / len(tokens)) < 0.68:
            return False

        return True

    def _parse_chunk_index(self, raw_value: Any) -> int | None:
        try:
            return int(str(raw_value).strip())
        except (TypeError, ValueError):
            return None

    def _entry_rank_tuple(self, entry: dict[str, Any]) -> tuple[int, float, int]:
        return (
            int(entry.get("hits", 0)),
            float(entry.get("score", -1.0)),
            -int(entry.get("first_seen", 0)),
        )

    def _apply_diversity_guard(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(entries) <= 2:
            return entries

        with_chunk_index = [entry for entry in entries if entry.get("chunk_index") is not None]
        without_chunk_index = [entry for entry in entries if entry.get("chunk_index") is None]
        if len(with_chunk_index) <= 2:
            return entries

        max_chunk_index = max(int(entry["chunk_index"]) for entry in with_chunk_index)
        target_windows = min(10, max(4, len(with_chunk_index) // 3))
        window_size = max(1, (max_chunk_index + target_windows) // target_windows)

        windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for entry in with_chunk_index:
            window_id = int(entry["chunk_index"]) // window_size
            windows[window_id].append(entry)

        for window_entries in windows.values():
            window_entries.sort(key=self._entry_rank_tuple, reverse=True)

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        window_counts: dict[int, int] = defaultdict(int)
        window_cap = 2 if len(windows) >= 4 else 3

        # Seed with top entry from each window to enforce coverage across document sections.
        for window_id in sorted(windows):
            best_entry = windows[window_id][0]
            entry_id = str(best_entry["dedupe_key"])
            selected.append(best_entry)
            selected_ids.add(entry_id)
            window_counts[window_id] += 1

        ranked_candidates = sorted(with_chunk_index, key=self._entry_rank_tuple, reverse=True)
        for entry in ranked_candidates:
            entry_id = str(entry["dedupe_key"])
            if entry_id in selected_ids:
                continue

            window_id = int(entry["chunk_index"]) // window_size
            if window_counts[window_id] >= window_cap:
                continue

            selected.append(entry)
            selected_ids.add(entry_id)
            window_counts[window_id] += 1

        for entry in without_chunk_index:
            entry_id = str(entry["dedupe_key"])
            if entry_id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry_id)

        return selected

    def get_context_bundle(
        self,
        session_id: str,
        queries: list[str],
        k: int | None = None,
        max_chars: int = 18000,
        allowed_sources: set[str] | None = None,
    ) -> str:
        vector_store = self.load_vector_store(session_id)
        normalized_queries = self._normalize_queries(queries)
        if not normalized_queries:
            normalized_queries = ["Comprehensive understanding of uploaded material"]

        result_k = k or self.retrieval_k
        source_whitelist = (
            {source.strip() for source in allowed_sources if source and source.strip()}
            if allowed_sources
            else None
        )

        deduped: dict[str, dict[str, Any]] = {}
        sequence = 0

        for query in normalized_queries:
            for document, score in self._search_with_scores(vector_store, query, result_k):
                content = document.page_content.strip()
                if not content:
                    continue
                if not self._is_informative_content(content):
                    continue

                metadata = document.metadata or {}
                source = str(metadata.get("source", "")).strip()
                if source_whitelist is not None and source not in source_whitelist:
                    continue

                chunk_index = str(metadata.get("chunk_index", "na")).strip() or "na"
                parsed_chunk_index = self._parse_chunk_index(chunk_index)

                block = self._build_context_block(document)
                if not block:
                    continue

                numeric_score = score if score is not None else -1.0
                dedupe_key = f"{source}::{chunk_index}" if source else content
                existing = deduped.get(dedupe_key)
                if existing is None:
                    deduped[dedupe_key] = {
                        "dedupe_key": dedupe_key,
                        "score": numeric_score,
                        "first_seen": sequence,
                        "hits": 1,
                        "source": source,
                        "chunk_index": parsed_chunk_index,
                        "block": block,
                    }
                else:
                    existing["hits"] = int(existing["hits"]) + 1
                    if numeric_score > float(existing["score"]):
                        existing["score"] = numeric_score
                        existing["block"] = block
                sequence += 1

        ranked_entries = sorted(
            deduped.values(),
            key=lambda entry: (
                int(entry["hits"]),
                float(entry["score"]),
                -int(entry["first_seen"]),
            ),
            reverse=True,
        )

        grouped_by_source: dict[str, list[dict[str, Any]]] = {}
        source_priority: dict[str, float] = {}
        for entry in ranked_entries:
            source = str(entry.get("source", "")).strip() or "uploaded_document"
            grouped_by_source.setdefault(source, []).append(entry)
            source_priority[source] = max(source_priority.get(source, -1.0), float(entry["score"]))

        for source, source_entries in grouped_by_source.items():
            source_entries.sort(key=self._entry_rank_tuple, reverse=True)
            grouped_by_source[source] = self._apply_diversity_guard(source_entries)

        ordered_sources = [
            source
            for source, _ in sorted(
                source_priority.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        ranked_blocks: list[str] = []
        while True:
            appended = False
            for source in ordered_sources:
                source_blocks = grouped_by_source.get(source, [])
                if not source_blocks:
                    continue
                ranked_blocks.append(str(source_blocks.pop(0)["block"]))
                appended = True
            if not appended:
                break

        selected: list[str] = []
        consumed_chars = 0

        for block in ranked_blocks:
            separator_chars = 2 if selected else 0
            projected = consumed_chars + separator_chars + len(block)
            if projected > max_chars:
                if not selected:
                    selected.append(block[:max_chars])
                break

            selected.append(block)
            consumed_chars = projected

        return "\n\n".join(selected).strip()

from pathlib import Path

from app.core.config import Settings
from app.core.logging import logger
from app.models.schemas import GenerationOptions
from app.services import jobs
from app.services.extraction_service import extract_file_text
from app.services.generation_service import GeminiGenerationService
from app.services.history_service import HistoryService
from app.services.ocr_service import VisionOCRService
from app.services.processing_service import clean_extracted_text, split_into_chunks
from app.services.rag_service import RAGService


class StudyGuidePipeline:
    def __init__(
        self,
        settings: Settings,
        rag_service: RAGService,
        generation_service: GeminiGenerationService,
        history_service: HistoryService,
    ) -> None:
        self.settings = settings
        self.rag_service = rag_service
        self.generation_service = generation_service
        self.history_service = history_service

    def run_job(
        self,
        job_id: str,
        session_id: str,
        file_paths: list[Path],
        options: GenerationOptions,
    ) -> None:
        try:
            jobs.update_job(
                job_id,
                status="processing",
                progress=10,
                stage="extracting",
                message="Extracting text from uploaded documents.",
            )

            ocr_service = self._build_ocr_service()

            all_chunks: list[str] = []
            all_metadata: list[dict] = []
            source_documents: list[str] = []

            for file_path in file_paths:
                extracted = extract_file_text(
                    file_path=file_path,
                    low_text_char_threshold=self.settings.low_text_char_threshold,
                    ocr_service=ocr_service,
                )

                cleaned_text = clean_extracted_text(extracted.text)
                if not cleaned_text:
                    logger.warning("No text detected in file %s", extracted.file_name)
                    continue

                chunks = split_into_chunks(
                    cleaned_text,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )

                source_documents.append(extracted.file_name)

                for index, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append(
                        {
                            "source": extracted.file_name,
                            "chunk_index": index,
                            "used_ocr": extracted.used_ocr,
                        }
                    )

            if not all_chunks:
                raise ValueError(
                    "No readable text could be extracted from the uploaded files."
                )

            jobs.update_job(
                job_id,
                progress=50,
                stage="embedding",
                message="Creating embeddings and storing vectors in ChromaDB.",
            )

            self.rag_service.build_vector_store(
                session_id=session_id,
                chunks=all_chunks,
                metadata=all_metadata,
            )

            jobs.update_job(
                job_id,
                progress=70,
                stage="retrieving",
                message="Retrieving relevant content for generation.",
            )

            retrieval_queries = self._build_retrieval_queries(options, source_documents)
            retrieval_k = self.settings.retrieval_k
            if "unit_summaries" in options.output_types:
                retrieval_k = max(retrieval_k, 10)
            context_bundle = self.rag_service.get_context_bundle(
                session_id=session_id,
                queries=retrieval_queries,
                k=retrieval_k,
                max_chars=self.settings.context_max_chars,
                allowed_sources=set(source_documents),
            )

            coverage_context = self._build_document_coverage_context(
                chunks=all_chunks,
                metadata=all_metadata,
                max_chars=max(6000, self.settings.context_max_chars // 2),
            )
            context_bundle = self._merge_context_bundles(
                primary_context=context_bundle,
                secondary_context=coverage_context,
                max_chars=self.settings.context_max_chars,
            )

            if not context_bundle.strip():
                context_bundle = self._build_context_from_chunks(
                    chunks=all_chunks,
                    metadata=all_metadata,
                    max_chars=self.settings.context_max_chars,
                )

            jobs.update_job(
                job_id,
                progress=85,
                stage="generating",
                message="Generating structured study material with Gemini.",
            )

            result = self.generation_service.generate_from_context(
                context=context_bundle,
                options=options,
                source_documents=source_documents,
            )

            self.history_service.save_session(
                session_id=session_id,
                job_id=job_id,
                file_names=source_documents,
                options=options,
                result=result,
            )

            jobs.complete_job(job_id=job_id, result=result)
        except Exception as exc:  # pragma: no cover
            logger.exception("Pipeline execution failed for job %s", job_id)
            jobs.fail_job(job_id, error=str(exc))

    def _build_ocr_service(self) -> VisionOCRService | None:
        credentials = self.settings.google_application_credentials.strip()
        if not credentials:
            return None

        try:
            return VisionOCRService(credentials_path=credentials)
        except Exception as exc:
            logger.warning("OCR initialization failed: %s", exc)
            return None

    def _build_retrieval_queries(
        self,
        options: GenerationOptions,
        source_documents: list[str],
    ) -> list[str]:
        topical_intents: list[str] = []

        if "summary" in options.output_types:
            topical_intents.append("high level summary and major ideas")
        if "unit_summaries" in options.output_types:
            topical_intents.append(
                "unit-wise and chapter-wise coverage from beginning to end"
            )
        if "key_concepts" in options.output_types:
            topical_intents.append(
                "important concepts, terms, definitions, and principles"
            )
        if "topic_notes" in options.output_types:
            topical_intents.append("topic hierarchy with subtopics and explanations")
        if "flashcards" in options.output_types:
            topical_intents.append(
                "short factual recall points suitable for flashcards"
            )
        if "qa" in options.output_types:
            topical_intents.append("exam-focused question and answer material")
        if "viva" in options.output_types:
            topical_intents.append("potential viva and oral discussion questions")
        if "difficulty" in options.output_types:
            topical_intents.append(
                "material that can be explained at beginner intermediate advanced levels"
            )
        if "flashcards" in options.output_types or "qa" in options.output_types:
            topical_intents.append(
                "content suitable for Bloom levels: " + ", ".join(options.bloom_levels)
            )

        source_hint = ", ".join(source_documents)

        # First prioritize broad comprehension of the full document before any output-specific shaping.
        document_understanding_queries = [
            (
                "Build a complete understanding of the uploaded files "
                f"({source_hint}) from beginning to end, preserving chapter/unit sequence, "
                "major themes, definitions, formulas, examples, and conclusions."
            ),
            (
                "Retrieve representative passages across early, middle, and late parts of the uploaded material "
                f"({source_hint}) so the overall document meaning is captured, not just isolated snippets."
            ),
            (
                "Identify what the document is primarily about, key subject areas, and supporting evidence "
                f"from uploaded files ({source_hint})."
            ),
        ]

        output_focused_query = (
            "Retrieve only text grounded in the uploaded files "
            f"({source_hint}) covering: {'; '.join(topical_intents)}."
            if topical_intents
            else f"Retrieve the most important text from uploaded files ({source_hint})."
        )

        queries = [*document_understanding_queries, output_focused_query]

        if options.custom_prompt.strip():
            queries.append(options.custom_prompt.strip())

        queries.append(
            "Definitions, formulas, examples, and exam-relevant insights from uploaded files"
        )

        return queries

    def _build_context_from_chunks(
        self,
        chunks: list[str],
        metadata: list[dict],
        max_chars: int,
    ) -> str:
        if not chunks:
            return ""

        context_blocks: list[str] = []
        consumed_chars = 0

        for chunk, chunk_metadata in zip(chunks, metadata):
            content = chunk.strip()
            if not content:
                continue

            source = (
                str(chunk_metadata.get("source", "uploaded_document")).strip()
                or "uploaded_document"
            )
            chunk_index = str(chunk_metadata.get("chunk_index", "na")).strip() or "na"
            block = f"[source:{source} | chunk:{chunk_index}]\n{content}"

            separator_chars = 2 if context_blocks else 0
            projected = consumed_chars + separator_chars + len(block)
            if projected > max_chars:
                if not context_blocks:
                    context_blocks.append(block[:max_chars])
                break

            context_blocks.append(block)
            consumed_chars = projected

        return "\n\n".join(context_blocks).strip()

    def _build_document_coverage_context(
        self,
        chunks: list[str],
        metadata: list[dict],
        max_chars: int,
    ) -> str:
        if not chunks:
            return ""

        by_source: dict[str, list[tuple[int, str]]] = {}
        for chunk, chunk_metadata in zip(chunks, metadata):
            source = (
                str(chunk_metadata.get("source", "uploaded_document")).strip()
                or "uploaded_document"
            )
            chunk_index_raw = chunk_metadata.get("chunk_index", 0)
            try:
                chunk_index = int(chunk_index_raw)
            except (TypeError, ValueError):
                chunk_index = 0
            content = chunk.strip()
            if not content:
                continue
            by_source.setdefault(source, []).append((chunk_index, content))

        selected_blocks: list[str] = []
        consumed_chars = 0

        for source, source_chunks in by_source.items():
            source_chunks.sort(key=lambda item: item[0])
            target_count = min(12, max(4, len(source_chunks) // 5))
            if len(source_chunks) <= target_count:
                sampled = source_chunks
            else:
                step = (len(source_chunks) - 1) / (target_count - 1)
                sampled_indices = sorted(
                    {int(round(step * idx)) for idx in range(target_count)}
                )
                sampled = [source_chunks[idx] for idx in sampled_indices]

            for chunk_index, content in sampled:
                block = f"[source:{source} | chunk:{chunk_index}]\n{content}"
                separator_chars = 2 if selected_blocks else 0
                projected = consumed_chars + separator_chars + len(block)
                if projected > max_chars:
                    if not selected_blocks:
                        selected_blocks.append(block[:max_chars])
                    break
                selected_blocks.append(block)
                consumed_chars = projected

        return "\n\n".join(selected_blocks).strip()

    def _merge_context_bundles(
        self,
        primary_context: str,
        secondary_context: str,
        max_chars: int,
    ) -> str:
        primary_blocks = [
            block.strip() for block in primary_context.split("\n\n") if block.strip()
        ]
        secondary_blocks = [
            block.strip() for block in secondary_context.split("\n\n") if block.strip()
        ]
        merged_blocks: list[str] = []
        seen: set[str] = set()
        consumed_chars = 0

        for block in [*primary_blocks, *secondary_blocks]:
            if block in seen:
                continue
            seen.add(block)
            separator_chars = 2 if merged_blocks else 0
            projected = consumed_chars + separator_chars + len(block)
            if projected > max_chars:
                if not merged_blocks:
                    merged_blocks.append(block[:max_chars])
                break
            merged_blocks.append(block)
            consumed_chars = projected

        return "\n\n".join(merged_blocks).strip()

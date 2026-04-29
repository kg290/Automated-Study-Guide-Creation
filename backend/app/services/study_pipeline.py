from pathlib import Path
import re

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
        rag_service: RAGService | None,
        generation_service: GeminiGenerationService,
        history_service: HistoryService,
    ) -> None:
        self.settings = settings
        self.rag_service = rag_service
        self.generation_service = generation_service
        self.history_service = history_service

    def collect_source_material(
        self,
        file_paths: list[Path],
    ) -> tuple[list[str], list[dict], list[str]]:
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
            if not chunks:
                logger.warning("No usable chunks produced for file %s", extracted.file_name)
                continue

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
            raise ValueError("No readable text could be extracted from the uploaded files.")

        return all_chunks, all_metadata, source_documents

    def build_generation_context(
        self,
        session_id: str,
        all_chunks: list[str],
        all_metadata: list[dict],
        options: GenerationOptions,
        source_documents: list[str],
    ) -> str:
        generation_context_budget = min(self.settings.context_max_chars, 12000)
        outline_budget = min(1800, max(900, generation_context_budget // 7))
        evidence_budget = max(3500, generation_context_budget - outline_budget)
        coverage_context_budget = min(max(4200, evidence_budget // 2), 6500)

        coverage_context = self._build_document_coverage_context(
            chunks=all_chunks,
            metadata=all_metadata,
            max_chars=coverage_context_budget,
        )
        outline_guidance = self._build_document_outline_guidance(
            chunks=all_chunks,
            metadata=all_metadata,
            max_chars=outline_budget,
        )
        compact_full_context = self._build_context_from_chunks(
            chunks=all_chunks,
            metadata=all_metadata,
            max_chars=evidence_budget,
        )
        focus_context = self._build_focus_context(
            chunks=all_chunks,
            metadata=all_metadata,
            focus_text=options.custom_prompt,
            max_chars=min(2600, max(1200, evidence_budget // 3)),
        )

        # Small documents fit comfortably in context. Avoid embedding/retrieval work and
        # give Gemini the whole cleaned document, which also makes custom focus prompts
        # much more reliable for exact table/section references.
        if len(all_chunks) <= max(8, self.settings.retrieval_k + 2):
            small_doc_evidence = self._merge_multiple_context_bundles(
                [focus_context, compact_full_context, coverage_context],
                max_chars=evidence_budget,
            )
            return self._combine_context_sections(
                [outline_guidance, small_doc_evidence],
                max_chars=generation_context_budget,
            )

        retrieval_context = ""
        if self.rag_service is not None:
            try:
                self.rag_service.build_vector_store(
                    session_id=session_id,
                    chunks=all_chunks,
                    metadata=all_metadata,
                )

                retrieval_queries = self._build_retrieval_queries(options, source_documents)
                retrieval_k = self.settings.retrieval_k
                if "unit_summaries" in options.output_types:
                    retrieval_k = max(retrieval_k, 10)

                retrieval_context = self.rag_service.get_context_bundle(
                    session_id=session_id,
                    queries=retrieval_queries,
                    k=retrieval_k,
                    max_chars=max(3500, generation_context_budget // 2),
                    allowed_sources=set(source_documents),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "RAG context unavailable for session %s, falling back to chunk coverage: %s",
                    session_id,
                    exc,
                )

        evidence_context = self._merge_multiple_context_bundles(
            [
                focus_context,
                retrieval_context,
                coverage_context,
                compact_full_context,
            ],
            max_chars=evidence_budget,
        )
        context_bundle = self._combine_context_sections(
            [outline_guidance, evidence_context],
            max_chars=generation_context_budget,
        )

        if context_bundle.strip():
            return context_bundle
        if coverage_context.strip():
            return coverage_context
        return compact_full_context

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

            all_chunks, all_metadata, source_documents = self.collect_source_material(file_paths)

            jobs.update_job(
                job_id,
                progress=50,
                stage="embedding",
                message="Preparing grounded context across the uploaded material.",
            )

            jobs.update_job(
                job_id,
                progress=70,
                stage="retrieving",
                message="Selecting the most representative passages for generation.",
            )

            context_bundle = self.build_generation_context(
                session_id=session_id,
                all_chunks=all_chunks,
                all_metadata=all_metadata,
                options=options,
                source_documents=source_documents,
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
            focus = options.custom_prompt.strip()
            queries.extend(
                [
                    focus,
                    f"Find the exact section, heading, table, figure, or passage about: {focus}",
                    f"Retrieve any text that explains or surrounds this focus: {focus}",
                ]
            )

        queries.append(
            "Definitions, formulas, examples, and exam-relevant insights from uploaded files"
        )

        return queries

    def _normalize_line(self, line: str) -> str:
        return re.sub(r"\s+", " ", (line or "").strip())

    def _looks_like_heading(self, line: str) -> bool:
        cleaned = self._normalize_line(line)
        if not cleaned or len(cleaned) < 5 or len(cleaned) > 90:
            return False
        if cleaned.endswith((".", ":", ";", ",")):
            return False

        word_count = len(cleaned.split())
        if word_count > 12:
            return False

        if re.match(
            r"^(chapter|unit|module|section|topic|part)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.match(r"^\d+(?:\.\d+)*[\)\.]?\s+[A-Za-z]", cleaned):
            return True

        alpha_chars = [char for char in cleaned if char.isalpha()]
        if not alpha_chars:
            return False
        uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
        title_like_ratio = sum(
            1 for word in cleaned.split() if word[:1].isupper()
        ) / max(1, word_count)

        return uppercase_ratio >= 0.72 or title_like_ratio >= 0.72

    def _summarize_chunk_for_outline(self, chunk: str, max_chars: int = 120) -> str:
        compact = " ".join(chunk.split()).strip()
        if not compact:
            return ""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact)
            if sentence.strip()
        ]
        snippet = sentences[0] if sentences else compact
        if len(snippet) <= max_chars:
            return snippet
        clipped = snippet[:max_chars].rsplit(" ", 1)[0].strip()
        return clipped or snippet[:max_chars].strip()

    def _infer_chunk_heading(self, chunk: str, chunk_index: int) -> str:
        lines = [
            self._normalize_line(line)
            for line in chunk.splitlines()
            if self._normalize_line(line)
        ]
        for line in lines[:8]:
            if self._looks_like_heading(line):
                return line

        compact = " ".join(chunk.split()).strip()
        if not compact:
            return f"Section {chunk_index + 1}"

        first_sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0].strip()
        title_tokens = first_sentence.split()[:7]
        if not title_tokens:
            return f"Section {chunk_index + 1}"
        return " ".join(title_tokens).strip(" -:;,.") or f"Section {chunk_index + 1}"

    def _build_document_outline_guidance(
        self,
        chunks: list[str],
        metadata: list[dict],
        max_chars: int,
    ) -> str:
        if not chunks or max_chars <= 0:
            return ""

        by_source: dict[str, list[tuple[int, str]]] = {}
        for chunk, chunk_metadata in zip(chunks, metadata):
            source = (
                str(chunk_metadata.get("source", "uploaded_document")).strip()
                or "uploaded_document"
            )
            try:
                chunk_index = int(chunk_metadata.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_index = 0
            content = chunk.strip()
            if not content:
                continue
            by_source.setdefault(source, []).append((chunk_index, content))

        lines = [
            "Document structure guidance:",
            "Infer units in the same order as the material appears; cover early, middle, and late sections.",
        ]

        for source, source_chunks in by_source.items():
            source_chunks.sort(key=lambda item: item[0])
            target_count = min(6, max(3, len(source_chunks) // 2))
            if len(source_chunks) <= target_count:
                sampled = source_chunks
            else:
                step = (len(source_chunks) - 1) / max(1, target_count - 1)
                sampled_indices = sorted(
                    {int(round(step * idx)) for idx in range(target_count)}
                )
                sampled = [source_chunks[idx] for idx in sampled_indices]

            lines.append(f"Source: {source}")
            for ordinal, (chunk_index, content) in enumerate(sampled, start=1):
                heading = self._infer_chunk_heading(content, chunk_index)
                summary = self._summarize_chunk_for_outline(content, max_chars=110)
                lines.append(
                    f"- Unit candidate {ordinal} around chunk {chunk_index}: {heading} -> {summary}"
                )

        guidance = "\n".join(lines).strip()
        if len(guidance) <= max_chars:
            return guidance
        clipped = guidance[:max_chars].rsplit("\n", 1)[0].strip()
        return clipped or guidance[:max_chars].strip()

    def _build_focus_context(
        self,
        chunks: list[str],
        metadata: list[dict],
        focus_text: str,
        max_chars: int,
    ) -> str:
        focus = " ".join((focus_text or "").split()).strip()
        if not focus or not chunks or max_chars <= 0:
            return ""

        focus_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]{1,}", focus)
        }
        if not focus_terms:
            return ""

        scored: list[tuple[tuple[int, int, int], str, dict]] = []
        for chunk, chunk_metadata in zip(chunks, metadata):
            lowered = chunk.lower()
            overlap = sum(1 for token in focus_terms if token in lowered)
            if overlap <= 0:
                continue
            exact_phrase = 1 if focus.lower() in lowered else 0
            score = (exact_phrase, overlap, len(chunk))
            scored.append((score, chunk, chunk_metadata))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        selected_chunks = [item[1] for item in scored[:5]]
        selected_metadata = [item[2] for item in scored[:5]]
        return self._build_context_from_chunks(
            chunks=selected_chunks,
            metadata=selected_metadata,
            max_chars=max_chars,
        )

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

    def _merge_multiple_context_bundles(
        self,
        contexts: list[str],
        max_chars: int,
    ) -> str:
        merged_blocks: list[str] = []
        seen: set[str] = set()
        consumed_chars = 0

        for context in contexts:
            blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
            for block in blocks:
                if block in seen:
                    continue
                seen.add(block)
                separator_chars = 2 if merged_blocks else 0
                projected = consumed_chars + separator_chars + len(block)
                if projected > max_chars:
                    if not merged_blocks:
                        merged_blocks.append(block[:max_chars])
                    return "\n\n".join(merged_blocks).strip()
                merged_blocks.append(block)
                consumed_chars = projected

        return "\n\n".join(merged_blocks).strip()

    def _combine_context_sections(
        self,
        sections: list[str],
        max_chars: int,
    ) -> str:
        combined: list[str] = []
        consumed_chars = 0

        for section in sections:
            cleaned = section.strip()
            if not cleaned:
                continue
            separator_chars = 2 if combined else 0
            projected = consumed_chars + separator_chars + len(cleaned)
            if projected > max_chars:
                if not combined:
                    combined.append(cleaned[:max_chars])
                break
            combined.append(cleaned)
            consumed_chars = projected

        return "\n\n".join(combined).strip()

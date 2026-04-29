import json
import random
import re
import time
from collections import Counter
from threading import Lock
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.models.schemas import (
    DifficultyExplanation,
    Flashcard,
    GenerationOptions,
    QAItem,
    StudyGuideResult,
    TopicNote,
    UnitSummary,
)


class _GeminiTopicNoteSchema(BaseModel):
    topic: str = ""
    notes: str = ""


class _GeminiFlashcardSchema(BaseModel):
    question: str = ""
    answer: str = ""
    bloom_level: str = ""


class _GeminiQASchema(BaseModel):
    question: str = ""
    answer: str = ""
    bloom_level: str = ""


class _GeminiUnitSummarySchema(BaseModel):
    unit_title: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)


class _GeminiDifficultySchema(BaseModel):
    mode: str = ""
    explanation: str = ""


class _GeminiStudyGuideSchema(BaseModel):
    summary_notes: str = ""
    unit_wise_summaries: list[_GeminiUnitSummarySchema] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    topic_wise_notes: list[_GeminiTopicNoteSchema] = Field(default_factory=list)
    flashcards: list[_GeminiFlashcardSchema] = Field(default_factory=list)
    qa_sets: list[_GeminiQASchema] = Field(default_factory=list)
    viva_questions: list[str] = Field(default_factory=list)
    difficulty_explanations: list[_GeminiDifficultySchema] = Field(default_factory=list)


class _GeminiSummaryConceptSchema(BaseModel):
    summary_notes: str = ""
    key_concepts: list[str] = Field(default_factory=list)


class _GeminiUnitTopicSchema(BaseModel):
    unit_wise_summaries: list[_GeminiUnitSummarySchema] = Field(default_factory=list)
    topic_wise_notes: list[_GeminiTopicNoteSchema] = Field(default_factory=list)


class _GeminiFlashcardBundleSchema(BaseModel):
    flashcards: list[_GeminiFlashcardSchema] = Field(default_factory=list)


class _GeminiQABundleSchema(BaseModel):
    qa_sets: list[_GeminiQASchema] = Field(default_factory=list)


class _GeminiVivaDifficultySchema(BaseModel):
    viva_questions: list[str] = Field(default_factory=list)
    difficulty_explanations: list[_GeminiDifficultySchema] = Field(default_factory=list)


class GeminiGenerationService:
    NOT_AVAILABLE = "Not available in uploaded material"
    BLOOM_LEVELS = ("remember", "understand", "apply", "analyze")
    MAX_RATE_LIMIT_RETRIES = 4
    MIN_GROUNDED_OVERLAP = 0.24
    MIN_BLOCK_MATCH_OVERLAP = 0.18
    CITATION_PATTERN = re.compile(
        r"\[source:(?P<source>[^\]|]+)\s*\|\s*chunk:(?P<chunk>[^\]]+)\]"
    )
    CONTEXT_BLOCK_PATTERN = re.compile(
        r"\[source:(?P<source>[^\]|]+)\s*\|\s*chunk:(?P<chunk>[^\]]+)\]\s*(?P<text>.*?)(?=(?:\n\s*\[source:)|\Z)",
        re.DOTALL,
    )
    STOP_WORDS = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "were",
        "their",
        "there",
        "about",
        "which",
        "into",
        "these",
        "those",
        "will",
        "would",
        "could",
        "should",
        "between",
        "during",
        "through",
        "while",
        "where",
        "when",
        "been",
        "being",
        "such",
        "also",
        "than",
        "them",
        "they",
        "then",
        "only",
        "very",
        "some",
        "each",
        "many",
        "most",
        "more",
        "less",
        "much",
        "over",
        "under",
        "your",
    }
    WEAK_CONCEPT_TERMS = {
        "topic",
        "topics",
        "notes",
        "note",
        "table",
        "contents",
        "source",
        "chapter",
        "value",
        "values",
        "message",
        "messages",
        "block",
        "blocks",
        "data",
        "using",
        "used",
        "return",
        "result",
        "results",
    }
    _REQUEST_LOCK = Lock()

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to backend .env.")

        self.client = genai.Client(api_key=api_key)
        if (
            model_name
            and model_name.strip()
            and model_name.strip() != "gemini-2.5-flash"
        ):
            logger.warning(
                "Configured GEMINI_MODEL '%s' is ignored. Using gemini-2.5-flash only.",
                model_name.strip(),
            )
        self.model_name = "gemini-2.5-flash"
        self.fallback_models: list[str] = []

    def generate_from_context(
        self,
        context: str,
        options: GenerationOptions,
        source_documents: list[str],
    ) -> StudyGuideResult:
        payload = {
            "output_types": options.output_types,
            "difficulty_modes": options.difficulty_modes,
            "bloom_levels": options.bloom_levels,
            "custom_prompt": options.custom_prompt,
        }

        with self._REQUEST_LOCK:
            filtered = self._generate_grouped_outputs(
                context=context,
                options=options,
                source_documents=source_documents,
            )

            if self._should_retry_underfilled_result(filtered, options):
                fallback_filtered = self._filter_by_options(
                    self._build_fallback_output(
                        context,
                        reason="post-generation quality rescue",
                        focus_text=options.custom_prompt,
                    ),
                    options,
                    context,
                    allowed_sources=set(source_documents),
                )
                filtered = self._merge_result_payloads(
                    filtered,
                    fallback_filtered,
                    prefer_existing=True,
                )

        return StudyGuideResult(
            summary_notes=filtered.get("summary_notes", ""),
            unit_wise_summaries=filtered.get("unit_wise_summaries", []),
            key_concepts=filtered.get("key_concepts", []),
            topic_wise_notes=filtered.get("topic_wise_notes", []),
            flashcards=filtered.get("flashcards", []),
            qa_sets=filtered.get("qa_sets", []),
            viva_questions=filtered.get("viva_questions", []),
            difficulty_explanations=filtered.get("difficulty_explanations", []),
            source_documents=source_documents,
        )

    def _build_prompts(
        self,
        *,
        context: str,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        custom_focus = str(payload.get("custom_prompt", "")).strip()
        requested_outputs = {
            item.strip().lower()
            for item in payload.get("output_types", [])
            if str(item).strip()
        }
        block_count = max(1, len(self._parse_context_blocks(context)))
        system_prompt = (
            "You are an academic study-guide generator. "
            "You must use ONLY the provided context blocks. "
            "Every factual claim must be supported by one or more chunk citations in this format: "
            "[source:FILE_NAME | chunk:INDEX]. "
            f"If evidence is missing, output exactly '{self.NOT_AVAILABLE}'. "
            "Do not invent facts, definitions, formulas, names, or examples. "
            "When a custom focus is supplied, prioritize that focus explicitly, but still produce the full requested outputs from the document if supporting evidence exists. "
            "Return only JSON and never include free text outside the JSON object."
        )

        focus_instructions = (
            f"""
Custom focus requirement:
- Prioritize this focus strongly when selecting relevant content: "{custom_focus}".
- If the focus exists in the context, include it explicitly in at least one unit summary, one topic note, and one QA/flashcard answer when those sections are requested.
- Do not leave all outputs unavailable just because the focus is narrow; still generate the broader requested study guide from the document.
""".strip()
            if custom_focus
            else ""
        )

        user_prompt = f"""
Context blocks:
{context}

Generation request:
{json.dumps(payload, indent=2)}

Output quality constraints:
{self._build_quality_constraints(requested_outputs, block_count)}

{focus_instructions}

Hard grounding rules:
- Use only the given context blocks and their chunk tags.
- Do not cite any source that is not present in the context.
- If a field cannot be supported, set that field value to exactly "{self.NOT_AVAILABLE}".
- Do not emit partial JSON, comments, markdown, or explanatory prose outside the JSON object.

Return strict JSON matching this schema exactly:
{self._build_requested_schema(requested_outputs)}
""".strip()

        return system_prompt, user_prompt

    def _build_quality_constraints(
        self,
        requested_outputs: set[str],
        block_count: int,
    ) -> str:
        unit_target = min(8, max(4, block_count))
        concept_target = min(12, max(6, block_count * 2))
        topic_target = min(8, max(4, block_count))
        flashcard_target = min(6, max(4, block_count))
        qa_target = min(6, max(4, block_count))
        viva_target = min(6, max(4, block_count))
        unit_summary_limit = 110
        unit_point_limit = "2 to 5"
        topic_note_limit = 90

        if requested_outputs == {"unit_summaries"}:
            unit_target = min(6, max(4, block_count // 2))
            unit_summary_limit = 85
            unit_point_limit = "2 to 3"
        if requested_outputs == {"topic_notes"}:
            topic_target = min(6, max(4, block_count // 2))
            topic_note_limit = 70
        constraints: list[str] = []
        if "summary" in requested_outputs:
            constraints.append(
                "- summary_notes: comprehensive narrative covering all units; 4 to 8 short paragraphs with citations; total <= 320 words."
            )
        if "unit_summaries" in requested_outputs:
            constraints.extend(
                [
                    f"- unit_wise_summaries: {unit_target} ordered units/chapters that together cover the document from beginning to end.",
                    "  Each unit must have:",
                    "  - unit_title",
                    f"  - summary (<= {unit_summary_limit} words, with citation)",
                    f"  - key_points ({unit_point_limit} concise bullet-style strings, each grounded and cited)",
                ]
            )
        if "key_concepts" in requested_outputs:
            constraints.append(
                f"- key_concepts: maximum {concept_target} grounded phrases (prefer 2 to 5 words each)."
            )
        if "topic_notes" in requested_outputs:
            constraints.append(
                f"- topic_wise_notes: maximum {topic_target} topics; each notes field <= {topic_note_limit} words and must include citation."
            )
        if "flashcards" in requested_outputs:
            constraints.extend(
                [
                    f"- flashcards: maximum {flashcard_target} cards; each answer <= 75 words and must include citation.",
                    "  Each flashcard must include bloom_level from requested bloom_levels.",
                ]
            )
        if "qa" in requested_outputs:
            constraints.extend(
                [
                    f"- qa_sets: maximum {qa_target} Q&A items; each answer <= 90 words and must include citation.",
                    "  Each QA item must include bloom_level from requested bloom_levels.",
                ]
            )
        if "viva" in requested_outputs:
            constraints.append(
                f"- viva_questions: maximum {viva_target} focused questions."
            )
        if "difficulty" in requested_outputs:
            constraints.append(
                "- difficulty_explanations: include requested modes only; each explanation <= 100 words and must include citation."
            )
        return "\n".join(constraints)

    def _build_requested_schema(self, requested_outputs: set[str]) -> str:
        schema_parts: list[str] = []
        if "summary" in requested_outputs:
            schema_parts.append('  "summary_notes": "string"')
        if "unit_summaries" in requested_outputs:
            schema_parts.append(
                '  "unit_wise_summaries": [{"unit_title": "string", "summary": "string", "key_points": ["string"]}]'
            )
        if "key_concepts" in requested_outputs:
            schema_parts.append('  "key_concepts": ["string"]')
        if "topic_notes" in requested_outputs:
            schema_parts.append(
                '  "topic_wise_notes": [{"topic": "string", "notes": "string"}]'
            )
        if "flashcards" in requested_outputs:
            schema_parts.append(
                '  "flashcards": [{"question": "string", "answer": "string", "bloom_level": "string"}]'
            )
        if "qa" in requested_outputs:
            schema_parts.append(
                '  "qa_sets": [{"question": "string", "answer": "string", "bloom_level": "string"}]'
            )
        if "viva" in requested_outputs:
            schema_parts.append('  "viva_questions": ["string"]')
        if "difficulty" in requested_outputs:
            schema_parts.append(
                '  "difficulty_explanations": [{"mode": "beginner", "explanation": "string"}, {"mode": "intermediate", "explanation": "string"}, {"mode": "advanced", "explanation": "string"}]'
            )

        return "{\n" + ",\n".join(schema_parts) + "\n}"

    def _build_generation_config(
        self,
        *,
        system_prompt: str,
        options: GenerationOptions,
        max_output_tokens_override: int | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
        use_response_schema: bool = True,
    ) -> types.GenerateContentConfig:
        output_types = {
            item.strip().lower()
            for item in options.output_types
            if item and item.strip()
        }
        requested_sections = max(1, len(output_types))
        base_max_output_tokens = min(5600, 1800 + (requested_sections * 420))
        max_output_tokens = max_output_tokens_override or base_max_output_tokens

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature_override
            if temperature_override is not None
            else 0.05,
            "top_p": top_p_override if top_p_override is not None else 0.85,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
        }
        response_schema = (
            self._response_schema_for_output_types(output_types)
            if use_response_schema
            else None
        )
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        automatic_function_calling_cls = getattr(
            types, "AutomaticFunctionCallingConfig", None
        )
        if automatic_function_calling_cls is not None:
            config_kwargs["automatic_function_calling"] = (
                automatic_function_calling_cls(disable=True)
            )

        thinking_config_cls = getattr(types, "ThinkingConfig", None)
        if thinking_config_cls is not None:
            config_kwargs["thinking_config"] = thinking_config_cls(
                include_thoughts=False,
                thinking_budget=0,
            )

        return types.GenerateContentConfig(**config_kwargs)

    def _response_schema_for_output_types(
        self,
        output_types: set[str],
    ) -> type[BaseModel] | None:
        full_schema_outputs = {
            "summary",
            "unit_summaries",
            "key_concepts",
            "topic_notes",
            "flashcards",
            "qa",
            "viva",
            "difficulty",
        }
        if output_types == full_schema_outputs:
            return _GeminiStudyGuideSchema
        if output_types == {"summary", "key_concepts"} or output_types == {"summary"}:
            return _GeminiSummaryConceptSchema
        if output_types == {"key_concepts"}:
            return _GeminiSummaryConceptSchema
        if output_types == {"unit_summaries", "topic_notes"}:
            return _GeminiUnitTopicSchema
        if output_types == {"unit_summaries"}:
            return _GeminiUnitTopicSchema
        if output_types == {"topic_notes"}:
            return _GeminiUnitTopicSchema
        if output_types == {"flashcards"}:
            return _GeminiFlashcardBundleSchema
        if output_types == {"qa"}:
            return _GeminiQABundleSchema
        if output_types == {"viva", "difficulty"}:
            return _GeminiVivaDifficultySchema
        if output_types == {"viva"}:
            return _GeminiVivaDifficultySchema
        if output_types == {"difficulty"}:
            return _GeminiVivaDifficultySchema
        return None

    def _generate_with_gemini(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        options: GenerationOptions,
    ) -> dict[str, Any]:
        candidate_models: list[str] = []
        for model_name in [self.model_name, *self.fallback_models]:
            cleaned = model_name.strip()
            if cleaned and cleaned not in candidate_models:
                candidate_models.append(cleaned)

        config = self._build_generation_config(
            system_prompt=system_prompt,
            options=options,
        )

        last_error: Exception | None = None
        for model_name in candidate_models:
            for attempt in range(self.MAX_RATE_LIMIT_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=config,
                    )

                    parsed_payload = self._extract_payload_from_response(response)
                    if parsed_payload is not None and not self._is_payload_effectively_empty(
                        parsed_payload
                    ):
                        return parsed_payload

                    if self._response_hit_max_tokens(response):
                        raise RuntimeError(
                            "Gemini response hit max tokens before producing valid JSON."
                        )

                    compact_retry_payload = self._retry_compact_generation(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        options=options,
                    )
                    if compact_retry_payload is not None:
                        return compact_retry_payload

                    json_retry_payload = self._retry_without_schema(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        options=options,
                    )
                    if json_retry_payload is not None:
                        return json_retry_payload

                    last_error = RuntimeError(
                        "Gemini returned an empty structured response."
                    )
                    if attempt < self.MAX_RATE_LIMIT_RETRIES - 1:
                        wait_seconds = min(6.0, 1.0 + attempt)
                        logger.warning(
                            "Gemini returned empty structured response on model %s. Retrying in %.2fs (attempt %d/%d).",
                            model_name,
                            wait_seconds,
                            attempt + 1,
                            self.MAX_RATE_LIMIT_RETRIES,
                        )
                        time.sleep(wait_seconds)
                        continue
                    break
                except Exception as exc:
                    error_text = str(exc).lower()
                    last_error = exc

                    if (
                        "404" in error_text
                        or "not_found" in error_text
                        or "max tokens" in error_text
                    ):
                        break

                    if "429" in error_text or "resource_exhausted" in error_text:
                        wait_seconds = min(
                            18.0, (2**attempt) + random.uniform(0.2, 0.9)
                        )
                        logger.warning(
                            "Gemini rate-limited on model %s. Retrying in %.2fs (attempt %d/%d).",
                            model_name,
                            wait_seconds,
                            attempt + 1,
                            self.MAX_RATE_LIMIT_RETRIES,
                        )
                        time.sleep(wait_seconds)
                        continue

                    # Retry a small number of transient failures before moving on.
                    if attempt < self.MAX_RATE_LIMIT_RETRIES - 1:
                        time.sleep(min(6.0, 1.2 + attempt))
                        continue
                    break

        raise RuntimeError(f"Gemini generation unavailable after retries: {last_error}")

    def _empty_result_payload(self) -> dict[str, Any]:
        return {
            "summary_notes": "",
            "unit_wise_summaries": [],
            "key_concepts": [],
            "topic_wise_notes": [],
            "flashcards": [],
            "qa_sets": [],
            "viva_questions": [],
            "difficulty_explanations": [],
        }

    def _generate_grouped_outputs(
        self,
        *,
        context: str,
        options: GenerationOptions,
        source_documents: list[str],
    ) -> dict[str, Any]:
        grouped_output_types = self._build_output_groups(options.output_types)
        merged = self._empty_result_payload()

        for output_group in grouped_output_types:
            group_options = GenerationOptions(
                output_types=output_group,
                difficulty_modes=options.difficulty_modes,
                bloom_levels=options.bloom_levels,
                custom_prompt=options.custom_prompt,
            )
            payload = {
                "output_types": group_options.output_types,
                "difficulty_modes": group_options.difficulty_modes,
                "bloom_levels": group_options.bloom_levels,
                "custom_prompt": group_options.custom_prompt,
            }
            system_prompt, user_prompt = self._build_prompts(
                context=context,
                payload=payload,
            )

            try:
                parsed = self._generate_with_gemini(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    options=group_options,
                )
            except Exception as exc:
                logger.warning(
                    "Grouped generation failed for outputs %s; using grounded fallback: %s",
                    output_group,
                    exc,
                )
                parsed = self._build_fallback_output(
                    context,
                    reason=f"grouped generation failure for {','.join(output_group)}",
                    focus_text=options.custom_prompt,
                )

            filtered = self._filter_by_options(
                parsed,
                group_options,
                context,
                allowed_sources=set(source_documents),
            )
            merged = self._merge_result_payloads(
                merged,
                filtered,
                prefer_existing=True,
            )

        return merged

    def _build_output_groups(self, output_types: list[str]) -> list[list[str]]:
        normalized = [
            item.strip().lower()
            for item in output_types
            if item and item.strip()
        ]
        groups: list[list[str]] = []
        preferred_groups = [
            ["summary", "key_concepts"],
            ["unit_summaries"],
            ["topic_notes"],
            ["flashcards"],
            ["qa"],
            ["viva", "difficulty"],
        ]

        for preferred_group in preferred_groups:
            selected = [item for item in preferred_group if item in normalized]
            if selected:
                groups.append(selected)

        covered = {item for group in groups for item in group}
        for item in normalized:
            if item not in covered:
                groups.append([item])

        return groups or [normalized]

    def _merge_result_payloads(
        self,
        base: dict[str, Any],
        incoming: dict[str, Any],
        *,
        prefer_existing: bool = False,
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if isinstance(value, list):
                if prefer_existing and merged.get(key):
                    continue
                if value:
                    merged[key] = value
            elif prefer_existing and merged.get(key) not in {"", self.NOT_AVAILABLE, None}:
                continue
            elif value not in {"", self.NOT_AVAILABLE, None}:
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        return merged

    def _response_hit_max_tokens(self, response: Any) -> bool:
        for candidate in getattr(response, "candidates", None) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is None:
                continue
            if str(finish_reason).upper().endswith("MAX_TOKENS"):
                return True
        return False

    def _is_payload_effectively_empty(self, payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return True

        scalar_fields = [
            str(payload.get("summary_notes", "") or "").strip(),
        ]
        list_fields = [
            payload.get("unit_wise_summaries", []),
            payload.get("key_concepts", []),
            payload.get("topic_wise_notes", []),
            payload.get("flashcards", []),
            payload.get("qa_sets", []),
            payload.get("viva_questions", []),
            payload.get("difficulty_explanations", []),
        ]

        if any(field and field != self.NOT_AVAILABLE for field in scalar_fields):
            return False

        for items in list_fields:
            if not isinstance(items, list):
                continue
            if any(bool(item) for item in items):
                return False

        return True

    def _build_rescue_prompt(
        self,
        *,
        base_user_prompt: str,
        custom_prompt: str,
    ) -> str:
        focus_line = (
            f'- Give special attention to the custom focus: "{custom_prompt.strip()}".\n'
            if custom_prompt.strip()
            else ""
        )
        return (
            f"{base_user_prompt}\n\n"
            "Rescue pass instructions:\n"
            "- First infer the structure of the document from the context blocks.\n"
            "- Then generate the requested study-guide sections from that understanding.\n"
            "- Do not return every requested field as unavailable when the context clearly contains relevant material.\n"
            f"{focus_line}"
            "- Prefer grounded, concise content over placeholders.\n"
            "Return strict JSON only."
        )

    def _result_score(
        self,
        result: dict[str, Any],
        options: GenerationOptions,
    ) -> int:
        output_types = {
            item.strip().lower()
            for item in options.output_types
            if item and item.strip()
        }
        score = 0
        unique_citations = self._unique_citation_count(result)
        focus_hits = self._focus_hit_count(result, options.custom_prompt)
        duplicate_penalty = self._duplicate_content_penalty(result)

        if "summary" in output_types and result.get("summary_notes") not in {"", self.NOT_AVAILABLE}:
            score += 3
        if "unit_summaries" in output_types:
            score += min(3, len(result.get("unit_wise_summaries", [])))
        if "key_concepts" in output_types:
            score += min(2, len(result.get("key_concepts", [])) // 3)
        if "topic_notes" in output_types:
            score += min(2, len(result.get("topic_wise_notes", [])) // 2)
        if "flashcards" in output_types:
            score += min(2, len(result.get("flashcards", [])) // 2)
        if "qa" in output_types:
            score += min(2, len(result.get("qa_sets", [])) // 2)
        if "viva" in output_types and len(result.get("viva_questions", [])) > 0:
            score += 1
        if "difficulty" in output_types and len(result.get("difficulty_explanations", [])) > 0:
            score += 1
        score += min(3, unique_citations // 3)
        score += min(2, focus_hits)
        score -= duplicate_penalty

        return score

    def _unique_citation_count(self, result: dict[str, Any]) -> int:
        citations: set[str] = set()
        values: list[str] = []
        summary = str(result.get("summary_notes", "") or "").strip()
        if summary:
            values.append(summary)
        for item in result.get("unit_wise_summaries", []):
            if isinstance(item, UnitSummary):
                values.append(item.summary)
                values.extend(item.key_points)
        for item in result.get("topic_wise_notes", []):
            if isinstance(item, TopicNote):
                values.append(item.notes)
        for item in result.get("flashcards", []):
            if isinstance(item, Flashcard):
                values.append(item.answer)
        for item in result.get("qa_sets", []):
            if isinstance(item, QAItem):
                values.append(item.answer)
        for item in result.get("difficulty_explanations", []):
            if isinstance(item, DifficultyExplanation):
                values.append(item.explanation)

        for value in values:
            for citation in self._extract_citations(value):
                citations.add(citation["tag"])
        return len(citations)

    def _focus_hit_count(self, result: dict[str, Any], focus_text: str) -> int:
        focus = " ".join((focus_text or "").split()).strip().lower()
        if not focus:
            return 0

        focus_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]{1,}", focus)
            if token.lower() not in self.STOP_WORDS
        }
        if not focus_terms:
            return 0

        values: list[str] = []
        values.append(str(result.get("summary_notes", "") or ""))
        for item in result.get("unit_wise_summaries", []):
            if isinstance(item, UnitSummary):
                values.append(f"{item.unit_title} {item.summary}")
        for item in result.get("topic_wise_notes", []):
            if isinstance(item, TopicNote):
                values.append(f"{item.topic} {item.notes}")
        for item in result.get("flashcards", []):
            if isinstance(item, Flashcard):
                values.append(f"{item.question} {item.answer}")
        for item in result.get("qa_sets", []):
            if isinstance(item, QAItem):
                values.append(f"{item.question} {item.answer}")

        hit_count = 0
        for value in values:
            lowered = value.lower()
            if focus in lowered:
                hit_count += 1
                continue
            overlap = sum(1 for token in focus_terms if token in lowered)
            if overlap >= max(2, len(focus_terms) // 2):
                hit_count += 1
        return hit_count

    def _duplicate_content_penalty(self, result: dict[str, Any]) -> int:
        normalized_values: list[str] = []
        for item in result.get("unit_wise_summaries", []):
            if isinstance(item, UnitSummary) and item.summary != self.NOT_AVAILABLE:
                normalized_values.append(self._strip_citations(item.summary).lower())
        for item in result.get("topic_wise_notes", []):
            if isinstance(item, TopicNote) and item.notes != self.NOT_AVAILABLE:
                normalized_values.append(self._strip_citations(item.notes).lower())
        duplicate_count = len(normalized_values) - len(set(normalized_values))
        return max(0, min(3, duplicate_count))

    def _should_retry_underfilled_result(
        self,
        result: dict[str, Any],
        options: GenerationOptions,
    ) -> bool:
        minimum_score = 4 if options.custom_prompt.strip() else 3
        return self._result_score(result, options) < minimum_score

    def _extract_payload_from_response(self, response: Any) -> dict[str, Any] | None:
        parsed_payload = self._extract_parsed_payload(response)
        if parsed_payload is not None:
            return parsed_payload

        text_response = self._response_to_text(response)
        if not text_response.strip():
            return self._extract_payload_from_candidates(response)

        try:
            return self._safe_json_parse(text_response)
        except ValueError:
            return self._extract_payload_from_candidates(response)

    def _retry_compact_generation(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        options: GenerationOptions,
    ) -> dict[str, Any] | None:
        compact_prompt = (
            f"{user_prompt}\n\n"
            "Compact regeneration constraints:\n"
            "- summary_notes: at most 220 words with citations and full document coverage.\n"
            "- unit_wise_summaries: 4 to 6 items in document order; each summary <= 70 words with citations and 1 to 2 key_points.\n"
            "- key_concepts: at most 10 concise phrases.\n"
            "- topic_wise_notes: at most 6 topics; each notes <= 60 words with citation.\n"
            "- flashcards: at most 6 items; each answer <= 45 words with citation and bloom_level.\n"
            "- qa_sets: at most 6 items; each answer <= 55 words with citation and bloom_level.\n"
            "- viva_questions: at most 5 items.\n"
            "- difficulty_explanations: 3 items only; each <= 70 words with citation.\n"
            "Return strict JSON only."
        )

        compact_config = self._build_generation_config(
            system_prompt=system_prompt,
            options=options,
            max_output_tokens_override=3200,
            temperature_override=0.0,
            top_p_override=0.8,
        )

        try:
            compact_response = self.client.models.generate_content(
                model=model_name,
                contents=compact_prompt,
                config=compact_config,
            )
        except Exception:
            return None

        return self._extract_payload_from_response(compact_response)

    def _retry_without_schema(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        options: GenerationOptions,
    ) -> dict[str, Any] | None:
        fallback_prompt = (
            f"{user_prompt}\n\n"
            "If schema validation is difficult, still return strict JSON with the exact keys and no markdown."
        )

        config = self._build_generation_config(
            system_prompt=system_prompt,
            options=options,
            max_output_tokens_override=3200,
            temperature_override=0.0,
            top_p_override=0.82,
            use_response_schema=False,
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=fallback_prompt,
                config=config,
            )
        except Exception:
            return None

        extracted_payload = self._extract_payload_from_response(response)
        if extracted_payload is not None:
            return extracted_payload

        text_response = self._response_to_text(response)
        if not text_response:
            return None

        try:
            return self._safe_json_parse(text_response)
        except ValueError:
            return self._extract_payload_from_candidates(response)

    def _extract_parsed_payload(self, response: Any) -> dict[str, Any] | None:
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            return None

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, BaseModel):
            dumped = parsed.model_dump()
            return dumped if isinstance(dumped, dict) else None

        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped

        as_dict = getattr(parsed, "dict", None)
        if callable(as_dict):
            try:
                dumped = as_dict()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped

        return None

    def _response_to_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        fragments: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", [])
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    fragments.append(str(part_text))

        return "\n".join(fragments).strip()

    def _extract_payload_from_candidates(self, response: Any) -> dict[str, Any] | None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None

        for candidate in candidates:
            direct_payload = self._extract_candidate_payload(candidate)
            if direct_payload is not None:
                return direct_payload

            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", [])
            for part in parts:
                part_payload = self._extract_part_payload(part)
                if part_payload is not None:
                    return part_payload

        return None

    def _extract_candidate_payload(self, candidate: Any) -> dict[str, Any] | None:
        for attribute_name in ("parsed", "json", "data", "output", "result"):
            payload = self._coerce_payload_to_dict(
                getattr(candidate, attribute_name, None)
            )
            if payload is not None:
                return payload

        model_dump_method = getattr(candidate, "model_dump", None)
        if callable(model_dump_method):
            try:
                dumped = model_dump_method()
            except Exception:
                dumped = None
            payload = self._coerce_payload_to_dict(dumped)
            if payload is not None:
                return payload

        dict_method = getattr(candidate, "dict", None)
        if callable(dict_method):
            try:
                dumped = dict_method()
            except Exception:
                dumped = None
            payload = self._coerce_payload_to_dict(dumped)
            if payload is not None:
                return payload

        return None

    def _extract_part_payload(self, part: Any) -> dict[str, Any] | None:
        for attribute_name in (
            "parsed",
            "json",
            "inline_data",
            "function_call",
            "function_response",
            "response",
            "data",
            "text",
        ):
            payload = self._coerce_payload_to_dict(getattr(part, attribute_name, None))
            if payload is not None:
                return payload

        model_dump_method = getattr(part, "model_dump", None)
        if callable(model_dump_method):
            try:
                dumped = model_dump_method()
            except Exception:
                dumped = None
            payload = self._coerce_payload_to_dict(dumped)
            if payload is not None:
                return payload

        dict_method = getattr(part, "dict", None)
        if callable(dict_method):
            try:
                dumped = dict_method()
            except Exception:
                dumped = None
            payload = self._coerce_payload_to_dict(dumped)
            if payload is not None:
                return payload

        return None

    def _coerce_payload_to_dict(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            return value

        if isinstance(value, BaseModel):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else None

        model_dump_method = getattr(value, "model_dump", None)
        if callable(model_dump_method):
            try:
                dumped = model_dump_method()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped
            if isinstance(dumped, str):
                try:
                    parsed = self._safe_json_parse(dumped)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                dumped = dict_method()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = self._safe_json_parse(stripped)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

        return None

    def _safe_json_parse(self, text: str) -> dict[str, Any]:
        stripped = text.strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        repaired = stripped
        # Remove markdown fences or accidental labels before JSON begins.
        first_brace = repaired.find("{")
        if first_brace > 0:
            repaired = repaired[first_brace:]

        # Trim trailing non-JSON noise after the last closing brace.
        last_brace = repaired.rfind("}")
        if last_brace >= 0:
            repaired = repaired[: last_brace + 1]

        # Remove trailing commas before closing braces/brackets.
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Balance braces for common truncation cases.
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired = repaired + ("}" * (open_braces - close_braces))

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        if fenced_match:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                pass

        object_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if object_match:
            try:
                return json.loads(object_match.group(1))
            except json.JSONDecodeError:
                pass

        # Recover from common escaped JSON payloads (double-encoded object string).
        escaped_match = re.search(r'"(\{.*\})"', stripped, re.DOTALL)
        if escaped_match:
            candidate = escaped_match.group(1).encode("utf-8").decode("unicode_escape")
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError("Failed to parse Gemini response as JSON.")

    def _normalize_source_name(self, source: str) -> str:
        return " ".join(str(source).split()).strip().lower()

    def _strip_citations(self, text: str) -> str:
        return " ".join(self.CITATION_PATTERN.sub("", text or "").split()).strip()

    def _extract_citations(self, text: str) -> list[dict[str, str]]:
        citations: list[dict[str, str]] = []
        for match in self.CITATION_PATTERN.finditer(text or ""):
            source = " ".join(match.group("source").split()).strip()
            chunk = " ".join(match.group("chunk").split()).strip()
            citations.append(
                {
                    "source": source,
                    "chunk": chunk,
                    "tag": f"[source:{source} | chunk:{chunk}]",
                }
            )
        return citations

    def _parse_context_blocks(self, context: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for match in self.CONTEXT_BLOCK_PATTERN.finditer(context):
            source = " ".join(match.group("source").split()).strip()
            chunk = " ".join(match.group("chunk").split()).strip()
            text = " ".join(match.group("text").split()).strip()
            if not text:
                continue

            terms = {
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
                if token not in self.STOP_WORDS
            }
            blocks.append(
                {
                    "source": source,
                    "chunk": chunk,
                    "tag": f"[source:{source} | chunk:{chunk}]",
                    "text": text,
                    "terms": terms,
                }
            )

        return blocks

    def _best_matching_citation(
        self,
        text: str,
        context_blocks: list[dict[str, Any]],
    ) -> str | None:
        clean_text = self._strip_citations(text)
        candidate_terms = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", clean_text.lower())
            if token not in self.STOP_WORDS
        }
        if not candidate_terms:
            return None

        best_block: dict[str, Any] | None = None
        best_overlap = 0.0
        best_common = 0

        for block in context_blocks:
            common = len(candidate_terms & block["terms"])
            if common <= 0:
                continue

            overlap = common / len(candidate_terms)
            if overlap > best_overlap or (
                overlap == best_overlap and common > best_common
            ):
                best_overlap = overlap
                best_common = common
                best_block = block

        if best_block is None or best_overlap < self.MIN_BLOCK_MATCH_OVERLAP:
            return None

        return str(best_block["tag"])

    def _summarize_block_text(self, text: str, max_chars: int = 300) -> str:
        compact = " ".join(text.split()).strip()
        if not compact:
            return ""

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact)
            if sentence.strip()
        ]
        if not sentences:
            snippet = compact
        else:
            snippet = " ".join(sentences[:2])

        if len(snippet) <= max_chars:
            return snippet

        clipped = snippet[:max_chars].rsplit(" ", 1)[0].strip()
        return clipped or snippet[:max_chars].strip()

    def _is_noisy_text(self, text: str) -> bool:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return True

        if re.search(r"(?:\b[a-zA-Z]\b\s+){6,}", compact):
            return True

        tokens = re.findall(r"[A-Za-z0-9\-]+", compact)
        if not tokens:
            return True

        single_letter_tokens = [
            token for token in tokens if len(token) == 1 and token.isalpha()
        ]
        if single_letter_tokens and (len(single_letter_tokens) / len(tokens)) > 0.22:
            return True

        return False

    def _truncate_text(self, text: str, max_chars: int) -> str:
        compact = " ".join((text or "").split()).strip()
        if len(compact) <= max_chars:
            return compact

        candidate = compact[:max_chars].rsplit(" ", 1)[0].strip()
        if candidate.count("[source:") > candidate.count("]"):
            candidate = candidate.rsplit("[source:", 1)[0].strip()
        return candidate or compact[:max_chars].strip()

    def _normalize_bloom_levels(self, bloom_levels: list[str]) -> list[str]:
        normalized = [
            level.strip().lower()
            for level in bloom_levels
            if level and level.strip().lower() in self.BLOOM_LEVELS
        ]
        if normalized:
            # Keep order while removing duplicates.
            return list(dict.fromkeys(normalized))
        return list(self.BLOOM_LEVELS)

    def _normalize_bloom_level(
        self,
        value: Any,
        allowed_levels: list[str],
        fallback_index: int,
    ) -> str:
        candidate = str(value or "").strip().lower()
        if candidate in allowed_levels:
            return candidate
        return allowed_levels[fallback_index % len(allowed_levels)]

    def _normalize_question_key(self, question: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = [token for token in cleaned.split() if token not in self.STOP_WORDS]
        return " ".join(tokens)

    def _dedupe_questions(self, questions: list[str], limit: int) -> list[str]:
        deduped: list[str] = []
        seen_keys: set[str] = set()

        for question in questions:
            cleaned = " ".join(question.split()).strip()
            if not cleaned:
                continue

            key = self._normalize_question_key(cleaned)
            if not key or key in seen_keys:
                continue

            seen_keys.add(key)
            deduped.append(cleaned)
            if len(deduped) >= limit:
                break

        return deduped

    def _extract_key_phrases(
        self,
        context_blocks: list[dict[str, Any]],
        limit: int = 12,
    ) -> list[str]:
        phrase_counts: Counter[str] = Counter()
        unigram_counts: Counter[str] = Counter()

        for block in context_blocks:
            tokens = [
                token
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9\-]{2,}", block["text"].lower()
                )
                if token not in self.STOP_WORDS and token not in self.WEAK_CONCEPT_TERMS
            ]
            unigram_counts.update(tokens)

            for ngram_size in (3, 2):
                if len(tokens) < ngram_size:
                    continue
                for index in range(len(tokens) - ngram_size + 1):
                    gram = tokens[index : index + ngram_size]
                    if any(token in self.WEAK_CONCEPT_TERMS for token in gram):
                        continue
                    if len(set(gram)) < 2:
                        continue
                    phrase_counts[" ".join(gram)] += 1

        concepts: list[str] = []
        seen: set[str] = set()

        for phrase, _ in phrase_counts.most_common(40):
            normalized = phrase.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            concepts.append(" ".join(part.capitalize() for part in normalized.split()))
            if len(concepts) >= limit:
                return concepts

        for token, _ in unigram_counts.most_common(60):
            normalized = token.strip().lower()
            if normalized in seen or normalized in self.WEAK_CONCEPT_TERMS:
                continue
            if len(normalized) < 5 and "-" not in normalized:
                continue
            seen.add(normalized)
            concepts.append(normalized.capitalize())
            if len(concepts) >= limit:
                return concepts

        return concepts

    def _context_terms(self, context: str) -> set[str]:
        context_without_citations = self._strip_citations(context)
        tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9\-]{2,}", context_without_citations.lower()
        )
        return set(tokens)

    def _is_grounded_text(self, text: str, context_terms: set[str]) -> bool:
        cleaned = self._strip_citations(" ".join(text.split()).strip())
        if not cleaned:
            return False

        if self.NOT_AVAILABLE.lower() in cleaned.lower():
            return True

        tokens = [
            token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", cleaned)
        ]
        if not tokens:
            return True

        if len(tokens) <= 3:
            return any(token in context_terms for token in tokens)

        content_tokens = [token for token in tokens if token not in self.STOP_WORDS]
        if not content_tokens:
            return True

        overlap_count = sum(1 for token in content_tokens if token in context_terms)
        overlap_ratio = overlap_count / len(content_tokens)
        return overlap_ratio >= self.MIN_GROUNDED_OVERLAP

    def _ground_text(
        self,
        value: Any,
        context_terms: set[str],
        context_blocks: list[dict[str, Any]],
        allowed_sources: set[str] | None,
        *,
        require_citation: bool,
    ) -> str:
        text = " ".join(str(value).split()).strip()
        if not text:
            return self.NOT_AVAILABLE

        if self.NOT_AVAILABLE.lower() in text.lower():
            return self.NOT_AVAILABLE

        source_whitelist = (
            {self._normalize_source_name(source) for source in allowed_sources}
            if allowed_sources
            else None
        )
        valid_context_citations = {
            (
                self._normalize_source_name(str(block["source"])),
                str(block["chunk"]).strip(),
            )
            for block in context_blocks
        }

        citations = self._extract_citations(text)
        if citations:
            for citation in citations:
                source_name = self._normalize_source_name(citation["source"])
                chunk_label = str(citation["chunk"]).strip()

                if source_whitelist is not None and source_name not in source_whitelist:
                    return self.NOT_AVAILABLE
                if (
                    valid_context_citations
                    and (source_name, chunk_label) not in valid_context_citations
                ):
                    return self.NOT_AVAILABLE

        plain_text = self._strip_citations(text)
        if self._is_noisy_text(plain_text):
            return self.NOT_AVAILABLE

        if not self._is_grounded_text(plain_text, context_terms):
            return self.NOT_AVAILABLE

        if not require_citation:
            return text if citations else plain_text

        if citations:
            return text

        inferred_citation = self._best_matching_citation(plain_text, context_blocks)
        if inferred_citation is None:
            return self.NOT_AVAILABLE

        inferred_parts = self._extract_citations(inferred_citation)
        if source_whitelist is not None and inferred_parts:
            inferred_source = self._normalize_source_name(inferred_parts[0]["source"])
            if inferred_source not in source_whitelist:
                return self.NOT_AVAILABLE

        return f"{plain_text} {inferred_citation}".strip()

    def _filter_by_options(
        self,
        data: dict[str, Any],
        options: GenerationOptions,
        context: str,
        allowed_sources: set[str] | None,
    ) -> dict[str, Any]:
        output_types = {
            item.strip().lower()
            for item in options.output_types
            if item and item.strip()
        }
        difficulty_modes = {
            mode.strip().lower()
            for mode in options.difficulty_modes
            if mode and mode.strip().lower() in {"beginner", "intermediate", "advanced"}
        }
        if not difficulty_modes:
            difficulty_modes = {"beginner", "intermediate", "advanced"}
        bloom_levels = self._normalize_bloom_levels(options.bloom_levels)

        context_blocks = self._parse_context_blocks(context)
        if context_blocks:
            context_terms = {
                term for block in context_blocks for term in block["terms"]
            }
        else:
            context_terms = self._context_terms(context)

        result: dict[str, Any] = {
            "summary_notes": "",
            "unit_wise_summaries": [],
            "key_concepts": [],
            "topic_wise_notes": [],
            "flashcards": [],
            "qa_sets": [],
            "viva_questions": [],
            "difficulty_explanations": [],
        }

        if "summary" in output_types:
            summary_text = self._ground_text(
                data.get("summary_notes", ""),
                context_terms,
                context_blocks,
                allowed_sources,
                require_citation=True,
            )
            if summary_text != self.NOT_AVAILABLE:
                summary_text = self._truncate_text(summary_text, 1700)
            result["summary_notes"] = summary_text

        if "unit_summaries" in output_types or "summary" in output_types:
            raw_units = data.get("unit_wise_summaries", [])
            units: list[UnitSummary] = []
            seen_titles: set[str] = set()
            for index, item in enumerate(raw_units):
                if not isinstance(item, dict):
                    continue

                title = (
                    " ".join(
                        str(item.get("unit_title", f"Unit {index + 1}")).split()
                    ).strip()
                    or f"Unit {index + 1}"
                )
                title_key = title.lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                unit_summary = self._ground_text(
                    item.get("summary", ""),
                    context_terms,
                    context_blocks,
                    allowed_sources,
                    require_citation=True,
                )

                raw_points = item.get("key_points", [])
                normalized_points: list[str] = []
                for raw_point in raw_points if isinstance(raw_points, list) else []:
                    grounded_point = self._ground_text(
                        raw_point,
                        context_terms,
                        context_blocks,
                        allowed_sources,
                        require_citation=True,
                    )
                    if grounded_point == self.NOT_AVAILABLE:
                        continue
                    normalized_points.append(self._truncate_text(grounded_point, 260))

                if not normalized_points and unit_summary != self.NOT_AVAILABLE:
                    normalized_points = [self._truncate_text(unit_summary, 260)]

                units.append(
                    UnitSummary(
                        unit_title=title,
                        summary=self._truncate_text(unit_summary, 560),
                        key_points=normalized_points[:5],
                    )
                )

            result["unit_wise_summaries"] = units[:12]

        if "key_concepts" in output_types:
            concepts: list[str] = []
            seen_concepts: set[str] = set()
            for item in data.get("key_concepts", []):
                concept = " ".join(str(item).split()).strip()
                if not concept:
                    continue
                concept_tokens = re.findall(
                    r"[A-Za-z][A-Za-z0-9\-]{1,}", concept.lower()
                )
                if concept_tokens and not any(
                    token in context_terms for token in concept_tokens
                ):
                    continue
                filtered_tokens = [
                    token
                    for token in concept_tokens
                    if token not in self.STOP_WORDS
                    and token not in self.WEAK_CONCEPT_TERMS
                ]
                if not filtered_tokens:
                    continue
                if (
                    len(filtered_tokens) == 1
                    and len(filtered_tokens[0]) < 6
                    and "-" not in filtered_tokens[0]
                ):
                    continue
                concept_key = concept.lower()
                if concept_key in seen_concepts:
                    continue
                seen_concepts.add(concept_key)
                concepts.append(concept)
            result["key_concepts"] = concepts[:10]

        if "topic_notes" in output_types:
            topic_notes_raw = data.get("topic_wise_notes", [])
            result["topic_wise_notes"] = [
                TopicNote(
                    topic=" ".join(
                        str(item.get("topic", "Untitled Topic")).split()
                    ).strip()
                    or "Untitled Topic",
                    notes=self._ground_text(
                        item.get("notes", ""),
                        context_terms,
                        context_blocks,
                        allowed_sources,
                        require_citation=True,
                    ),
                )
                for item in topic_notes_raw
                if isinstance(item, dict)
            ][:10]

        if "flashcards" in output_types:
            flashcards_raw = data.get("flashcards", [])
            flashcards: list[Flashcard] = []
            seen_answers: set[str] = set()
            for index, item in enumerate(flashcards_raw):
                if not isinstance(item, dict):
                    continue

                question = (
                    " ".join(str(item.get("question", "")).split()).strip()
                    or self.NOT_AVAILABLE
                )
                if question != self.NOT_AVAILABLE and not self._is_grounded_text(
                    question, context_terms
                ):
                    question = self.NOT_AVAILABLE

                answer = self._ground_text(
                    item.get("answer", ""),
                    context_terms,
                    context_blocks,
                    allowed_sources,
                    require_citation=True,
                )
                if answer != self.NOT_AVAILABLE:
                    answer_key = self._strip_citations(answer).lower()
                    if answer_key in seen_answers:
                        continue
                    seen_answers.add(answer_key)
                flashcards.append(
                    Flashcard(
                        question=question,
                        answer=answer,
                        bloom_level=self._normalize_bloom_level(
                            item.get("bloom_level", ""),
                            bloom_levels,
                            index,
                        ),
                    )
                )

            result["flashcards"] = flashcards[:12]

        if "qa" in output_types:
            qa_raw = data.get("qa_sets", [])
            qa_items: list[QAItem] = []
            seen_answers: set[str] = set()
            for index, item in enumerate(qa_raw):
                if not isinstance(item, dict):
                    continue

                question = (
                    " ".join(str(item.get("question", "")).split()).strip()
                    or self.NOT_AVAILABLE
                )
                if question != self.NOT_AVAILABLE and not self._is_grounded_text(
                    question, context_terms
                ):
                    question = self.NOT_AVAILABLE

                answer = self._ground_text(
                    item.get("answer", ""),
                    context_terms,
                    context_blocks,
                    allowed_sources,
                    require_citation=True,
                )
                if answer != self.NOT_AVAILABLE:
                    answer_key = self._strip_citations(answer).lower()
                    if answer_key in seen_answers:
                        continue
                    seen_answers.add(answer_key)
                qa_items.append(
                    QAItem(
                        question=question,
                        answer=answer,
                        bloom_level=self._normalize_bloom_level(
                            item.get("bloom_level", ""),
                            bloom_levels,
                            index,
                        ),
                    )
                )

            result["qa_sets"] = qa_items[:12]

        if "viva" in output_types:
            viva_questions: list[str] = []
            for item in data.get("viva_questions", []):
                question = " ".join(str(item).split()).strip()
                if not question:
                    continue
                if self._is_grounded_text(question, context_terms):
                    viva_questions.append(question)
            result["viva_questions"] = self._dedupe_questions(viva_questions, limit=8)

        if "difficulty" in output_types:
            difficulty_raw = data.get("difficulty_explanations", [])
            result["difficulty_explanations"] = [
                DifficultyExplanation(
                    mode=str(item.get("mode", "beginner")).lower(),
                    explanation=self._ground_text(
                        item.get("explanation", ""),
                        context_terms,
                        context_blocks,
                        allowed_sources,
                        require_citation=True,
                    ),
                )
                for item in difficulty_raw
                if isinstance(item, dict)
                and str(item.get("mode", "")).lower() in difficulty_modes
            ]

        return self._backfill_missing_fields(result, bloom_levels)

    def _backfill_missing_fields(
        self,
        result: dict[str, Any],
        allowed_bloom_levels: list[str] | None = None,
    ) -> dict[str, Any]:
        unit_items = [
            item
            for item in result.get("unit_wise_summaries", [])
            if isinstance(item, UnitSummary)
        ]

        topic_notes = [
            note
            for note in result.get("topic_wise_notes", [])
            if isinstance(note, TopicNote) and note.notes != self.NOT_AVAILABLE
        ]

        topic_note_texts = [
            self._truncate_text(note.notes, 360) for note in topic_notes
        ]

        topic_items = [
            item
            for item in result.get("topic_wise_notes", [])
            if isinstance(item, TopicNote)
        ]
        if topic_note_texts:
            replacement_index = 0
            for item in topic_items:
                if item.notes != self.NOT_AVAILABLE:
                    continue
                item.notes = topic_note_texts[replacement_index % len(topic_note_texts)]
                replacement_index += 1

        topic_note_texts = [
            self._truncate_text(item.notes, 360)
            for item in topic_items
            if item.notes != self.NOT_AVAILABLE
        ]

        if not unit_items and topic_items:
            generated_units: list[UnitSummary] = []
            for index, topic_item in enumerate(topic_items, start=1):
                generated_units.append(
                    UnitSummary(
                        unit_title=topic_item.topic or f"Unit {index}",
                        summary=self._truncate_text(topic_item.notes, 520),
                        key_points=[self._truncate_text(topic_item.notes, 240)],
                    )
                )
            result["unit_wise_summaries"] = generated_units[:12]
            unit_items = generated_units[:12]

        topic_lookup: dict[str, TopicNote] = {}
        for topic_item in topic_items:
            topic_key = self._normalize_question_key(topic_item.topic)
            if topic_key and topic_item.notes != self.NOT_AVAILABLE:
                topic_lookup[topic_key] = topic_item

        for unit_item in unit_items:
            if unit_item.summary != self.NOT_AVAILABLE:
                continue
            unit_key = self._normalize_question_key(unit_item.unit_title)
            topic_match = topic_lookup.get(unit_key)
            if topic_match is None:
                for topic_key, candidate in topic_lookup.items():
                    common_terms = set(unit_key.split()) & set(topic_key.split())
                    if len(common_terms) >= 2:
                        topic_match = candidate
                        break
            if topic_match is None:
                continue
            unit_item.summary = self._truncate_text(topic_match.notes, 520)
            if not unit_item.key_points:
                unit_item.key_points = [self._truncate_text(topic_match.notes, 240)]
        if unit_items:
            result["unit_wise_summaries"] = unit_items

        summary_text = str(result.get("summary_notes", "") or "").strip()
        if summary_text == self.NOT_AVAILABLE and topic_note_texts:
            result["summary_notes"] = self._truncate_text(
                " ".join(topic_note_texts[:5]), 1700
            )
            summary_text = result["summary_notes"]

        if unit_items:
            unit_based_summary = " ".join(
                self._truncate_text(f"{item.unit_title}: {item.summary}", 230)
                for item in unit_items[:8]
                if item.summary and item.summary != self.NOT_AVAILABLE
            )
            if unit_based_summary and (
                summary_text == self.NOT_AVAILABLE
                or len(self._strip_citations(summary_text)) < 220
            ):
                result["summary_notes"] = self._truncate_text(unit_based_summary, 1700)
                summary_text = result["summary_notes"]

        flashcard_items = [
            card for card in result.get("flashcards", []) if isinstance(card, Flashcard)
        ]
        bloom_levels = allowed_bloom_levels or list(self.BLOOM_LEVELS)
        for index, card in enumerate(flashcard_items):
            if card.bloom_level not in bloom_levels:
                card.bloom_level = bloom_levels[index % len(bloom_levels)]

        qa_items = [
            item for item in result.get("qa_sets", []) if isinstance(item, QAItem)
        ]
        for index, qa_item in enumerate(qa_items):
            if qa_item.bloom_level not in bloom_levels:
                qa_item.bloom_level = bloom_levels[index % len(bloom_levels)]

        summary_after = str(result.get("summary_notes", "") or "").strip()
        summary_seed = (
            summary_after
            if summary_after and summary_after != self.NOT_AVAILABLE
            else (topic_note_texts[0] if topic_note_texts else self.NOT_AVAILABLE)
        )

        difficulty_items = [
            item
            for item in result.get("difficulty_explanations", [])
            if isinstance(item, DifficultyExplanation)
        ]

        if not difficulty_items and summary_seed != self.NOT_AVAILABLE:
            result["difficulty_explanations"] = [
                DifficultyExplanation(
                    mode="beginner",
                    explanation=self._truncate_text(summary_seed, 380),
                ),
                DifficultyExplanation(
                    mode="intermediate",
                    explanation=self._truncate_text(
                        " ".join(topic_note_texts[:3]) or summary_seed, 480
                    ),
                ),
                DifficultyExplanation(
                    mode="advanced",
                    explanation=self._truncate_text(
                        " ".join(topic_note_texts[:5]) or summary_seed, 560
                    ),
                ),
            ]
            return result

        for index, item in enumerate(difficulty_items):
            if item.explanation != self.NOT_AVAILABLE:
                continue
            if summary_seed == self.NOT_AVAILABLE:
                continue
            if index == 0:
                item.explanation = self._truncate_text(summary_seed, 380)
            elif index == 1:
                item.explanation = self._truncate_text(
                    " ".join(topic_note_texts[:3]) or summary_seed, 480
                )
            else:
                item.explanation = self._truncate_text(
                    " ".join(topic_note_texts[:5]) or summary_seed, 560
                )

        return result

    def _build_fallback_output(
        self,
        context: str,
        reason: str,
        focus_text: str = "",
    ) -> dict[str, Any]:
        logger.warning("Using grounded fallback generation path: %s", reason)

        context_blocks = self._parse_context_blocks(context)
        if not context_blocks:
            return {
                "summary_notes": self.NOT_AVAILABLE,
                "unit_wise_summaries": [],
                "key_concepts": [],
                "topic_wise_notes": [],
                "flashcards": [],
                "qa_sets": [],
                "viva_questions": [],
                "difficulty_explanations": [
                    {"mode": "beginner", "explanation": self.NOT_AVAILABLE},
                    {"mode": "intermediate", "explanation": self.NOT_AVAILABLE},
                    {"mode": "advanced", "explanation": self.NOT_AVAILABLE},
                ],
            }

        quality_blocks = [
            block
            for block in context_blocks
            if not self._is_noisy_text(str(block.get("text", "")))
        ]
        candidate_blocks = quality_blocks or context_blocks
        focus_terms = {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]{1,}", focus_text.lower())
            if token not in self.STOP_WORDS
        }
        if focus_terms:
            candidate_blocks = sorted(
                candidate_blocks,
                key=lambda block: (
                    len(focus_terms & set(block.get("terms", set()))),
                    len(str(block.get("text", ""))),
                ),
                reverse=True,
            )
        selected_blocks = candidate_blocks[:14]

        key_concepts = self._extract_key_phrases(selected_blocks, limit=10)

        summary_parts: list[str] = []
        seen_snippets: set[str] = set()
        for block in selected_blocks[:8]:
            snippet = self._summarize_block_text(block["text"], max_chars=220)
            snippet_key = snippet.lower()
            if not snippet or snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
            summary_parts.append(f"{snippet} {block['tag']}")

        summary = self._truncate_text(" ".join(summary_parts), 1700)
        if not summary:
            summary = self.NOT_AVAILABLE

        unit_wise_summaries: list[dict[str, Any]] = []
        for index, block in enumerate(selected_blocks[:10], start=1):
            snippet = self._summarize_block_text(block["text"], max_chars=260)
            if not snippet:
                continue
            unit_wise_summaries.append(
                {
                    "unit_title": f"Unit {index}",
                    "summary": self._truncate_text(f"{snippet} {block['tag']}", 520),
                    "key_points": [
                        self._truncate_text(f"{snippet} {block['tag']}", 260),
                    ],
                }
            )

        topic_wise_notes: list[dict[str, str]] = []
        for index, block in enumerate(selected_blocks[:10], start=1):
            topic_label = (
                key_concepts[index - 1]
                if index - 1 < len(key_concepts)
                else f"Topic {index}"
            )
            snippet = self._summarize_block_text(block["text"], max_chars=260)
            notes_text = (
                f"{snippet} {block['tag']}".strip() if snippet else self.NOT_AVAILABLE
            )
            topic_wise_notes.append(
                {
                    "topic": topic_label,
                    "notes": self._truncate_text(notes_text, 360),
                }
            )

        if not key_concepts:
            key_concepts = [
                item["topic"] for item in topic_wise_notes if item["topic"]
            ][:8]

        block_usage: dict[int, int] = {
            index: 0 for index in range(len(selected_blocks))
        }

        def pick_block_for_concept(concept: str) -> dict[str, Any]:
            concept_terms = {
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", concept.lower())
                if token not in self.STOP_WORDS
            }

            best_index: int | None = None
            best_score: tuple[int, int] | None = None

            for index, block in enumerate(selected_blocks):
                overlap = len(concept_terms & block["terms"]) if concept_terms else 0
                usage_penalty = block_usage[index]
                score = (overlap, -usage_penalty)
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index

            if best_index is None:
                best_index = min(block_usage, key=block_usage.get)

            block_usage[best_index] += 1
            return selected_blocks[best_index]

        flashcards: list[dict[str, str]] = []
        for index, concept in enumerate(key_concepts[:12]):
            block = pick_block_for_concept(concept)
            answer_snippet = self._summarize_block_text(block["text"], max_chars=210)
            answer_text = (
                f"{answer_snippet} {block['tag']}".strip()
                if answer_snippet
                else self.NOT_AVAILABLE
            )
            flashcards.append(
                {
                    "question": f"What is the role of {concept} in this document?",
                    "answer": self._truncate_text(answer_text, 300),
                    "bloom_level": self.BLOOM_LEVELS[index % len(self.BLOOM_LEVELS)],
                }
            )

        qa_sets: list[dict[str, str]] = []
        for index, concept in enumerate(key_concepts[:12], start=1):
            block = pick_block_for_concept(concept)
            answer_snippet = self._summarize_block_text(block["text"], max_chars=220)
            answer_text = (
                f"{answer_snippet} {block['tag']}".strip()
                if answer_snippet
                else self.NOT_AVAILABLE
            )
            qa_sets.append(
                {
                    "question": f"Q{index}: Explain {concept} with one key implication from the uploaded material.",
                    "answer": self._truncate_text(answer_text, 320),
                    "bloom_level": self.BLOOM_LEVELS[
                        (index - 1) % len(self.BLOOM_LEVELS)
                    ],
                }
            )

        viva_templates = [
            "How would you explain {concept} in your own words?",
            "Why is {concept} important in this document?",
            "What practical implication does {concept} have?",
            "How does {concept} connect to other units in the material?",
        ]
        viva_questions = [
            viva_templates[index % len(viva_templates)].format(concept=concept)
            for index, concept in enumerate(key_concepts[:10])
        ]
        viva_questions = self._dedupe_questions(viva_questions, limit=8)

        beginner_text = " ".join(
            note["notes"]
            for note in topic_wise_notes[:2]
            if note["notes"] != self.NOT_AVAILABLE
        )
        intermediate_text = " ".join(
            note["notes"]
            for note in topic_wise_notes[:4]
            if note["notes"] != self.NOT_AVAILABLE
        )
        advanced_text = " ".join(
            note["notes"]
            for note in topic_wise_notes[:6]
            if note["notes"] != self.NOT_AVAILABLE
        )

        difficulty_explanations = [
            {
                "mode": "beginner",
                "explanation": self._truncate_text(beginner_text or summary, 420)
                if (beginner_text or summary)
                else self.NOT_AVAILABLE,
            },
            {
                "mode": "intermediate",
                "explanation": self._truncate_text(intermediate_text or summary, 520)
                if (intermediate_text or summary)
                else self.NOT_AVAILABLE,
            },
            {
                "mode": "advanced",
                "explanation": self._truncate_text(advanced_text or summary, 620)
                if (advanced_text or summary)
                else self.NOT_AVAILABLE,
            },
        ]

        return {
            "summary_notes": summary,
            "unit_wise_summaries": unit_wise_summaries,
            "key_concepts": key_concepts,
            "topic_wise_notes": topic_wise_notes,
            "flashcards": flashcards,
            "qa_sets": qa_sets,
            "viva_questions": viva_questions,
            "difficulty_explanations": difficulty_explanations,
        }

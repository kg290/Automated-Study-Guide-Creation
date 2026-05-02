import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from threading import Lock
from typing import Any, Iterable

from google import genai
from google.genai import types

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


@dataclass
class ContextBlock:
    source: str
    chunk: str
    tag: str
    section_title: str
    text: str
    terms: set[str]


@dataclass
class SectionArtifact:
    unit_title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)
    topic_wise_notes: list[TopicNote] = field(default_factory=list)


class GeminiGenerationService:
    NOT_AVAILABLE = "Not available in uploaded material"
    BLOOM_LEVELS = ("remember", "understand", "apply", "analyze")
    MAX_RATE_LIMIT_RETRIES = 4
    CITATION_PATTERN = re.compile(
        r"\[source:(?P<source>[^\]|]+)\s*\|\s*chunk:(?P<chunk>[^\]]+)\]"
    )
    CONTEXT_BLOCK_PATTERN = re.compile(
        r"\[source:(?P<source>[^\]|]+)\s*\|\s*chunk:(?P<chunk>[^\]|]+?)(?:\s*\|\s*section:(?P<section>[^\]]+))?\]\s*(?P<text>.*?)(?=(?:\n\s*\[source:)|\Z)",
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
        "into",
        "than",
        "onto",
        "across",
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
        requested_outputs = self._normalize_output_types(options.output_types)
        context_blocks = self._parse_context_blocks(context)

        if not context_blocks:
            raise ValueError("No context blocks available for study-guide generation.")

        with self._REQUEST_LOCK:
            section_artifacts = self._build_section_artifacts(
                context_blocks=context_blocks,
                custom_prompt=options.custom_prompt,
            )
            result_payload = self._build_result_payload(
                section_artifacts=section_artifacts,
                requested_outputs=requested_outputs,
                options=options,
                context_blocks=context_blocks,
            )

        return StudyGuideResult(
            summary_notes=result_payload["summary_notes"],
            unit_wise_summaries=result_payload["unit_wise_summaries"],
            key_concepts=result_payload["key_concepts"],
            topic_wise_notes=result_payload["topic_wise_notes"],
            flashcards=result_payload["flashcards"],
            qa_sets=result_payload["qa_sets"],
            viva_questions=result_payload["viva_questions"],
            difficulty_explanations=result_payload["difficulty_explanations"],
            source_documents=source_documents,
        )

    def _build_result_payload(
        self,
        *,
        section_artifacts: list[SectionArtifact],
        requested_outputs: set[str],
        options: GenerationOptions,
        context_blocks: list[ContextBlock],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary_notes": "",
            "unit_wise_summaries": [],
            "key_concepts": [],
            "topic_wise_notes": [],
            "flashcards": [],
            "qa_sets": [],
            "viva_questions": [],
            "difficulty_explanations": [],
        }

        section_digest = self._build_section_digest(section_artifacts)
        concept_seed = self._aggregate_section_key_concepts(section_artifacts, limit=16)

        if "summary" in requested_outputs or "key_concepts" in requested_outputs:
            try:
                summary_text, summary_concepts = self._generate_summary_bundle(
                    section_digest=section_digest,
                    options=options,
                )
            except Exception as exc:
                logger.warning("Summary bundle generation failed: %s", exc)
                summary_text = self._fallback_summary_text(section_artifacts)
                summary_concepts = []
        else:
            summary_text = self._fallback_summary_text(section_artifacts)
            summary_concepts = []

        merged_concepts = self._dedupe_strings(
            [*summary_concepts, *concept_seed, *self._extract_key_phrases(context_blocks, 10)],
            limit=14,
        )

        if "summary" in requested_outputs:
            payload["summary_notes"] = self._ensure_summary_quality(
                summary_text=summary_text,
                section_artifacts=section_artifacts,
                context_blocks=context_blocks,
                custom_prompt=options.custom_prompt,
            )

        if "unit_summaries" in requested_outputs:
            payload["unit_wise_summaries"] = self._normalize_unit_summaries(section_artifacts)

        if "key_concepts" in requested_outputs:
            payload["key_concepts"] = merged_concepts

        if "topic_notes" in requested_outputs:
            if len(section_artifacts) <= 2:
                payload["topic_wise_notes"] = self._fallback_topic_notes(
                    section_artifacts=section_artifacts,
                    context_blocks=context_blocks,
                    key_concepts=merged_concepts,
                )
            else:
                try:
                    payload["topic_wise_notes"] = self._generate_topic_notes_bundle(
                        section_digest=section_digest,
                        options=options,
                        key_concepts=merged_concepts,
                    )
                except Exception as exc:
                    logger.warning("Topic note generation failed: %s", exc)
                    payload["topic_wise_notes"] = self._fallback_topic_notes(
                        section_artifacts=section_artifacts,
                        context_blocks=context_blocks,
                        key_concepts=merged_concepts,
                    )

        if "flashcards" in requested_outputs or "qa" in requested_outputs:
            generated_flashcards: list[Flashcard] = []
            generated_qa_sets: list[QAItem] = []
            try:
                generated_flashcards, generated_qa_sets = self._generate_learning_artifacts(
                    section_digest=section_digest,
                    options=options,
                    key_concepts=merged_concepts,
                )
            except Exception as exc:
                logger.warning("Learning artifact generation failed: %s", exc)

            if "flashcards" in requested_outputs:
                payload["flashcards"] = generated_flashcards or self._fallback_flashcards(
                    concepts=merged_concepts,
                    section_artifacts=section_artifacts,
                    context_blocks=context_blocks,
                    bloom_levels=options.bloom_levels,
                )

            if "qa" in requested_outputs:
                payload["qa_sets"] = generated_qa_sets or self._fallback_qa_sets(
                    concepts=merged_concepts,
                    section_artifacts=section_artifacts,
                    context_blocks=context_blocks,
                    bloom_levels=options.bloom_levels,
                )

        if "viva" in requested_outputs or "difficulty" in requested_outputs:
            try:
                viva_questions, difficulty_items = self._generate_viva_and_difficulty(
                    section_digest=section_digest,
                    options=options,
                    summary_text=payload["summary_notes"] or summary_text,
                )
            except Exception as exc:
                logger.warning("Viva/difficulty generation failed: %s", exc)
                viva_questions = self._fallback_viva_questions(merged_concepts)
                difficulty_items = self._fallback_difficulty_explanations(
                    summary_text=payload["summary_notes"] or summary_text,
                    options=options,
                )
            if "viva" in requested_outputs:
                payload["viva_questions"] = viva_questions
            if "difficulty" in requested_outputs:
                payload["difficulty_explanations"] = difficulty_items

        payload = self._augment_result_with_general_knowledge(
            payload,
            options=options,
            context_blocks=context_blocks,
        )
        payload = self._validate_and_repair_payload(
            payload,
            section_artifacts=section_artifacts,
            context_blocks=context_blocks,
            options=options,
        )
        payload = self._finalize_payload(payload, requested_outputs, options)
        return payload

    def _build_section_artifacts(
        self,
        *,
        context_blocks: list[ContextBlock],
        custom_prompt: str,
    ) -> list[SectionArtifact]:
        section_packs = self._build_section_packs(context_blocks)
        artifacts: list[SectionArtifact] = []

        for section_index, pack in enumerate(section_packs, start=1):
            try:
                artifact = self._generate_section_artifact(
                    section_blocks=pack,
                    section_index=section_index,
                    custom_prompt=custom_prompt,
                )
            except Exception as exc:
                logger.warning(
                    "Section artifact generation failed for section %d: %s",
                    section_index,
                    exc,
                )
                artifact = self._fallback_section_artifact(pack, section_index)
            artifacts.append(self._sanitize_section_artifact(artifact, pack, section_index))

        return self._merge_small_sections(artifacts)

    def _build_section_packs(
        self,
        context_blocks: list[ContextBlock],
    ) -> list[list[ContextBlock]]:
        context_blocks = self._preferred_study_blocks(context_blocks)
        if len(context_blocks) <= 4:
            return [context_blocks]

        packs: list[list[ContextBlock]] = []
        current: list[ContextBlock] = []
        current_source = context_blocks[0].source
        current_section = context_blocks[0].section_title
        target_sections = min(8, max(4, ceil(len(context_blocks) / 2)))
        group_size = max(2, ceil(len(context_blocks) / target_sections))

        for block in context_blocks:
            should_break = (
                current
                and (
                    block.source != current_source
                    or (
                        block.section_title
                        and current_section
                        and block.section_title != current_section
                        and len(current) >= 1
                    )
                    or len(current) >= group_size + 1
                )
            )
            if should_break:
                packs.append(current)
                current = []
                current_source = block.source
                current_section = block.section_title

            current.append(block)
            current_source = block.source
            current_section = block.section_title or current_section

        if current:
            packs.append(current)

        max_packs = 4 if len(context_blocks) <= 8 else 5
        return self._rebalance_section_packs(packs, max_packs=max_packs)

    def _rebalance_section_packs(
        self,
        packs: list[list[ContextBlock]],
        *,
        max_packs: int,
    ) -> list[list[ContextBlock]]:
        packs = [pack for pack in packs if pack]
        while len(packs) > max_packs:
            merge_index = 0
            smallest_size = None
            for index in range(len(packs) - 1):
                combined_size = sum(len(block.text) for block in packs[index]) + sum(
                    len(block.text) for block in packs[index + 1]
                )
                if smallest_size is None or combined_size < smallest_size:
                    smallest_size = combined_size
                    merge_index = index
            packs[merge_index] = [*packs[merge_index], *packs[merge_index + 1]]
            del packs[merge_index + 1]
        return packs

    def _generate_section_artifact(
        self,
        *,
        section_blocks: list[ContextBlock],
        section_index: int,
        custom_prompt: str,
    ) -> SectionArtifact:
        section_context = self._blocks_to_context(section_blocks, max_chars=4400)
        suggested_title = self._guess_section_title(section_blocks, section_index)
        focus_line = (
            f'Explicitly address this focus if it is relevant to the section: "{custom_prompt.strip()}".'
            if custom_prompt.strip()
            else "Focus on the section's main teaching content."
        )
        system_prompt = (
            "You are building a clean study artifact for one section of academic material. "
            "Use the section text as the backbone. "
            "If the source is terse, add brief textbook-style clarification, but do not invent document-specific claims. "
            "Return only the requested XML-like tags, with no markdown fences."
        )
        user_prompt = f"""
Suggested section label:
{suggested_title}

Section context:
{section_context}

Rules:
- Keep the original teaching order and main terminology.
- Prefer real concept names over generic labels.
- Summary should be 120 to 220 words and genuinely teachable.
- Key points should be concise and non-repetitive.
- Topic notes should each explain one focused idea.
- Ignore raw code fragments, syntax snippets, and implementation boilerplate unless they are directly necessary to explain the concept.
- Keep questions, formulas, and claims grounded in the section.
- {focus_line}

Return exactly this structure:
<artifact>
<unit_title>...</unit_title>
<summary>...</summary>
<key_points>
<item>...</item>
<item>...</item>
</key_points>
<key_concepts>
<item>...</item>
<item>...</item>
</key_concepts>
<topic_notes>
<topic_note>
<topic>...</topic>
<notes>...</notes>
</topic_note>
</topic_notes>
</artifact>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2200,
            temperature=0.18,
            top_p=0.9,
        )
        return self._parse_section_artifact(response_text)

    def _generate_summary_bundle(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
    ) -> tuple[str, list[str]]:
        focus_line = (
            f'Explicitly include this focus when it matters: "{options.custom_prompt.strip()}".'
            if options.custom_prompt.strip()
            else "Cover the whole document coherently from beginning to end."
        )
        system_prompt = (
            "You are writing a polished study-guide summary from section digests. "
            "Write clearly enough that a student can learn the material from the summary alone. "
            "You may add brief general academic clarification where the notes are terse. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Rules:
- Write 420 to 760 words.
- Keep the document order.
- Explain the major mechanisms, contrasts, stages, and important terminology.
- Avoid vague filler and avoid repeating the same sentence pattern.
- {focus_line}

Return exactly this structure:
<summary_bundle>
<summary_notes>...</summary_notes>
<key_concepts>
<item>...</item>
<item>...</item>
</key_concepts>
</summary_bundle>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3200,
            temperature=0.24,
            top_p=0.92,
        )
        summary_notes = self._extract_tag_text(response_text, "summary_notes")
        key_concepts = self._extract_item_list(response_text, "key_concepts")
        if not summary_notes:
            fallback_summary = self._extract_plain_summary_from_response(response_text)
            if len(self._strip_citations(fallback_summary)) >= 320:
                summary_notes = fallback_summary
            else:
                raise RuntimeError("Summary bundle did not contain summary_notes.")
        if not key_concepts:
            key_concepts = self._derive_key_concepts_from_text(summary_notes or section_digest, limit=10)
        return summary_notes, key_concepts

    def _generate_topic_notes_bundle(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
        key_concepts: list[str],
    ) -> list[TopicNote]:
        concept_hint = ", ".join(key_concepts[:12]) if key_concepts else "Use the most important document concepts."
        focus_line = (
            f'Pay special attention to this focus where relevant: "{options.custom_prompt.strip()}".'
            if options.custom_prompt.strip()
            else "Cover the most teachable topics from the document."
        )
        system_prompt = (
            "You are creating topic-wise study notes from structured section digests. "
            "Each topic should read like a mini study note, not copied bullet fragments. "
            "Write concise, explanatory notes that help a student understand the topic for revision. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Key concept hints:
{concept_hint}

Rules:
- Generate 3 to 6 topic notes.
- Each topic title should be 2 to 6 words and sound like a real study topic.
- Each note should be around 60 to 130 words.
- Explain what the topic is, why it matters, and how it works when applicable.
- Avoid raw code, list fragments, table-like text, or formula dumps.
- {focus_line}

Return exactly this structure:
<topic_notes>
<topic_note>
<topic>...</topic>
<notes>...</notes>
</topic_note>
</topic_notes>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2200,
            temperature=0.24,
            top_p=0.92,
        )
        topic_notes = self._parse_topic_notes(response_text)
        if not topic_notes:
            raise RuntimeError("Topic note generation produced no notes.")
        return topic_notes

    def _generate_flashcards(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
        key_concepts: list[str],
    ) -> list[Flashcard]:
        concept_hint = ", ".join(key_concepts[:10]) if key_concepts else "Use the most important concepts."
        system_prompt = (
            "You are creating flashcards for revision. "
            "Flashcards should feel like compact concept-detail revision cards. "
            "They are not exam-style questions and they are not essay prompts. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Key concept hints:
{concept_hint}

Rules:
- Generate 8 to 12 flashcards.
- Use the <question> tag as a concise front-of-card title because of the existing schema.
- Each title should be 2 to 8 words and should NOT be phrased as a question.
- Titles should directly name the concept, property, mechanism, standard, stage, or process.
- Each answer should be around 35 to 95 words.
- Each card should focus on one idea only.
- Each answer should give useful revision detail, not just a dictionary-style fragment.
- Avoid compare/explain essay prompts, question-mark phrasing, and multi-part prompts.
- Use bloom levels only from: {", ".join(options.bloom_levels)}.

Return exactly this structure:
<flashcards>
<card>
<question>...</question>
<answer>...</answer>
<bloom_level>remember</bloom_level>
</card>
</flashcards>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2600,
            temperature=0.16,
            top_p=0.88,
        )
        cards = self._parse_flashcards(response_text, options.bloom_levels)
        if not cards:
            raise RuntimeError("Flashcard generation produced no cards.")
        return cards

    def _generate_learning_artifacts(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
        key_concepts: list[str],
    ) -> tuple[list[Flashcard], list[QAItem]]:
        concept_hint = ", ".join(key_concepts[:12]) if key_concepts else "Use the most important concepts."
        system_prompt = (
            "You are creating two different kinds of study artifacts from the same academic material. "
            "Flashcards are for quick concept review and compact recall-ready explanations. "
            "Q&A items are for deeper explanation and conceptual understanding. "
            "Do not make them feel the same. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Key concept hints:
{concept_hint}

Rules for flashcards:
- Generate 6 to 10 flashcards.
- Use the <question> tag as a compact flashcard title because of the existing schema.
- Each flashcard title should be 2 to 8 words and should NOT be written as a question.
- The title should name the concept, process, standard, property, or mechanism directly.
- Each answer should be around 45 to 110 words.
- The answer should give a compact but informative explanation for revision.
- The flashcard should work like a concept-detail card, not like a hidden exam question.
- Avoid compare/explain essay prompts, multi-part prompts, and question-mark phrasing.
- Flashcards should feel like concept cards for active recall and spaced repetition.

Rules for Q&A:
- Generate 5 to 7 Q&A items.
- These are not flashcards.
- Prefer How, Why, Explain, Compare, Distinguish, or What is the difference between.
- Each answer should be around 80 to 170 words.
- Each answer should teach the concept clearly enough for revision.
- Q&A should focus on mechanism, contrast, reasoning, or application.
- Do not reuse or lightly paraphrase the flashcard prompts.

Use bloom levels only from: {", ".join(options.bloom_levels)}.

Return exactly this structure:
<learning_artifacts>
<flashcards>
<card>
<question>...</question>
<answer>...</answer>
<bloom_level>remember</bloom_level>
</card>
</flashcards>
<qa_sets>
<qa_item>
<question>...</question>
<answer>...</answer>
<bloom_level>understand</bloom_level>
</qa_item>
</qa_sets>
</learning_artifacts>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3800,
            temperature=0.2,
            top_p=0.9,
        )
        cards = self._parse_flashcards(response_text, options.bloom_levels)
        qa_items = self._parse_qa_sets(response_text, options.bloom_levels)
        if not cards and not qa_items:
            raise RuntimeError("Learning artifact generation produced no usable flashcards or Q&A.")
        return cards, qa_items

    def _generate_qa_sets(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
        flashcards: list[Flashcard],
        key_concepts: list[str],
    ) -> list[QAItem]:
        flashcard_questions = "\n".join(f"- {item.question}" for item in flashcards[:8])
        concept_hint = ", ".join(key_concepts[:10]) if key_concepts else "Use the most important concepts."
        system_prompt = (
            "You are creating study Q&A for deeper understanding. "
            "These are not flashcards. Questions should test explanation, mechanism, reasoning, contrast, or application. "
            "Answers should be full enough to revise from. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Key concept hints:
{concept_hint}

Do not reuse or lightly paraphrase these flashcard prompts:
{flashcard_questions or "- None"}

Rules:
- Generate 6 to 9 Q&A items.
- Prefer How, Why, Explain, Compare, Distinguish, or What is the difference between.
- Answers should be around 90 to 180 words.
- Each answer should teach the concept clearly, even if the source notes are terse.
- Use bloom levels only from: {", ".join(options.bloom_levels)}.

Return exactly this structure:
<qa_sets>
<qa_item>
<question>...</question>
<answer>...</answer>
<bloom_level>understand</bloom_level>
</qa_item>
</qa_sets>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3400,
            temperature=0.24,
            top_p=0.92,
        )
        qa_items = self._parse_qa_sets(response_text, options.bloom_levels)
        if not qa_items:
            raise RuntimeError("Q&A generation produced no items.")
        return qa_items

    def _generate_viva_and_difficulty(
        self,
        *,
        section_digest: str,
        options: GenerationOptions,
        summary_text: str,
    ) -> tuple[list[str], list[DifficultyExplanation]]:
        requested_modes = ", ".join(options.difficulty_modes)
        system_prompt = (
            "You are generating viva prompts and layered explanations from study material. "
            "Return only the requested XML-like tags."
        )
        user_prompt = f"""
Grounded section digest:
{section_digest}

Summary backbone:
{self._truncate_text(summary_text, 2200)}

Rules:
- Viva questions should test explanation and oral reasoning.
- Difficulty explanations should reframe the same material for the requested learner levels.
- Requested modes: {requested_modes}.

Return exactly this structure:
<study_extras>
<viva_questions>
<item>...</item>
</viva_questions>
<difficulty_explanations>
<difficulty>
<mode>beginner</mode>
<explanation>...</explanation>
</difficulty>
</difficulty_explanations>
</study_extras>
""".strip()

        response_text = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2600,
            temperature=0.22,
            top_p=0.9,
        )
        viva_questions = self._extract_item_list(response_text, "viva_questions")
        difficulty_items = self._parse_difficulty_explanations(
            response_text,
            options.difficulty_modes,
        )
        if not difficulty_items:
            difficulty_items = self._fallback_difficulty_explanations(
                summary_text=summary_text,
                options=options,
            )
        return viva_questions[:8], difficulty_items

    def _generate_text_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        candidate_models: list[str] = []
        for model_name in [self.model_name, *self.fallback_models]:
            cleaned = model_name.strip()
            if cleaned and cleaned not in candidate_models:
                candidate_models.append(cleaned)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

        automatic_function_calling_cls = getattr(
            types, "AutomaticFunctionCallingConfig", None
        )
        if automatic_function_calling_cls is not None:
            config.automatic_function_calling = automatic_function_calling_cls(
                disable=True
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
                    text = self._strip_code_fences(self._response_to_text(response))
                    if text.strip():
                        return text.strip()
                    last_error = RuntimeError("Gemini returned an empty text response.")
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc).lower()
                    if "429" in error_text or "resource_exhausted" in error_text:
                        wait_seconds = min(18.0, (2**attempt) + random.uniform(0.2, 0.9))
                        logger.warning(
                            "Gemini rate-limited on model %s. Retrying in %.2fs (attempt %d/%d).",
                            model_name,
                            wait_seconds,
                            attempt + 1,
                            self.MAX_RATE_LIMIT_RETRIES,
                        )
                        time.sleep(wait_seconds)
                        continue
                    if attempt < self.MAX_RATE_LIMIT_RETRIES - 1:
                        time.sleep(min(6.0, 1.2 + attempt))
                        continue
                    break
                break

        raise RuntimeError(f"Gemini generation unavailable after retries: {last_error}")

    def _response_to_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        fragments: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    fragments.append(str(part_text))

        return "\n".join(fragments).strip()

    def _parse_section_artifact(self, text: str) -> SectionArtifact:
        body = self._extract_tag_text(text, "artifact") or text
        unit_title = self._extract_tag_text(body, "unit_title")
        summary = self._extract_tag_text(body, "summary")
        key_points_block = self._extract_tag_text(body, "key_points")
        concepts_block = self._extract_tag_text(body, "key_concepts")
        topic_notes_block = self._extract_tag_text(body, "topic_notes")

        key_points = self._extract_inline_items(key_points_block)
        key_concepts = self._extract_inline_items(concepts_block)
        topic_notes = self._parse_topic_notes(topic_notes_block)

        return SectionArtifact(
            unit_title=self._sanitize_label(unit_title) or "Study Section",
            summary=self._normalize_generated_text(self._truncate_text(self._prepare_block_text(summary), 1400)),
            key_points=key_points,
            key_concepts=key_concepts,
            topic_wise_notes=topic_notes,
        )

    def _parse_flashcards(
        self,
        text: str,
        allowed_bloom_levels: list[str],
    ) -> list[Flashcard]:
        cards: list[Flashcard] = []
        normalized_levels = self._normalize_bloom_levels(allowed_bloom_levels)

        for match in re.finditer(r"<card>\s*(.*?)\s*</card>", text, re.DOTALL | re.IGNORECASE):
            block = match.group(1)
            question = self._clean_text(self._extract_tag_text(block, "question"))
            answer = self._clean_text(self._extract_tag_text(block, "answer"))
            bloom_level = self._clean_text(self._extract_tag_text(block, "bloom_level")).lower()
            if not question or not answer:
                continue
            if bloom_level not in normalized_levels:
                bloom_level = normalized_levels[len(cards) % len(normalized_levels)]
            cards.append(
                Flashcard(
                    question=self._repair_flashcard_title(question),
                    answer=self._normalize_generated_text(self._truncate_text(answer, 520)),
                    bloom_level=bloom_level,
                )
            )

        return cards

    def _parse_qa_sets(
        self,
        text: str,
        allowed_bloom_levels: list[str],
    ) -> list[QAItem]:
        items: list[QAItem] = []
        normalized_levels = self._normalize_bloom_levels(allowed_bloom_levels)

        for match in re.finditer(r"<qa_item>\s*(.*?)\s*</qa_item>", text, re.DOTALL | re.IGNORECASE):
            block = match.group(1)
            question = self._clean_text(self._extract_tag_text(block, "question"))
            answer = self._clean_text(self._extract_tag_text(block, "answer"))
            bloom_level = self._clean_text(self._extract_tag_text(block, "bloom_level")).lower()
            if not question or not answer:
                continue
            if bloom_level not in normalized_levels:
                bloom_level = normalized_levels[len(items) % len(normalized_levels)]
            items.append(
                QAItem(
                    question=self._repair_question(question, "qa"),
                    answer=self._normalize_generated_text(self._truncate_text(answer, 1400)),
                    bloom_level=bloom_level,
                )
            )

        return items

    def _parse_difficulty_explanations(
        self,
        text: str,
        requested_modes: list[str],
    ) -> list[DifficultyExplanation]:
        requested = {
            mode.strip().lower()
            for mode in requested_modes
            if mode and mode.strip()
        } or {"beginner", "intermediate", "advanced"}
        items: list[DifficultyExplanation] = []

        for match in re.finditer(
            r"<difficulty>\s*(.*?)\s*</difficulty>",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            block = match.group(1)
            mode = self._clean_text(self._extract_tag_text(block, "mode")).lower()
            explanation = self._clean_text(self._extract_tag_text(block, "explanation"))
            if mode not in requested or not explanation:
                continue
            items.append(
                DifficultyExplanation(
                    mode=mode,
                    explanation=self._normalize_generated_text(self._truncate_text(explanation, 1200)),
                )
            )

        return items

    def _parse_topic_notes(self, text: str) -> list[TopicNote]:
        notes: list[TopicNote] = []
        for match in re.finditer(
            r"<topic_note>\s*(.*?)\s*</topic_note>",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            block = match.group(1)
            topic = self._clean_text(self._extract_tag_text(block, "topic"))
            note_text = self._prepare_block_text(self._extract_tag_text(block, "notes"))
            title = self._sanitize_label(topic)
            if self._looks_like_noise_title(title):
                title = self._derive_topic_title_from_text(note_text)
            if len(title.split()) == 1 and not title.isupper():
                title = self._derive_topic_title_from_text(note_text)
            if note_text:
                notes.append(
                    TopicNote(
                        topic=title or "Core Topic",
                        notes=self._normalize_generated_text(self._truncate_text(note_text, 900)),
                    )
                )
        return notes

    def _extract_tag_text(self, text: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return self._clean_text(match.group(1)) if match else ""

    def _extract_item_list(self, text: str, tag: str) -> list[str]:
        return self._extract_inline_items(self._extract_tag_text(text, tag))

    def _extract_inline_items(self, text: str) -> list[str]:
        item_matches = re.findall(r"<item>\s*(.*?)\s*</item>", text, re.DOTALL | re.IGNORECASE)
        if item_matches:
            return self._dedupe_strings(
                [self._clean_text(item) for item in item_matches],
                limit=20,
            )

        values: list[str] = []
        for line in re.split(r"[\r\n]+", text):
            cleaned = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
            if cleaned:
                values.append(self._clean_text(cleaned))
        return self._dedupe_strings(values, limit=20)

    def _extract_plain_summary_from_response(self, text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(
            r"<key_concepts>.*?</key_concepts>",
            " ",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"</?summary_bundle>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?summary_notes>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?item>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        prepared = self._prepare_block_text(cleaned)
        sentences = [
            self._clean_text(sentence)
            for sentence in re.split(r"(?<=[.!?])\s+", prepared)
            if self._clean_text(sentence)
        ]
        if not sentences:
            return self._normalize_generated_text(prepared)

        kept: list[str] = []
        for sentence in sentences:
            if self._looks_like_code_like_content(sentence):
                continue
            if self._looks_like_tabular_noise(sentence):
                continue
            if len(sentence.split()) < 7:
                continue
            kept.append(sentence)
        summary = " ".join(kept[:10]) if kept else prepared
        return self._normalize_generated_text(self._truncate_summary_text(summary, 5200))

    def _derive_key_concepts_from_text(self, text: str, limit: int) -> list[str]:
        prepared = self._prepare_block_text(text)
        concepts: list[str] = []

        title_phrases = re.findall(
            r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,3}",
            prepared,
        )
        for phrase in title_phrases:
            label = self._sanitize_label(phrase)
            if label and not self._looks_like_noise_title(label):
                concepts.append(label)

        counter: Counter[str] = Counter()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", prepared):
            lowered = token.lower()
            if lowered in self.STOP_WORDS or self._is_noise_token(lowered):
                continue
            counter[lowered] += 1

        for token, _count in counter.most_common(limit * 2):
            label = self._sanitize_label(token.title())
            if label and not self._looks_like_noise_title(label):
                concepts.append(label)

        return self._dedupe_strings(concepts, limit=limit)

    def _derive_topic_title_from_text(self, text: str) -> str:
        concepts = self._derive_key_concepts_from_text(text, limit=4)
        for concept in concepts:
            label = self._sanitize_label(concept)
            if label and not self._looks_like_noise_title(label):
                return label

        prepared = self._prepare_block_text(text)
        phrase_match = re.search(
            r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,3})\b",
            prepared,
        )
        if phrase_match:
            label = self._sanitize_label(phrase_match.group(1))
            if label and not self._looks_like_noise_title(label):
                return label
        return "Core Topic"

    def _fallback_section_artifact(
        self,
        blocks: list[ContextBlock],
        section_index: int,
    ) -> SectionArtifact:
        title = self._guess_section_title(blocks, section_index)
        snippets = [self._summarize_block_text(block.text, 280) for block in blocks[:3]]
        summary = " ".join(snippet for snippet in snippets if snippet).strip()
        concepts = self._extract_key_phrases(blocks, 7)
        key_points = []
        for block in blocks[:4]:
            sentence = self._summarize_block_text(block.text, 220)
            if sentence:
                key_points.append(sentence)
        topic_notes = [
            TopicNote(
                topic=concept,
                notes=self._summarize_block_text(blocks[min(index, len(blocks) - 1)].text, 260),
            )
            for index, concept in enumerate(concepts[:3])
            if blocks
        ]
        return SectionArtifact(
            unit_title=title,
            summary=summary or title,
            key_points=self._dedupe_strings(key_points, limit=5),
            key_concepts=concepts,
            topic_wise_notes=topic_notes,
        )

    def _sanitize_section_artifact(
        self,
        artifact: SectionArtifact,
        blocks: list[ContextBlock],
        section_index: int,
    ) -> SectionArtifact:
        artifact.unit_title = self._truncate_text(
            self._sanitize_label(
                artifact.unit_title or self._guess_section_title(blocks, section_index)
            ),
            140,
        )
        artifact.summary = self._truncate_summary_text(
            self._normalize_generated_text(
                artifact.summary or self._fallback_section_artifact(blocks, section_index).summary
            ),
            1600,
        )
        artifact.key_points = self._dedupe_strings(
            [
                self._normalize_generated_text(self._truncate_text(item, 420))
                for item in artifact.key_points
                if self._clean_text(item)
            ],
            limit=5,
        )
        if not artifact.key_points:
            artifact.key_points = self._fallback_section_artifact(blocks, section_index).key_points[:3]

        artifact.key_concepts = self._dedupe_strings(
            [
                self._sanitize_label(self._truncate_text(item, 100))
                for item in artifact.key_concepts
                if self._clean_text(item)
            ],
            limit=8,
        )
        if not artifact.key_concepts:
            artifact.key_concepts = self._extract_key_phrases(blocks, 6)

        cleaned_topic_notes: list[TopicNote] = []
        for note in artifact.topic_wise_notes:
            topic = self._sanitize_label(self._truncate_text(note.topic, 120))
            notes = self._normalize_generated_text(self._truncate_text(note.notes, 900))
            if topic and notes and not self._looks_like_noise_title(topic):
                cleaned_topic_notes.append(TopicNote(topic=topic, notes=notes))
        if not cleaned_topic_notes:
            cleaned_topic_notes = self._fallback_section_artifact(blocks, section_index).topic_wise_notes
        artifact.topic_wise_notes = cleaned_topic_notes[:4]
        return artifact

    def _merge_small_sections(
        self,
        artifacts: list[SectionArtifact],
    ) -> list[SectionArtifact]:
        if len(artifacts) <= 1:
            return artifacts

        merged: list[SectionArtifact] = []
        buffer_item: SectionArtifact | None = None

        for artifact in artifacts:
            if buffer_item is None:
                buffer_item = artifact
                continue

            if len(self._strip_citations(buffer_item.summary)) < 160:
                buffer_item = SectionArtifact(
                    unit_title=buffer_item.unit_title,
                    summary=self._truncate_summary_text(
                        f"{buffer_item.summary} {artifact.summary}",
                        1800,
                    ),
                    key_points=self._dedupe_strings(
                        [*buffer_item.key_points, *artifact.key_points],
                        limit=6,
                    ),
                    key_concepts=self._dedupe_strings(
                        [*buffer_item.key_concepts, *artifact.key_concepts],
                        limit=10,
                    ),
                    topic_wise_notes=[*buffer_item.topic_wise_notes, *artifact.topic_wise_notes][:5],
                )
                merged.append(buffer_item)
                buffer_item = None
            else:
                merged.append(buffer_item)
                buffer_item = artifact

        if buffer_item is not None:
            merged.append(buffer_item)

        return merged

    def _normalize_unit_summaries(
        self,
        section_artifacts: list[SectionArtifact],
    ) -> list[UnitSummary]:
        return [
            UnitSummary(
                unit_title=self._truncate_text(item.unit_title, 140),
                summary=self._truncate_summary_text(item.summary, 1200),
                key_points=self._dedupe_strings(
                    [self._truncate_text(point, 420) for point in item.key_points],
                    limit=5,
                ),
            )
            for item in section_artifacts
            if item.summary
        ][:8]

    def _normalize_topic_notes(
        self,
        section_artifacts: list[SectionArtifact],
    ) -> list[TopicNote]:
        notes: list[TopicNote] = []
        seen: set[str] = set()
        for artifact in section_artifacts:
            for item in artifact.topic_wise_notes:
                key = self._normalize_question_key(item.topic)
                if not key or key in seen:
                    continue
                seen.add(key)
                notes.append(
                    TopicNote(
                        topic=self._truncate_text(item.topic, 120),
                        notes=self._truncate_text(item.notes, 900),
                    )
                )
                if len(notes) >= 12:
                    return notes
        return notes

    def _fallback_topic_notes(
        self,
        *,
        section_artifacts: list[SectionArtifact],
        context_blocks: list[ContextBlock],
        key_concepts: list[str],
    ) -> list[TopicNote]:
        notes: list[TopicNote] = []
        seen: set[str] = set()

        topic_candidates = self._dedupe_strings(
            [
                *key_concepts,
                *[artifact.unit_title for artifact in section_artifacts],
                *[
                    item.topic
                    for artifact in section_artifacts
                    for item in artifact.topic_wise_notes
                ],
            ],
            limit=16,
        )

        for candidate in topic_candidates:
            topic = self._sanitize_label(candidate)
            if not topic or self._looks_like_noise_title(topic):
                continue
            if len(topic.split()) == 1 and not topic.isupper():
                continue
            key = self._normalize_question_key(topic)
            if not key or key in seen:
                continue

            snippet = self._build_plain_context_snippet_for_query(
                query=topic,
                context_blocks=context_blocks,
                limit=2,
                max_chars=900,
            )
            supporting_summary = self._find_best_supporting_summary(topic, section_artifacts)
            prepared = self._prepare_block_text(supporting_summary or snippet)
            note_text = self._normalize_generated_text(
                self._truncate_text(self._summarize_block_text(prepared, 420), 420)
            )
            if not note_text or self._looks_like_low_quality_study_text(note_text):
                continue

            seen.add(key)
            notes.append(TopicNote(topic=topic, notes=note_text))
            if len(notes) >= 8:
                return notes

        if notes:
            return notes
        return self._normalize_topic_notes(section_artifacts)

    def _find_best_supporting_summary(
        self,
        topic: str,
        section_artifacts: list[SectionArtifact],
    ) -> str:
        topic_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", topic)
            if token.lower() not in self.STOP_WORDS
        }
        best_summary = ""
        best_score = -1
        first_summary = ""
        for artifact in section_artifacts:
            summary = self._prepare_block_text(artifact.summary)
            if not summary:
                continue
            if not first_summary:
                first_summary = summary
            summary_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", summary)
                if token.lower() not in self.STOP_WORDS
            }
            score = len(topic_terms & summary_terms)
            if artifact.unit_title and topic.lower() in artifact.unit_title.lower():
                score += 2
            if score > best_score:
                best_score = score
                best_summary = summary
        return best_summary or first_summary

    def _build_section_digest(
        self,
        section_artifacts: list[SectionArtifact],
    ) -> str:
        lines: list[str] = []
        for index, artifact in enumerate(section_artifacts, start=1):
            lines.append(f"Section {index}: {artifact.unit_title}")
            lines.append(f"Summary: {self._truncate_text(artifact.summary, 760)}")
            if artifact.key_points:
                lines.append("Key points:")
                lines.extend(f"- {self._truncate_text(point, 260)}" for point in artifact.key_points[:4])
            if artifact.key_concepts:
                lines.append("Key concepts: " + ", ".join(artifact.key_concepts[:8]))
            for note in artifact.topic_wise_notes[:2]:
                lines.append(
                    f"Topic note - {note.topic}: {self._truncate_text(note.notes, 240)}"
                )
            lines.append("")
        return "\n".join(lines).strip()

    def _aggregate_section_key_concepts(
        self,
        section_artifacts: list[SectionArtifact],
        limit: int,
    ) -> list[str]:
        values: list[str] = []
        for artifact in section_artifacts:
            values.extend(artifact.key_concepts)
            values.append(artifact.unit_title)
            for note in artifact.topic_wise_notes:
                values.append(note.topic)
        return self._dedupe_strings(values, limit=limit)

    def _fallback_summary_text(self, section_artifacts: list[SectionArtifact]) -> str:
        sections = []
        for artifact in section_artifacts:
            part = self._truncate_text(artifact.summary, 520)
            if part:
                sections.append(part)
        return self._truncate_summary_text(" ".join(sections), 4200)

    def _fallback_flashcards(
        self,
        *,
        concepts: list[str],
        section_artifacts: list[SectionArtifact],
        context_blocks: list[ContextBlock],
        bloom_levels: list[str],
    ) -> list[Flashcard]:
        cards: list[Flashcard] = []
        normalized_levels = self._normalize_bloom_levels(bloom_levels)
        concept_pool = concepts or self._extract_key_phrases(context_blocks, 10)
        summaries = [artifact.summary for artifact in section_artifacts if artifact.summary]

        for index, concept in enumerate(concept_pool[:10]):
            answer_seed = summaries[index % len(summaries)] if summaries else concept
            cards.append(
                Flashcard(
                    question=self._repair_flashcard_title(self._sanitize_label(concept)),
                    answer=self._truncate_text(self._summarize_block_text(self._prepare_block_text(answer_seed), 320), 420),
                    bloom_level=normalized_levels[index % len(normalized_levels)],
                )
            )
        return cards

    def _fallback_qa_sets(
        self,
        *,
        concepts: list[str],
        section_artifacts: list[SectionArtifact],
        context_blocks: list[ContextBlock],
        bloom_levels: list[str],
    ) -> list[QAItem]:
        qa_items: list[QAItem] = []
        normalized_levels = self._normalize_bloom_levels(bloom_levels)
        concept_pool = concepts or self._extract_key_phrases(context_blocks, 10)

        for index, concept in enumerate(concept_pool[:8]):
            supporting_summary = self._find_best_supporting_summary(concept, section_artifacts)
            snippet = self._build_plain_context_snippet_for_query(
                query=concept,
                context_blocks=context_blocks,
                limit=2,
                max_chars=900,
            )
            artifact = section_artifacts[index % len(section_artifacts)] if section_artifacts else None
            answer_seed = self._prepare_block_text(supporting_summary or snippet) or (artifact.summary if artifact else concept)
            answer_text = self._truncate_text(self._summarize_block_text(answer_seed, 720), 1000)
            question = self._build_fallback_qa_question(concept, index)
            qa_items.append(
                QAItem(
                    question=self._repair_question(question, "qa"),
                    answer=answer_text,
                    bloom_level=normalized_levels[index % len(normalized_levels)],
                )
            )
        return qa_items

    def _build_fallback_qa_question(self, concept: str, index: int) -> str:
        cleaned = self._sanitize_label(concept) or "this concept"
        lowered = cleaned.lower()
        if " vs " in lowered or " versus " in lowered:
            return f"Compare and contrast {cleaned}."
        templates = [
            f"Why is {cleaned} important in this material?",
            f"How does {cleaned} work?",
            f"Explain the role of {cleaned}.",
            f"How is {cleaned} used in practice?",
        ]
        return templates[index % len(templates)]

    def _repair_flashcard_title(self, text: str) -> str:
        cleaned = self._sanitize_label(text)
        cleaned = re.sub(
            r"^(what|which|why|how|when|where|define|state|identify|name|explain|compare|distinguish)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(is|are|was|were|does|do|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(importance|role|purpose|function|advantage|difference)\s+of\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip(" ?!;:.,")
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8]).strip(" -:;,")
        if not cleaned:
            return "Core Concept"
        title_words: list[str] = []
        for word in cleaned.split():
            if re.fullmatch(r"[A-Z0-9\-]{2,}", word):
                title_words.append(word)
            else:
                title_words.append(word[:1].upper() + word[1:])
        return " ".join(title_words)

    def _fallback_viva_questions(self, concepts: list[str]) -> list[str]:
        return [
            f"Explain {concept} in your own words."
            for concept in concepts[:6]
        ]

    def _fallback_difficulty_explanations(
        self,
        *,
        summary_text: str,
        options: GenerationOptions,
    ) -> list[DifficultyExplanation]:
        plain = self._strip_citations(summary_text) or "The uploaded material covers the core topic and its main ideas."
        explanations: dict[str, str] = {
            "beginner": self._truncate_text(plain, 480),
            "intermediate": self._truncate_text(plain, 720),
            "advanced": self._truncate_text(plain, 980),
        }
        results: list[DifficultyExplanation] = []
        for mode in options.difficulty_modes:
            cleaned = mode.strip().lower()
            if cleaned in explanations:
                results.append(
                    DifficultyExplanation(
                        mode=cleaned,
                        explanation=explanations[cleaned],
                    )
                )
        return results

    def _augment_result_with_general_knowledge(
        self,
        result: dict[str, Any],
        *,
        options: GenerationOptions,
        context_blocks: list[ContextBlock],
    ) -> dict[str, Any]:
        summary_text = str(result.get("summary_notes", "") or "").strip()
        if "summary" in self._normalize_output_types(options.output_types) and self._needs_summary_augmentation(summary_text):
            try:
                result["summary_notes"] = self._augment_summary_text(
                    result=result,
                    context_blocks=context_blocks,
                    custom_prompt=options.custom_prompt,
                )
            except Exception as exc:
                logger.warning("Summary augmentation failed: %s", exc)

        flashcards = [item for item in result.get("flashcards", []) if isinstance(item, Flashcard)]
        for item in flashcards:
            if not self._needs_answer_augmentation(item.answer):
                continue
            try:
                item.answer = self._augment_study_answer(
                    question=item.question,
                    current_answer=item.answer,
                    context_blocks=context_blocks,
                    style="flashcard",
                )
            except Exception as exc:
                logger.warning("Flashcard augmentation failed for '%s': %s", item.question, exc)
        result["flashcards"] = flashcards

        qa_items = [item for item in result.get("qa_sets", []) if isinstance(item, QAItem)]
        for item in qa_items:
            if not self._needs_answer_augmentation(item.answer):
                continue
            try:
                item.answer = self._augment_study_answer(
                    question=item.question,
                    current_answer=item.answer,
                    context_blocks=context_blocks,
                    style="qa",
                )
            except Exception as exc:
                logger.warning("Q&A augmentation failed for '%s': %s", item.question, exc)
        result["qa_sets"] = qa_items
        return result

    def _augment_summary_text(
        self,
        *,
        result: dict[str, Any],
        context_blocks: list[ContextBlock],
        custom_prompt: str,
    ) -> str:
        unit_lines = [
            f"- {self._sanitize_label(item.unit_title)}: {self._truncate_text(item.summary, 260)}"
            for item in result.get("unit_wise_summaries", [])
            if isinstance(item, UnitSummary)
        ]
        topic_lines = [
            f"- {self._sanitize_label(item.topic)}: {self._truncate_text(item.notes, 180)}"
            for item in result.get("topic_wise_notes", [])
            if isinstance(item, TopicNote)
        ]
        context_snippet = self._build_context_snippet_for_query(
            query=custom_prompt or "overall document understanding",
            context_blocks=context_blocks,
            limit=5,
            max_chars=2400,
        )
        focus_line = (
            f'Explicitly include this focus where relevant: "{custom_prompt.strip()}".'
            if custom_prompt.strip()
            else "Cover the whole material coherently."
        )

        system_prompt = (
            "You are strengthening an academic study summary. "
            "Use the document evidence first. "
            "When the document is terse, add helpful textbook-level explanation so the result reads like a real study guide. "
            "Return plain text only."
        )
        user_prompt = f"""
Document evidence:
{context_snippet}

Existing units:
{chr(10).join(unit_lines) or "- None"}

Existing topics:
{chr(10).join(topic_lines) or "- None"}

Rules:
- Write 450 to 760 words.
- Explain the topic as if the student will study from this summary alone.
- Preserve the document's overall direction, but fill in missing connective explanation where needed.
- {focus_line}
""".strip()

        return self._truncate_summary_text(
            self._generate_text_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=3200,
                temperature=0.24,
                top_p=0.92,
            ),
            5200,
        )

    def _augment_study_answer(
        self,
        *,
        question: str,
        current_answer: str,
        context_blocks: list[ContextBlock],
        style: str,
    ) -> str:
        context_snippet = self._build_context_snippet_for_query(
            query=question,
            context_blocks=context_blocks,
            limit=4,
            max_chars=1800,
        )
        current_plain = self._strip_citations(current_answer or "").strip()
        system_prompt = (
            "You are improving a study-guide answer. "
            "Use the document snippet first. "
            "If the snippet is incomplete, fill the gap with clear, reliable general academic explanation. "
            "Return plain text only."
        )
        if style == "flashcard":
            instruction = (
                "Write a concise flashcard answer in 2 to 4 sentences, around 35 to 85 words. "
                "Directly answer the question and make the concept memorable."
            )
            max_output_tokens = 520
        else:
            instruction = (
                "Write a fuller study answer in 4 to 7 sentences, around 100 to 190 words. "
                "Teach the concept clearly enough for revision."
            )
            max_output_tokens = 840

        user_prompt = f"""
Question:
{question}

Current answer:
{current_plain or self.NOT_AVAILABLE}

Relevant document evidence:
{context_snippet}

Task:
{instruction}
Do not answer with "{self.NOT_AVAILABLE}".
""".strip()

        answer = self._generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=0.22 if style == "qa" else 0.18,
            top_p=0.9,
        )
        return self._truncate_text(answer, 1200 if style == "qa" else 420)

    def _build_context_snippet_for_query(
        self,
        *,
        query: str,
        context_blocks: list[ContextBlock],
        limit: int,
        max_chars: int,
    ) -> str:
        query_terms = {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]{1,}", query.lower())
            if token not in self.STOP_WORDS
        }
        ranked = []
        for index, block in enumerate(context_blocks):
            overlap = len(query_terms & block.terms)
            ranked.append((overlap, -index, block))
        ranked.sort(reverse=True)
        selected = [item[2] for item in ranked[:limit] if item[0] > 0]
        if not selected:
            selected = context_blocks[:limit]

        lines: list[str] = []
        used = 0
        for block in selected:
            snippet = self._truncate_text(block.text, 420)
            line = f"{block.tag} {snippet}".strip()
            if lines and used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _build_plain_context_snippet_for_query(
        self,
        *,
        query: str,
        context_blocks: list[ContextBlock],
        limit: int,
        max_chars: int,
    ) -> str:
        tagged = self._build_context_snippet_for_query(
            query=query,
            context_blocks=context_blocks,
            limit=limit,
            max_chars=max_chars,
        )
        return self._prepare_block_text(self._strip_citations(tagged))

    def _ensure_summary_quality(
        self,
        *,
        summary_text: str,
        section_artifacts: list[SectionArtifact],
        context_blocks: list[ContextBlock],
        custom_prompt: str,
    ) -> str:
        candidate = self._truncate_summary_text(summary_text, 5200)
        if not self._needs_summary_augmentation(candidate):
            return candidate
        try:
            return self._augment_summary_text(
                result={
                    "unit_wise_summaries": self._normalize_unit_summaries(section_artifacts),
                    "topic_wise_notes": self._normalize_topic_notes(section_artifacts),
                },
                context_blocks=context_blocks,
                custom_prompt=custom_prompt,
            )
        except Exception:
            return self._fallback_summary_text(section_artifacts)

    def _needs_answer_augmentation(self, answer: str) -> bool:
        plain = self._strip_citations(answer or "").strip()
        return not plain or plain.lower() == self.NOT_AVAILABLE.lower() or len(plain) < 42

    def _needs_summary_augmentation(self, summary_text: str) -> bool:
        plain = self._strip_citations(summary_text or "").strip()
        return not plain or plain.lower() == self.NOT_AVAILABLE.lower() or len(plain) < 520

    def _validate_and_repair_payload(
        self,
        result: dict[str, Any],
        *,
        section_artifacts: list[SectionArtifact],
        context_blocks: list[ContextBlock],
        options: GenerationOptions,
    ) -> dict[str, Any]:
        result["summary_notes"] = self._normalize_generated_text(
            self._truncate_summary_text(str(result.get("summary_notes", "") or ""), 5200)
        )

        repaired_units: list[UnitSummary] = []
        for index, item in enumerate(result.get("unit_wise_summaries", [])):
            if not isinstance(item, UnitSummary):
                continue
            fallback = section_artifacts[min(index, len(section_artifacts) - 1)] if section_artifacts else None
            title = self._sanitize_label(item.unit_title)
            if not title or self._looks_like_noise_title(title):
                title = fallback.unit_title if fallback else f"Section {index + 1}"
            summary = self._normalize_generated_text(
                self._truncate_summary_text(self._prepare_block_text(item.summary), 1200)
            )
            if self._looks_like_low_quality_study_text(summary):
                summary = fallback.summary if fallback else summary
            key_points = []
            for point in item.key_points:
                cleaned_point = self._normalize_generated_text(self._truncate_text(point, 420))
                if cleaned_point and not self._looks_like_code_like_content(cleaned_point):
                    key_points.append(cleaned_point)
            if not key_points and fallback:
                key_points = fallback.key_points[:3]
            repaired_units.append(UnitSummary(unit_title=title, summary=summary, key_points=key_points[:5]))
        result["unit_wise_summaries"] = repaired_units

        repaired_topics: list[TopicNote] = []
        seen_topics: set[str] = set()
        fallback_topics = self._normalize_topic_notes(section_artifacts)
        for item in result.get("topic_wise_notes", []):
            if not isinstance(item, TopicNote):
                continue
            notes = self._normalize_generated_text(
                self._truncate_text(self._prepare_block_text(item.notes), 900)
            )
            topic = self._sanitize_label(item.topic)
            if self._looks_like_noise_title(topic) or len(topic.split()) == 1:
                topic = self._derive_topic_title_from_text(notes)
            key = self._normalize_question_key(topic)
            if (
                not topic
                or not notes
                or key in seen_topics
                or self._looks_like_noise_title(topic)
                or self._looks_like_low_quality_study_text(notes)
            ):
                continue
            seen_topics.add(key)
            repaired_topics.append(TopicNote(topic=topic, notes=notes))
        if len(repaired_topics) < 4:
            for item in fallback_topics:
                key = self._normalize_question_key(item.topic)
                if key in seen_topics or self._looks_like_noise_title(item.topic):
                    continue
                seen_topics.add(key)
                repaired_topics.append(item)
                if len(repaired_topics) >= 10:
                    break
        result["topic_wise_notes"] = repaired_topics

        result["key_concepts"] = [
            concept
            for concept in (
                self._sanitize_label(item)
                for item in result.get("key_concepts", [])
            )
            if concept and not self._looks_like_noise_title(concept)
        ]

        for item in result.get("flashcards", []):
            if isinstance(item, Flashcard):
                item.question = self._repair_flashcard_title(item.question)
                item.answer = self._normalize_generated_text(self._truncate_text(item.answer, 420))

        for item in result.get("qa_sets", []):
            if isinstance(item, QAItem):
                item.question = self._repair_question(item.question, "qa")
                item.answer = self._normalize_generated_text(self._truncate_text(item.answer, 1400))

        return result

    def _finalize_payload(
        self,
        result: dict[str, Any],
        requested_outputs: set[str],
        options: GenerationOptions,
    ) -> dict[str, Any]:
        if "summary" not in requested_outputs:
            result["summary_notes"] = ""
        if "unit_summaries" not in requested_outputs:
            result["unit_wise_summaries"] = []
        if "key_concepts" not in requested_outputs:
            result["key_concepts"] = []
        if "topic_notes" not in requested_outputs:
            result["topic_wise_notes"] = []
        if "flashcards" not in requested_outputs:
            result["flashcards"] = []
        if "qa" not in requested_outputs:
            result["qa_sets"] = []
        if "viva" not in requested_outputs:
            result["viva_questions"] = []
        if "difficulty" not in requested_outputs:
            result["difficulty_explanations"] = []

        result["flashcards"] = self._dedupe_flashcards(result.get("flashcards", []))[:10]
        result["qa_sets"] = self._dedupe_qa_sets(result.get("qa_sets", []), result["flashcards"])[:8]
        result["key_concepts"] = self._dedupe_strings(result.get("key_concepts", []), limit=14)
        result["unit_wise_summaries"] = result.get("unit_wise_summaries", [])[:8]
        result["topic_wise_notes"] = result.get("topic_wise_notes", [])[:12]
        result["viva_questions"] = self._dedupe_strings(result.get("viva_questions", []), limit=8)

        allowed_bloom = self._normalize_bloom_levels(options.bloom_levels)
        for index, item in enumerate(result["flashcards"]):
            if item.bloom_level not in allowed_bloom:
                item.bloom_level = allowed_bloom[index % len(allowed_bloom)]
        for index, item in enumerate(result["qa_sets"]):
            if item.bloom_level not in allowed_bloom:
                item.bloom_level = allowed_bloom[index % len(allowed_bloom)]

        requested_modes = {
            mode.strip().lower()
            for mode in options.difficulty_modes
            if mode and mode.strip()
        }
        if requested_modes:
            result["difficulty_explanations"] = [
                item
                for item in result.get("difficulty_explanations", [])
                if isinstance(item, DifficultyExplanation) and item.mode in requested_modes
            ]
        return result

    def _dedupe_flashcards(self, items: list[Flashcard]) -> list[Flashcard]:
        filtered: list[Flashcard] = []
        seen_questions: set[str] = set()
        for item in items:
            if not isinstance(item, Flashcard):
                continue
            item.question = self._repair_flashcard_title(item.question)
            item.answer = self._normalize_generated_text(self._truncate_text(item.answer, 420))
            key = self._normalize_question_key(item.question)
            if not key or key in seen_questions:
                continue
            if key.startswith(("how ", "why ", "explain ", "compare ", "what ", "which ", "define ", "name ", "identify ", "state ")):
                continue
            seen_questions.add(key)
            filtered.append(item)
        return filtered

    def _dedupe_qa_sets(self, items: list[QAItem], flashcards: list[Flashcard]) -> list[QAItem]:
        flashcard_keys = {self._normalize_question_key(item.question) for item in flashcards}
        filtered: list[QAItem] = []
        seen_questions: set[str] = set()
        for item in items:
            if not isinstance(item, QAItem):
                continue
            item.question = self._repair_question(item.question, "qa")
            item.answer = self._normalize_generated_text(self._truncate_text(item.answer, 1400))
            key = self._normalize_question_key(item.question)
            if not key or key in seen_questions or key in flashcard_keys:
                continue
            if not any(
                key.startswith(prefix)
                for prefix in (
                    "how ",
                    "why ",
                    "explain ",
                    "compare ",
                    "distinguish ",
                    "what is the difference",
                    "what are the differences",
                    "when would ",
                )
            ):
                continue
            seen_questions.add(key)
            filtered.append(item)
        return filtered

    def _parse_context_blocks(self, context: str) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        for match in self.CONTEXT_BLOCK_PATTERN.finditer(context):
            source = " ".join(match.group("source").split()).strip()
            chunk = " ".join(match.group("chunk").split()).strip()
            section_title = " ".join((match.group("section") or "").split()).strip()
            text = self._prepare_block_text(match.group("text"))
            if not text:
                continue
            terms = {
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
                if token not in self.STOP_WORDS
            }
            if section_title:
                terms.update(
                    token
                    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", section_title.lower())
                    if token not in self.STOP_WORDS
                )
            blocks.append(
                ContextBlock(
                    source=source,
                    chunk=chunk,
                    tag=f"[source:{source} | chunk:{chunk}]",
                    section_title=section_title,
                    text=text,
                    terms=terms,
                )
            )
        return blocks

    def _blocks_to_context(
        self,
        blocks: list[ContextBlock],
        *,
        max_chars: int,
    ) -> str:
        lines: list[str] = []
        used = 0
        for block in blocks:
            entry = f"{block.tag} {self._prepare_block_text(block.text)}".strip()
            if not entry or entry == block.tag:
                continue
            if lines and used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry)
        return "\n".join(lines)

    def _guess_section_title(self, blocks: list[ContextBlock], section_index: int) -> str:
        for block in blocks:
            label = self._sanitize_label(block.section_title)
            if label and not self._looks_like_noise_title(label):
                return label
        study_blocks = self._preferred_study_blocks(blocks)
        joined = " ".join(block.text for block in study_blocks[:2])
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,4}", joined)
        for candidate in candidates:
            cleaned = self._sanitize_label(candidate)
            if 4 <= len(cleaned) <= 90:
                return cleaned
        key_phrases = self._extract_key_phrases(study_blocks or blocks, 3)
        if key_phrases:
            return key_phrases[0].title()
        return f"Section {section_index}"

    def _extract_key_phrases(
        self,
        blocks: Iterable[ContextBlock],
        limit: int,
    ) -> list[str]:
        unigram_counter: Counter[str] = Counter()
        bigram_counter: Counter[str] = Counter()

        for block in blocks:
            if self._looks_like_code_like_content(block.text):
                continue
            words = [
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", block.text)
                if token.lower() not in self.STOP_WORDS and not self._is_noise_token(token)
            ]
            unigram_counter.update(words)
            bigrams = [
                f"{words[index]} {words[index + 1]}"
                for index in range(len(words) - 1)
                if words[index] != words[index + 1]
            ]
            bigram_counter.update(bigrams)

        phrases: list[str] = []
        for phrase, count in bigram_counter.most_common(limit * 2):
            if count < 2:
                continue
            cleaned_phrase = self._sanitize_label(phrase.title())
            if not cleaned_phrase or self._looks_like_noise_title(cleaned_phrase):
                continue
            phrases.append(cleaned_phrase)
            if len(phrases) >= limit:
                return phrases

        for word, count in unigram_counter.most_common(limit * 3):
            if count < 2:
                continue
            cleaned_word = self._sanitize_label(word.upper() if word.isupper() else word.title())
            if not cleaned_word or self._looks_like_noise_title(cleaned_word):
                continue
            phrases.append(cleaned_word)
            if len(phrases) >= limit:
                return self._dedupe_strings(phrases, limit=limit)

        return self._dedupe_strings(phrases, limit=limit)

    def _summarize_block_text(self, text: str, max_chars: int) -> str:
        compact = self._clean_text(text)
        if len(compact) <= max_chars:
            return compact
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact)
            if sentence.strip()
        ]
        if not sentences:
            return self._truncate_text(compact, max_chars)
        snippet = " ".join(sentences[:2])
        return self._truncate_text(snippet, max_chars)

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _normalize_generated_text(self, text: str) -> str:
        cleaned = self._clean_text(text)
        cleaned = cleaned.replace("**", "").replace("`", "")
        cleaned = cleaned.replace("\\(", "(").replace("\\)", ")")
        cleaned = cleaned.replace("\\[", "[").replace("\\]", "]")
        cleaned = re.sub(r"</?(?:summary_bundle|summary_notes|key_concepts|item|flashcards|card|question|answer|bloom_level|qa_sets?|qa|viva_questions?|difficulty_modes?|mode|explanation|topic_notes?|topic_note|topic|notes|artifact|unit_title|summary|key_points?)>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _strip_citations(self, text: str) -> str:
        return self._clean_text(self.CITATION_PATTERN.sub("", text or ""))

    def _sanitize_label(self, text: str) -> str:
        cleaned = self._normalize_generated_text(text)
        if ":" in cleaned:
            left, right = cleaned.split(":", 1)
            if len(right.split()) >= 4 and (
                len(left.split()) >= 2 or re.search(r"[A-Z0-9\-]{3,}", left)
            ):
                cleaned = left.strip()
        cleaned = re.sub(r"^[\-\*\d\.\)\(:\s]+", "", cleaned)
        cleaned = re.sub(r"\b(?:lc|leetcode)\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:;,")
        words = cleaned.split()
        if len(words) > 10:
            cleaned = " ".join(words[:10]).strip(" -:;,")
        return cleaned

    def _repair_question(self, question: str, style: str) -> str:
        cleaned = self._normalize_generated_text(question)
        if ":" in cleaned and len(cleaned.split(":")[1].split()) > 5:
            cleaned = cleaned.split(":", 1)[0].strip()
        cleaned = re.sub(
            r",?\s*as (?:seen|used|shown|illustrated|exemplified)\s+in[^?!.]*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r",?\s*highlighting\s+why[^?!.]*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\s*\([^)]*code[^)]*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(and|or|with|including|such as|based on|for example|by|in|on|of|to|from|like)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"[,:;]\s*$", "", cleaned).strip()
        cleaned = re.sub(r"\s+(and|or|with|including|by|in|on|of|to|from|like)\?$", "?", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip(" .;:")
        if style == "flashcard":
            starters = ("what ", "which ", "name ", "define ", "state ", "identify ", "when ")
            if not cleaned.lower().startswith(starters):
                cleaned = f"What is {cleaned.lstrip('What is ').strip()}".strip()
        else:
            qa_starters = ("how ", "why ", "explain ", "compare ", "distinguish ", "what is the difference", "what are the differences", "when would ")
            if not cleaned.lower().startswith(qa_starters):
                subject = re.sub(r"^(what is|what are)\s+", "", cleaned, flags=re.IGNORECASE).strip()
                cleaned = f"Explain {subject}".strip()
        return self._truncate_question(cleaned if cleaned.endswith("?") else f"{cleaned}?", 170 if style == "qa" else 120)

    def _is_noise_token(self, token: str) -> bool:
        lowered = token.lower()
        if len(lowered) <= 2:
            return True
        noise_tokens = {
            "trienode",
            "priorityqueueint",
            "vectorint",
            "nullptr",
            "board",
            "cols",
            "diag",
            "diag2",
            "minheap",
            "maxheap",
            "existsvectorvectorchar",
        }
        return lowered in noise_tokens

    def _looks_like_code_like_content(self, text: str) -> bool:
        compact = self._normalize_generated_text(text)
        if not compact:
            return False
        code_markers = ("::", "->", "();", "return ", "class ", "public:", "private:", "vector", "bool ", "int ", "nullptr")
        marker_hits = sum(1 for marker in code_markers if marker in compact)
        punctuation = sum(compact.count(symbol) for symbol in (";", "{", "}", "[", "]"))
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", compact)
        natural = [word for word in words if word.isalpha() and len(word) >= 4]
        natural_ratio = len(natural) / max(1, len(words))
        return marker_hits >= 2 or (punctuation >= 4 and natural_ratio < 0.45) or self._looks_like_tabular_noise(compact)

    def _looks_like_noise_title(self, text: str) -> bool:
        cleaned = self._sanitize_label(text)
        if not cleaned:
            return True
        if self._looks_like_code_like_content(cleaned):
            return True
        if len(cleaned.split()) > 10:
            return True
        words = cleaned.lower().split()
        if len(words) == 1 and words[0] in {"word", "node", "array", "string", "priority", "median", "nums", "prev", "else", "mid"}:
            return True
        if re.search(r"\b(?:node|return|vector|priority|queueint|nullptr)\b", cleaned, re.IGNORECASE) and len(cleaned.split()) <= 3:
            return True
        return False

    def _prepare_block_text(self, text: str) -> str:
        normalized = self._normalize_generated_text(text)
        lines = re.split(r"[\r\n]+", normalized)
        kept: list[str] = []
        for line in lines:
            for fragment in re.split(r"(?<=[.!?])\s+|\s{2,}|;\s+", line):
                candidate = self._clean_text(fragment)
                if not candidate:
                    continue
                candidate = re.sub(
                    r"\b(?:class|public|private|protected|return|bool|int|void|string|vector|nullptr)\b[^.!?]{0,220}",
                    " ",
                    candidate,
                    flags=re.IGNORECASE,
                )
                candidate = self._clean_text(candidate)
                if not candidate:
                    continue
                if self._looks_like_code_like_content(candidate):
                    continue
                if self._looks_like_tabular_noise(candidate):
                    continue
                if re.search(r"\btable of contents\b", candidate, re.IGNORECASE):
                    continue
                if not self._looks_like_natural_study_sentence(candidate):
                    continue
                kept.append(candidate)
        if kept:
            return " ".join(kept)
        return normalized

    def _preferred_study_blocks(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        preferred = [
            block
            for block in blocks
            if not self._looks_like_code_like_content(block.text)
            and not self._looks_like_tabular_noise(block.text)
        ]
        return preferred if len(preferred) >= max(2, len(blocks) // 3) else blocks

    def _looks_like_tabular_noise(self, text: str) -> bool:
        compact = self._normalize_generated_text(text)
        if not compact:
            return False
        complexity_hits = len(re.findall(r"\bO\([^)]*\)", compact))
        slash_hits = compact.count("/") + compact.count("|")
        digit_hits = len(re.findall(r"\b\d+\b", compact))
        short_chunks = len(re.findall(r"\b[A-Za-z]{1,3}\b", compact))
        codeish_identifiers = len(re.findall(r"\b[a-z]+[A-Z][A-Za-z0-9_]*\b|\b[A-Za-z_]+::[A-Za-z_]+\b", compact))
        verb_hits = len(re.findall(r"\b(is|are|uses|explains|shows|describes|maintains|works|processes|stores|computes|helps|allows|supports|means)\b", compact, flags=re.IGNORECASE))
        if complexity_hits >= 2:
            return True
        if slash_hits >= 4 and short_chunks >= 6:
            return True
        if digit_hits >= 6 and short_chunks >= 8 and len(compact.split()) < 45:
            return True
        if codeish_identifiers >= 2 and verb_hits == 0:
            return True
        if slash_hits >= 2 and verb_hits == 0 and len(compact.split()) >= 8:
            return True
        return False

    def _looks_like_natural_study_sentence(self, text: str) -> bool:
        compact = self._normalize_generated_text(text)
        if not compact:
            return False
        words = compact.split()
        if len(words) < 5:
            return False
        alpha_words = re.findall(r"[A-Za-z]{3,}", compact)
        if len(alpha_words) < 4:
            return False
        if self._looks_like_code_like_content(compact) or self._looks_like_tabular_noise(compact):
            return False
        return True

    def _looks_like_low_quality_study_text(self, text: str) -> bool:
        compact = self._normalize_generated_text(text)
        if not compact:
            return True
        if self._looks_like_code_like_content(compact) or self._looks_like_tabular_noise(compact):
            return True
        words = compact.split()
        if len(words) < 10:
            return True
        verb_hits = len(
            re.findall(
                r"\b(is|are|uses|explains|shows|describes|maintains|works|processes|stores|computes|helps|allows|supports|means|focuses|covers|introduces|demonstrates)\b",
                compact,
                flags=re.IGNORECASE,
            )
        )
        if verb_hits == 0:
            return True
        return False

    def _truncate_text(self, text: str, max_chars: int) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        clipped = cleaned[:max_chars]
        sentence_boundary = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
        if sentence_boundary > max_chars * 0.72:
            return clipped[: sentence_boundary + 1].strip()
        space_boundary = clipped.rfind(" ")
        return clipped[:space_boundary].strip() if space_boundary > 0 else clipped.strip()

    def _truncate_summary_text(self, text: str, max_chars: int) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        clipped = cleaned[:max_chars]
        sentence_endings = [
            clipped.rfind(". "),
            clipped.rfind("? "),
            clipped.rfind("! "),
        ]
        sentence_boundary = max(sentence_endings)
        if sentence_boundary > max_chars * 0.68:
            return clipped[: sentence_boundary + 1].strip()
        paragraph_boundary = clipped.rfind("  ")
        if paragraph_boundary > max_chars * 0.6:
            return clipped[:paragraph_boundary].strip()
        space_boundary = clipped.rfind(" ")
        return clipped[:space_boundary].strip() if space_boundary > 0 else clipped.strip()

    def _truncate_question(self, question: str, max_chars: int) -> str:
        cleaned = self._clean_text(question)
        if len(cleaned) <= max_chars:
            return cleaned
        words = cleaned.split()
        while words and len(" ".join(words)) > max_chars:
            words.pop()
        truncated = " ".join(words).rstrip(" ,;:")
        if truncated and not truncated.endswith("?"):
            truncated += "?"
        return truncated or cleaned[:max_chars].rstrip() + "?"

    def _normalize_output_types(self, output_types: list[str]) -> set[str]:
        mapping = {
            "summary": "summary",
            "unit_summaries": "unit_summaries",
            "key_concepts": "key_concepts",
            "topic_notes": "topic_notes",
            "flashcards": "flashcards",
            "qa": "qa",
            "viva": "viva",
            "difficulty": "difficulty",
        }
        return {
            mapping[item.strip().lower()]
            for item in output_types
            if item and item.strip().lower() in mapping
        }

    def _normalize_bloom_levels(self, bloom_levels: list[str]) -> list[str]:
        levels = [
            level.strip().lower()
            for level in bloom_levels
            if level and level.strip().lower() in self.BLOOM_LEVELS
        ]
        return levels or list(self.BLOOM_LEVELS)

    def _normalize_question_key(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def _dedupe_strings(self, values: Iterable[str], *, limit: int) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._clean_text(value)
            key = self._normalize_question_key(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            results.append(cleaned)
            if len(results) >= limit:
                break
        return results

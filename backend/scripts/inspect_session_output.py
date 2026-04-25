import sys

from app.core.config import get_settings
from app.models.schemas import GenerationOptions
from app.services.generation_service import GeminiGenerationService
from app.services.rag_service import RAGService
from app.services.study_pipeline import StudyGuidePipeline


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: inspect_session_output.py <session_id> <source_filename>")

    session_id = sys.argv[1]
    source_filename = sys.argv[2]

    settings = get_settings()
    rag = RAGService(vector_root=settings.vector_db_path, retrieval_k=settings.retrieval_k)
    generation = GeminiGenerationService(api_key=settings.gemini_api_key, model_name=settings.gemini_model)

    options = GenerationOptions(
        output_types=["summary", "key_concepts", "topic_notes", "flashcards", "qa", "viva", "difficulty"],
        difficulty_modes=["beginner", "intermediate", "advanced"],
        custom_prompt="",
    )

    pipeline_stub = StudyGuidePipeline(settings=settings, rag_service=rag, generation_service=generation, history_service=None)  # type: ignore[arg-type]
    queries = pipeline_stub._build_retrieval_queries(options, [source_filename])

    context = rag.get_context_bundle(
        session_id=session_id,
        queries=queries,
        k=settings.retrieval_k,
        max_chars=settings.context_max_chars,
        allowed_sources={source_filename},
    )

    payload = {
        "output_types": options.output_types,
        "difficulty_modes": options.difficulty_modes,
        "custom_prompt": options.custom_prompt,
    }
    system_prompt, user_prompt = generation._build_prompts(context=context, payload=payload)

    raw_generated = generation._generate_with_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        options=options,
    )

    print("RAW summary:")
    print(raw_generated.get("summary_notes", ""))
    print("\nRAW difficulty:")
    for item in raw_generated.get("difficulty_explanations", []):
        print(item)

    filtered = generation._filter_by_options(
        raw_generated,
        options,
        context,
        allowed_sources={source_filename},
    )

    print("\nFILTERED summary:")
    print(filtered.get("summary_notes", ""))
    print("\nFILTERED difficulty:")
    for item in filtered.get("difficulty_explanations", []):
        print(item)


if __name__ == "__main__":
    main()

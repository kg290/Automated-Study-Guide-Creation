import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import GenerationOptions
from app.services.generation_service import GeminiGenerationService
from app.services.rag_service import RAGService


def main() -> None:
    settings = get_settings()
    session_id = sys.argv[1] if len(sys.argv) > 1 else "18e1996258c943dba59ae43c5fbb6878"
    model_name = sys.argv[2] if len(sys.argv) > 2 else settings.gemini_model

    rag = RAGService(vector_root=settings.vector_db_path, retrieval_k=settings.retrieval_k)
    context = rag.get_context_bundle(
        session_id=session_id,
        queries=["summary,key concepts,topic notes,flashcards,qa,viva,difficulty"],
        k=settings.retrieval_k,
        max_chars=settings.context_max_chars,
        allowed_sources={"ComputerNetworks-CompleteNotes.pdf"},
    )

    service = GeminiGenerationService(api_key=settings.gemini_api_key, model_name=model_name)
    options = GenerationOptions(
        output_types=["summary", "key_concepts", "topic_notes", "flashcards", "qa", "viva", "difficulty"],
        difficulty_modes=["beginner", "intermediate", "advanced"],
        custom_prompt="",
    )

    payload = {
        "output_types": options.output_types,
        "difficulty_modes": options.difficulty_modes,
        "custom_prompt": options.custom_prompt,
    }

    system_prompt, user_prompt = service._build_prompts(context=context, payload=payload)
    config = service._build_generation_config(system_prompt=system_prompt, options=options)

    response = service.client.models.generate_content(
        model=service.model_name,
        contents=user_prompt,
        config=config,
    )

    parsed = getattr(response, "parsed", None)
    print("parsed_type:", type(parsed).__name__ if parsed is not None else None)
    if parsed is not None:
        try:
            if hasattr(parsed, "model_dump"):
                print("parsed_keys:", list(parsed.model_dump().keys()))
            elif isinstance(parsed, dict):
                print("parsed_keys:", list(parsed.keys()))
            else:
                print("parsed_value:", parsed)
        except Exception as exc:
            print("parsed_inspect_error:", exc)

    text = getattr(response, "text", "") or ""
    print("text_preview:")
    print(text[:2200])
    print("\ntext_length:", len(text))

    if text:
        try:
            loaded = json.loads(text)
            print("json_parse: success", list(loaded.keys()))
        except Exception as exc:
            print("json_parse: failed", exc)


if __name__ == "__main__":
    main()

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

CITATION_RE = re.compile(r"\[source:(?P<source>[^\]|]+)\s*\|\s*chunk:(?P<chunk>[^\]]+)\]")
NOT_AVAILABLE = "Not available in uploaded material"


@dataclass
class FieldCheck:
    name: str
    text: str
    requires_citation: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _collect_required_fields(result: dict[str, Any]) -> list[FieldCheck]:
    fields: list[FieldCheck] = []

    fields.append(FieldCheck("summary_notes", _clean(result.get("summary_notes", "")), True))

    for index, item in enumerate(result.get("topic_wise_notes", []), start=1):
        fields.append(FieldCheck(f"topic_wise_notes[{index}].notes", _clean(item.get("notes", "")), True))

    for index, item in enumerate(result.get("flashcards", []), start=1):
        fields.append(FieldCheck(f"flashcards[{index}].answer", _clean(item.get("answer", "")), True))

    for index, item in enumerate(result.get("qa_sets", []), start=1):
        fields.append(FieldCheck(f"qa_sets[{index}].answer", _clean(item.get("answer", "")), True))

    for index, item in enumerate(result.get("difficulty_explanations", []), start=1):
        fields.append(FieldCheck(f"difficulty_explanations[{index}].explanation", _clean(item.get("explanation", "")), True))

    for index, item in enumerate(result.get("viva_questions", []), start=1):
        fields.append(FieldCheck(f"viva_questions[{index}]", _clean(item), False))

    return fields


def _collect_citations(text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in CITATION_RE.finditer(text):
        source = _clean(match.group("source"))
        chunk = _clean(match.group("chunk"))
        items.append({"source": source, "chunk": chunk})
    return items


def _analyze_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    source_documents = [str(item).strip() for item in result.get("source_documents", []) if str(item).strip()]
    source_set = {item.lower() for item in source_documents}

    fields = _collect_required_fields(result)
    required_fields = [field for field in fields if field.requires_citation]

    required_with_citation = 0
    required_not_available = 0
    invalid_source_citations = 0
    missing_citation_fields: list[str] = []
    not_available_fields: list[str] = []
    all_citations: list[dict[str, str]] = []

    for field in required_fields:
        if not field.text:
            missing_citation_fields.append(field.name)
            continue

        if NOT_AVAILABLE.lower() in field.text.lower():
            required_not_available += 1
            not_available_fields.append(field.name)
            continue

        citations = _collect_citations(field.text)
        all_citations.extend(citations)

        if citations:
            required_with_citation += 1
            for citation in citations:
                if citation["source"].lower() not in source_set:
                    invalid_source_citations += 1
        else:
            missing_citation_fields.append(field.name)

    summary = _clean(result.get("summary_notes", ""))
    key_concepts = result.get("key_concepts", [])
    topic_wise_notes = result.get("topic_wise_notes", [])
    flashcards = result.get("flashcards", [])
    qa_sets = result.get("qa_sets", [])

    fallback_phrase = "fallback mode" in summary.lower()
    long_token_count = len([token for token in re.findall(r"\S+", summary) if len(token) > 35])

    return {
        "source_documents": source_documents,
        "required_field_count": len(required_fields),
        "required_with_citation": required_with_citation,
        "citation_coverage": round((required_with_citation / len(required_fields)) * 100, 2)
        if required_fields
        else 0.0,
        "required_not_available": required_not_available,
        "required_not_available_ratio": round((required_not_available / len(required_fields)) * 100, 2)
        if required_fields
        else 0.0,
        "invalid_source_citations": invalid_source_citations,
        "total_citations": len(all_citations),
        "missing_citation_fields": missing_citation_fields,
        "not_available_fields": not_available_fields,
        "fallback_phrase_present": fallback_phrase,
        "summary_char_count": len(summary),
        "summary_long_token_count": long_token_count,
        "section_counts": {
            "key_concepts": len(key_concepts),
            "topic_wise_notes": len(topic_wise_notes),
            "flashcards": len(flashcards),
            "qa_sets": len(qa_sets),
            "viva_questions": len(result.get("viva_questions", [])),
            "difficulty_explanations": len(result.get("difficulty_explanations", [])),
        },
        "summary_preview": summary[:380],
    }


def _submit_job(api_base: str, pdf_path: Path) -> dict[str, Any]:
    with pdf_path.open("rb") as file_handle:
        response = requests.post(
            f"{api_base}/study-guides/generate",
            files=[("files", (pdf_path.name, file_handle, "application/pdf"))],
            data={
                "output_types": "summary,key_concepts,topic_notes,flashcards,qa,viva,difficulty",
                "difficulty_modes": "beginner,intermediate,advanced",
                "custom_prompt": "",
            },
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def _poll_job(api_base: str, job_id: str, timeout_seconds: int = 1400) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    started = time.time()
    timeline: list[dict[str, Any]] = []
    seen = set()

    while True:
        response = requests.get(f"{api_base}/study-guides/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        payload = response.json()

        signature = (payload.get("status"), payload.get("stage"), payload.get("progress"), payload.get("message"))
        if signature not in seen:
            seen.add(signature)
            timeline.append(
                {
                    "elapsed_seconds": round(time.time() - started, 2),
                    "status": payload.get("status"),
                    "stage": payload.get("stage"),
                    "progress": payload.get("progress"),
                    "message": payload.get("message"),
                }
            )

        status = payload.get("status")
        if status in {"completed", "failed"}:
            duration = time.time() - started
            return payload, timeline, duration

        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for job {job_id}.")

        time.sleep(2.5)


def main() -> None:
    project_root = _project_root()
    input_dir = project_root / "Input"
    output_dir = project_root / "backend" / "evaluation_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base = "http://127.0.0.1:8001/api/v1"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in Input directory.")
        return

    print(f"Found {len(pdf_files)} PDFs. Running batch evaluation against {api_base}.")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": api_base,
        "pdf_count": len(pdf_files),
        "items": [],
    }

    for index, pdf_path in enumerate(pdf_files, start=1):
        print("=" * 92)
        print(f"[{index}/{len(pdf_files)}] Processing: {pdf_path.name}")

        item: dict[str, Any] = {
            "pdf": pdf_path.name,
            "path": str(pdf_path),
        }

        try:
            submitted = _submit_job(api_base, pdf_path)
            job_id = submitted["job_id"]
            session_id = submitted["session_id"]
            item["job_id"] = job_id
            item["session_id"] = session_id

            print(f"  Job queued: {job_id}")
            payload, timeline, duration = _poll_job(api_base, job_id)
            item["duration_seconds"] = round(duration, 2)
            item["timeline"] = timeline
            item["final_status"] = payload.get("status")
            item["final_stage"] = payload.get("stage")
            item["error"] = payload.get("error")
            item["raw_job_payload"] = payload

            if payload.get("status") == "completed":
                analysis = _analyze_result(payload)
                item["analysis"] = analysis
                print(
                    "  Completed in "
                    f"{item['duration_seconds']}s | citations {analysis['citation_coverage']}% | "
                    f"not-available {analysis['required_not_available_ratio']}% | "
                    f"invalid citations {analysis['invalid_source_citations']}"
                )
                print(f"  Summary preview: {analysis['summary_preview']}")
            else:
                print(f"  Failed with error: {item['error']}")

        except Exception as exc:
            item["final_status"] = "error"
            item["error"] = str(exc)
            print(f"  Error while processing {pdf_path.name}: {exc}")

        report["items"].append(item)

    output_path = output_dir / "all_pdfs_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    completed_items = [item for item in report["items"] if item.get("final_status") == "completed"]
    failed_items = [item for item in report["items"] if item.get("final_status") != "completed"]

    if completed_items:
        avg_citation = sum(item["analysis"]["citation_coverage"] for item in completed_items) / len(completed_items)
        avg_not_available = (
            sum(item["analysis"]["required_not_available_ratio"] for item in completed_items)
            / len(completed_items)
        )
        print("=" * 92)
        print("Aggregate summary:")
        print(f"  Completed: {len(completed_items)} | Failed: {len(failed_items)}")
        print(f"  Avg citation coverage (required fields): {round(avg_citation, 2)}%")
        print(f"  Avg required-field not-available ratio: {round(avg_not_available, 2)}%")
    else:
        print("No completed jobs to summarize.")

    print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()

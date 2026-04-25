import os
from pathlib import Path

import fitz

try:
    from google.cloud import vision
except ImportError:  # pragma: no cover
    vision = None


class VisionOCRService:
    def __init__(self, credentials_path: str = "") -> None:
        if vision is None:
            raise RuntimeError(
                "google-cloud-vision is not installed. Install dependencies before using OCR."
            )

        if credentials_path:
            credentials = Path(credentials_path)
            if not credentials.exists():
                raise RuntimeError(
                    "Google Vision credentials path does not exist."
                )
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)

        self.client = vision.ImageAnnotatorClient()

    def ocr_image_bytes(self, image_bytes: bytes) -> str:
        image = vision.Image(content=image_bytes)
        response = self.client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"Vision OCR error: {response.error.message}")

        if response.full_text_annotation and response.full_text_annotation.text:
            return response.full_text_annotation.text
        return ""

    def ocr_pdf_pages(
        self,
        pdf_path: Path,
        page_numbers: list[int] | None = None,
        *,
        zoom: float = 1.6,
    ) -> dict[int, str]:
        document = fitz.open(pdf_path)
        extracted_pages: dict[int, str] = {}
        allowed_pages = set(page_numbers) if page_numbers else None

        try:
            for page_index, page in enumerate(document):
                if allowed_pages is not None and page_index not in allowed_pages:
                    continue

                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                page_text = self.ocr_image_bytes(pix.tobytes("png")).strip()
                if page_text:
                    extracted_pages[page_index] = page_text
        finally:
            document.close()

        return extracted_pages

    def ocr_pdf(self, pdf_path: Path) -> str:
        page_map = self.ocr_pdf_pages(pdf_path=pdf_path)
        ordered_pages = [page_map[index] for index in sorted(page_map)]
        return "\n\n".join(ordered_pages)

# Backend Service

FastAPI backend for document ingestion, OCR fallback, text processing, RAG retrieval, Gemini generation, and session history.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Required Environment Variables

- GEMINI_API_KEY
- GOOGLE_APPLICATION_CREDENTIALS (for scanned PDF OCR)

## API Prefix

- /api/v1

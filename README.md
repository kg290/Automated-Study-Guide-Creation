# Generative AI Agent for Automated Study Guide Creation

Full-stack capstone project that converts uploaded PDF/DOCX academic material into structured study assets using OCR, RAG, and Gemini.

## What This Project Delivers

- Multi-file upload for PDF and DOCX documents.
- Smart extraction pipeline:
  - DOCX direct parsing.
  - PDF text extraction using PyMuPDF.
  - Automatic low-text PDF detection and OCR fallback using Google Vision API.
- Content processing:
  - Text cleaning.
  - Header/footer repetition reduction.
  - Chunking for retrieval.
- RAG architecture:
  - Embeddings via sentence-transformers (all-MiniLM-L6-v2).
  - Vector storage in ChromaDB.
  - Semantic retrieval before generation.
- Gemini-powered study outputs grounded in uploaded context:
  - Summary notes
  - Key concepts
  - Topic-wise notes
  - Flashcards
  - Q&A sets
  - Viva questions
  - Beginner/intermediate/advanced explanations
- Modern Next.js dashboard experience:
  - Landing page
  - Upload dashboard
  - Processing status page
  - Results page with tabs and copy/download actions
  - History page with saved sessions (SQLite-backed)

## Tech Stack

### Frontend
- Next.js (React, TypeScript)
- Framer Motion
- Axios
- React Dropzone

### Backend
- Python
- FastAPI
- LangChain
- ChromaDB
- sentence-transformers
- Gemini API
- Google Cloud Vision API
- PyMuPDF
- python-docx
- SQLite

## Project Structure

```text
.
├── backend
│   ├── app
│   │   ├── api/v1/endpoints
│   │   ├── core
│   │   ├── models
│   │   └── services
│   ├── uploads
│   ├── vector_store
│   ├── .env.example
│   └── requirements.txt
├── frontend
│   ├── app
│   ├── components
│   ├── lib
│   ├── .env.example
│   └── package.json
└── README.md
```

## Setup Instructions

## 1. Backend Setup

Open terminal in project root and run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit backend .env with your keys and paths:

- GEMINI_API_KEY=your_gemini_api_key
- GOOGLE_APPLICATION_CREDENTIALS=absolute_or_relative_path_to_service_account_json

Run backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend base URL:

- http://localhost:8000
- API base: http://localhost:8000/api/v1

## 2. Frontend Setup

In a second terminal:

```powershell
cd frontend
npm install
copy .env.example .env
```

Ensure frontend .env points to backend:

- NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1

Run frontend:

```powershell
npm run dev
```

Frontend URL:

- http://localhost:3000

## Environment and Security Notes

- API keys are read from backend environment variables only.
- Never expose Gemini or Google credentials in frontend code.
- Keep .env and service account files out of version control.
- Existing .gitignore already excludes common secret/runtime files.

## Main API Endpoints

- GET /api/v1/health
- POST /api/v1/study-guides/generate
- GET /api/v1/study-guides/jobs/{job_id}
- POST /api/v1/study-guides/sessions/{session_id}/regenerate
- GET /api/v1/history
- GET /api/v1/history/{session_id}

## User Flow

1. Upload one or many PDF/DOCX files.
2. System extracts text and runs OCR for low-text PDFs.
3. Content is cleaned and chunked.
4. Chunks are embedded and stored in ChromaDB.
5. Relevant chunks are retrieved based on selected output type.
6. Gemini generates source-grounded study outputs.
7. Results are displayed in tabs with copy/download support.
8. Sessions are stored in SQLite for history revisit and regeneration.

## Optional Premium Extensions (Roadmap)

- Export generated study guide as PDF.
- MCQ generator.
- Chat with uploaded notes.
- Unit-wise generation controls.
- Role-based user accounts and cloud storage.

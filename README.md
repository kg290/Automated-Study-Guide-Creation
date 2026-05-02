# Automated Study Guide Creation

Generative AI study-guide platform that turns uploaded academic material into structured revision outputs for students and professors. The system extracts content from PDFs and DOCX files, organizes the material, retrieves relevant context, and generates study assets such as summaries, unit guides, concept cards, Q&A, viva prompts, and difficulty-based explanations.

## Problem Statement

Professors often spend significant time preparing concise study guides and summaries for students from large sets of course materials. This project aims to automate that workflow by reading educational content and producing a structured study guide within minutes.

## What The Project Does

- Accepts one or more uploaded `PDF` or `DOCX` files
- Extracts text using direct parsing first
- Applies OCR for low-text and mixed-content PDF pages
- Cleans and chunks the source material for retrieval
- Builds a retrieval context using embeddings and ChromaDB
- Generates grounded study outputs from the uploaded material
- Stores sessions in history for revisit and regeneration

## Generated Outputs

- Summary notes
- Unit and chapter guide
- Key concepts
- Topic-wise notes
- Flashcards as concept-detail revision cards
- Q&A as explanation-oriented question-answer pairs
- Viva questions
- Difficulty explanations:
  - Beginner
  - Intermediate
  - Advanced

## Current Product Experience

### Frontend

- Light notebook-style dashboard
- Dedicated processing page with active pipeline stages
- Results workspace with top-level tabs:
  - Summary
  - Flashcards
  - Q&A
- Animated study-card presentation for Flashcards and Q&A
- Copy and markdown download actions
- Session history view

### Backend

- FastAPI service
- PDF extraction with PyMuPDF
- DOCX extraction with `python-docx`
- OCR fallback with Google Cloud Vision
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Vector retrieval using ChromaDB
- Gemini-powered study-guide generation
- SQLite-backed history persistence

## Tech Stack

### Frontend

- Next.js
- React
- TypeScript
- Framer Motion
- React Dropzone
- Lucide React

### Backend

- Python
- FastAPI
- Gemini API
- ChromaDB
- sentence-transformers
- PyMuPDF
- python-docx
- Google Cloud Vision API
- SQLite

## Project Structure

```text
.
├── backend
│   ├── app
│   │   ├── api
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

## Setup

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Set the required backend environment variables in `.env`:

- `GEMINI_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS`

Run the backend:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env
```

Set:

- `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1`

Run the frontend:

```powershell
npm run dev
```

## Main API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/study-guides/generate`
- `GET /api/v1/study-guides/jobs/{job_id}`
- `POST /api/v1/study-guides/sessions/{session_id}/regenerate`
- `GET /api/v1/history`
- `GET /api/v1/history/{session_id}`

## Typical Flow

1. Upload source files
2. Extract and OCR text where needed
3. Clean and chunk the content
4. Build embeddings and retrieval context
5. Generate study outputs
6. Review outputs in Summary, Flashcards, and Q&A tabs
7. Revisit or regenerate from History

## Notes

- API keys are kept on the backend only
- Uploaded study material should stay grounded to the session
- Old saved sessions preserve the output from the time they were generated, so regeneration is required to see newer generation logic

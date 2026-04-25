import Link from "next/link";
import { ArrowRight, BrainCircuit, Database, FileText, Sparkles } from "lucide-react";

const FEATURES = [
  {
    icon: FileText,
    title: "Smart Extraction",
    description: "Reads DOCX directly, parses text PDFs, and auto-switches to OCR for scanned pages.",
  },
  {
    icon: Database,
    title: "Grounded RAG Engine",
    description:
      "Chunks and embeds your notes with all-MiniLM-L6-v2, stores vectors in ChromaDB, and retrieves source-tagged context.",
  },
  {
    icon: BrainCircuit,
    title: "Structured Generation",
    description:
      "Gemini produces summaries, flashcards, Q&A, viva prompts, and difficulty-tiered explanations with strict source grounding.",
  },
];

const WORKFLOW = [
  "Upload PDF/DOCX",
  "Extract text + OCR fallback",
  "Clean and chunk content",
  "Embed + store in ChromaDB",
  "Retrieve relevant chunks",
  "Generate structured study guide",
];

export default function HomePage() {
  return (
    <div className="page-stack">
      <section className="hero-grid glass-card hero-card">
        <div>
          <p className="eyebrow">Capstone Full-Stack Project</p>
          <h1>
            Generative AI Agent for <span>Automated Study Guide Creation</span>
          </h1>
          <p className="hero-copy">
            Upload educational documents and generate polished study material in seconds. Built for students
            and professors with OCR, RAG retrieval, and source-grounded LLM generation.
          </p>

          <div className="hero-actions">
            <Link href="/dashboard" className="primary-button">
              Launch Dashboard
              <ArrowRight size={16} />
            </Link>
            <Link href="/history" className="secondary-button">
              View History
            </Link>
          </div>
        </div>

        <div className="hero-panel">
          <div className="hero-stat">
            <span>Outputs</span>
            <strong>Summary, Flashcards, Q and A, Difficulty Modes</strong>
          </div>
          <div className="hero-stat">
            <span>Pipeline</span>
            <strong>{"Extraction -> OCR -> Embedding -> Retrieval -> Gemini"}</strong>
          </div>
          <div className="hero-stat">
            <span>Model Stack</span>
            <strong>LangChain + ChromaDB + Gemini + Google Vision</strong>
          </div>
        </div>
      </section>

      <section className="feature-grid">
        {FEATURES.map((feature) => (
          <article key={feature.title} className="glass-card feature-card">
            <feature.icon size={20} />
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </article>
        ))}
      </section>

      <section className="glass-card workflow-card">
        <div className="workflow-header">
          <Sparkles size={18} />
          <h2>Expected User Flow</h2>
        </div>
        <div className="workflow-list">
          {WORKFLOW.map((item, index) => (
            <div key={item} className="workflow-row">
              <span>{index + 1}</span>
              <p>{item}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

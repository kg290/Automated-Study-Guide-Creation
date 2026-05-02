"use client";

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpenText,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Clipboard,
  Download,
  Files,
  HelpCircle,
  Layers3,
  LibraryBig,
  RefreshCcw,
  Sparkles,
} from "lucide-react";

import { StudyGuideResult } from "@/lib/types";
import ElectricBorder from "@/components/ElectricBorder";

type ViewTabId = "summary" | "flashcards" | "qa";

interface StudyGuideTabsProps {
  result: StudyGuideResult;
}

const VIEW_TABS: Array<{ id: ViewTabId; label: string }> = [
  { id: "summary", label: "Summary" },
  { id: "flashcards", label: "Flashcards" },
  { id: "qa", label: "Q&A" },
];

const CITATION_TAG_REGEX = /\s*\[(?:source:)?[^\]|]+\|\s*chunk:[^\]]+\]/gi;

function stripCitationTags(value: string): string {
  if (!value) {
    return "";
  }

  return value
    .replace(CITATION_TAG_REGEX, "")
    .replace(/\*\*/g, "")
    .replace(/`+/g, "")
    .replace(/\\\(/g, "(")
    .replace(/\\\)/g, ")")
    .replace(/\\\[/g, "[")
    .replace(/\\\]/g, "]")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function splitIntoParagraphs(value: string): string[] {
  const cleaned = stripCitationTags(value);
  if (!cleaned) {
    return [];
  }

  const explicitParagraphs = cleaned
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  if (explicitParagraphs.length > 1) {
    return explicitParagraphs;
  }

  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const grouped: string[] = [];
  for (let index = 0; index < sentences.length; index += 2) {
    grouped.push(sentences.slice(index, index + 2).join(" ").trim());
  }

  return grouped.filter(Boolean);
}

function sanitizeResult(result: StudyGuideResult): StudyGuideResult {
  const unitWiseSummaries = Array.isArray(result.unit_wise_summaries) ? result.unit_wise_summaries : [];
  const keyConcepts = Array.isArray(result.key_concepts) ? result.key_concepts : [];
  const topicWiseNotes = Array.isArray(result.topic_wise_notes) ? result.topic_wise_notes : [];
  const flashcards = Array.isArray(result.flashcards) ? result.flashcards : [];
  const qaSets = Array.isArray(result.qa_sets) ? result.qa_sets : [];
  const vivaQuestions = Array.isArray(result.viva_questions) ? result.viva_questions : [];
  const difficultyExplanations = Array.isArray(result.difficulty_explanations)
    ? result.difficulty_explanations
    : [];

  return {
    ...result,
    summary_notes: stripCitationTags(result.summary_notes),
    unit_wise_summaries: unitWiseSummaries.map((item) => ({
      ...item,
      unit_title: stripCitationTags(item.unit_title),
      summary: stripCitationTags(item.summary),
      key_points: (Array.isArray(item.key_points) ? item.key_points : []).map((point) => stripCitationTags(point)),
    })),
    key_concepts: keyConcepts.map((item) => stripCitationTags(item)),
    topic_wise_notes: topicWiseNotes.map((item) => ({
      ...item,
      topic: stripCitationTags(item.topic),
      notes: stripCitationTags(item.notes),
    })),
    flashcards: flashcards.map((item) => ({
      ...item,
      question: stripCitationTags(item.question),
      answer: stripCitationTags(item.answer),
      bloom_level: item.bloom_level || "remember",
    })),
    qa_sets: qaSets.map((item) => ({
      ...item,
      question: stripCitationTags(item.question),
      answer: stripCitationTags(item.answer),
      bloom_level: item.bloom_level || "remember",
    })),
    viva_questions: vivaQuestions.map((item) => stripCitationTags(item)),
    difficulty_explanations: difficultyExplanations.map((item) => ({
      ...item,
      explanation: stripCitationTags(item.explanation),
    })),
  };
}

function toMarkdown(result: StudyGuideResult): string {
  const lines: string[] = [];

  lines.push("# Automated Study Guide");
  lines.push("");
  lines.push("## Summary Notes");
  lines.push(result.summary_notes || "Not available in uploaded material");
  lines.push("");

  lines.push("## Unit and Chapter Guide");
  if (result.unit_wise_summaries.length === 0) {
    lines.push("Not available in uploaded material");
  } else {
    result.unit_wise_summaries.forEach((item) => {
      lines.push(`### ${item.unit_title}`);
      lines.push(item.summary);
      if (item.key_points.length > 0) {
        item.key_points.forEach((point) => lines.push(`- ${point}`));
      }
      lines.push("");
    });
  }

  lines.push("## Key Concepts");
  if (result.key_concepts.length === 0) {
    lines.push("- Not available in uploaded material");
  } else {
    result.key_concepts.forEach((item) => lines.push(`- ${item}`));
  }
  lines.push("");

  lines.push("## Topic-wise Notes");
  if (result.topic_wise_notes.length === 0) {
    lines.push("Not available in uploaded material");
  } else {
    result.topic_wise_notes.forEach((topic) => {
      lines.push(`### ${topic.topic}`);
      lines.push(topic.notes);
      lines.push("");
    });
  }

  lines.push("## Flashcards");
  if (result.flashcards.length === 0) {
    lines.push("Not available in uploaded material");
  } else {
    result.flashcards.forEach((item, index) => {
      lines.push(`${index + 1}. Card: ${item.question} (${item.bloom_level.toUpperCase()})`);
      lines.push(`   Details: ${item.answer}`);
    });
  }
  lines.push("");

  lines.push("## Q&A Sets");
  if (result.qa_sets.length === 0) {
    lines.push("Not available in uploaded material");
  } else {
    result.qa_sets.forEach((item, index) => {
      lines.push(`${index + 1}. Q: ${item.question} (${item.bloom_level.toUpperCase()})`);
      lines.push(`   A: ${item.answer}`);
    });
  }
  lines.push("");

  return lines.join("\n");
}

function notebookTitle(result: StudyGuideResult): string {
  if (result.source_documents.length > 0) {
    return result.source_documents[0].replace(/\.[^.]+$/, "");
  }

  if (result.unit_wise_summaries.length > 0) {
    return result.unit_wise_summaries[0].unit_title;
  }

  return "Study Guide";
}

export function StudyGuideTabs({ result }: StudyGuideTabsProps) {
  const cleanResult = useMemo(() => sanitizeResult(result), [result]);
  const markdown = useMemo(() => toMarkdown(cleanResult), [cleanResult]);
  const summaryParagraphs = useMemo(() => splitIntoParagraphs(cleanResult.summary_notes), [cleanResult.summary_notes]);
  const title = useMemo(() => notebookTitle(cleanResult), [cleanResult]);

  const [activeTab, setActiveTab] = useState<ViewTabId>("summary");
  const [flashcardIndex, setFlashcardIndex] = useState<number>(0);
  const [qaIndex, setQaIndex] = useState<number>(0);
  const [showFlashcardAnswer, setShowFlashcardAnswer] = useState<boolean>(false);
  const [showQaAnswer, setShowQaAnswer] = useState<boolean>(false);
  const [copyMessage, setCopyMessage] = useState<string>("");

  const promptSuggestions = useMemo(
    () =>
      [
        ...cleanResult.qa_sets.slice(0, 3).map((item) => item.question),
        ...cleanResult.flashcards.slice(0, 3).map((item) => item.question),
      ].filter(Boolean),
    [cleanResult.flashcards, cleanResult.qa_sets],
  );

  const activeFlashcard = cleanResult.flashcards[flashcardIndex] || null;
  const activeQa = cleanResult.qa_sets[qaIndex] || null;

  const handleCopy = async () => {
    const text =
      activeTab === "flashcards"
        ? cleanResult.flashcards
            .map((item, index) => `${index + 1}. ${item.question}\nDetails: ${item.answer}`)
            .join("\n\n")
        : activeTab === "qa"
          ? cleanResult.qa_sets.map((item, index) => `${index + 1}. ${item.question}\n${item.answer}`).join("\n\n")
          : markdown;

    await navigator.clipboard.writeText(text || markdown);
    setCopyMessage("Copied");
    window.setTimeout(() => setCopyMessage(""), 1300);
  };

  const handleDownload = () => {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "study-guide.md";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const moveFlashcard = (direction: -1 | 1) => {
    if (cleanResult.flashcards.length === 0) {
      return;
    }

    setShowFlashcardAnswer(false);
    setFlashcardIndex((current) => (current + direction + cleanResult.flashcards.length) % cleanResult.flashcards.length);
  };

  const moveQa = (direction: -1 | 1) => {
    if (cleanResult.qa_sets.length === 0) {
      return;
    }

    setShowQaAnswer(false);
    setQaIndex((current) => (current + direction + cleanResult.qa_sets.length) % cleanResult.qa_sets.length);
  };

  return (
    <section className="study-tabs-shell glass-card">
      <div className="study-tabs-header">
        <div>
          <span className="eyebrow">Study Workspace</span>
          <h2 className="study-tabs-title">{title}</h2>
          <p className="study-tabs-subtitle">
            {cleanResult.source_documents.length} source{cleanResult.source_documents.length === 1 ? "" : "s"} grounded into this session
          </p>
        </div>

        <div className="studio-actions">
          <button type="button" className="mini-button" onClick={handleCopy}>
            <Clipboard size={14} />
            {copyMessage || "Copy"}
          </button>
          <button type="button" className="mini-button" onClick={handleDownload}>
            <Download size={14} />
            Download
          </button>
        </div>
      </div>

      <div className="study-view-tabs">
        {VIEW_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`study-view-tab ${activeTab === tab.id ? "study-view-tab-active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "summary" ? (
        <div className="summary-workspace">
          <div className="summary-main">
            <section className="summary-hero-card">
              <span className="workspace-label">Summary</span>
              <h3>{title}</h3>
              {summaryParagraphs.length > 0 ? (
                summaryParagraphs.map((paragraph, index) => <p key={`summary-${index}`}>{paragraph}</p>)
              ) : (
                <p>Not available in uploaded material.</p>
              )}
            </section>

            <section className="summary-section-card">
              <div className="workspace-section-header">
                <Sparkles size={16} />
                <h3>Key Concepts</h3>
              </div>
              <div className="concept-cloud">
                {cleanResult.key_concepts.length > 0 ? (
                  cleanResult.key_concepts.map((concept) => (
                    <span key={concept} className="concept-pill">
                      {concept}
                    </span>
                  ))
                ) : (
                  <span className="concept-pill">Not available in uploaded material</span>
                )}
              </div>
            </section>

            <section className="summary-section-card">
              <div className="workspace-section-header">
                <LibraryBig size={16} />
                <h3>Unit Guide</h3>
              </div>
              <div className="unit-stack">
                {cleanResult.unit_wise_summaries.length > 0 ? (
                  cleanResult.unit_wise_summaries.map((item, index) => (
                    <article className="unit-card" key={`${item.unit_title}-${index}`}>
                      <div className="unit-card-head">
                        <span className="unit-index">{index + 1}</span>
                        <div>
                          <h4>{item.unit_title}</h4>
                          <p>{item.summary || "Not available in uploaded material"}</p>
                        </div>
                      </div>
                      {item.key_points.length > 0 ? (
                        <ul className="unit-points">
                          {item.key_points.map((point, pointIndex) => (
                            <li key={`${item.unit_title}-${pointIndex}`}>{point}</li>
                          ))}
                        </ul>
                      ) : null}
                    </article>
                  ))
                ) : (
                  <article className="unit-card">
                    <p>Not available in uploaded material.</p>
                  </article>
                )}
              </div>
            </section>

            <section className="summary-section-card">
              <div className="workspace-section-header">
                <Layers3 size={16} />
                <h3>Topic Notes</h3>
              </div>
              <div className="topic-note-grid">
                {cleanResult.topic_wise_notes.length > 0 ? (
                  cleanResult.topic_wise_notes.map((item, index) => (
                    <article className="topic-note-card" key={`${item.topic}-${index}`}>
                      <h4>{item.topic}</h4>
                      <p>{item.notes}</p>
                    </article>
                  ))
                ) : (
                  <article className="topic-note-card">
                    <p>Not available in uploaded material.</p>
                  </article>
                )}
              </div>
            </section>
          </div>

          <aside className="summary-side">
            <section className="summary-side-card">
              <div className="workspace-section-header">
                <Files size={16} />
                <h3>Sources</h3>
              </div>
              <div className="source-list">
                {cleanResult.source_documents.map((source) => (
                  <article key={source} className="source-item">
                    <span className="source-bullet" />
                    <div>
                      <strong>{source}</strong>
                      <p>Uploaded source grounded into the current study session.</p>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="summary-side-card">
              <div className="source-summary-grid">
                <div className="source-stat-card">
                  <span>Units</span>
                  <strong>{cleanResult.unit_wise_summaries.length}</strong>
                </div>
                <div className="source-stat-card">
                  <span>Topics</span>
                  <strong>{cleanResult.topic_wise_notes.length}</strong>
                </div>
                <div className="source-stat-card">
                  <span>Cards</span>
                  <strong>{cleanResult.flashcards.length + cleanResult.qa_sets.length}</strong>
                </div>
              </div>
            </section>

            <section className="summary-side-card">
              <div className="workspace-section-header">
                <BrainCircuit size={16} />
                <h3>Study Prompts</h3>
              </div>
              <div className="prompt-chip-grid">
                {promptSuggestions.length > 0 ? (
                  promptSuggestions.map((prompt, index) => (
                    <button
                      key={`${prompt}-${index}`}
                      type="button"
                      className="prompt-chip"
                      onClick={() => {
                        const qaMatch = cleanResult.qa_sets.findIndex((item) => item.question === prompt);
                        if (qaMatch >= 0) {
                          setActiveTab("qa");
                          setQaIndex(qaMatch);
                          setShowQaAnswer(false);
                          return;
                        }

                        const flashcardMatch = cleanResult.flashcards.findIndex((item) => item.question === prompt);
                        if (flashcardMatch >= 0) {
                          setActiveTab("flashcards");
                          setFlashcardIndex(flashcardMatch);
                          setShowFlashcardAnswer(false);
                        }
                      }}
                    >
                      {prompt}
                    </button>
                  ))
                ) : (
                  <p className="empty-copy">No prompts available.</p>
                )}
              </div>
            </section>

            <section className="summary-side-card">
              <div className="workspace-section-header">
                <BookOpenText size={16} />
                <h3>Grounding</h3>
              </div>
              <p className="side-note-copy">
                Outputs in this workspace are constrained to the uploaded study material and cleaned for direct revision use.
              </p>
            </section>
          </aside>
        </div>
      ) : null}

      {activeTab === "flashcards" ? (
        <section className="studio-page">
          <div className="studio-page-copy">
            <span className="workspace-label">Flashcards</span>
            <h3>{title} Flashcards</h3>
            <p>Use concept-detail cards to review one important idea at a time and reveal the key explanation only when needed.</p>
          </div>

          <div className="studio-frame">
            <div className="studio-ambient studio-ambient-a" />
            <div className="studio-ambient studio-ambient-b" />

            <div className="qa-stage flashcard-stage">
              <div className="study-card-stage qa-question-row">
                <button
                  type="button"
                  className="study-card-nav"
                  onClick={() => moveFlashcard(-1)}
                  disabled={cleanResult.flashcards.length === 0}
                >
                  <ChevronLeft size={18} />
                </button>

                <AnimatePresence mode="wait">
                  <motion.div
                    className="study-card-motion-shell"
                    key={`flashcard-question-${flashcardIndex}`}
                    initial={{ opacity: 0, y: 16, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -12, scale: 0.98 }}
                    transition={{ duration: 0.22, ease: "easeOut" }}
                  >
                    <ElectricBorder
                      color="#7df9ff"
                      speed={1}
                      chaos={0.12}
                      borderRadius={32}
                      className="electric-border-card electric-border-card-flashcard"
                      style={{ borderRadius: 32 }}
                    >
                      <article className="study-card study-card-qa study-card-flashcard-qa">
                        {activeFlashcard ? (
                          <>
                            <div className="study-card-meta">
                              <span className="bloom-badge">{activeFlashcard.bloom_level}</span>
                              <span>
                                {flashcardIndex + 1} / {cleanResult.flashcards.length}
                              </span>
                            </div>
                            <div className="study-card-face study-card-face-qa">
                              <p className="study-card-question study-card-question-qa study-card-question-flashcard-title">
                                {activeFlashcard.question}
                              </p>
                              <button type="button" className="mini-ghost-button" onClick={() => setShowFlashcardAnswer((current) => !current)}>
                                {showFlashcardAnswer ? "Hide details" : "Open details"}
                              </button>
                            </div>
                          </>
                        ) : (
                          <div className="study-card-face">
                            <p className="study-card-question">No flashcards available for this material.</p>
                          </div>
                        )}
                      </article>
                    </ElectricBorder>
                  </motion.div>
                </AnimatePresence>

                <button
                  type="button"
                  className="study-card-nav"
                  onClick={() => moveFlashcard(1)}
                  disabled={cleanResult.flashcards.length === 0}
                >
                  <ChevronRight size={18} />
                </button>
              </div>

              <AnimatePresence>
                {showFlashcardAnswer && activeFlashcard ? (
                  <motion.section
                    key={`flashcard-answer-${flashcardIndex}`}
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 12 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                  >
                    <div className="qa-answer-panel flashcard-answer-panel">
                      <div className="qa-answer-panel-header">
                        <span className="workspace-label">Details</span>
                        <span className="qa-answer-length">Concept snapshot</span>
                      </div>
                      <p>{activeFlashcard.answer}</p>
                    </div>
                  </motion.section>
                ) : null}
              </AnimatePresence>
            </div>

            <div className="studio-footer">
              <div className="studio-progress">
                <button
                  type="button"
                  className="progress-reset"
                  onClick={() => {
                    setFlashcardIndex(0);
                    setShowFlashcardAnswer(false);
                  }}
                >
                  <RefreshCcw size={14} />
                  Restart
                </button>
                <div className="progress-line">
                  <span
                    style={{
                      width: cleanResult.flashcards.length > 0 ? `${((flashcardIndex + 1) / cleanResult.flashcards.length) * 100}%` : "0%",
                    }}
                  />
                </div>
                <span className="progress-count">
                  {cleanResult.flashcards.length > 0 ? `${flashcardIndex + 1} / ${cleanResult.flashcards.length}` : "0 / 0"}
                </span>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {activeTab === "qa" ? (
        <section className="studio-page">
          <div className="studio-page-copy">
            <span className="workspace-label">Q&A</span>
            <h3>{title} Q&A</h3>
            <p>Practice longer explanation-style questions with grounded answers that teach the reasoning behind the topic.</p>
          </div>

          <div className="studio-frame">
            <div className="studio-ambient studio-ambient-a" />
            <div className="studio-ambient studio-ambient-b" />

            <div className="qa-stage">
              <div className="study-card-stage qa-question-row">
                <button
                  type="button"
                  className="study-card-nav"
                  onClick={() => moveQa(-1)}
                  disabled={cleanResult.qa_sets.length === 0}
                >
                  <ChevronLeft size={18} />
                </button>

                <AnimatePresence mode="wait">
                  <motion.div
                    className="study-card-motion-shell"
                    key={`qa-question-${qaIndex}`}
                    initial={{ opacity: 0, y: 16, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -12, scale: 0.98 }}
                    transition={{ duration: 0.22, ease: "easeOut" }}
                  >
                    <ElectricBorder
                      color="#7df9ff"
                      speed={1}
                      chaos={0.12}
                      borderRadius={32}
                      className="electric-border-card electric-border-card-qa"
                      style={{ borderRadius: 32 }}
                    >
                      <article className="study-card study-card-qa">
                        {activeQa ? (
                          <>
                            <div className="study-card-meta">
                              <span className="bloom-badge">{activeQa.bloom_level}</span>
                              <span>
                                {qaIndex + 1} / {cleanResult.qa_sets.length}
                              </span>
                            </div>
                            <div className="study-card-face study-card-face-qa">
                              <p className="study-card-question study-card-question-qa">{activeQa.question}</p>
                              <button type="button" className="mini-ghost-button" onClick={() => setShowQaAnswer((current) => !current)}>
                                {showQaAnswer ? "Hide answer" : "Explain answer"}
                              </button>
                            </div>
                          </>
                        ) : (
                          <div className="study-card-face">
                            <p className="study-card-question">No Q&A prompts available for this material.</p>
                          </div>
                        )}
                      </article>
                    </ElectricBorder>
                  </motion.div>
                </AnimatePresence>

                <button
                  type="button"
                  className="study-card-nav"
                  onClick={() => moveQa(1)}
                  disabled={cleanResult.qa_sets.length === 0}
                >
                  <ChevronRight size={18} />
                </button>
              </div>

              <AnimatePresence>
                {showQaAnswer && activeQa ? (
                  <motion.section
                    key={`qa-answer-${qaIndex}`}
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 12 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                  >
                    <div className="qa-answer-panel">
                      <div className="qa-answer-panel-header">
                        <span className="workspace-label">Answer</span>
                        <span className="qa-answer-length">Detailed explanation</span>
                      </div>
                      <p>{activeQa.answer}</p>
                    </div>
                  </motion.section>
                ) : null}
              </AnimatePresence>
            </div>

            <div className="studio-footer">
              <div className="studio-progress">
                <button
                  type="button"
                  className="progress-reset"
                  onClick={() => {
                    setQaIndex(0);
                    setShowQaAnswer(false);
                  }}
                >
                  <RefreshCcw size={14} />
                  Restart
                </button>
                <div className="progress-line">
                  <span
                    style={{
                      width: cleanResult.qa_sets.length > 0 ? `${((qaIndex + 1) / cleanResult.qa_sets.length) * 100}%` : "0%",
                    }}
                  />
                </div>
                <span className="progress-count">
                  {cleanResult.qa_sets.length > 0 ? `${qaIndex + 1} / ${cleanResult.qa_sets.length}` : "0 / 0"}
                </span>
              </div>

              <div className="feedback-row">
                <button type="button" className="mini-button">
                  <Sparkles size={14} />
                  Good content
                </button>
                <button type="button" className="mini-button">
                  <HelpCircle size={14} />
                  Bad content
                </button>
              </div>
            </div>
          </div>
        </section>
      ) : null}
    </section>
  );
}

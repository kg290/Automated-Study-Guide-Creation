"use client";

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Clipboard, Download } from "lucide-react";

import { StudyGuideResult } from "@/lib/types";

type TabId = "summary" | "units" | "flashcards" | "qa" | "difficulty";

interface StudyGuideTabsProps {
  result: StudyGuideResult;
}

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "summary", label: "Summary" },
  { id: "units", label: "Unit Guide" },
  { id: "flashcards", label: "Flashcards" },
  { id: "qa", label: "Q&A" },
  { id: "difficulty", label: "Difficulty Modes" },
];

const CITATION_TAG_REGEX = /\s*\[source:[^\]|]+\|\s*chunk:[^\]]+\]/gi;

function stripCitationTags(value: string): string {
  if (!value) {
    return "";
  }

  return value
    .replace(CITATION_TAG_REGEX, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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
      key_points: (Array.isArray(item.key_points) ? item.key_points : []).map((point) =>
        stripCitationTags(point),
      ),
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
      lines.push(`${index + 1}. Q: ${item.question} (${item.bloom_level.toUpperCase()})`);
      lines.push(`   A: ${item.answer}`);
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

  lines.push("## Viva Questions");
  if (result.viva_questions.length === 0) {
    lines.push("- Not available in uploaded material");
  } else {
    result.viva_questions.forEach((item) => lines.push(`- ${item}`));
  }
  lines.push("");

  lines.push("## Difficulty Explanations");
  if (result.difficulty_explanations.length === 0) {
    lines.push("Not available in uploaded material");
  } else {
    result.difficulty_explanations.forEach((item) => {
      lines.push(`### ${item.mode}`);
      lines.push(item.explanation);
      lines.push("");
    });
  }

  return lines.join("\n");
}

function sectionContent(tabId: TabId, result: StudyGuideResult): string {
  if (tabId === "summary") {
    return [
      "Summary Notes",
      result.summary_notes,
      "",
      "Unit and Chapter Guide",
      ...result.unit_wise_summaries.map(
        (item) =>
          `${item.unit_title}: ${item.summary}${
            item.key_points.length ? `\n- ${item.key_points.join("\n- ")}` : ""
          }`,
      ),
      "",
      "Key Concepts",
      ...result.key_concepts,
      "",
      "Topic-wise Notes",
      ...result.topic_wise_notes.map((item) => `${item.topic}: ${item.notes}`),
      "",
      "Viva Questions",
      ...result.viva_questions,
    ].join("\n");
  }

  if (tabId === "units") {
    return result.unit_wise_summaries
      .map(
        (item) =>
          `${item.unit_title}\n${item.summary}${
            item.key_points.length ? `\n- ${item.key_points.join("\n- ")}` : ""
          }`,
      )
      .join("\n\n");
  }

  if (tabId === "flashcards") {
    return result.flashcards
      .map((item, index) => `${index + 1}. [${item.bloom_level.toUpperCase()}] ${item.question}\n${item.answer}`)
      .join("\n\n");
  }

  if (tabId === "qa") {
    return result.qa_sets
      .map((item, index) => `${index + 1}. [${item.bloom_level.toUpperCase()}] ${item.question}\n${item.answer}`)
      .join("\n\n");
  }

  return result.difficulty_explanations
    .map((item) => `${item.mode.toUpperCase()}\n${item.explanation}`)
    .join("\n\n");
}

export function StudyGuideTabs({ result }: StudyGuideTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("summary");
  const [copyMessage, setCopyMessage] = useState<string>("");

  const cleanResult = useMemo(() => sanitizeResult(result), [result]);
  const markdown = useMemo(() => toMarkdown(cleanResult), [cleanResult]);

  const handleCopy = async () => {
    const text = sectionContent(activeTab, cleanResult);
    await navigator.clipboard.writeText(text);
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

  const tabContent =
    activeTab === "summary" ? (
      <div className="content-stack">
        <article className="result-block">
          <h3>Summary Notes</h3>
          <p>{cleanResult.summary_notes || "Not available in uploaded material"}</p>
        </article>

        <article className="result-block">
          <h3>Unit and Chapter Guide</h3>
          <div className="card-grid">
            {cleanResult.unit_wise_summaries.length > 0 ? (
              cleanResult.unit_wise_summaries.map((item, index) => (
                <motion.div
                  className="result-card"
                  key={`${item.unit_title}-${index}`}
                  whileHover={{ y: -4 }}
                  transition={{ duration: 0.2 }}
                >
                  <h4>{item.unit_title}</h4>
                  <p>{item.summary}</p>
                  {item.key_points.length > 0 ? (
                    <ul className="list-grid">
                      {item.key_points.map((point, pointIndex) => (
                        <li key={`${item.unit_title}-point-${pointIndex}`}>{point}</li>
                      ))}
                    </ul>
                  ) : null}
                </motion.div>
              ))
            ) : (
              <div className="result-card">
                <p>Not available in uploaded material</p>
              </div>
            )}
          </div>
        </article>

        <article className="result-block">
          <h3>Key Concepts</h3>
          <ul className="list-grid">
            {cleanResult.key_concepts.length > 0 ? (
              cleanResult.key_concepts.map((item) => <li key={item}>{item}</li>)
            ) : (
              <li>Not available in uploaded material</li>
            )}
          </ul>
        </article>

        <article className="result-block">
          <h3>Topic-wise Notes</h3>
          <div className="card-grid">
            {cleanResult.topic_wise_notes.length > 0 ? (
              cleanResult.topic_wise_notes.map((item) => (
                <motion.div className="result-card" key={item.topic} whileHover={{ y: -4 }} transition={{ duration: 0.2 }}>
                  <h4>{item.topic}</h4>
                  <p>{item.notes}</p>
                </motion.div>
              ))
            ) : (
              <div className="result-card">
                <p>Not available in uploaded material</p>
              </div>
            )}
          </div>
        </article>

        <article className="result-block">
          <h3>Viva Questions</h3>
          <ul className="list-grid">
            {cleanResult.viva_questions.length > 0 ? (
              cleanResult.viva_questions.map((item) => <li key={item}>{item}</li>)
            ) : (
              <li>Not available in uploaded material</li>
            )}
          </ul>
        </article>
      </div>
    ) : null;

  return (
    <section className="glass-card">
      <div className="tabs-header">
        <div className="tabs-row">
          {TABS.map((tab) => (
            <motion.button
              key={tab.id}
              type="button"
              className={`tab-pill ${activeTab === tab.id ? "tab-pill-active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.97 }}
            >
              {tab.label}
            </motion.button>
          ))}
        </div>
        <div className="action-row">
          <motion.button type="button" className="mini-button" onClick={handleCopy} whileHover={{ y: -2 }}>
            <Clipboard size={14} />
            {copyMessage || "Copy"}
          </motion.button>
          <motion.button type="button" className="mini-button" onClick={handleDownload} whileHover={{ y: -2 }}>
            <Download size={14} />
            Download
          </motion.button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          className="tab-panel"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.24, ease: "easeOut" }}
        >
          {activeTab === "summary" ? tabContent : null}

          {activeTab === "units" ? (
            <div className="card-grid">
              {cleanResult.unit_wise_summaries.length > 0 ? (
                cleanResult.unit_wise_summaries.map((item, index) => (
                  <motion.article
                    className="result-card"
                    key={`${item.unit_title}-${index}`}
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                  >
                    <h4>{item.unit_title}</h4>
                    <p>{item.summary}</p>
                    {item.key_points.length > 0 ? (
                      <ul className="list-grid">
                        {item.key_points.map((point, pointIndex) => (
                          <li key={`${item.unit_title}-tab-point-${pointIndex}`}>{point}</li>
                        ))}
                      </ul>
                    ) : null}
                  </motion.article>
                ))
              ) : (
                <div className="result-card">
                  <p>No unit or chapter summaries generated for this session.</p>
                </div>
              )}
            </div>
          ) : null}

          {activeTab === "flashcards" ? (
            <div className="card-grid">
              {cleanResult.flashcards.length > 0 ? (
                cleanResult.flashcards.map((card, index) => (
                  <motion.article
                    className="result-card"
                    key={`${card.question}-${index}`}
                    whileHover={{ y: -4, rotate: -0.4 }}
                    transition={{ duration: 0.2 }}
                  >
                    <span className="badge">{card.bloom_level}</span>
                    <h4>{card.question}</h4>
                    <p>{card.answer}</p>
                  </motion.article>
                ))
              ) : (
                <div className="result-card">
                  <p>No flashcards generated for this session.</p>
                </div>
              )}
            </div>
          ) : null}

          {activeTab === "qa" ? (
            <div className="card-grid">
              {cleanResult.qa_sets.length > 0 ? (
                cleanResult.qa_sets.map((item, index) => (
                  <motion.article
                    className="result-card"
                    key={`${item.question}-${index}`}
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                  >
                    <span className="badge">{item.bloom_level}</span>
                    <h4>{item.question}</h4>
                    <p>{item.answer}</p>
                  </motion.article>
                ))
              ) : (
                <div className="result-card">
                  <p>No Q&A content generated for this session.</p>
                </div>
              )}
            </div>
          ) : null}

          {activeTab === "difficulty" ? (
            <div className="card-grid">
              {cleanResult.difficulty_explanations.length > 0 ? (
                cleanResult.difficulty_explanations.map((item) => (
                  <motion.article
                    className="result-card"
                    key={item.mode}
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                  >
                    <span className="badge">{item.mode}</span>
                    <p>{item.explanation}</p>
                  </motion.article>
                ))
              ) : (
                <div className="result-card">
                  <p>No difficulty-mode explanation generated for this session.</p>
                </div>
              )}
            </div>
          ) : null}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

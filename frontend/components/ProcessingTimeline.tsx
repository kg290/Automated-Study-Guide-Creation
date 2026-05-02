"use client";

import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { BookOpenText, BrainCircuit, ScanSearch, Sparkles } from "lucide-react";

interface ProcessingTimelineProps {
  stage: string;
  progress: number;
  message: string;
  elapsedSeconds?: number;
}

const STAGES = ["queued", "extracting", "structuring", "embedding", "retrieving", "mapping", "generating", "validating", "completed"];
const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  extracting: "Extracting Text",
  structuring: "Structuring Sections",
  embedding: "Building Embeddings",
  retrieving: "Retrieving Context",
  mapping: "Mapping Evidence",
  generating: "Generating Output",
  validating: "Validating Result",
  completed: "Completed",
};

const LIVE_MESSAGES: Partial<Record<string, string[]>> = {
  extracting: ["Reading pages and text layers.", "Checking for OCR-worthy regions."],
  structuring: ["Grouping content into sections.", "Organizing the material for downstream generation."],
  embedding: ["Preparing retrieval support.", "Linking chunks to semantic search space."],
  retrieving: ["Collecting evidence across the document.", "Pulling representative material from multiple sections."],
  mapping: ["Combining structure with retrieved evidence.", "Preparing the final grounded generation context."],
  generating: ["Writing summaries, flashcards, and Q&A.", "Composing final study artifacts from the document map."],
  validating: ["Cleaning and validating output quality.", "Preparing the final result for display."],
};

export function ProcessingTimeline({ stage, progress, message, elapsedSeconds = 0 }: ProcessingTimelineProps) {
  const currentStageIndex = STAGES.indexOf(stage);
  const stageLabel = STAGE_LABELS[stage] || "Processing";
  const prefersReducedMotion = useReducedMotion();
  const [messageIndex, setMessageIndex] = useState(0);
  const floatingSteps = [
    { icon: ScanSearch, label: "Extract", className: "processing-chip-a" },
    { icon: BrainCircuit, label: "Retrieve", className: "processing-chip-b" },
    { icon: Sparkles, label: "Compose", className: "processing-chip-c" },
  ];
  const liveMessages = LIVE_MESSAGES[stage] || ["Pipeline is actively working through your material."];
  const liveMessage = liveMessages[messageIndex % liveMessages.length];
  const showSlowHint = elapsedSeconds >= 45 && ["retrieving", "mapping", "generating", "validating"].includes(stage);

  useEffect(() => {
    setMessageIndex(0);
    if (prefersReducedMotion || liveMessages.length <= 1) {
      return;
    }

    const timer = window.setInterval(() => {
      setMessageIndex((current) => (current + 1) % liveMessages.length);
    }, 2200);

    return () => window.clearInterval(timer);
  }, [liveMessages, prefersReducedMotion]);

  return (
    <section className="glass-card processing-card processing-shell">
      <div className="processing-copy">
        <div className="processing-meta">
          <span className="eyebrow">Pipeline In Motion</span>
          <h2>Processing Your Documents</h2>
          <p>{message || "Preparing your study guide pipeline..."}</p>
          <div className="submit-row">
            <span className="status-chip status-chip-live">
              <span className="live-dot" />
              Current stage: {stageLabel}
            </span>
            <span className="status-chip file-counter">{progress}% mapped</span>
          </div>
          <div className="processing-live-note">
            <motion.span
              key={`${stage}-${messageIndex}`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              {liveMessage}
            </motion.span>
            {showSlowHint ? <span className="processing-slow-hint">Longer technical PDFs can spend extra time in the final generation phase.</span> : null}
          </div>
        </div>

        <div className="progress-track">
          <motion.span
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.55, ease: "easeOut" }}
          />
        </div>
        <p className="progress-value">{progress}% complete</p>

        <div className="processing-stage-list">
          {STAGES.map((item, index) => {
            const completed = index < currentStageIndex || stage === "completed";
            const current = index === currentStageIndex && stage !== "completed";
            return (
              <div
                className={`processing-stage-pill ${completed ? "processing-stage-pill-active" : ""} ${current ? "processing-stage-pill-current" : ""}`}
                key={item}
              >
                {STAGE_LABELS[item]}
              </div>
            );
          })}
        </div>

        <div className="timeline-grid">
          {STAGES.map((item, index) => {
            const completed = index < currentStageIndex || stage === "completed";
            const current = index === currentStageIndex && stage !== "completed";
            return (
              <motion.div
                className={`timeline-node ${completed ? "timeline-node-active" : ""} ${current ? "timeline-node-current" : ""}`}
                key={item}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.26, delay: index * 0.06, ease: "easeOut" }}
              >
                <div className="timeline-dot" />
                <span>{item}</span>
              </motion.div>
            );
          })}
        </div>
      </div>

      <div className="processing-visual" aria-hidden="true">
        <motion.div
          className="book-stage"
          animate={
            prefersReducedMotion
              ? undefined
              : {
                  y: [0, -8, 0],
                }
          }
          transition={{ duration: 4.2, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
        >
          <motion.div
            className="book-glow"
            animate={
              prefersReducedMotion
                ? undefined
                : {
                    scale: [1, 1.08, 1],
                    opacity: [0.45, 0.8, 0.45],
                  }
            }
            transition={{ duration: 3.6, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
          />
          <motion.div
            className="signal-ring"
            animate={prefersReducedMotion ? undefined : { scale: [0.9, 1.14], opacity: [0.6, 0] }}
            transition={{ duration: 2.2, repeat: Number.POSITIVE_INFINITY, ease: "easeOut" }}
          />
          <div className="book-base" />
          <div className="book-spine" />
          <motion.div
            className="book-page book-page-left"
            animate={prefersReducedMotion ? undefined : { rotate: [-8, -18, -8], x: [0, -3, 0] }}
            transition={{ duration: 2.8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
          />
          <motion.div
            className="book-page book-page-right"
            animate={prefersReducedMotion ? undefined : { rotate: [8, 18, 8], x: [0, 3, 0] }}
            transition={{ duration: 2.8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut", delay: 0.15 }}
          />
          <div className="book-core">
            <BookOpenText size={34} />
          </div>

          {floatingSteps.map((item, index) => (
            <motion.div
              key={item.label}
              className={`processing-chip ${item.className}`}
              animate={
                prefersReducedMotion
                  ? undefined
                  : {
                      y: [0, -10, 0],
                      rotate: [0, index % 2 === 0 ? 4 : -4, 0],
                    }
              }
              transition={{
                duration: 3.4 + index * 0.4,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
                delay: index * 0.25,
              }}
            >
              <item.icon size={14} />
              {item.label}
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

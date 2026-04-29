"use client";

import { motion, useReducedMotion } from "framer-motion";
import { BookOpenText, BrainCircuit, ScanSearch, Sparkles } from "lucide-react";

interface ProcessingTimelineProps {
  stage: string;
  progress: number;
  message: string;
}

const STAGES = ["queued", "extracting", "embedding", "retrieving", "generating", "completed"];
const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  extracting: "Extracting Text",
  embedding: "Building Embeddings",
  retrieving: "Retrieving Context",
  generating: "Generating Output",
  completed: "Completed",
};

export function ProcessingTimeline({ stage, progress, message }: ProcessingTimelineProps) {
  const currentStageIndex = STAGES.indexOf(stage);
  const stageLabel = STAGE_LABELS[stage] || "Processing";
  const prefersReducedMotion = useReducedMotion();
  const floatingSteps = [
    { icon: ScanSearch, label: "Extract", className: "processing-chip-a" },
    { icon: BrainCircuit, label: "Retrieve", className: "processing-chip-b" },
    { icon: Sparkles, label: "Compose", className: "processing-chip-c" },
  ];

  return (
    <section className="glass-card processing-card processing-shell">
      <div className="processing-copy">
        <div className="processing-meta">
          <span className="eyebrow">Pipeline In Motion</span>
          <h2>Processing Your Documents</h2>
          <p>{message || "Preparing your study guide pipeline..."}</p>
          <div className="submit-row">
            <span className="status-chip">Current stage: {stageLabel}</span>
            <span className="status-chip file-counter">{progress}% mapped</span>
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
            const completed = index <= currentStageIndex || stage === "completed";
            return (
              <div
                className={`processing-stage-pill ${completed ? "processing-stage-pill-active" : ""}`}
                key={item}
              >
                {STAGE_LABELS[item]}
              </div>
            );
          })}
        </div>

        <div className="timeline-grid">
          {STAGES.map((item, index) => {
            const completed = index <= currentStageIndex || stage === "completed";
            return (
              <motion.div
                className={`timeline-node ${completed ? "timeline-node-active" : ""}`}
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

"use client";

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

  return (
    <section className="glass-card processing-card">
      <div className="processing-meta">
        <h2>Processing Your Documents</h2>
        <p>{message || "Preparing your study guide pipeline..."}</p>
        <span className="status-chip">Current stage: {stageLabel}</span>
      </div>

      <div className="progress-track">
        <span style={{ width: `${progress}%` }} />
      </div>
      <p className="progress-value">{progress}% complete</p>

      <div className="timeline-grid">
        {STAGES.map((item, index) => {
          const completed = index <= currentStageIndex || stage === "completed";
          return (
            <div className={`timeline-node ${completed ? "timeline-node-active" : ""}`} key={item}>
              <div className="timeline-dot" />
              <span>{item}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

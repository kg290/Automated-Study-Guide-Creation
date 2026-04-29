"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { BookOpenText, Files, Layers3 } from "lucide-react";

import { StudyGuideTabs } from "@/components/StudyGuideTabs";
import { getJobStatus, regenerateSession } from "@/lib/api";
import { DEFAULT_BLOOM_LEVELS, DEFAULT_DIFFICULTY_MODES, DEFAULT_OUTPUT_TYPES } from "@/lib/constants";
import { JobStatusResponse } from "@/lib/types";

export default function ResultsPage() {
  const params = useParams<{ jobId: string }>();
  const router = useRouter();
  const jobId = useMemo(
    () => (Array.isArray(params.jobId) ? params.jobId[0] : params.jobId),
    [params.jobId],
  );

  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [regenerating, setRegenerating] = useState<boolean>(false);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        if (cancelled) {
          return;
        }

        setJob(status);
        setLoading(false);

        if (status.status === "processing" || status.status === "queued") {
          return;
        }

        if (status.status === "failed") {
          setError(status.error || "Generation failed for this job.");
        }
      } catch {
        if (!cancelled) {
          setError("Unable to load generated result.");
          setLoading(false);
        }
      }
    };

    poll();
    timer = setInterval(poll, 2500);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId]);

  const handleRegenerate = async () => {
    if (!job?.session_id) {
      return;
    }

    setRegenerating(true);
    try {
      const response = await regenerateSession(job.session_id, {
        output_types: DEFAULT_OUTPUT_TYPES,
        difficulty_modes: DEFAULT_DIFFICULTY_MODES,
        bloom_levels: DEFAULT_BLOOM_LEVELS,
        custom_prompt: "",
      });
      router.push(`/processing/${response.job_id}`);
    } catch {
      setError("Regeneration failed. Please try again.");
    } finally {
      setRegenerating(false);
    }
  };

  if (loading) {
    return (
      <div className="page-stack">
        <section className="glass-card">
          <h2>Loading result...</h2>
          <p>Please wait while we fetch your generated output.</p>
        </section>
      </div>
    );
  }

  if (error || !job || job.status === "failed") {
    return (
      <div className="page-stack">
        <section className="glass-card">
          <h2>Result Unavailable</h2>
          <p className="form-error">{error || job?.error || "This job did not complete successfully."}</p>
          <div className="submit-row">
            <Link href="/dashboard" className="secondary-button">
              Back to Dashboard
            </Link>
          </div>
        </section>
      </div>
    );
  }

  if (!job.result) {
    return (
      <div className="page-stack">
        <section className="glass-card">
          <h2>Still Processing</h2>
          <p>The job is not complete yet.</p>
          <Link className="primary-button" href={`/processing/${jobId}`}>
            Open Processing Page
          </Link>
        </section>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <motion.section
        className="page-title-block"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        <div>
          <h1>Generated Study Guide</h1>
          <p>Output grounded in your uploaded documents through retrieval-augmented generation.</p>
        </div>
      </motion.section>

      <section className="results-overview-grid">
        <div className="glass-card overview-stat">
          <Files size={18} />
          <strong>{job.result.source_documents.length}</strong>
          <span>source document{job.result.source_documents.length === 1 ? "" : "s"}</span>
        </div>
        <div className="glass-card overview-stat">
          <BookOpenText size={18} />
          <strong>{job.result.unit_wise_summaries.length}</strong>
          <span>unit summaries</span>
        </div>
        <div className="glass-card overview-stat">
          <Layers3 size={18} />
          <strong>{job.result.flashcards.length + job.result.qa_sets.length}</strong>
          <span>study prompts</span>
        </div>
      </section>

      <section className="info-banner">
        <strong>Source Files:</strong> {job.result.source_documents.join(", ") || "Unknown source"}
      </section>
      <section className="info-banner info-banner-strong">
        <strong>Grounding:</strong> Responses are constrained to retrieved chunks from the uploaded session only.
      </section>

      <section className="submit-row">
        <button type="button" className="primary-button" onClick={handleRegenerate} disabled={regenerating}>
          {regenerating ? "Regenerating..." : "Regenerate Outputs"}
        </button>
        <Link className="secondary-button" href={`/history/${job.session_id}`}>
          Open History Session
        </Link>
      </section>

      <StudyGuideTabs result={job.result} />
    </div>
  );
}

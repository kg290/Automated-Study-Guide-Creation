"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

import { ProcessingTimeline } from "@/components/ProcessingTimeline";
import { getJobStatus } from "@/lib/api";
import { JobStatusResponse } from "@/lib/types";

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  if (minutes <= 0) {
    return `${remaining}s`;
  }
  return `${minutes}m ${remaining.toString().padStart(2, "0")}s`;
}

export default function ProcessingPage() {
  const params = useParams<{ jobId: string }>();
  const router = useRouter();
  const jobId = useMemo(
    () => (Array.isArray(params.jobId) ? params.jobId[0] : params.jobId),
    [params.jobId],
  );

  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

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

        if (status.status === "completed") {
          router.replace(`/results/${jobId}`);
          return;
        }

        if (status.status === "failed") {
          setError(status.error || "Generation failed.");
        }
      } catch {
        if (!cancelled) {
          setError("Unable to fetch processing status.");
        }
      }
    };

    poll();
    timer = setInterval(poll, 2200);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId, router]);

  useEffect(() => {
    const start = Date.now();
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - start) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="page-stack">
      <section className="page-title-block">
        <div>
          <h1>Processing</h1>
          <p>The AI pipeline is extracting, indexing, retrieving, and generating your study guide.</p>
        </div>
      </section>

      <ProcessingTimeline
        stage={job?.stage || "queued"}
        progress={job?.progress || 0}
        message={job?.message || "Initializing pipeline..."}
        elapsedSeconds={elapsedSeconds}
      />
      <section className="info-banner">
        <strong>Elapsed:</strong> {formatElapsed(elapsedSeconds)}
      </section>

      {error ? (
        <section className="glass-card">
          <p className="form-error">{error}</p>
          <Link className="secondary-button" href="/dashboard">
            Back to Dashboard
          </Link>
        </section>
      ) : null}
    </div>
  );
}

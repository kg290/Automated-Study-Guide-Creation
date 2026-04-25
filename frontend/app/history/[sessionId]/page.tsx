"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { StudyGuideTabs } from "@/components/StudyGuideTabs";
import { getHistoryDetail, regenerateSession } from "@/lib/api";
import { DEFAULT_BLOOM_LEVELS, DEFAULT_DIFFICULTY_MODES, DEFAULT_OUTPUT_TYPES } from "@/lib/constants";
import { HistoryDetail } from "@/lib/types";

export default function HistoryDetailPage() {
  const params = useParams<{ sessionId: string }>();
  const router = useRouter();
  const sessionId = useMemo(
    () => (Array.isArray(params.sessionId) ? params.sessionId[0] : params.sessionId),
    [params.sessionId],
  );

  const [session, setSession] = useState<HistoryDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [regenerating, setRegenerating] = useState<boolean>(false);

  useEffect(() => {
    const loadSession = async () => {
      try {
        const detail = await getHistoryDetail(sessionId);
        setSession(detail);
      } catch {
        setError("Unable to load this history session.");
      } finally {
        setLoading(false);
      }
    };

    loadSession();
  }, [sessionId]);

  const handleRegenerate = async () => {
    setRegenerating(true);
    try {
      const response = await regenerateSession(sessionId, {
        output_types: DEFAULT_OUTPUT_TYPES,
        difficulty_modes: DEFAULT_DIFFICULTY_MODES,
        bloom_levels: session?.options.bloom_levels?.length
          ? session.options.bloom_levels
          : DEFAULT_BLOOM_LEVELS,
        custom_prompt: session?.options.custom_prompt || "",
      });
      router.push(`/processing/${response.job_id}`);
    } catch {
      setError("Failed to regenerate this session.");
    } finally {
      setRegenerating(false);
    }
  };

  if (loading) {
    return (
      <div className="page-stack">
        <section className="glass-card">
          <h2>Loading session...</h2>
        </section>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="page-stack">
        <section className="glass-card">
          <h2>Session Unavailable</h2>
          <p className="form-error">{error || "Unable to open this history item."}</p>
        </section>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <section className="page-title-block">
        <div>
          <h1>History Session Detail</h1>
          <p>Session ID: {session.session_id}</p>
        </div>
      </section>

      <section className="info-banner">
        <strong>Files:</strong> {session.file_names.join(", ")}
      </section>

      <section className="submit-row">
        <button type="button" className="primary-button" onClick={handleRegenerate} disabled={regenerating}>
          {regenerating ? "Regenerating..." : "Regenerate From This Session"}
        </button>
      </section>

      <StudyGuideTabs result={session.result} />
    </div>
  );
}

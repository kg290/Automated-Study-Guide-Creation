"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { getHistory } from "@/lib/api";
import { HistoryItem } from "@/lib/types";

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const response = await getHistory(60);
        setItems(response.items);
      } catch {
        setError("Failed to load history.");
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  return (
    <div className="page-stack">
      <section className="page-title-block">
        <div>
          <h1>History</h1>
          <p>Revisit previously generated study guide sessions.</p>
        </div>
      </section>

      <section className="glass-card history-grid">
        {loading ? <p>Loading history...</p> : null}
        {error ? <p className="form-error">{error}</p> : null}

        {!loading && !error && items.length === 0 ? (
          <p>No history yet. Generate your first study guide from the dashboard.</p>
        ) : null}

        {items.map((item) => (
          <article key={item.session_id} className="history-card">
            <div className="history-card-header">
              <h3>Session {item.session_id.slice(0, 10)}...</h3>
              <span className="status-chip">{new Date(item.created_at).toLocaleString()}</span>
            </div>

            <div className="history-files">
              {item.file_names.map((file) => (
                <span key={file}>{file}</span>
              ))}
            </div>

            <div className="submit-row">
              <Link href={`/history/${item.session_id}`} className="primary-button">
                Open Session
              </Link>
              <Link href={`/results/${item.job_id}`} className="secondary-button">
                Open Last Result Job
              </Link>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}

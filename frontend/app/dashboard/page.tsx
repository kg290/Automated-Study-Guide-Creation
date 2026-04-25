"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { FileDropzone } from "@/components/FileDropzone";
import { OutputConfigurator } from "@/components/OutputConfigurator";
import { DEFAULT_BLOOM_LEVELS, DEFAULT_DIFFICULTY_MODES, DEFAULT_OUTPUT_TYPES } from "@/lib/constants";
import { generateStudyGuide } from "@/lib/api";

export default function DashboardPage() {
  const router = useRouter();

  const [files, setFiles] = useState<File[]>([]);
  const [outputTypes, setOutputTypes] = useState<string[]>(DEFAULT_OUTPUT_TYPES);
  const [difficultyModes, setDifficultyModes] = useState<string[]>(DEFAULT_DIFFICULTY_MODES);
  const [bloomLevels, setBloomLevels] = useState<Array<"remember" | "understand" | "apply" | "analyze">>(
    DEFAULT_BLOOM_LEVELS,
  );
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (files.length === 0) {
      setError("Please upload at least one PDF or DOCX file.");
      return;
    }
    if (outputTypes.length === 0) {
      setError("Choose at least one output type.");
      return;
    }
    if (bloomLevels.length === 0) {
      setError("Choose at least one Bloom taxonomy level.");
      return;
    }

    setError("");
    setSubmitting(true);
    setUploadProgress(0);

    try {
      const response = await generateStudyGuide(
        {
          files,
          outputTypes,
          difficultyModes,
          bloomLevels,
          customPrompt,
        },
        setUploadProgress,
      );

      router.push(`/processing/${response.job_id}`);
    } catch (requestError: unknown) {
      const message =
        typeof requestError === "object" &&
        requestError !== null &&
        "response" in requestError &&
        typeof (requestError as { response?: { data?: { detail?: string } } }).response?.data?.detail ===
          "string"
          ? (requestError as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : "Failed to start generation. Check backend configuration and try again.";
      setError(message || "Unexpected error while starting generation.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page-stack">
      <section className="page-title-block">
        <div>
          <h1>Upload Dashboard</h1>
          <p>Upload materials, choose generation outputs, and start the grounded study guide pipeline.</p>
        </div>
      </section>

      <form className="page-stack" onSubmit={handleSubmit}>
        <FileDropzone files={files} onFilesChange={setFiles} />

        <OutputConfigurator
          outputTypes={outputTypes}
          onOutputTypesChange={setOutputTypes}
          difficultyModes={difficultyModes}
          onDifficultyModesChange={setDifficultyModes}
          bloomLevels={bloomLevels}
          onBloomLevelsChange={setBloomLevels}
          customPrompt={customPrompt}
          onCustomPromptChange={setCustomPrompt}
        />

        <section className="glass-card submit-row">
          <button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Starting..." : "Generate Study Guide"}
          </button>
          {submitting ? <span className="status-chip">Upload progress: {uploadProgress}%</span> : null}
          {error ? <p className="form-error">{error}</p> : null}
        </section>
      </form>
    </div>
  );
}

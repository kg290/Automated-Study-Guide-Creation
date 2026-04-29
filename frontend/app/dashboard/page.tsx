"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { BookOpenText, Layers3, Sparkles } from "lucide-react";

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
      <motion.section
        className="page-title-block dashboard-hero"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <div>
          <h1>Upload Dashboard</h1>
          <p>Upload materials, choose generation outputs, and start the grounded study guide pipeline.</p>
        </div>
        <div className="dashboard-pills">
          <span className="dashboard-pill">
            <BookOpenText size={15} />
            {files.length || 0} file{files.length === 1 ? "" : "s"} ready
          </span>
          <span className="dashboard-pill">
            <Layers3 size={15} />
            {outputTypes.length} outputs selected
          </span>
          <span className="dashboard-pill">
            <Sparkles size={15} />
            {bloomLevels.length} Bloom levels active
          </span>
        </div>
      </motion.section>

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

        <motion.section
          className="glass-card launch-card"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.15, ease: "easeOut" }}
        >
          <div className="launch-copy">
            <span className="eyebrow">Ready To Run</span>
            <h3>Start the study guide engine</h3>
            <p>
              The pipeline will extract, retrieve, and compose a clean study guide from your uploaded material.
            </p>
          </div>

          <div className="launch-actions">
            <button
              className={`primary-button launch-button ${submitting ? "launch-button-active" : ""}`}
              type="submit"
              disabled={submitting}
            >
              <Sparkles size={16} />
              {submitting ? "Launching Pipeline..." : "Generate Study Guide"}
            </button>
            {submitting ? (
              <span className="status-chip status-chip-live">Upload progress: {uploadProgress}%</span>
            ) : (
              <span className="status-chip">{files.length ? `${files.length} source file(s) queued` : "Waiting for files"}</span>
            )}
          </div>
          {error ? <p className="form-error">{error}</p> : null}
        </motion.section>
      </form>
    </div>
  );
}

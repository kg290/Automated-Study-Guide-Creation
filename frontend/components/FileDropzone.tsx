"use client";

import { useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { BookOpenText, FileText, FileUp, ScanSearch, UploadCloud, X } from "lucide-react";

interface FileDropzoneProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

export function FileDropzone({ files, onFilesChange }: FileDropzoneProps) {
  const [error, setError] = useState<string>("");
  const [appendMode, setAppendMode] = useState<boolean>(false);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
    onDropAccepted: (acceptedFiles) => {
      if (!appendMode) {
        onFilesChange(acceptedFiles);
      } else {
        const merged = [...files];
        const mergedKeys = new Set(merged.map((file) => `${file.name}-${file.size}`));
        for (const incoming of acceptedFiles) {
          const key = `${incoming.name}-${incoming.size}`;
          if (!mergedKeys.has(key)) {
            merged.push(incoming);
            mergedKeys.add(key);
          }
        }
        onFilesChange(merged);
      }
      setError("");
    },
    onDropRejected: () => {
      setError("Only PDF and DOCX files are allowed.");
    },
    multiple: true,
  });

  const removeFile = (index: number) => {
    const next = files.filter((_, fileIndex) => fileIndex !== index);
    onFilesChange(next);
  };

  const dropzoneRootProps = getRootProps({
    className: `dropzone ${isDragActive ? "dropzone-active" : ""}`,
  });

  return (
    <motion.section
      className="glass-card dropzone-shell"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <h3 className="section-title">Upload Academic Material</h3>
      <p className="section-subtitle">
        Drag and drop PDF or DOCX files. The system auto-detects scanned PDFs and switches to OCR.
      </p>

      <div className="dropzone-badges">
        <span className="dropzone-badge">
          <BookOpenText size={14} />
          PDF + DOCX
        </span>
        <span className="dropzone-badge">
          <ScanSearch size={14} />
          OCR fallback
        </span>
        <span className="dropzone-badge">
          <FileUp size={14} />
          Multi-file ready
        </span>
      </div>

      <motion.div animate={isDragActive ? { scale: 1.01, y: -3 } : { scale: 1, y: 0 }} transition={{ duration: 0.22, ease: "easeOut" }}>
        <div {...dropzoneRootProps}>
          <input {...getInputProps()} />
          <UploadCloud size={30} />
          <h4>{isDragActive ? "Drop files here" : "Drop files or click to browse"}</h4>
          <p>
            {appendMode
              ? "Append mode: new uploads will be added to current selection."
              : "Replace mode: new uploads will replace current selection."}
          </p>
          <div className="dropzone-meter" aria-hidden="true">
            <span />
          </div>
        </div>
      </motion.div>

      {error ? <p className="form-error">{error}</p> : null}

      <div className="dropzone-tools">
        <label className="toggle-line" htmlFor="appendMode">
          <input
            id="appendMode"
            type="checkbox"
            checked={appendMode}
            onChange={(event) => setAppendMode(event.target.checked)}
          />
          <span>Add to existing selection</span>
        </label>
        {files.length > 0 ? (
          <div className="submit-row">
            <span className="file-counter">{files.length} selected</span>
            <button type="button" className="secondary-button" onClick={() => onFilesChange([])}>
              Clear Selected Files
            </button>
          </div>
        ) : null}
      </div>

      {files.length > 0 ? (
        <div className="file-list">
          {files.map((file, index) => (
            <motion.div
              className="file-item"
              key={`${file.name}-${file.size}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: index * 0.04, ease: "easeOut" }}
            >
              <div className="file-item-meta">
                <FileText size={16} />
                <div>
                  <strong>{file.name}</strong>
                  <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
              </div>
              <button type="button" className="icon-button" onClick={() => removeFile(index)}>
                <X size={16} />
              </button>
            </motion.div>
          ))}
        </div>
      ) : null}
    </motion.section>
  );
}

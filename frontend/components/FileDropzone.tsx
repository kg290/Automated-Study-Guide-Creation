"use client";

import { useState } from "react";
import { useDropzone } from "react-dropzone";
import { FileText, UploadCloud, X } from "lucide-react";

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

  return (
    <section className="glass-card">
      <h3 className="section-title">Upload Academic Material</h3>
      <p className="section-subtitle">
        Drag and drop PDF or DOCX files. The system auto-detects scanned PDFs and switches to OCR.
      </p>

      <div {...getRootProps()} className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}>
        <input {...getInputProps()} />
        <UploadCloud size={30} />
        <h4>{isDragActive ? "Drop files here" : "Drop files or click to browse"}</h4>
        <p>
          {appendMode
            ? "Append mode: new uploads will be added to current selection."
            : "Replace mode: new uploads will replace current selection."}
        </p>
      </div>

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
          <button type="button" className="secondary-button" onClick={() => onFilesChange([])}>
            Clear Selected Files
          </button>
        ) : null}
      </div>

      {files.length > 0 ? (
        <div className="file-list">
          {files.map((file, index) => (
            <div className="file-item" key={`${file.name}-${file.size}`}>
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
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

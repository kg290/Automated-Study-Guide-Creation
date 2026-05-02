"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Layers3, Wand2 } from "lucide-react";

import { DIFFICULTY_OPTIONS, OUTPUT_OPTIONS } from "@/lib/constants";

interface OutputConfiguratorProps {
  outputTypes: string[];
  onOutputTypesChange: (values: string[]) => void;
  difficultyModes: string[];
  onDifficultyModesChange: (values: string[]) => void;
  bloomLevels: Array<"remember" | "understand" | "apply" | "analyze">;
  onBloomLevelsChange: (values: Array<"remember" | "understand" | "apply" | "analyze">) => void;
  customPrompt: string;
  onCustomPromptChange: (value: string) => void;
}

export function OutputConfigurator({
  outputTypes,
  onOutputTypesChange,
  difficultyModes,
  onDifficultyModesChange,
  customPrompt,
  onCustomPromptChange,
}: OutputConfiguratorProps) {
  const toggleOutputType = (id: string) => {
    if (outputTypes.includes(id)) {
      onOutputTypesChange(outputTypes.filter((item) => item !== id));
      return;
    }
    onOutputTypesChange([...outputTypes, id]);
  };

  const toggleDifficultyMode = (id: string) => {
    if (difficultyModes.includes(id)) {
      onDifficultyModesChange(difficultyModes.filter((item) => item !== id));
      return;
    }
    onDifficultyModesChange([...difficultyModes, id]);
  };

  return (
    <motion.section
      className="glass-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.08, ease: "easeOut" }}
    >
      <h3 className="section-title">Choose Generated Outputs</h3>
      <p className="section-subtitle">Configure exactly what the AI should generate from your uploaded material.</p>

      <div className="selection-summary">
        <span className="selection-chip">
          <Layers3 size={14} />
          {outputTypes.length} output modes
        </span>
        <span className="selection-chip">
          <Wand2 size={14} />
          {difficultyModes.length} difficulty tracks
        </span>
      </div>

      <div className="grounding-banner">
        <strong>Strict Grounding Mode</strong>
        <p>Outputs are forced to stay within uploaded document context and include source chunk citations.</p>
      </div>

      <div className="option-grid">
        {OUTPUT_OPTIONS.map((option) => {
          const active = outputTypes.includes(option.id);
          return (
            <motion.button
              key={option.id}
              type="button"
              onClick={() => toggleOutputType(option.id)}
              className={`pill-option ${active ? "pill-option-active" : ""}`}
              aria-pressed={active}
              whileHover={{ y: -3, scale: 1.01 }}
              whileTap={{ scale: 0.985 }}
            >
              <span className={`pill-option-status ${active ? "pill-option-status-active" : ""}`}>
                <CheckCircle2 size={14} />
                {active ? "Selected" : "Tap to select"}
              </span>
              <strong>{option.label}</strong>
              <span>{option.description}</span>
            </motion.button>
          );
        })}
      </div>

      <h4 className="minor-heading">Difficulty Modes</h4>
      <div className="option-row">
        {DIFFICULTY_OPTIONS.map((option) => {
          const active = difficultyModes.includes(option.id);
          return (
            <motion.button
              key={option.id}
              type="button"
              className={`chip-option ${active ? "chip-option-active" : ""}`}
              aria-pressed={active}
              onClick={() => toggleDifficultyMode(option.id)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.97 }}
            >
              {active ? <CheckCircle2 size={14} /> : null}
              {option.label}
            </motion.button>
          );
        })}
      </div>

      <label className="text-input-label" htmlFor="customPrompt">
        Custom Generation Focus (Optional)
      </label>
      <textarea
        id="customPrompt"
        className="text-input"
        rows={4}
        placeholder="Example: prioritize derivations, edge cases, and exam-style contrast questions"
        value={customPrompt}
        onChange={(event) => onCustomPromptChange(event.target.value)}
      />
    </motion.section>
  );
}

"use client";

import { BLOOM_OPTIONS, DIFFICULTY_OPTIONS, OUTPUT_OPTIONS } from "@/lib/constants";

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
  bloomLevels,
  onBloomLevelsChange,
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

  const toggleBloomLevel = (id: "remember" | "understand" | "apply" | "analyze") => {
    if (bloomLevels.includes(id)) {
      onBloomLevelsChange(bloomLevels.filter((item) => item !== id));
      return;
    }
    onBloomLevelsChange([...bloomLevels, id]);
  };

  return (
    <section className="glass-card">
      <h3 className="section-title">Choose Generated Outputs</h3>
      <p className="section-subtitle">Configure exactly what the AI should generate from your uploaded material.</p>

      <div className="grounding-banner">
        <strong>Strict Grounding Mode</strong>
        <p>Outputs are forced to stay within uploaded document context and include source chunk citations.</p>
      </div>

      <div className="option-grid">
        {OUTPUT_OPTIONS.map((option) => {
          const active = outputTypes.includes(option.id);
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => toggleOutputType(option.id)}
              className={`pill-option ${active ? "pill-option-active" : ""}`}
            >
              <strong>{option.label}</strong>
              <span>{option.description}</span>
            </button>
          );
        })}
      </div>

      <h4 className="minor-heading">Difficulty Modes</h4>
      <div className="option-row">
        {DIFFICULTY_OPTIONS.map((option) => {
          const active = difficultyModes.includes(option.id);
          return (
            <button
              key={option.id}
              type="button"
              className={`chip-option ${active ? "chip-option-active" : ""}`}
              onClick={() => toggleDifficultyMode(option.id)}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      <h4 className="minor-heading">Bloom Taxonomy for Q&A and Flashcards</h4>
      <div className="option-row">
        {BLOOM_OPTIONS.map((option) => {
          const active = bloomLevels.includes(option.id);
          return (
            <button
              key={option.id}
              type="button"
              className={`chip-option ${active ? "chip-option-active" : ""}`}
              onClick={() => toggleBloomLevel(option.id)}
            >
              {option.label}
            </button>
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
    </section>
  );
}

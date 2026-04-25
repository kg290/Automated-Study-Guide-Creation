export const OUTPUT_OPTIONS = [
  {
    id: "summary",
    label: "Summary Notes",
    description: "Condensed overview of the complete material.",
  },
  {
    id: "unit_summaries",
    label: "Unit and Chapter Guide",
    description: "Sequential unit/chapter summaries to cover the full document.",
  },
  {
    id: "key_concepts",
    label: "Key Concepts",
    description: "Top terms, ideas, and definitions.",
  },
  {
    id: "topic_notes",
    label: "Topic-wise Notes",
    description: "Structured notes segmented by topic.",
  },
  {
    id: "flashcards",
    label: "Flashcards",
    description: "Quick recall cards for revision.",
  },
  {
    id: "qa",
    label: "Question and Answer",
    description: "Potential exam-ready Q&A pairs.",
  },
  {
    id: "viva",
    label: "Viva Questions",
    description: "Oral examination style prompts.",
  },
  {
    id: "difficulty",
    label: "Difficulty Modes",
    description: "Beginner/intermediate/advanced explanations.",
  },
] as const;

export const DIFFICULTY_OPTIONS = [
  { id: "beginner", label: "Beginner" },
  { id: "intermediate", label: "Intermediate" },
  { id: "advanced", label: "Advanced" },
] as const;

export const BLOOM_OPTIONS = [
  { id: "remember", label: "Remember" },
  { id: "understand", label: "Understand" },
  { id: "apply", label: "Apply" },
  { id: "analyze", label: "Analyze" },
] as const;

export const DEFAULT_OUTPUT_TYPES = OUTPUT_OPTIONS.map((item) => item.id);
export const DEFAULT_DIFFICULTY_MODES = DIFFICULTY_OPTIONS.map((item) => item.id);
export const DEFAULT_BLOOM_LEVELS = BLOOM_OPTIONS.map((item) => item.id);

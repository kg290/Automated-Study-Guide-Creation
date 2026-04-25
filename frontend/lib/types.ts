export type JobStatus = "queued" | "processing" | "completed" | "failed";

export interface Flashcard {
  question: string;
  answer: string;
  bloom_level: "remember" | "understand" | "apply" | "analyze";
}

export interface QAItem {
  question: string;
  answer: string;
  bloom_level: "remember" | "understand" | "apply" | "analyze";
}

export interface TopicNote {
  topic: string;
  notes: string;
}

export interface UnitSummary {
  unit_title: string;
  summary: string;
  key_points: string[];
}

export interface DifficultyExplanation {
  mode: "beginner" | "intermediate" | "advanced";
  explanation: string;
}

export interface StudyGuideResult {
  summary_notes: string;
  unit_wise_summaries: UnitSummary[];
  key_concepts: string[];
  topic_wise_notes: TopicNote[];
  flashcards: Flashcard[];
  qa_sets: QAItem[];
  viva_questions: string[];
  difficulty_explanations: DifficultyExplanation[];
  source_documents: string[];
  generated_at: string;
}

export interface JobCreateResponse {
  job_id: string;
  session_id: string;
  status: JobStatus;
}

export interface JobStatusResponse {
  job_id: string;
  session_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  error?: string | null;
  result?: StudyGuideResult | null;
  created_at: string;
  updated_at: string;
}

export interface HistoryItem {
  session_id: string;
  job_id: string;
  file_names: string[];
  created_at: string;
}

export interface HistoryDetail {
  session_id: string;
  job_id: string;
  file_names: string[];
  created_at: string;
  options: {
    output_types: string[];
    difficulty_modes: string[];
    bloom_levels?: Array<"remember" | "understand" | "apply" | "analyze">;
    custom_prompt: string;
  };
  result: StudyGuideResult;
}

export interface HistoryListResponse {
  items: HistoryItem[];
}

export interface GenerationPayload {
  files: File[];
  outputTypes: string[];
  difficultyModes: string[];
  bloomLevels: Array<"remember" | "understand" | "apply" | "analyze">;
  customPrompt: string;
}

export interface RegeneratePayload {
  output_types: string[];
  difficulty_modes: string[];
  bloom_levels: Array<"remember" | "understand" | "apply" | "analyze">;
  custom_prompt: string;
}

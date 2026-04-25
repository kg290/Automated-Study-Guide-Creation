import axios from "axios";

import {
  GenerationPayload,
  HistoryDetail,
  HistoryListResponse,
  JobCreateResponse,
  JobStatusResponse,
  RegeneratePayload,
} from "@/lib/types";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1",
  timeout: 300000,
});

export async function generateStudyGuide(
  payload: GenerationPayload,
  onProgress?: (value: number) => void,
): Promise<JobCreateResponse> {
  const formData = new FormData();

  payload.files.forEach((file) => formData.append("files", file));
  formData.append("output_types", payload.outputTypes.join(","));
  formData.append("difficulty_modes", payload.difficultyModes.join(","));
  formData.append("bloom_levels", payload.bloomLevels.join(","));
  formData.append("custom_prompt", payload.customPrompt);

  const response = await api.post<JobCreateResponse>("/study-guides/generate", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (event) => {
      if (!onProgress || !event.total) {
        return;
      }
      const progress = Math.round((event.loaded / event.total) * 100);
      onProgress(progress);
    },
  });

  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await api.get<JobStatusResponse>(`/study-guides/jobs/${jobId}`);
  return response.data;
}

export async function regenerateSession(
  sessionId: string,
  payload: RegeneratePayload,
): Promise<JobCreateResponse> {
  const response = await api.post<JobCreateResponse>(
    `/study-guides/sessions/${sessionId}/regenerate`,
    payload,
  );
  return response.data;
}

export async function getHistory(limit = 30): Promise<HistoryListResponse> {
  const response = await api.get<HistoryListResponse>("/history", {
    params: { limit },
  });
  return response.data;
}

export async function getHistoryDetail(sessionId: string): Promise<HistoryDetail> {
  const response = await api.get<HistoryDetail>(`/history/${sessionId}`);
  return response.data;
}

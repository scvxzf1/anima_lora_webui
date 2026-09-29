import type {
  TrainingConfigFile,
  TrainingConfigGroup,
} from "../../api/trainingContext";
import { apiRequest } from "../../api/client";

export type GroupQueueResponse = {
  ok: boolean;
  message?: string;
  queued_count?: number;
  failures?: {
    index?: number;
    config_file?: string;
    label?: string;
    error?: string;
  }[];
  failed_index?: number;
  failed_item?: { config_file?: string; label?: string };
  error?: string;
};

export function queueableTrainingFiles(
  group: TrainingConfigGroup,
): TrainingConfigFile[] {
  return group.files.filter(
    (file) =>
      file.path &&
      file.trainable === true &&
      !file.path.replace(/\\/g, "/").startsWith("configs/datasets/"),
  );
}

export function groupQueueItems(
  files: TrainingConfigFile[],
  preset: string,
  group: TrainingConfigGroup,
) {
  return files.map((file) => {
    const filename = file.filename || file.path.split("/").pop() || "";
    return {
      variant: file.method || filename.replace(/\.toml$/i, ""),
      preset,
      methods_subdir: file.methods_subdir || group.methods_subdir || "imported",
      config_file: file.path,
      filename,
      label: file.label || filename,
      confirm_preprocess: true,
    };
  });
}

export function enqueueTrainingGroup(
  group: TrainingConfigGroup,
  files: TrainingConfigFile[],
  preset: string,
  gpuIds: string[],
) {
  return apiRequest<GroupQueueResponse>("/api/training/queue/batch/start", {
    method: "POST",
    body: JSON.stringify({
      items: groupQueueItems(files, preset, group),
      preset,
      gpu_whitelist: gpuIds,
      start_paused: true,
    }),
  });
}

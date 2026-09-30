import { apiRequest } from "../../api/client";
import type { TrainingConfigFile } from "../../api/trainingContext";

export type HotstartTarget = {
  configFile: string;
  preset: string;
  variant: string;
  subdir: string;
};

export function hotstartTarget(file: TrainingConfigFile, preset: string): HotstartTarget {
  return {
    configFile: file.path,
    preset,
    variant: file.method || "lora",
    subdir: file.methods_subdir || "gui-methods",
  };
}

export type HotstartInspection = {
  ok: boolean;
  abs_path: string;
  kind: string;
  compatible: boolean;
  message?: string;
};

export async function inspectHotstart(path: string, target: HotstartTarget, signal?: AbortSignal): Promise<HotstartInspection> {
  const result = await apiRequest<HotstartInspection>("/api/training/continue-lora/inspect", {
    method: "POST",
    signal,
    body: JSON.stringify({
      path,
      variant: target.variant,
      preset: target.preset,
      methods_subdir: target.subdir,
      config_file: target.configFile,
    }),
  });
  const absolute = typeof result?.abs_path === "string" &&
    /^(?:\/(?!\/)|[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+)/.test(result.abs_path);
  if (result?.ok !== true || result.compatible !== true || !absolute)
    throw new Error(typeof result?.message === "string" ? result.message : "权重检查响应无效或不兼容");
  return result;
}

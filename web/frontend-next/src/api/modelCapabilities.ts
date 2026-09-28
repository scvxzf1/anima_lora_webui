import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "./client";

export type ModelCapability = {
  name: string;
  display_name: string;
  aliases: string[];
  supported_tasks?: string[];
  plain_lora_only?: boolean;
  supported_preview_tasks?: string[];
  max_preview_references?: number;
};

export function useModelCapabilities(enabled = true) {
  return useQuery({
    queryKey: ["model-capabilities"],
    queryFn: ({ signal }) => apiRequest<{ items: ModelCapability[] }>("/api/config/model-families", { signal }),
    retry: false,
    enabled,
  });
}

export function findModelCapability(items: ModelCapability[] | undefined, family: unknown) {
  const key = String(family || "anima").trim().toLowerCase().replaceAll("-", "_");
  return items?.find((item) => item.name === key || item.aliases?.includes(key));
}

export function taskLabel(task: string) {
  return ({ t2i: "普通文生图", edit: "编辑训练", unknown: "尚未确认" } as Record<string, string>)[task] || task;
}

export function capabilityLabels(capability: ModelCapability | undefined): string[] {
  if (!capability?.supported_tasks) return ["能力未知"];
  return [...capability.supported_tasks.map(taskLabel), ...(capability.plain_lora_only ? ["仅支持 LoRA"] : [])];
}

export function editCapabilityIssue(capability: ModelCapability | undefined): string | null {
  if (!capability?.supported_tasks) return "模型能力信息不可用，请重新读取或更新后端";
  return capability.supported_tasks.includes("edit") ? null : `当前模型 ${capability.display_name} 不支持编辑数据集训练`;
}

export function previewCapabilityLabels(capability: ModelCapability | undefined): string[] {
  if (!capability?.supported_preview_tasks) return ["采样能力未知"];
  return capability.supported_preview_tasks.map((task) => ({ t2i: "文生图采样", edit: "编辑采样" } as Record<string, string>)[task] || task);
}

import { CONFIG_FIELD_CATALOG } from "./domain/config-field-catalog.js";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";

export const RESOURCE_GROUPS = [
  ["compute", "精度与计算后端"],
  ["residency", "块交换与内存卸载"],
  ["activation", "梯度检查点"],
  ["compile", "编译加速"],
  ["adaptive_precision", "自适应精度"],
  ["oom_retry", "OOM 重试"],
  ["diagnostics", "诊断与预检"],
  ["topology", "流水线参数（实验审计）"],
  ["other", "其他资源参数"],
] as const;

export function resourceGroup(key: string) {
  if (["mixed_precision"].includes(key)) return "compute";
  if (["adaptive_precision", "adaptive_fp32_modules", "adaptive_loss_scale"].includes(key))
    return "adaptive_precision";
  if ([
    "adaptive_oom_retry",
    "adaptive_oom_retry_max_attempts",
    "adaptive_oom_retry_swap_increment",
    "adaptive_oom_retry_max_swap",
    "adaptive_oom_retry_timeout",
  ].includes(key)) return "oom_retry";
  if (key === "debug_finite_checks" || key === "v100_flash_stability") return "diagnostics";
  const cluster = CONFIG_FIELD_CATALOG[key]?.cluster;
  return RESOURCE_GROUPS.some(([id]) => id === cluster) ? cluster : "other";
}

export function groupResourceFields(fields: TrainingFieldSpec[]) {
  return RESOURCE_GROUPS.map(([id, title]) => ({
    id,
    title,
    fields: fields.filter((field) => resourceGroup(field.key) === id).sort((a, b) =>
      (CONFIG_FIELD_CATALOG[a.key]?.siblingOrder ?? 0) -
      (CONFIG_FIELD_CATALOG[b.key]?.siblingOrder ?? 0)),
  })).filter((group) => group.fields.length > 0);
}

export function resourceSummary(id: string, draft: TrainingDraft) {
  const value = (key: string) => String(draft[key] ?? "未设置");
  const enabled = (key: string) => draft[key] === true || draft[key] === "true";
  switch (id) {
    case "compute":
      return `${value("mixed_precision")} · ${value("base_compute")} · ${value("attn_mode")}`;
    case "adaptive_precision":
      return value("adaptive_precision");
    case "oom_retry":
      return enabled("adaptive_oom_retry")
        ? `开启 · ${value("adaptive_oom_retry_max_attempts")} 次 · +${value("adaptive_oom_retry_swap_increment")} 块`
        : "关闭";
    case "residency":
      return enabled("auto_block_swap")
        ? `AUTO · ${value("auto_block_swap_mode")}`
        : `手动 · ${value("blocks_to_swap")} 块`;
    case "activation":
      return `梯度检查点${enabled("gradient_checkpointing") ? "开启" : "关闭"} · ${value("selective_checkpoint")}`;
    case "compile":
      return enabled("torch_compile") ? `开启 · ${value("compile_block_scope")}` : "关闭";
    case "preprocess":
      return `${value("preprocess_memory_profile")} · workers ${value("max_data_loader_n_workers")}`;
    case "diagnostics":
      return `数值检查${enabled("debug_finite_checks") ? "开启" : "关闭"}`;
    case "topology":
      return `流水线并行${enabled("pipeline_parallel") ? "开启" : "关闭"}`;
    default:
      return "";
  }
}

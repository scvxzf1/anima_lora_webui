import { CONFIG_FIELD_CATALOG, CONFIG_STAGE_META } from "../../../../static/js/dragon-ui/pages/config-field-catalog.js";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";
import { groupResourceFields, resourceSummary } from "./resourceGroups";

const OVERRIDES: Record<string, string> = {
  caption_extension: "captions",
  output_dir: "output",
  save_every_n_steps: "output",
};
const TITLES: Record<string, string> = {
  models: "模型与编码器", dataset: "数据集配置", captions: "标注与遮罩",
  filters: "输入筛选", cache: "缓存策略", contract: "基础方法",
  capacity: "容量与训练目标", "warm-start": "权重热启动",
  adapter: "适配器变体", orthogonal: "正交与时间步适配",
  raw: "网络额外参数", identity: "任务与输出名称", volume: "训练量与批次",
};
const SUMMARY_KEYS: Record<string, string[]> = {
  models: ["model_family"], dataset: ["dataset_config"], captions: ["caption_extension"],
  cache: ["use_vae_cache", "use_text_cache"], contract: ["network_module"],
  capacity: ["network_dim", "network_alpha"], "warm-start": ["network_weights"],
  identity: ["output_name"], volume: ["max_train_epochs", "max_train_steps", "train_batch_size"],
  optimizer: ["optimizer_type", "learning_rate"], loss: ["timestep_sampling"],
  preview: ["sample_every_n_steps", "sample_every_n_epochs"], output: ["save_model_as", "save_precision"],
  observability: ["log_with"],
};

export function groupStageFields(stage: string, fields: TrainingFieldSpec[]) {
  if (stage === "resources") return groupResourceFields(fields);
  const clusters = [...(CONFIG_STAGE_META.find((item) => item.id === stage)?.clusters || []),
    ...(stage === "input" ? [{ id: "preprocess", label: "预处理与数据加载" }] : [])];
  const owner = (key: string) => {
    const id = OVERRIDES[key] || CONFIG_FIELD_CATALOG[key]?.cluster;
    return clusters.some((cluster) => cluster.id === id) ? id : "other";
  };
  return [...clusters, { id: "other", label: "其他配置与审计" }].map(({ id, label }) => ({
    id, title: TITLES[id] || label,
    fields: fields.filter((field) => owner(field.key) === id).sort((a, b) =>
      (CONFIG_FIELD_CATALOG[a.key]?.siblingOrder ?? 0) - (CONFIG_FIELD_CATALOG[b.key]?.siblingOrder ?? 0)),
  })).filter((group) => group.fields.length > 0);
}

export function stageSummary(stage: string, id: string, draft: TrainingDraft) {
  if (stage === "resources") return resourceSummary(id, draft);
  return (SUMMARY_KEYS[id] || []).filter((key) => draft[key] !== undefined && draft[key] !== "")
    .map((key) => {
      const value = draft[key];
      if (key === "network_dim") return `rank ${value}`;
      if (key === "network_alpha") return `alpha ${value}`;
      if (key === "max_train_epochs") return `${value} 轮`;
      if (key === "max_train_steps") return `${value} 步`;
      if (key === "train_batch_size") return `batch ${value}`;
      if (key === "sample_every_n_steps") return `每 ${value} 步`;
      if (key === "sample_every_n_epochs") return `每 ${value} 轮`;
      if (typeof value === "boolean") return `${key === "use_vae_cache" ? "VAE" : "文本"}缓存${value ? "开启" : "关闭"}`;
      return String(value);
    }).join(" · ");
}

export function stageGroupDefaultOpen(stage: string, id: string) {
  return ({ input: ["models", "dataset"], method: ["contract", "capacity"],
    training: ["identity", "volume"], resources: ["compute"] } as Record<string, string[]>)[stage]?.includes(id) || false;
}

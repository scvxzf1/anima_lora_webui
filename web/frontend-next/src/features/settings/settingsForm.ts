import type { GlobalSettings } from "./api";

type Field = {
  key: string;
  label: string;
  kind: "text" | "number" | "boolean";
  min?: number;
  max?: number;
  optional?: boolean;
};
const scales = [
  ["config", "训练参数"],
  ["datasets", "数据集"],
  ["training", "当前监控"],
  ["model_config", "模型配置"],
  ["settings", "全局设置"],
  ["weight_analysis", "权重分析"],
  ["image_test", "生图测试"],
  ["environment", "环境检测"],
  ["history_overview", "历史概览"],
  ["history_analysis", "历史分析"],
  ["history_preview", "历史预览"],
  ["history_logs", "历史日志"],
  ["history_config_files", "历史配置"],
];
export const SETTINGS_GROUPS: { title: string; fields: Field[] }[] = [
  {
    title: "文件与任务",
    fields: [
      { key: "output_root", label: "训练输出目录", kind: "text" },
      { key: "configs_root", label: "配置根目录", kind: "text" },
      { key: "history_root", label: "历史目录", kind: "text" },
      { key: "queue_root", label: "队列目录", kind: "text" },
      {
        key: "tagging_max_retained_jobs",
        label: "打标任务保留数量",
        kind: "number",
        min: 1,
        max: 500,
      },
    ],
  },
  {
    title: "生图测试",
    fields: [
      { key: "image_test_save_root", label: "生图保存目录", kind: "text" },
      {
        key: "image_test_allow_home_search",
        label: "允许搜索用户主目录",
        kind: "boolean",
      },
    ],
  },
  {
    title: "界面显示",
    fields: [
      { key: "dragon_motion_enabled", label: "界面动效", kind: "boolean" },
      {
        key: "dragon_config_help_always_visible",
        label: "参数帮助常显",
        kind: "boolean",
      },
      {
        key: "dragon_config_tags_always_visible",
        label: "参数标签常显",
        kind: "boolean",
      },
      {
        key: "ui_scale",
        label: "全局缩放 (%)",
        kind: "number",
        min: 25,
        max: 400,
      },
    ],
  },
  {
    title: "分页面缩放",
    fields: scales.map(([key, label]) => ({
      key: `ui_scale_${key}`,
      label: `${label} (%)`,
      kind: "number",
      min: 25,
      max: 400,
      optional: true,
    })),
  },
];

export function settingsDraft(
  data: GlobalSettings,
): Record<string, string | number | boolean> {
  return Object.fromEntries(
    SETTINGS_GROUPS.flatMap((group) =>
      group.fields.map((field) => {
        const value = data.path_overrides?.[field.key] ?? data[field.key];
        return [
          field.key,
          typeof value === "boolean" ||
          typeof value === "number" ||
          typeof value === "string"
            ? value
            : field.kind === "boolean"
              ? false
              : "",
        ];
      }),
    ),
  );
}

export function settingsPatch(
  draft: Record<string, unknown>,
  baseline: Record<string, unknown>,
) {
  return Object.fromEntries(
    Object.entries(draft).filter(
      ([key, value]) => !Object.is(value, baseline[key]),
    ),
  );
}

export function applySettingsDefaults(
  draft: ReturnType<typeof settingsDraft>,
  defaults: GlobalSettings | undefined,
) {
  const next = { ...draft };
  if (!defaults) return next;
  for (const key of Object.keys(next)) {
    const value = defaults.path_overrides?.[key] ?? defaults[key];
    if (
      typeof value === "string" ||
      typeof value === "boolean" ||
      typeof value === "number"
    )
      next[key] = value;
  }
  return next;
}

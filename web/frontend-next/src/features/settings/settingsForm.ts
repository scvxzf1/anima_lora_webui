import type { GlobalSettings } from "./api";

type Field = {
  key: string;
  label: string;
  description: string;
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
export const SETTINGS_GROUPS: {
  title: string;
  description: string;
  fields: Field[];
}[] = [
  {
    title: "文件与任务",
    description: "管理运行数据的目录和打标任务保留数量。更改目录不会迁移已有文件。",
    fields: [
      { key: "output_root", label: "训练输出目录", description: "训练产物和运行文件的默认保存位置。", kind: "text" },
      { key: "configs_root", label: "配置根目录", description: "配置读取与保存所使用的根目录。", kind: "text" },
      { key: "history_root", label: "历史目录", description: "训练历史记录的存储位置。", kind: "text" },
      { key: "queue_root", label: "队列目录", description: "WebUI 训练队列文件的存储位置。", kind: "text" },
      {
        key: "tagging_max_retained_jobs",
        label: "打标任务保留数量",
        description: "限制保留的打标任务记录数，超出后清理较早记录。",
        kind: "number",
        min: 1,
        max: 500,
      },
    ],
  },
  {
    title: "生图测试",
    description: "设置生图测试结果的保存目录，以及模型文件的搜索范围。",
    fields: [
      { key: "image_test_save_root", label: "生图保存目录", description: "测试生成图像的默认输出位置。", kind: "text" },
      {
        key: "image_test_allow_home_search",
        label: "允许搜索用户主目录",
        description: "查找模型时是否包含当前用户的主目录。",
        kind: "boolean",
      },
    ],
  },
  {
    title: "界面显示",
    description: "控制动效、参数辅助信息和所有页面的默认缩放比例。",
    fields: [
      { key: "dragon_motion_enabled", label: "界面动效", description: "启用或关闭界面过渡与装饰动效。", kind: "boolean" },
      {
        key: "dragon_config_help_always_visible",
        label: "参数帮助常显",
        description: "在训练参数表单中持续显示帮助说明。",
        kind: "boolean",
      },
      {
        key: "dragon_config_tags_always_visible",
        label: "参数标签常显",
        description: "在训练参数表单中持续显示参数标签。",
        kind: "boolean",
      },
      {
        key: "ui_scale",
        label: "全局缩放 (%)",
        description: "设置未单独覆盖页面的默认界面缩放。",
        kind: "number",
        min: 25,
        max: 400,
      },
    ],
  },
  {
    title: "分页面缩放",
    description: "为常用页面单独设置缩放；留空时沿用全局缩放。",
    fields: scales.map(([key, label]) => ({
      key: `ui_scale_${key}`,
      label: `${label} (%)`,
      description: "留空时使用全局缩放比例。",
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

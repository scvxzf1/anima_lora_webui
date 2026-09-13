import type { ModelConfigItem, ModelConfigResponse } from "./api";

export const MODEL_PATHS = [
  ["pretrained_model_name_or_path", "DiT 模型"],
  ["qwen3", "Qwen3 文本编码器"],
  ["vae", "VAE"],
] as const;

export function modelGroups(data: ModelConfigResponse) {
  return data.groups?.length
    ? data.groups
    : [
        {
          id: "ungrouped",
          label: "未分组",
          item_ids: data.items.map((item) => item.id),
        },
      ];
}
export function addModel(
  data: ModelConfigResponse,
  groupId?: string,
): ModelConfigResponse {
  const groups = modelGroups(data);
  const item: ModelConfigItem = {
    id: `model-${crypto.randomUUID().slice(0, 12)}`,
    name: `模型配置 ${data.items.length + 1}`,
    model_family: "anima",
    pretrained_model_name_or_path: "",
    qwen3: "",
    vae: "",
  };
  return {
    ...data,
    items: [...data.items, item],
    groups: groups.map((group) =>
      group.id === (groupId || groups[0].id)
        ? { ...group, item_ids: [...group.item_ids, item.id] }
        : group,
    ),
  };
}
export function deleteModel(
  data: ModelConfigResponse,
  id: string,
): ModelConfigResponse {
  if (data.default_id === id || data.items.length <= 1) return data;
  return {
    ...data,
    items: data.items.filter((item) => item.id !== id),
    groups: modelGroups(data).map((group) => ({
      ...group,
      item_ids: group.item_ids.filter((itemId) => itemId !== id),
    })),
  };
}
export function moveModel(
  data: ModelConfigResponse,
  id: string,
  target: string,
) {
  if (
    !modelGroups(data).some((group) => group.id === target) ||
    !data.items.some((item) => item.id === id)
  )
    return data;
  return {
    ...data,
    groups: modelGroups(data).map((group) => ({
      ...group,
      item_ids: [
        ...group.item_ids.filter((itemId) => itemId !== id),
        ...(group.id === target ? [id] : []),
      ],
    })),
  };
}
export function reorderModel(
  data: ModelConfigResponse,
  id: string,
  direction: number,
) {
  return {
    ...data,
    groups: modelGroups(data).map((group) => {
      const index = group.item_ids.indexOf(id);
      const next = index + direction;
      if (index < 0 || next < 0 || next >= group.item_ids.length) return group;
      const item_ids = [...group.item_ids];
      [item_ids[index], item_ids[next]] = [item_ids[next], item_ids[index]];
      return { ...group, item_ids };
    }),
  };
}

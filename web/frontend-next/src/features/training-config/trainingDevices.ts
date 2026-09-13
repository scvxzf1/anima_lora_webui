export type TrainingDevice = { id: string; name: string; memoryGb?: number };
export type DeviceSelection = { mode: "single" | "ddp"; devices: TrainingDevice[] };
export const DEVICE_STORAGE_KEY = "dragon-next.training-devices.v1";

export function readDeviceSelection(): DeviceSelection | null {
  try {
    const value = JSON.parse(localStorage.getItem(DEVICE_STORAGE_KEY) || "null");
    if (!value || !["single", "ddp"].includes(value.mode) || !Array.isArray(value.devices)) return null;
    if (!value.devices.every((gpu: TrainingDevice) => typeof gpu?.id === "string" && /^\d+$/.test(gpu.id)
      && typeof gpu.name === "string")) return null;
    const devices: TrainingDevice[] = value.devices;
    if (new Set(devices.map((gpu) => gpu.id)).size !== devices.length) return null;
    if (value.mode === "single" && devices.length > 1) return null;
    return { mode: value.mode, devices };
  } catch {
    return null;
  }
}

export function normalizeDevices(rows: Record<string, unknown>[]): TrainingDevice[] {
  return rows.filter((row) => /^\d+$/.test(String(row.index))).map((row) => ({
    id: String(row.index), name: String(row.name || `GPU ${row.index}`),
    memoryGb: Number(row.memory_total_gb) > 0 ? Number(row.memory_total_gb) : undefined,
  }));
}

export function selectionIssue(selection: DeviceSelection, devices: TrainingDevice[]) {
  const missing = selection.devices.filter((saved) => !devices.some((gpu) => gpu.id === saved.id && gpu.name === saved.name));
  if (missing.length) return `已选设备不可用：${missing.map((gpu) => `GPU ${gpu.id} · ${gpu.name}`).join("、")}。请重新选择。`;
  if (!devices.length) return "未检测到可用 GPU。";
  if (selection.mode === "single" && selection.devices.length !== 1) return "请选择一张 GPU。";
  if (selection.mode === "ddp" && selection.devices.length < 2) return "数据并行至少需要选择两张 GPU。";
  return "";
}

export function deviceSummary(selection: DeviceSelection) {
  return `${selection.mode === "single" ? "单卡" : "数据并行"} · ${selection.devices.map((gpu) => `GPU ${gpu.id} · ${gpu.name}`).join(" / ") || "未选择"}`;
}

import { describe, expect, it } from "vitest";
import { historyGpuChartPoints, historyGpuDevices } from "./historyGpuMetrics";

describe("history GPU metrics", () => {
  const points = [
    { ts: 100, gpu_index: 0, vram_used_gb: 6, per_gpu: [
      { index: 0, uuid: "GPU-a", name: "Alpha", vram_used_gb: 4, gpu_util: 90, gpu_temp: 60 },
      { index: 1, uuid: "GPU-b", name: "Beta", vram_used_gb: 2, gpu_util: 40, gpu_temp: 51 },
    ] },
    { ts: 102, gpu_index: 0, vram_used_gb: 7, per_gpu: [
      { index: 0, uuid: "GPU-c", name: "Replacement", vram_used_gb: 5, gpu_util: 80 },
      { index: 1, uuid: "GPU-b", name: "Beta", vram_used_gb: 2, gpu_util: 0, gpu_temp: 49 },
    ] },
  ];

  it("keeps physical identities distinct when an index is reused", () => {
    expect(historyGpuDevices(points)).toEqual([
      { key: "uuid:GPU-a", index: 0, name: "Alpha" },
      { key: "uuid:GPU-c", index: 0, name: "Replacement" },
      { key: "uuid:GPU-b", index: 1, name: "Beta" },
    ]);
    expect(historyGpuChartPoints(points, "uuid:GPU-a")).toEqual([
      { index: 0, uuid: "GPU-a", name: "Alpha", vram_used_gb: 4, gpu_util: 90, gpu_temp: 60, ts: 100 },
      { ts: 102 },
    ]);
  });

  it("preserves zero readings and legacy aggregate samples", () => {
    expect(historyGpuChartPoints(points, "uuid:GPU-b")[1]).toMatchObject({ gpu_util: 0, ts: 102 });
    const legacy = [{ timestamp: 100, vram_used_gb: 11.2, gpu_util: 88 }];
    expect(historyGpuDevices(legacy)).toEqual([]);
    expect(historyGpuChartPoints(legacy)).toEqual([{ ts: 100, timestamp: 100, vram_used_gb: 11.2, gpu_util: 88 }]);
  });
});

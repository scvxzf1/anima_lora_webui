import { describe, expect, it } from "vitest";
import { HISTORY_GPU_MAX_POINTS, historyGpuChartPoints, historyGpuDevices, historyGpuInspectionIndices, historyGpuRecentPoints, historyGpuSummary } from "./historyGpuMetrics";

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

  it("summarizes only recorded fields and exposes valid joint sample points", () => {
    const aggregate = [
      { ts: 100, vram_used_gb: 4, vram_total_gb: 8, gpu_util: 0 },
      { ts: 200, vram_used_gb: 6, gpu_total: 2, gpu_temp: 72 },
    ];
    expect(historyGpuSummary(aggregate)).toEqual({
      vram_used_gb: { latest: 6, peak: 6 },
      vram_total_gb: { latest: 8, peak: 8 },
      gpu_util: { latest: 0, peak: 0 },
      gpu_temp: { latest: 72, peak: 72 },
    });
    expect(historyGpuInspectionIndices(aggregate)).toEqual([0, 1]);
    expect(historyGpuSummary([{ ts: 1, vram_used_gb: 3 }])).toEqual({
      vram_used_gb: { latest: 3, peak: 3 },
      vram_total_gb: {}, gpu_util: {}, gpu_temp: {},
    });
    expect(historyGpuInspectionIndices([{ ts: 1 }, { ts: 2, vram_total_gb: 0 }])).toEqual([1]);
  });

  it("summarizes the selected GPU independently without inventing total memory", () => {
    const selected = historyGpuChartPoints(points, "uuid:GPU-b");
    expect(historyGpuSummary(selected)).toEqual({
      vram_used_gb: { latest: 2, peak: 2 },
      vram_total_gb: {},
      gpu_util: { latest: 0, peak: 40 },
      gpu_temp: { latest: 49, peak: 51 },
    });
  });

  it("limits inspected history to the most recent 1000 samples", () => {
    const samples = Array.from({ length: HISTORY_GPU_MAX_POINTS + 5 }, (_, index) => ({ ts: index }));
    const recent = historyGpuRecentPoints(samples);

    expect(recent).toHaveLength(HISTORY_GPU_MAX_POINTS);
    expect(recent[0].ts).toBe(5);
    expect(recent.at(-1)?.ts).toBe(HISTORY_GPU_MAX_POINTS + 4);
  });
});

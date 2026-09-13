import { afterEach, expect, it } from "vitest";
import { DEVICE_STORAGE_KEY, normalizeDevices, readDeviceSelection, selectionIssue } from "./trainingDevices";

const devices = [{ id: "0", name: "A" }, { id: "1", name: "B" }];
afterEach(() => localStorage.clear());

it("restores valid preferences and rejects malformed or ambiguous selections", () => {
  expect(readDeviceSelection()).toBeNull();
  for (const value of ["broken", "{}", JSON.stringify({ mode: "single", devices }),
    JSON.stringify({ mode: "ddp", devices: [devices[0], devices[0]] })]) {
    localStorage.setItem(DEVICE_STORAGE_KEY, value);
    expect(readDeviceSelection()).toBeNull();
  }
  localStorage.setItem(DEVICE_STORAGE_KEY, JSON.stringify({ mode: "ddp", devices }));
  expect(readDeviceSelection()).toEqual({ mode: "ddp", devices });
});

it("requires explicit valid devices and never silently replaces an absent GPU", () => {
  expect(selectionIssue({ mode: "single", devices: [devices[1]] }, devices)).toBe("");
  expect(selectionIssue({ mode: "ddp", devices: [devices[0]] }, devices)).toContain("至少");
  expect(selectionIssue({ mode: "single", devices: [] }, devices)).toContain("请选择");
  expect(selectionIssue({ mode: "single", devices: [devices[1]] }, [devices[0]])).toContain("GPU 1");
  expect(selectionIssue({ mode: "single", devices: [devices[0]] }, [{ id: "0", name: "Changed" }])).toContain("不可用");
  expect(selectionIssue({ mode: "single", devices: [] }, [])).toContain("未检测");
});

it("uses explicit GPU indices and total memory, not array positions", () => {
  expect(normalizeDevices([{ index: 3, name: "A", memory_total_gb: 24 }, { name: "Invalid" }]))
    .toEqual([{ id: "3", name: "A", memoryGb: 24 }]);
});

import { afterEach, expect, it, vi } from "vitest";
import { cleanup, screen, waitFor } from "@testing-library/react";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { LiveMonitorPage } from "./LiveMonitorPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it.each([
  ["interrupted", "已中断", "error", false],
  ["failed", "失败", "error", false],
  ["error", "异常", "error", false],
  ["unavailable", "不可用", "error", false],
  ["running", "运行中", "running", true],
])("renders %s as the correct non-running/running state", async (status, label, dataState, canStop) => {
  vi.stubGlobal("WebSocket", class {
    close() {}
  });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/training/status") return jsonResponse({ status, task_id: "task-1" });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
    if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
    throw new Error(`Unexpected ${path}`);
  }));

  renderInApp(<LiveMonitorPage />);
  const state = await screen.findByRole("status", { name: "当前任务状态" });
  await waitFor(() => expect(state).toHaveTextContent(label));
  expect(state).toHaveAttribute("data-state", dataState);
  expect(screen.getByRole("button", { name: "停止训练" })).toHaveProperty("disabled", !canStop);
});

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useMonitorRefresh } from "./useMonitorRefresh";

const mocks = vi.hoisted(() => ({
  event: (_event: Record<string, unknown>) => {},
  open: () => {},
  invalidateQueries: vi.fn(),
  getQueryState: vi.fn(),
  getQueryData: vi.fn(),
}));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => mocks }));
vi.mock("../../app/useWebSocket", () => ({
  useWebSocket: (_path: string, event: typeof mocks.event, _enabled: boolean, open: typeof mocks.open) => {
    mocks.event = event;
    mocks.open = open;
    return { status: "open" };
  },
}));

describe("monitor event refresh", () => {
  afterEach(() => { cleanup(); vi.useRealTimers(); vi.clearAllMocks(); });
  function setup() {
    vi.useFakeTimers();
    mocks.invalidateQueries.mockResolvedValue(undefined);
    mocks.getQueryData.mockReturnValue({ task_id: "run-A" });
    mocks.getQueryState.mockReturnValue({ status: "success", data: { task_id: "run-B" } });
    return renderHook(() => useMonitorRefresh());
  }

  it("coalesces event bursts and reads the confirmed task after the status barrier", async () => {
    setup();
    for (let i = 0; i < 50; i++) mocks.event({ type: "progress", task_id: "run-A", current: 99999 });
    await act(() => vi.advanceTimersByTimeAsync(749));
    expect(mocks.invalidateQueries).not.toHaveBeenCalled();
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(mocks.invalidateQueries.mock.calls.map(([filter]) => filter.queryKey)).toEqual([
      ["live-monitor", "status"], ["live-monitor", "metrics", "run-B"], ["live-monitor", "logs", "run-B"],
    ]);
  });

  it("reconnect immediately confirms identity and never refreshes task data on failed status", async () => {
    setup();
    mocks.getQueryState.mockReturnValue({ status: "error", data: { task_id: "run-A" } });
    await act(async () => mocks.open());
    expect(mocks.invalidateQueries.mock.calls.map(([filter]) => filter.queryKey)).toEqual([
      ["live-monitor", "status"], ["training-queue"],
    ]);
  });

  it("drops old non-status events and cancels pending work on unmount", async () => {
    const hook = setup();
    mocks.event({ type: "log", task_id: "old-run" });
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(mocks.invalidateQueries).not.toHaveBeenCalled();
    mocks.event({ type: "status", task_id: "new-run" });
    hook.unmount();
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(mocks.invalidateQueries).not.toHaveBeenCalled();
  });
});

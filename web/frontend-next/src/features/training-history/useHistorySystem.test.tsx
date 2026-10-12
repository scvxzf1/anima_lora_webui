import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useHistorySystem } from "./useHistorySystem";
const { request } = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock("../../api/client", () => ({ apiRequest: request }));
afterEach(() => { cleanup(); request.mockReset(); });
it("loads from zero, respects page indices and reports the snapshot extent", async () => {
  request.mockResolvedValueOnce({ system: [{ gpu_util: 5 }], indices: [0], total: 2001, next_offset: 2000, has_more_after: true })
    .mockResolvedValueOnce({ system: [{ gpu_util: 9 }], indices: [2000], total: 2001, next_offset: 2001, has_more_after: false });
  const { result } = renderHook(() => useHistorySystem("a", [{ gpu_util: 99 }], 2001));
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(request.mock.calls[0][0]).toContain("offset=0"); expect(request.mock.calls[1][0]).toContain("offset=2000");
  expect(result.current.points[0].gpu_util).toBe(5); expect(result.current.points[1999]).toEqual({});
  expect(result.current.read).toBe(2001);
});
it("keeps partial results on failure and cancels a pending request", async () => {
  request.mockResolvedValueOnce({ system: [{ gpu_util: 5 }], indices: [0], total: 4000, next_offset: 2000, has_more_after: true }).mockRejectedValueOnce(new Error("offline"));
  const { result } = renderHook(() => useHistorySystem("a", [], 4000));
  await waitFor(() => expect(result.current.error).toBe("offline")); expect(result.current.read).toBe(2000);
  request.mockImplementation(() => new Promise(() => {}));
  act(() => result.current.retry());
  await waitFor(() => expect(request).toHaveBeenCalledTimes(3));
  const signal = request.mock.calls[2][1].signal;
  act(() => result.current.cancel()); expect(signal.aborted).toBe(true); expect(result.current.loading).toBe(false);
});
it("aborts old task requests on switch and unmount", async () => {
  request.mockImplementation(() => new Promise(() => {}));
  const { rerender, unmount } = renderHook(({ id }) => useHistorySystem(id, [], 9000), { initialProps: { id: "a" } });
  const first = request.mock.calls[0][1].signal;
  rerender({ id: "b" }); expect(first.aborted).toBe(true);
  expect(request.mock.calls[1][0]).toContain("/b/system");
  const second = request.mock.calls[1][1].signal; unmount(); expect(second.aborted).toBe(true);
});
it("stops at the first page snapshot when the running history grows", async () => {
  request.mockResolvedValueOnce({ system: [{ ts: 1 }], indices: [0], total: 2001, next_offset: 2000, has_more_after: true })
    .mockResolvedValueOnce({ system: [{ ts: 2 }, { ts: 3 }], indices: [2000, 2001], total: 5000, next_offset: 4000, has_more_after: true });
  const { result } = renderHook(() => useHistorySystem("a", [], 1000));
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.read).toBe(2001); expect(result.current.points).toHaveLength(2001);
  expect(result.current.total).toBe(2001); expect(request).toHaveBeenCalledTimes(2);
});
it.each([
  { indices: [-1] }, { indices: [2000] }, { indices: [] }, { total: NaN }, { next_offset: 0 }, { next_offset: 2001 },
])("rejects invalid paging metadata %j", async (invalid) => {
  request.mockResolvedValue({ system: [{ ts: 1 }], indices: [0], total: 4000, next_offset: 2000, has_more_after: true, ...invalid });
  const { result } = renderHook(() => useHistorySystem("a", [], 4000));
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.error).toBe("GPU 历史分页响应无效"); expect(result.current.read).toBe(0);
});
it("ignores a late response from an aborted task after switching", async () => {
  let resolveOld!: (value: unknown) => void;
  request.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
    .mockResolvedValueOnce({ system: [{ gpu_util: 9 }], indices: [0], total: 1, next_offset: 1, has_more_after: false });
  const { result, rerender } = renderHook(({ id }) => useHistorySystem(id, [], 1), { initialProps: { id: "a" } });
  rerender({ id: "b" });
  await waitFor(() => expect(result.current.points[0]?.gpu_util).toBe(9));
  await act(async () => resolveOld({ system: [{ gpu_util: 99 }], indices: [0], total: 1, next_offset: 1, has_more_after: false }));
  expect(result.current.points[0].gpu_util).toBe(9);
});

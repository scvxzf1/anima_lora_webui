import { useEffect, useRef, useState } from "react";
import { apiRequest } from "../../api/client";

type Point = Record<string, unknown>;
type Page = { system: Point[]; indices: number[]; total: number; next_offset: number; has_more_after: boolean };
export function useHistorySystem(taskId: string | undefined, initial: Point[], initialTotal?: number) {
  const [state, setState] = useState({ taskId, points: taskId ? [] as Point[] : initial, total: initialTotal, read: taskId ? 0 : initial.length, loading: Boolean(taskId), error: "" });
  const controller = useRef<AbortController | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!taskId) return;
    const abort = new AbortController(); controller.current = abort;
    setState({ taskId, points: [], total: initialTotal, read: 0, loading: true, error: "" });
    void (async () => {
      let offset = 0;
      const points: Point[] = [];
      try {
        let target: number | undefined;
        while (!abort.signal.aborted) {
          const page = await apiRequest<Page>(`/api/training/history/${encodeURIComponent(taskId)}/system?offset=${offset}&limit=2000`, { signal: abort.signal });
          if (abort.signal.aborted) return;
          if (!Array.isArray(page.system) || !Array.isArray(page.indices)
            || page.system.length !== page.indices.length
            || !Number.isSafeInteger(page.total) || page.total < 0
            || !Number.isSafeInteger(page.next_offset) || page.next_offset < offset
            || page.next_offset > Math.min(page.total, offset + 2000)
            || typeof page.has_more_after !== "boolean"
            || page.has_more_after !== (page.next_offset < page.total)
            || (page.has_more_after && page.next_offset === offset)
            || page.indices.some((index, i) => !Number.isSafeInteger(index) || index < offset
              || index >= page.next_offset || (i > 0 && index <= page.indices[i - 1]))
            || page.system.some((point) => !point || typeof point !== "object" || Array.isArray(point))) {
            throw new Error("GPU 历史分页响应无效");
          }
          target ??= page.total; // Snapshot the extent; do not chase a running task forever.
          if (page.total < target) throw new Error("GPU 历史长度已变化，请重新读取");
          const end = Math.min(target, page.next_offset);
          for (let index = offset; index < end; index++) points[index] = {};
          page.system.forEach((point, i) => { if (page.indices[i] < end) points[page.indices[i]] = point; });
          offset = end;
          const done = offset >= target || !page.has_more_after;
          setState({ taskId, points: [...points], total: target, read: offset, loading: !done, error: "" });
          if (done) return;
        }
      } catch (error) {
        if (!abort.signal.aborted) setState((state) => ({ ...state, loading: false, error: error instanceof Error ? error.message : "GPU 历史读取失败" }));
      }
    })();
    return () => abort.abort();
  }, [taskId, attempt, initialTotal]);
  const current = taskId && state.taskId !== taskId
    ? { points: [] as Point[], total: initialTotal, read: 0, loading: true, error: "" }
    : state;
  return { ...current, points: taskId ? current.points : initial, cancel: () => { controller.current?.abort(); setState((state) => ({ ...state, loading: false, error: "读取已取消，当前仅显示已读取区间" })); }, retry: () => setAttempt((value) => value + 1) };
}

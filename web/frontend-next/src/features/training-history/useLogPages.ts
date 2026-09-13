import { useEffect, useRef, useState } from "react";
import { fetchLogPage, LOG_PAGE_SIZE, type LogRecord } from "./logApi";

export function useLogPages(taskId: string, first: number, last: number, total: number, revision: number) {
  const cache = useRef(new Map<number, Map<number, LogRecord>>());
  const [, render] = useState(0);
  const [error, setError] = useState("");
  const start = Math.floor(first / LOG_PAGE_SIZE) * LOG_PAGE_SIZE;
  const end = Math.floor(last / LOG_PAGE_SIZE) * LOG_PAGE_SIZE;
  useEffect(() => { cache.current.clear(); }, [taskId, total, revision]);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    async function load() {
      for (let offset = start; offset <= end && offset < total; offset += LOG_PAGE_SIZE) {
        if (cache.current.has(offset)) continue;
        const page = await fetchLogPage(taskId, offset, LOG_PAGE_SIZE, controller.signal);
        if (controller.signal.aborted) return;
        cache.current.set(offset, new Map(page.logs.map((record, i) => [page.indices?.[i] ?? page.offset + i, record])));
        while (cache.current.size > 12) cache.current.delete(cache.current.keys().next().value!);
        render((value) => value + 1);
      }
    }
    void load().catch((reason: Error) => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [taskId, start, end, total, revision]);
  return { error, recordAt: (index: number) => cache.current.get(Math.floor(index / LOG_PAGE_SIZE) * LOG_PAGE_SIZE)?.get(index),
    loadedAt: (index: number) => cache.current.has(Math.floor(index / LOG_PAGE_SIZE) * LOG_PAGE_SIZE) };
}

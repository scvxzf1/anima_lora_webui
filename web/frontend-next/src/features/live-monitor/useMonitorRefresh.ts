import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocket } from "../../app/useWebSocket";
import { liveMonitorKeys, type TrainingStatus } from "./api";

const EVENT_TYPES = new Set(["status", "progress", "metrics", "log", "system"]);

export function useMonitorRefresh() {
  const client = useQueryClient();
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const active = useRef(false);
  useEffect(() => () => clearTimeout(timer.current), []);

  // Legacy broadcasts have no run identity. They are hints, never authoritative data.
  async function refresh() {
    if (active.current) return;
    active.current = true;
    try {
      await client.invalidateQueries({ queryKey: liveMonitorKeys.status }, { cancelRefetch: false });
      const status = client.getQueryState<TrainingStatus>(liveMonitorKeys.status);
      if (status?.status !== "success" || !status.data?.task_id) return;
      for (const key of [liveMonitorKeys.metrics, liveMonitorKeys.logs]) {
        void client.invalidateQueries({ queryKey: [...key, status.data.task_id] }, { cancelRefetch: false });
      }
    } finally {
      active.current = false;
    }
  }

  function schedule() {
    if (timer.current || active.current) return;
    timer.current = setTimeout(() => {
      timer.current = undefined;
      void refresh();
    }, 750);
  }

  return useWebSocket("/ws/training", (event) => {
    if (!EVENT_TYPES.has(String(event.type))) return;
    const current = client.getQueryData<TrainingStatus>(liveMonitorKeys.status)?.task_id;
    // A different status may announce the next run. Confirm it through HTTP first.
    if (event.type !== "status" && event.task_id && event.task_id !== current) return;
    schedule();
  }, true, () => {
    clearTimeout(timer.current);
    timer.current = undefined;
    void refresh();
    void client.invalidateQueries({ queryKey: ["training-queue"] });
  });
}

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocket } from "../../app/useWebSocket";
import { liveMonitorKeys, type TrainingStatus } from "./api";

const EVENT_TYPES = new Set(["status", "progress", "metrics", "log", "system"]);

export function useMonitorRefresh() {
  const client = useQueryClient();
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const active = useRef(false);
  const rerun = useRef(false);
  const visibilityGeneration = useRef(0);
  useEffect(() => () => clearTimeout(timer.current), []);

  function isVisible() {
    return document.visibilityState === "visible";
  }

  // Legacy broadcasts have no run identity. They are hints, never authoritative data.
  async function refresh() {
    if (!isVisible()) return;
    if (active.current) {
      rerun.current = true;
      return;
    }
    active.current = true;
    const generation = visibilityGeneration.current;
    try {
      await client.invalidateQueries({ queryKey: liveMonitorKeys.status }, { cancelRefetch: false });
      if (!isVisible() || generation !== visibilityGeneration.current) return;
      const status = client.getQueryState<TrainingStatus>(liveMonitorKeys.status);
      if (status?.status !== "success" || !status.data?.task_id) return;
      for (const key of [liveMonitorKeys.metrics, liveMonitorKeys.logs]) {
        void client.invalidateQueries({ queryKey: [...key, status.data.task_id] }, { cancelRefetch: false });
      }
      void client.invalidateQueries({ queryKey: liveMonitorKeys.gpus }, { cancelRefetch: false });
    } finally {
      active.current = false;
      if (rerun.current && isVisible()) {
        rerun.current = false;
        void refresh();
      }
    }
  }

  function schedule() {
    if (!isVisible() || timer.current || active.current) return;
    timer.current = setTimeout(() => {
      timer.current = undefined;
      void refresh();
    }, 750);
  }

  useEffect(() => {
    function onVisibilityChange() {
      visibilityGeneration.current += 1;
      clearTimeout(timer.current);
      timer.current = undefined;
      if (isVisible()) void refresh();
      else rerun.current = false;
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

  return useWebSocket("/ws/training", (event) => {
    if (!isVisible() || !EVENT_TYPES.has(String(event.type))) return;
    const current = client.getQueryData<TrainingStatus>(liveMonitorKeys.status)?.task_id;
    // A different status may announce the next run. Confirm it through HTTP first.
    if (event.type !== "status" && event.task_id && event.task_id !== current) return;
    schedule();
  }, true, () => {
    if (!isVisible()) return;
    clearTimeout(timer.current);
    timer.current = undefined;
    void refresh();
    void client.invalidateQueries({ queryKey: ["training-queue"] });
  });
}

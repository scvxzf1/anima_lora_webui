import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchGpus, liveMonitorKeys } from "../live-monitor/api";
import { DEVICE_STORAGE_KEY, deviceSummary, normalizeDevices, readDeviceSelection, selectionIssue,
  type DeviceSelection, type TrainingDevice } from "./trainingDevices";

export function useTrainingDevices() {
  const forceNext = useRef(false);
  const query = useQuery({ queryKey: liveMonitorKeys.gpus, queryFn: ({ signal }) => {
    const force = forceNext.current;
    forceNext.current = false;
    return fetchGpus(signal, force);
  }, retry: false });
  const devices = useMemo(() => normalizeDevices(query.error || query.data?.stale ? [] : query.data?.gpus || []), [query.data, query.error]);
  const [saved, setSaved] = useState(readDeviceSelection);
  const selection: DeviceSelection = saved || { mode: "single", devices: devices.slice(0, 1) };
  useEffect(() => {
    if (saved === null && devices.length) setSaved({ mode: "single", devices: devices.slice(0, 1) });
  }, [devices, saved]);
  useEffect(() => {
    if (!saved) return;
    try { localStorage.setItem(DEVICE_STORAGE_KEY, JSON.stringify(saved)); } catch { /* Storage may be unavailable. */ }
  }, [saved]);
  const issue = query.error ? `无法读取 GPU：${query.error.message}`
    : query.isPending ? "正在读取 GPU…" : query.data?.stale ? "GPU 采样暂不可用，请刷新列表。" : selectionIssue(selection, devices);
  function refreshDevices() {
    forceNext.current = true;
    return query.refetch();
  }
  function selectDevice(gpu: TrainingDevice) {
    setSaved({ ...selection, devices: selection.mode === "single" ? [gpu]
      : selection.devices.some((item) => item.id === gpu.id)
        ? selection.devices.filter((item) => item.id !== gpu.id) : [...selection.devices, gpu] });
  }
  function setMode(mode: DeviceSelection["mode"]) {
    // Never infer a second GPU or replace a missing device on a mode switch.
    setSaved({ mode, devices: mode === "single" ? selection.devices.slice(0, 1) : selection.devices });
  }
  return { devices, selection, selectDevice, setMode, issue, query, refreshDevices,
    gpuIds: selection.devices.map((gpu) => gpu.id), summary: deviceSummary(selection),
    resetSelection: () => setSaved({ ...selection, devices: selection.devices.filter((saved) =>
      devices.some((gpu) => gpu.id === saved.id && gpu.name === saved.name)) }) };
}

import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { fetchTrainingConfigGroups, fetchTrainingPresets, trainingContextKeys } from "../../api/trainingContext";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { useHotstartIntent } from "../training-config/hotstartIntent";
import { InlineConfirmDialog } from "../../components/InlineConfirmDialog";
import { hotstartTarget, inspectHotstart } from "./historyHotstartApi";
import type { HistoryWeight } from "./api";

export function HistoryHotstart({ weight }: { weight: HistoryWeight }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const selection = useTrainingContextStore();
  const groupsQuery = useQuery({ queryKey: trainingContextKeys.files(), queryFn: ({ signal }) => fetchTrainingConfigGroups(signal) });
  const presetsQuery = useQuery({ queryKey: trainingContextKeys.presets(), queryFn: ({ signal }) => fetchTrainingPresets(signal) });
  const offer = useHotstartIntent((state) => state.offer);
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const groups = Array.isArray(groupsQuery.data) ? groupsQuery.data : [];
  const presets = Array.isArray(presetsQuery.data) ? presetsQuery.data : [];
  const file = groups.flatMap((group) => (group.files || []).map((item) => ({ ...item, methods_subdir: item.methods_subdir || group.methods_subdir }))).find((item) => item.path === selection.configFile && item.trainable !== false);
  const preset = selection.preset;
  const available = Boolean(file && !file.locked && !file.readonly && presets.includes(preset));
  const target = file && hotstartTarget(file, preset);
  const targetKey = target && JSON.stringify(target);
  const latest = useRef({ available, target, path: weight.abs_path, fetching: groupsQuery.isFetching || presetsQuery.isFetching });
  latest.current = { available, target, path: weight.abs_path, fetching: groupsQuery.isFetching || presetsQuery.isFetching };
  const generation = useRef(0);
  useEffect(() => { generation.current++; setOpen(false); setError(""); }, [targetKey, weight.abs_path, available]);

  function stillAvailable(expected: typeof target, path: string | undefined) {
    const current = useTrainingContextStore.getState();
    const groupsState = queryClient.getQueryState(trainingContextKeys.files());
    const presetsState = queryClient.getQueryState(trainingContextKeys.presets());
    if (!expected || !path || !latest.current.available || latest.current.fetching ||
      latest.current.path !== path || JSON.stringify(latest.current.target) !== JSON.stringify(expected) ||
      current.configFile !== expected.configFile || current.preset !== expected.preset ||
      groupsState?.status !== "success" || groupsState.fetchStatus !== "idle" || groupsState.isInvalidated ||
      presetsState?.status !== "success" || presetsState.fetchStatus !== "idle" || presetsState.isInvalidated) return false;
    const groups = queryClient.getQueryData<Awaited<ReturnType<typeof fetchTrainingConfigGroups>>>(trainingContextKeys.files()) || [];
    const presets = queryClient.getQueryData<string[]>(trainingContextKeys.presets()) || [];
    const file = groups.flatMap((group) => (group.files || []).map((item) => ({ ...item, methods_subdir: item.methods_subdir || group.methods_subdir })))
      .find((item) => item.path === expected.configFile && item.trainable !== false && !item.locked && !item.readonly);
    return Boolean(file && presets.includes(expected.preset) && JSON.stringify(hotstartTarget(file, expected.preset)) === JSON.stringify(expected));
  }

  async function confirm() {
    if (!target || !weight.abs_path || pending || !stillAvailable(target, weight.abs_path)) return;
    const path = weight.abs_path;
    const requestGeneration = generation.current;
    setPending(true);
    setError("");
    try {
      const inspected = await inspectHotstart(path, target);
      if (generation.current !== requestGeneration || !stillAvailable(target, path))
        throw new Error("权重、训练配置或预设已改变，请重新检查");
      offer({ ...target, path: inspected.abs_path });
      setOpen(false);
      navigate("/training");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "权重检查失败");
    } finally {
      setPending(false);
    }
  }

  return <>
    <button type="button" disabled={!weight.abs_path || !available || groupsQuery.isFetching || presetsQuery.isFetching} onClick={() => { if (!stillAvailable(target, weight.abs_path)) return; setError(""); setOpen(true); }}>
      应用到训练草稿
    </button>
    {open && target && <InlineConfirmDialog
      title="检查并应用裸权重"
      message={`当前可训练配置：${target.configFile}；硬件预设：${target.preset}。只载入权重并从 step 0 开始，不恢复优化器、调度器或训练步数。确认后仍需在训练页再次检查并手动保存。${error ? `检查失败：${error}` : ""}`}
      confirmLabel="检查权重并前往草稿"
      busy={pending}
      confirmDisabled={!available || groupsQuery.isFetching || presetsQuery.isFetching}
      onCancel={() => setOpen(false)}
      onConfirm={confirm}
    />}
  </>;
}

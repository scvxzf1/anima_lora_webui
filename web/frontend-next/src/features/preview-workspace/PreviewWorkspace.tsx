import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useBlocker, useNavigate } from "react-router-dom";
import { useTrainingContext } from "../../app/useTrainingContext";
import { ApiError } from "../../api/client";
import { fetchRawTrainingConfig, saveTrainingConfigPatch, trainingConfigKeys } from "../training-config/api";
import { PreviewAssets } from "./PreviewAssets";
import { PreviewSettings } from "./PreviewSettings";
import {
  deletePreviewImages, fetchPreviewImages, fetchPreviewSettings, fetchPreviewTasks,
  fetchPreviewWeights, makePreviewGroups, previewKeys, savePreviewSettings,
  type PreviewSettings as Settings, type PreviewSource,
} from "./api";
import "./PreviewWorkspace.css";

export function PreviewWorkspace() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const context = useTrainingContext();
  const [source, setSource] = useState<PreviewSource>("training");
  const [scope, setScope] = useState("latest");
  const [taskId, setTaskId] = useState("");
  const [groupKey, setGroupKey] = useState("");
  const [days, setDays] = useState("all");
  const [draft, setDraft] = useState<Settings | null>(null);
  const [notice, setNotice] = useState("");
  const settings = useQuery({ queryKey: previewKeys.settings, queryFn: ({ signal }) => fetchPreviewSettings(signal) });
  const history = useQuery({ queryKey: previewKeys.tasks, queryFn: ({ signal }) => fetchPreviewTasks(signal) });
  const tasks = useMemo(() => (history.data?.tasks || []).filter((task) => task.job === "training"), [history.data]);
  const groups = useMemo(() => makePreviewGroups(tasks), [tasks]);
  const selectedGroup = groups.find((group) => group.key === groupKey);
  const draftSettings = draft || settings.data || {};
  const settingsDirty = Boolean(draft) && ["training_dir", "inference_dir", "custom_dir"].some((key) =>
    String(draftSettings[key as keyof Settings] || "").trim() !== String(settings.data?.[key as keyof Settings] || "").trim());
  const blocker = useBlocker(settingsDirty);
  useEffect(() => {
    if (settings.data && !draft) setDraft(settings.data);
  }, [settings.data, draft]);
  useEffect(() => {
    if (blocker.state !== "blocked") return;
    if (window.confirm("预览路径有未保存修改。离开页面会丢弃这些修改，是否继续？")) blocker.proceed();
    else blocker.reset();
  }, [blocker]);
  useEffect(() => {
    const listener = (event: BeforeUnloadEvent) => {
      if (!settingsDirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", listener);
    return () => window.removeEventListener("beforeunload", listener);
  }, [settingsDirty]);
  const selectedTaskId = scope === "task" ? taskId : "";
  const assetsKey = previewKeys.assets(source, `${scope}:${selectedTaskId}:${groupKey}`, days);
  const images = useQuery({
    queryKey: [...assetsKey, "images"],
    queryFn: ({ signal }) => fetchPreviewImages(source, scope, taskId, selectedGroup, days, signal),
    enabled: (source !== "training" || scope !== "task" || Boolean(taskId)) && (scope !== "group" || Boolean(groupKey)),
  });
  const weights = useQuery({
    queryKey: [...assetsKey, "weights"],
    queryFn: ({ signal }) => fetchPreviewWeights(scope, taskId, selectedGroup, signal),
    enabled: source === "training" && (scope !== "task" || Boolean(taskId)) && (scope !== "group" || Boolean(groupKey)),
  });
  const save = useMutation({
    mutationFn: () => savePreviewSettings({ training_dir: draftSettings.training_dir, inference_dir: draftSettings.inference_dir, custom_dir: draftSettings.custom_dir }),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: previewKeys.settings });
      setNotice(result.message || "预览路径设置已保存。");
      await queryClient.invalidateQueries({ queryKey: assetsKey });
    },
    onError: (error) => setNotice(error.message),
  });
  const remove = useMutation({
    mutationFn: (files: string[]) => deletePreviewImages(source, files, source === "training" && scope === "task" ? taskId : undefined),
    onSuccess: async (result) => {
      setNotice(result.message || "图片删除操作已完成。");
      await queryClient.invalidateQueries({ queryKey: assetsKey });
    },
    onError: (error) => setNotice(error.message),
  });
  const imagesData = images.data?.images || [];
  const weightsData = weights.data?.weights || [];
  const effectiveDirectory = source === "training"
    ? settings.data?.effective_training_dir || draftSettings.training_dir
    : source === "inference" ? draftSettings.inference_dir : draftSettings.custom_dir;

  async function hotstart(path: string) {
    const file = context.selectedFile;
    if (!file || file.locked || file.readonly) {
      setNotice("当前没有可编辑的训练配置；请先在训练配置页选择可写配置。");
      return;
    }
    if (!window.confirm(`检查兼容性并将此权重写入 ${file.path}？此操作会立即保存配置。`)) return;
    try {
      const check = await fetch("/api/training/continue-lora/inspect", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, variant: file.method || "lora", preset: context.selectedPreset, methods_subdir: file.methods_subdir || "gui-methods", config_file: file.path }),
      }).then((response) => response.json());
      if (!check.compatible || check.ok === false) throw new Error(check.message || check.error || "权重与训练配置不兼容");
      const raw = await fetchRawTrainingConfig(file.path);
      const saved = await saveTrainingConfigPatch(file.path, { network_weights: check.abs_path || path, dim_from_weights: true }, raw.revision);
      queryClient.setQueryData(trainingConfigKeys.raw(file.path), { ...raw, content: saved.content, revision: saved.revision });
      setNotice(`热启动权重已写入 ${file.path}。`);
      navigate("/training");
    } catch (error) {
      setNotice(error instanceof ApiError && error.status === 409 ? "训练配置在操作期间已变化，请刷新后重试。" : error instanceof Error ? error.message : "设置热启动失败");
    }
  }

  return <main className="preview-workspace">
    <header className="preview-header"><div><p className="eyebrow">ASSET BROWSER</p><h1>预览工作区</h1></div><span className="preview-directory">{effectiveDirectory || "尚未设置目录"}</span></header>
    <section className="preview-controls" aria-label="预览筛选">
      <div className="preview-source-tabs" role="group" aria-label="预览来源">{([ ["training", "训练样张"], ["inference", "推理输出"], ["custom", "自定义目录"] ] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={source === value} onClick={() => setSource(value)}>{label}</button>)}</div>
      <label>训练范围<select value={scope} disabled={source !== "training"} onChange={(event) => setScope(event.target.value)}><option value="latest">最新任务</option><option value="task">指定任务</option><option value="group">配置分组</option></select></label>
      {source === "training" && scope === "task" && <label>任务<select value={taskId} onChange={(event) => setTaskId(event.target.value)}><option value="">选择训练任务</option>{tasks.map((task) => <option key={task.id} value={task.id}>{task.name || task.variant || task.id}</option>)}</select></label>}
      {source === "training" && scope === "group" && <label>分组<select value={groupKey} onChange={(event) => setGroupKey(event.target.value)}><option value="">选择配置分组</option>{groups.map((group) => <option key={group.key} value={group.key}>{group.label} · {group.tasks.length} 次</option>)}</select></label>}
      <label>图片时间<select value={days} onChange={(event) => setDays(event.target.value)}><option value="7">最近 7 天</option><option value="14">最近 14 天</option><option value="30">最近 30 天</option><option value="all">全部时间</option></select></label>
      <button type="button" onClick={() => { void images.refetch(); if (source === "training") void weights.refetch(); }}>刷新</button>
    </section>
    <PreviewSettings settings={draftSettings} dirty={settingsDirty} saving={save.isPending} onChange={setDraft} onSave={() => save.mutate()} onDefaults={() => setDraft({ ...draftSettings, ...(draftSettings.defaults || {}) })} />
    {notice && <p role="status" className="preview-notice">{notice}</p>}
    {history.error && <p role="alert">{history.error.message} <button type="button" onClick={() => void history.refetch()}>重试任务列表</button></p>}
    {images.error && <p role="alert">{images.error.message} <button type="button" onClick={() => void images.refetch()}>重试图片</button></p>}
    {weights.error && <p role="alert">{weights.error.message} <button type="button" onClick={() => void weights.refetch()}>重试权重</button></p>}
    {(images.isPending || (source === "training" && weights.isPending)) && <p role="status">正在读取预览资源…</p>}
    {!images.isPending && !images.error && <p className="preview-note">{images.data?.message || `${images.data?.count ?? imagesData.length} / ${images.data?.total ?? imagesData.length} 张 · ${images.data?.directory || effectiveDirectory || "目录未设置"}`}</p>}
    {!weights.isPending && source === "training" && !weights.error && weights.data?.message && <p className="preview-note">{weights.data.message}</p>}
    <PreviewAssets images={imagesData} weights={source === "training" ? weightsData : []} taskId={selectedTaskId || undefined} readOnlyGroup={source === "training" && scope === "group"} onDelete={(files) => {
      if (source === "training" && scope === "group") return;
      if (window.confirm(`永久删除所选 ${files.length} 张图片？此操作无法撤销。`)) remove.mutate(files);
    }} onHotstart={(path) => void hotstart(path)} />
  </main>;
}

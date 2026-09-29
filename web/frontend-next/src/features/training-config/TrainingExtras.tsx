import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { apiRequest, ApiError } from "../../api/client";
import { findModelCapability, useModelCapabilities } from "../../api/modelCapabilities";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { CommandDialog } from "../../components/CommandDialog";
import { useServerDraftState } from "../../components/useServerDraft";
import {
  fetchModelConfigs,
  settingsKeys,
  type ModelConfigItem,
} from "../settings/api";
import { fetchRawTrainingConfig, saveTrainingConfigPatch } from "./api";
import { PromptVisualEditor } from "./PromptVisualEditor";

type PromptsResponse = {
  ok: boolean;
  file: string;
  content: string;
  prompts: string[];
  exists?: boolean;
  revision?: string;
  targetRevision?: string;
};

export function TrainingModelPicker({
  disabled,
  onSelect,
}: {
  disabled: boolean;
  onSelect: (item: ModelConfigItem) => void;
}) {
  const query = useQuery({
    queryKey: settingsKeys.models,
    queryFn: ({ signal }) => fetchModelConfigs(signal),
  });
  return (
    <div className="toolbar">
      <label className="training-model-picker">
        <span>模型组合</span>
        <select
          aria-label="快速选择模型组合"
          disabled={disabled}
          value=""
          onChange={(e) => {
            const item = query.data?.items.find(
              (entry) => entry.id === e.target.value,
            );
            if (item) onSelect(item);
          }}
        >
          <option value="">选择模型组合</option>
          {query.data?.items.map((item) => (
            <option
              key={item.id}
              value={item.id}
              disabled={item.complete === false}
            >
              {item.name} · {item.model_family}{item.capability_labels?.length ? ` · ${item.capability_labels.join(" / ")}` : ""}
            </option>
          ))}
        </select>
      </label>
      <Link to="/models">模型库</Link>
      {query.error && (
        <small role="alert" className="form-error">
          {query.error.message}
        </small>
      )}
    </div>
  );
}

export function TrainingSamplePrompts({
  file,
  promptFile,
  configRevision,
  modelFamily = "anima",
  trainingTask = "t2i",
  onClose,
  onSaved,
}: {
  file: TrainingConfigFile;
  promptFile: string;
  configRevision?: string;
  modelFamily?: string;
  trainingTask?: string;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const qc = useQueryClient();
  const capabilities = useModelCapabilities();
  const capability = findModelCapability(capabilities.data?.items, modelFamily);
  const key = ["training-config", "sample-prompts", promptFile, file.path];
  const query = useQuery({
    queryKey: key,
    queryFn: async ({ signal }) => {
      const read = (path: string) => apiRequest<PromptsResponse>(
        `/api/config/sample-prompts?${new URLSearchParams({ file: path })}`, { signal });
      if (promptFile) return read(promptFile);
      const fork = file.path.replace(/^configs\//, "configs/sample-prompts/").replace(/\.toml$/i, ".txt");
      const saved = await read(fork);
      if (saved.exists === undefined && !saved.content) throw new Error("当前后端未提供文件存在状态，请重启 WebUI 后读取旧提示词。不会覆盖已有文件。");
      if (saved.exists !== false) return saved;
      const fallback = await read("configs/sample_prompts.txt");
      return { ...fallback, targetRevision: saved.revision };
    },
  });
  const editor = useServerDraftState(query.data?.content);
  const [notice, setNotice] = useState("");
  const [editingRow, setEditingRow] = useState(false);
  const [rowDirty, setRowDirty] = useState(false);
  const [confirmClose, setConfirmClose] = useState(false);
  const continueRef = useRef<HTMLButtonElement>(null);
  const closeOrigin = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (confirmClose) continueRef.current?.focus();
    else closeOrigin.current?.focus();
  }, [confirmClose]);
  const save = useMutation({
    mutationFn: async () => {
      const saved = await apiRequest<PromptsResponse>(
        "/api/config/sample-prompts",
        {
          method: "PUT",
          body: JSON.stringify({
            file: query.data?.file,
            train_config_file: file.path,
            content: editor.draft,
            revision: query.data?.targetRevision || query.data?.revision,
          }),
        },
      );
      setNotice(`提示词已保存至 ${saved.file}；正在更新配置引用`);
      await saveTrainingConfigPatch(file.path, { sample_prompts: saved.file }, configRevision);
      return saved;
    },
    retry: false,
    onSuccess: async (data) => {
      qc.setQueryData(key, data);
      editor.accept(data.content, data.content);
      await onSaved();
      onClose();
    },
  });
  const saveConflict = save.error instanceof ApiError && save.error.status === 409;
  async function reloadAfterConflict() {
    if (!window.confirm("重新加载会放弃当前未保存的提示词修改。是否继续？")) return;
    const refreshed = await query.refetch();
    await fetchRawTrainingConfig(file.path);
    if (refreshed.data) editor.replace(refreshed.data.content);
    save.reset();
    setNotice("");
  }
  const close = () => {
    if (confirmClose) { setConfirmClose(false); return; }
    if (!editor.dirty && !rowDirty) { onClose(); return; }
    closeOrigin.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setConfirmClose(true);
  };
  return (
    <CommandDialog title={confirmClose ? "放弃样张提示词修改？" : "采样样张"} busy={save.isPending} onClose={close}>
      {confirmClose && <section>
        <p>存在未保存的修改，放弃后无法恢复。</p>
        <footer className="toolbar">
          <button type="button" ref={continueRef} onClick={() => setConfirmClose(false)}>继续编辑</button>
          <button type="button" onClick={onClose}>放弃修改</button>
        </footer>
      </section>}
      <div hidden={confirmClose}>
      <p className="effective-path">{promptFile ? "已关联" : "未关联 · 保存后关联当前配置"}: {query.data?.file || "正在读取"}</p>
      <p className="effective-path">采样模型：{capability?.display_name || modelFamily}</p>
      {capabilities.error && <p role="alert">采样能力读取失败，图像编辑暂不可用。<button type="button" onClick={() => void capabilities.refetch()}>重试</button></p>}
      <PromptVisualEditor
        modelFamily={modelFamily}
        supportedPreviewTasks={capabilities.error ? ["t2i"] : (capability?.supported_preview_tasks || ["t2i"])}
        maxPreviewReferences={capability?.max_preview_references ?? 4}
        defaultTask={trainingTask}
        content={editor.draft ?? ""}
        disabled={query.isPending || save.isPending || Boolean(query.error)}
        onChange={editor.setDraft}
        onEditing={(active, dirty) => { setEditingRow(active); setRowDirty(dirty); }}
      />
      {(query.error || save.error) && (
        <p role="alert" className="form-error">
          {(query.error || save.error)?.message}
          {saveConflict && <button type="button" onClick={() => void reloadAfterConflict()}>重新加载提示词</button>}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      <footer className="toolbar">
        <button type="button" onClick={close} disabled={save.isPending}>
          取消
        </button>
        <button
          type="button"
          className="primary-command"
          onClick={() => save.mutate()}
          disabled={
            save.isPending ||
            editingRow ||
            query.isPending ||
            Boolean(query.error) ||
            saveConflict ||
            (Boolean(promptFile) && !editor.dirty) ||
            Boolean(file.locked || file.readonly)
          }
        >
          保存提示词与配置引用
        </button>
      </footer>
      </div>
    </CommandDialog>
  );
}

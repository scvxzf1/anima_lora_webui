import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { apiRequest } from "../../api/client";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { CommandDialog } from "../../components/CommandDialog";
import { useServerDraftState } from "../../components/useServerDraft";
import {
  fetchModelConfigs,
  settingsKeys,
  type ModelConfigItem,
} from "../settings/api";
import { saveTrainingConfigPatch } from "./api";
import { PromptVisualEditor } from "./PromptVisualEditor";

type PromptsResponse = {
  ok: boolean;
  file: string;
  content: string;
  prompts: string[];
  exists?: boolean;
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
              {item.name} · {item.model_family}
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
  onClose,
  onSaved,
}: {
  file: TrainingConfigFile;
  promptFile: string;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const qc = useQueryClient();
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
      return saved.exists === false ? read("configs/sample_prompts.txt") : saved;
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
          }),
        },
      );
      setNotice(`提示词已保存至 ${saved.file}`);
      await saveTrainingConfigPatch(file.path, { sample_prompts: saved.file });
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
  const close = () => {
    if (confirmClose) { setConfirmClose(false); return; }
    if (!editor.dirty && !rowDirty) { onClose(); return; }
    closeOrigin.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setConfirmClose(true);
  };
  return (
    <CommandDialog title={confirmClose ? "放弃样张提示词修改？" : "样张提示词"} busy={save.isPending} onClose={close}>
      {confirmClose && <section>
        <p>存在未保存的修改，放弃后无法恢复。</p>
        <footer className="toolbar">
          <button type="button" ref={continueRef} onClick={() => setConfirmClose(false)}>继续编辑</button>
          <button type="button" onClick={onClose}>放弃修改</button>
        </footer>
      </section>}
      <div hidden={confirmClose}>
      <p className="effective-path">{promptFile ? "已关联" : "未关联 · 保存后关联当前配置"}: {query.data?.file || "正在读取"}</p>
      <PromptVisualEditor
        content={editor.draft ?? ""}
        disabled={query.isPending || save.isPending || Boolean(query.error)}
        onChange={editor.setDraft}
        onEditing={(active, dirty) => { setEditingRow(active); setRowDirty(dirty); }}
      />
      {(query.error || save.error) && (
        <p role="alert" className="form-error">
          {(query.error || save.error)?.message}
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

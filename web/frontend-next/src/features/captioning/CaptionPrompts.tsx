import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { useServerDraft } from "../../components/useServerDraft";
import {
  captioningKeys,
  fetchCaptionPrompts,
  saveCaptionPrompt,
  deleteCaptionPrompt,
  type PromptPreset,
} from "./api";

export function CaptionPrompts() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: captioningKeys.prompts,
    queryFn: ({ signal }) => fetchCaptionPrompts(signal),
  });
  const [editing, setEditing] = useState<PromptPreset | "new" | null>(null);
  const remove = useMutation({
    mutationFn: deleteCaptionPrompt,
    retry: false,
    onSuccess: (data) => qc.setQueryData(captioningKeys.prompts, data),
  });
  return (
    <section className="settings-section">
      <header className="review-heading">
        <h2>提示词预设</h2>
        <button type="button" onClick={() => setEditing("new")}>
          <Plus size={16} />
          新建提示词
        </button>
      </header>
      <div className="provider-list">
        {query.data?.presets.map((preset) => (
          <article key={preset.id}>
            <div>
              <h3>{preset.name}</h3>
              <p>{preset.builtin ? "内置预设" : "自定义预设"}</p>
            </div>
            <div className="toolbar">
              <button type="button" onClick={() => setEditing(preset)}>
                <Pencil size={15} />
                {preset.builtin ? "复制编辑" : "编辑"}
              </button>
              <button
                type="button"
                disabled={preset.builtin || remove.isPending}
                onClick={() => {
                  if (window.confirm(`删除提示词预设“${preset.name}”？`))
                    remove.mutate(preset.id);
                }}
              >
                <Trash2 size={15} />
                删除
              </button>
            </div>
          </article>
        ))}
      </div>
      {query.error && (
        <p className="form-error" role="alert">
          {query.error.message}
          <button
            type="button"
            onClick={() => void query.refetch()}
            disabled={query.isFetching}
          >
            重试提示词
          </button>
        </p>
      )}
      {remove.error && (
        <p className="form-error" role="alert">
          {remove.error.message}
        </p>
      )}
      {editing && (
        <PromptEditor
          preset={editing === "new" ? undefined : editing}
          onClose={() => setEditing(null)}
        />
      )}
    </section>
  );
}

function PromptEditor({
  preset,
  onClose,
}: {
  preset?: PromptPreset;
  onClose: () => void;
}) {
  const initial = useMemo(
    () => ({
      name: preset ? `${preset.name}${preset.builtin ? " 副本" : ""}` : "",
      user_prompt: preset?.user_prompt || "",
      system_prompt: preset?.system_prompt || "",
    }),
    [preset],
  );
  const editor = useServerDraft(initial);
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: () =>
      saveCaptionPrompt(
        editor.draft!,
        preset?.builtin ? undefined : preset?.id,
      ),
    retry: false,
    onSuccess: (data) => {
      qc.setQueryData(captioningKeys.prompts, data);
      onClose();
    },
  });
  function close() {
    if (!editor.dirty || window.confirm("放弃提示词修改？")) onClose();
  }
  return (
    <CommandDialog title="提示词预设" onClose={close} busy={save.isPending}>
      {editor.draft && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            save.mutate();
          }}
        >
          <fieldset
            className="unframed-fieldset settings-grid"
            disabled={save.isPending}
          >
            <label className="full-width">
              <span>名称</span>
              <input
                required
                value={editor.draft.name}
                onChange={(e) =>
                  editor.setDraft((d) => ({ ...d!, name: e.target.value }))
                }
              />
            </label>
            <label className="full-width">
              <span>系统提示词</span>
              <textarea
                rows={5}
                value={editor.draft.system_prompt}
                onChange={(e) =>
                  editor.setDraft((d) => ({
                    ...d!,
                    system_prompt: e.target.value,
                  }))
                }
              />
            </label>
            <label className="full-width">
              <span>用户提示词</span>
              <textarea
                rows={7}
                value={editor.draft.user_prompt}
                onChange={(e) =>
                  editor.setDraft((d) => ({
                    ...d!,
                    user_prompt: e.target.value,
                  }))
                }
              />
            </label>
          </fieldset>
          <footer className="toolbar">
            <button type="button" onClick={close} disabled={save.isPending}>
              取消
            </button>
            <button
              type="submit"
              className="primary-command"
              disabled={save.isPending}
            >
              保存提示词
            </button>
          </footer>
        </form>
      )}
      {save.error && (
        <p className="form-error" role="alert">
          {save.error.message}
        </p>
      )}
    </CommandDialog>
  );
}

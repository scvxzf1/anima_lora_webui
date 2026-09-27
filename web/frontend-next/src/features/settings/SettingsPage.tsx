import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Save, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import { useServerDraft } from "../../components/useServerDraft";
import { ApiError } from "../../api/client";
import { QueryFeedback } from "../../components/QueryFeedback";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { fetchGlobalSettings, saveGlobalSettings, settingsKeys } from "./api";
import {
  SETTINGS_GROUPS,
  settingsDraft,
  settingsPatch,
  applySettingsDefaults,
} from "./settingsForm";
import "./settings.css";

export function SettingsPage() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: settingsKeys.global,
    queryFn: ({ signal }) => fetchGlobalSettings(signal),
    retry: false,
  });
  const incoming = useMemo(
    () => (query.data ? settingsDraft(query.data) : undefined),
    [query.data],
  );
  const editor = useServerDraft(incoming);
  const [notice, setNotice] = useState("");
  const save = useMutation({
    mutationFn: (submitted: NonNullable<typeof editor.draft>) =>
      saveGlobalSettings({ ...settingsPatch(submitted, editor.baseline!), revision: query.data?.revision }),
    onSuccess: async (data, submitted) => {
      qc.setQueryData(settingsKeys.global, data);
      editor.accept(settingsDraft(data), submitted);
      if (data.requires_reload) {
        await qc.cancelQueries({
          predicate: (q) => q.queryKey[0] !== "settings",
        });
        qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "settings" });
        useTrainingContextStore.getState().selectConfigFile("");
      }
      setNotice(
        data.requires_reload
          ? "配置根已切换，现有文件未迁移。请刷新页面载入新目录。"
          : "全局设置已保存",
      );
      await qc.invalidateQueries({
        predicate: (q) => q.queryKey[0] !== "settings",
      });
      await qc.invalidateQueries({ queryKey: settingsKeys.models });
    },
    retry: false,
  });
  const saveConflict = save.error instanceof ApiError && save.error.status === 409;
  async function reloadAfterConflict() {
    if (!window.confirm("重新读取会放弃当前未保存的设置修改。是否继续？")) return;
    const refreshed = await query.refetch();
    if (refreshed.data) editor.replace(settingsDraft(refreshed.data));
    save.reset();
  }
  function submit() {
    if (!editor.draft || !editor.baseline || save.isPending || saveConflict || query.error) return;
    const patch = settingsPatch(editor.draft, editor.baseline);
    const pathsChanged = [
      "configs_root",
      "history_root",
      "queue_root",
      "output_root",
    ].some((key) => key in patch);
    if (
      pathsChanged &&
      !window.confirm(
        "更改目录会切换后续读写范围，不迁移现有配置、队列或产物。确认保存吗？",
      )
    )
      return;
    save.mutate(editor.draft);
  }
  return (
    <div className="settings-shell">
      <main className="settings-page">
        <header className="page-heading">
          <div>
            <p className="eyebrow">SYSTEM</p>
            <h1>全局设置</h1>
          </div>
          <span className="state-label">
            {editor.dirty ? "未保存" : query.error ? "读取失败" : !query.data ? "读取中" : "已同步"}
          </span>
        </header>
        <QueryFeedback query={query} label="设置" hasData={Boolean(query.data)} />
        {editor.draft ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <fieldset disabled={save.isPending} className="unframed-fieldset">
              {SETTINGS_GROUPS.map((group) => (
                <section className="settings-section" key={group.title}>
                  <h2>{group.title}</h2>
                  <div className="settings-grid">
                    {group.fields.map((field) => (
                      <label
                        key={field.key}
                        className={
                          field.kind === "boolean" ? "checkbox-row" : ""
                        }
                      >
                        <span>{field.label}</span>
                        {field.kind === "boolean" ? (
                          <input
                            type="checkbox"
                            checked={Boolean(editor.draft![field.key])}
                            onChange={(e) =>
                              editor.setDraft((d) => ({
                                ...d!,
                                [field.key]: e.target.checked,
                              }))
                            }
                          />
                        ) : (
                          <input
                            type={field.kind}
                            min={field.min}
                            max={field.max}
                            required={
                              field.kind === "number" && !field.optional
                            }
                            placeholder={field.optional ? "跟随全局" : ""}
                            value={String(editor.draft![field.key])}
                            onChange={(e) =>
                              editor.setDraft((d) => ({
                                ...d!,
                                [field.key]:
                                  field.kind === "number" &&
                                  e.target.value !== ""
                                    ? Number(e.target.value)
                                    : e.target.value,
                              }))
                            }
                          />
                        )}
                        {query.data?.effective_paths?.[field.key] && (
                          <small className="effective-path">
                            实际生效：{query.data.effective_paths[field.key]}
                          </small>
                        )}
                      </label>
                    ))}
                  </div>
                </section>
              ))}
            </fieldset>
            <footer className="settings-actions">
              <button
                type="submit"
                className="primary-command"
                disabled={!editor.dirty || save.isPending || saveConflict || Boolean(query.error)}
              >
                <Save size={16} />
                {save.isPending ? "保存中" : "保存设置"}
              </button>
              <button
                type="button"
                disabled={!editor.dirty || save.isPending}
                onClick={editor.reset}
              >
                <RotateCcw size={16} />
                还原修改
              </button>
              <button
                type="button"
                disabled={save.isPending || !query.data?.defaults}
                onClick={() => {
                  if (
                    window.confirm(
                      "将当前表单恢复为服务器提供的默认值？保存后才会生效。",
                    )
                  )
                    editor.setDraft(
                      applySettingsDefaults(
                        editor.draft!,
                        query.data?.defaults,
                      ),
                    );
                }}
              >
                恢复默认
              </button>
            </footer>
          </form>
        ) : null}
        {save.error && (
          <p role="alert" className="form-error">
            {save.error.message}
            {saveConflict && <button type="button" onClick={() => void reloadAfterConflict()}>重新读取设置</button>}
          </p>
        )}
        {notice && !editor.dirty && !query.error && <p role="status">{notice}</p>}
      </main>
    </div>
  );
}

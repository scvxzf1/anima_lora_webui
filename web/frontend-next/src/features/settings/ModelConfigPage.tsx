import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowUp,
  FolderPlus,
  Plus,
  Save,
  Trash2,
  RotateCcw,
} from "lucide-react";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { useServerDraft } from "../../components/useServerDraft";
import { QueryFeedback } from "../../components/QueryFeedback";
import {
  fetchModelConfigs,
  saveModelConfigs,
  settingsKeys,
  type ModelConfigItem,
  type ModelConfigResponse,
} from "./api";
import {
  addModel,
  deleteModel,
  modelGroups,
  MODEL_PATHS,
  moveModel,
  reorderModel,
} from "./modelLibrary";
import "./settings.css";

export function ModelConfigPage() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: settingsKeys.models,
    queryFn: ({ signal }) => fetchModelConfigs(signal),
    retry: false,
  });
  const editor = useServerDraft(query.data);
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [notice, setNotice] = useState("");
  const save = useMutation({
    mutationFn: (submitted: ModelConfigResponse) => saveModelConfigs(submitted),
    retry: false,
    onSuccess: async (data, submitted) => {
      qc.setQueryData(settingsKeys.models, data);
      editor.accept(data, submitted);
      setNotice("模型配置已保存");
      await qc.invalidateQueries({ queryKey: settingsKeys.global });
    },
  });
  const draft = editor.draft;
  const item =
    draft?.items.find((entry) => entry.id === selectedId) || draft?.items[0];
  const groups = draft ? modelGroups(draft) : [];
  const groupId = groups.find((group) =>
    group.item_ids.includes(item?.id || ""),
  )?.id;
  function update(patch: Partial<ModelConfigItem>) {
    editor.setDraft(
      (d) =>
        d && {
          ...d,
          items: d.items.map((entry) =>
            entry.id === item?.id ? { ...entry, ...patch } : entry,
          ),
        },
    );
    setNotice("");
  }
  function addGroup() {
    const label = window.prompt("分组名称")?.trim();
    if (!label || !draft) return;
    if (
      groups.some((group) => group.label.toLowerCase() === label.toLowerCase())
    ) {
      setNotice("分组名称已存在");
      return;
    }
    editor.setDraft({
      ...draft,
      groups: [
        ...groups,
        {
          id: `group-${crypto.randomUUID().slice(0, 12)}`,
          label,
          item_ids: [],
        },
      ],
    });
  }
  function deleteGroup(id: string) {
    if (
      !draft ||
      groups.length < 2 ||
      !window.confirm(
        "删除分组后，其中的模型配置会移入其他分组。不会删除模型文件。",
      )
    )
      return;
    const removed = groups.find((group) => group.id === id)!;
    const remaining = groups.filter((group) => group.id !== id);
    remaining[0] = {
      ...remaining[0],
      item_ids: [...remaining[0].item_ids, ...removed.item_ids],
    };
    editor.setDraft({ ...draft, groups: remaining });
  }
  return (
    <div className="settings-shell">
      <main className="settings-page">
        <header className="page-heading">
          <div>
            <p className="eyebrow">MODEL LIBRARY</p>
            <h1>模型配置</h1>
          </div>
          <span className="state-label">{editor.dirty ? "未保存" : query.error ? "读取失败" : !query.data ? "读取中" : "已同步"}</span>
        </header>
        <QueryFeedback query={query} label="模型配置" hasData={Boolean(query.data)} />
        {!draft ? null : !item ? (
          <button
            type="button"
            onClick={() => editor.setDraft(addModel(draft))}
          >
            <Plus size={16} />
            新建模型配置
          </button>
        ) : (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (!save.isPending && !query.error) save.mutate(draft);
            }}
          >
            <fieldset
              disabled={save.isPending}
              className="unframed-fieldset model-workspace"
            >
              <aside className="object-library">
                <header>
                  <h2>模型组合</h2>
                  <button
                    type="button"
                    title="新建模型配置"
                    aria-label="新建模型配置"
                    onClick={() => {
                      const next = addModel(draft, groupId);
                      editor.setDraft(next);
                      setSelectedId(next.items.at(-1)!.id);
                    }}
                  >
                    <Plus size={17} />
                  </button>
                  <button
                    type="button"
                    title="新建模型分组"
                    aria-label="新建模型分组"
                    onClick={addGroup}
                  >
                    <FolderPlus size={17} />
                  </button>
                </header>
                <input
                  aria-label="搜索模型配置"
                  placeholder="搜索模型配置"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                {groups.map((group) => (
                  <section key={group.id}>
                    <div className="library-group-heading">
                      <button
                        type="button"
                        title="重命名分组"
                        onClick={() => {
                          const label = window
                            .prompt("分组名称", group.label)
                            ?.trim();
                          if (label)
                            editor.setDraft({
                              ...draft,
                              groups: groups.map((g) =>
                                g.id === group.id ? { ...g, label } : g,
                              ),
                            });
                        }}
                      >
                        {group.label}
                      </button>
                      <button
                        type="button"
                        title="删除分组"
                        aria-label={`删除分组 ${group.label}`}
                        disabled={groups.length < 2}
                        onClick={() => deleteGroup(group.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                    {group.item_ids
                      .map((id) => draft.items.find((entry) => entry.id === id))
                      .filter(
                        (entry) =>
                          entry &&
                          `${entry.name} ${entry.model_family}`
                            .toLowerCase()
                            .includes(search.toLowerCase()),
                      )
                      .map((entry) => (
                        <button
                          className="object-row"
                          type="button"
                          key={entry!.id}
                          data-selected={entry!.id === item.id}
                          onClick={() => setSelectedId(entry!.id)}
                        >
                          <span>{entry!.name}</span>
                          <small>
                            {entry!.model_family}
                            {draft.default_id === entry!.id ? " · 默认" : ""}
                          </small>
                        </button>
                      ))}
                  </section>
                ))}
              </aside>
              <section className="model-editor">
                <header>
                  <h2>{item.name || "未命名配置"}</h2>
                  <div className="toolbar">
                    <button
                      type="button"
                      aria-label="上移模型配置"
                      title="上移"
                      onClick={() =>
                        editor.setDraft(reorderModel(draft, item.id, -1))
                      }
                    >
                      <ArrowUp size={16} />
                    </button>
                    <button
                      type="button"
                      aria-label="下移模型配置"
                      title="下移"
                      onClick={() =>
                        editor.setDraft(reorderModel(draft, item.id, 1))
                      }
                    >
                      <ArrowDown size={16} />
                    </button>
                    <button
                      type="button"
                      className="danger"
                      disabled={item.id === draft.default_id}
                      onClick={() => {
                        if (
                          window.confirm(
                            `删除模型配置“${item.name}”？不会删除模型文件。`,
                          )
                        )
                          editor.setDraft(deleteModel(draft, item.id));
                      }}
                    >
                      <Trash2 size={16} />
                      删除配置
                    </button>
                  </div>
                </header>
                <div className="settings-grid">
                  <label>
                    <span>名称</span>
                    <input
                      required
                      maxLength={80}
                      value={item.name}
                      onChange={(e) => update({ name: e.target.value })}
                    />
                  </label>
                  <label>
                    <span>模型族</span>
                    <select
                      value={item.model_family}
                      onChange={(e) => update({ model_family: e.target.value })}
                    >
                      <option value="anima">Anima</option>
                      <option value="krea2_raw">Krea-2 Raw</option>
                      <option value="z_image">Z-Image</option>
                    </select>
                  </label>
                  <label>
                    <span>所属分组</span>
                    <select
                      value={groupId}
                      onChange={(e) =>
                        editor.setDraft(
                          moveModel(draft, item.id, e.target.value),
                        )
                      }
                    >
                      {groups.map((group) => (
                        <option key={group.id} value={group.id}>
                          {group.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={draft.default_id === item.id}
                      disabled={draft.default_id === item.id}
                      onChange={() =>
                        editor.setDraft({ ...draft, default_id: item.id })
                      }
                    />
                    <span>默认模型组合</span>
                  </label>
                  {MODEL_PATHS.map(([key, label]) => (
                    <label className="full-width" key={key}>
                      <span>{label}</span>
                      <input
                        required
                        value={item[key]}
                        onChange={(e) => update({ [key]: e.target.value })}
                      />
                    </label>
                  ))}
                </div>
              </section>
            </fieldset>
            <footer className="settings-actions">
              <button
                className="primary-command"
                type="submit"
                disabled={!editor.dirty || save.isPending || Boolean(query.error)}
              >
                <Save size={16} />
                保存模型配置
              </button>
              <button
                type="button"
                disabled={!editor.dirty || save.isPending}
                onClick={editor.reset}
              >
                <RotateCcw size={16} />
                还原修改
              </button>
            </footer>
          </form>
        )}
        {save.error && (
          <p className="form-error" role="alert">
            {save.error.message}
            {save.error instanceof ApiError && save.error.status === 409 && (
              <button
                type="button"
                disabled={query.isFetching}
                onClick={async () => {
                  if (
                    !window.confirm(
                      "重新载入服务器模型库会丢弃当前未保存修改，确认继续吗？",
                    )
                  )
                    return;
                  const result = await query.refetch();
                  if (result.data && !result.error) {
                    editor.accept(result.data, editor.draft!);
                    save.reset();
                  }
                }}
              >
                重新载入服务器版本
              </button>
            )}
          </p>
        )}
        {notice && !editor.dirty && !query.error && <p role="status">{notice}</p>}
      </main>
    </div>
  );
}

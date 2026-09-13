import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { useServerDraft } from "../../components/useServerDraft";
import { LocalProfileFields } from "./LocalProfileFields";
import {
  captioningKeys,
  saveCaptionProfile,
  type CaptionProfile,
  type ProviderType,
} from "./api";

export function CaptionProfileEditor({
  profile,
  types,
  onClose,
}: {
  profile?: CaptionProfile;
  types: ProviderType[];
  onClose: () => void;
}) {
  const initial = useMemo(
    () => ({
      name: profile?.name || "",
      provider: profile?.provider || "openai_compatible",
      config: profile?.config || ({} as Record<string, unknown>),
    }),
    [profile],
  );
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [extra, setExtra] = useState("");
  const editor = useServerDraft(initial, Boolean(apiKey || clearKey || extra));
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: async () => {
      const additional: unknown = extra.trim() ? JSON.parse(extra) : {};
      if (
        !additional ||
        typeof additional !== "object" ||
        Array.isArray(additional)
      )
        throw new Error("高级配置必须是 JSON 对象");
      return saveCaptionProfile(
        {
          ...editor.draft,
          config: { ...editor.draft!.config, ...additional },
          ...(apiKey ? { api_key: apiKey } : {}),
          ...(clearKey ? { clear_api_key: true } : {}),
        },
        profile?.id,
      );
    },
    retry: false,
    gcTime: 0,
    onSuccess: (data) => {
      setApiKey("");
      qc.setQueryData(captioningKeys.profiles, data);
      onClose();
    },
  });
  const draft = editor.draft;
  function close() {
    if (
      !(editor.dirty || apiKey || extra || clearKey) ||
      window.confirm("放弃未保存的接入配置？")
    )
      onClose();
  }
  function update(key: string, value: unknown) {
    editor.setDraft((d) => ({ ...d!, config: { ...d!.config, [key]: value } }));
  }
  return (
    <CommandDialog
      title={profile ? "编辑接入预设" : "新建接入预设"}
      onClose={close}
      busy={save.isPending}
    >
      {draft && (
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
            <label>
              <span>名称</span>
              <input
                required
                value={draft.name}
                onChange={(e) =>
                  editor.setDraft({ ...draft, name: e.target.value })
                }
              />
            </label>
            <label>
              <span>接入类型</span>
              <select
                value={draft.provider}
                disabled={Boolean(profile)}
                onChange={(e) =>
                  editor.setDraft({
                    ...draft,
                    provider: e.target.value,
                    config: {},
                  })
                }
              >
                {types.map((type) => (
                  <option value={type.id} key={type.id}>
                    {type.label}
                  </option>
                ))}
              </select>
            </label>
            {draft.provider === "openai_compatible" ? (
              <>
                <label className="full-width">
                  <span>API 地址</span>
                  <input
                    type="url"
                    required
                    value={String(draft.config.base_url || "")}
                    onChange={(e) => update("base_url", e.target.value)}
                  />
                </label>
                <label>
                  <span>模型名称</span>
                  <input
                    required
                    value={String(draft.config.model || "")}
                    onChange={(e) => update("model", e.target.value)}
                  />
                </label>
                <label>
                  <span>
                    API Key {profile?.api_key_configured ? "（已配置）" : ""}
                  </span>
                  <input
                    type="password"
                    autoComplete="new-password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={Boolean(draft.config.allow_private_network)}
                    onChange={(e) =>
                      update("allow_private_network", e.target.checked)
                    }
                  />
                  <span>允许私有网络 API</span>
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={clearKey}
                    onChange={(e) => setClearKey(e.target.checked)}
                  />
                  <span>清除已保存密钥</span>
                </label>
              </>
            ) : (
              <LocalProfileFields
                provider={draft.provider}
                config={draft.config}
                update={update}
              />
            )}
            <details className="full-width">
              <summary>高级配置</summary>
              <pre className="config-json">
                {JSON.stringify(draft.config, null, 2)}
              </pre>
              <label>
                <span>配置覆盖 (JSON)</span>
                <textarea
                  rows={5}
                  value={extra}
                  onChange={(e) => setExtra(e.target.value)}
                  placeholder="{}"
                />
              </label>
            </details>
          </fieldset>
          <footer className="toolbar">
            <button type="button" onClick={close} disabled={save.isPending}>
              取消
            </button>
            <button
              className="primary-command"
              type="submit"
              disabled={save.isPending}
            >
              保存接入
            </button>
          </footer>
        </form>
      )}
      {save.error && (
        <p role="alert" className="form-error">
          {save.error.message}
        </p>
      )}
    </CommandDialog>
  );
}

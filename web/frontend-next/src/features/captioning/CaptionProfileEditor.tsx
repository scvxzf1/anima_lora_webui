import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { useServerDraft } from "../../components/useServerDraft";
import { fetchGpus } from "../live-monitor/api";
import { LocalProfileFields } from "./LocalProfileFields";
import { captionProfileGpuKey } from "./gpu-query";
import {
  captioningKeys,
  saveCaptionProfile,
  type CaptionProfile,
  type ProviderType,
} from "./api";

const SECRET_CONFIG_KEYS = new Set([
  "api_key",
  "apikey",
  "client_secret",
  "password",
  "refresh_token",
  "secret",
  "token",
  "access_token",
]);

function stripSecretConfig(config: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(config).filter(
      ([key]) => !SECRET_CONFIG_KEYS.has(key.toLowerCase()),
    ),
  );
}

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
      config: stripSecretConfig(profile?.config || {}),
    }),
    [profile],
  );
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [extra, setExtra] = useState("");
  const extraConfig = useMemo(() => {
    if (!extra.trim()) return { value: {} as Record<string, unknown> };
    try {
      const value: unknown = JSON.parse(extra);
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        return { error: "高级配置必须是 JSON 对象" };
      }
      return { value: value as Record<string, unknown> };
    } catch {
      return { error: "高级配置 JSON 格式无效" };
    }
  }, [extra]);
  const editor = useServerDraft(initial, Boolean(apiKey || clearKey || extra));
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: async () => {
      if (extraConfig.error) throw new Error(extraConfig.error);
      const additional = extraConfig.value || {};
      const payload = {
        ...editor.draft!,
        config: { ...editor.draft!.config, ...additional },
        ...(apiKey ? { api_key: apiKey } : {}),
        ...(clearKey ? { clear_api_key: true } : {}),
      };
      if (payload.provider !== "openai_compatible") {
        const config = payload.config as Record<string, unknown>;
        const device = config.device ?? "auto";
        if (device !== "auto" && device !== "cpu" && device !== "cuda") {
          throw new Error("执行设备必须是 auto、cpu 或 cuda");
        }
        config.device = device;
        if (device === "cuda") {
          const index = config.gpu_index;
          if (
            (typeof index !== "number" &&
              !(typeof index === "string" && index.trim() !== "")) ||
            !Number.isInteger(Number(index)) ||
            Number(index) < 0
          ) {
            throw new Error("CUDA 必须选择当前可用的 GPU");
          }
          let inventory;
          try {
            inventory = await fetchGpus(undefined, true);
          } catch {
            throw new Error("无法读取 GPU 列表，请重试后再保存 CUDA 配置");
          }
          if (
            inventory.stale ||
            !inventory.gpus?.some((gpu) => gpu.index === Number(index))
          ) {
            throw new Error("CUDA 必须选择当前可用的 GPU");
          }
          config.gpu_index = Number(index);
        } else {
          delete config.gpu_index;
        }
      }
      return saveCaptionProfile(payload, profile?.id);
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
  const effectiveConfig = useMemo(
    () => ({ ...(draft?.config || {}), ...(extraConfig.value || {}) }),
    [draft?.config, extraConfig.value],
  );
  const effectiveDraft = draft
    ? { ...draft, config: effectiveConfig }
    : undefined;
  const gpuQuery = useQuery({
    queryKey: captionProfileGpuKey,
    queryFn: ({ signal }) => fetchGpus(signal, true),
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    enabled: Boolean(
      effectiveDraft &&
      effectiveDraft.provider !== "openai_compatible" &&
      effectiveConfig.device === "cuda",
    ),
  });
  const cudaNeedsGpu = Boolean(
    effectiveDraft &&
    effectiveDraft.provider !== "openai_compatible" &&
    effectiveConfig.device === "cuda",
  );
  const gpuIndex = Number(effectiveConfig.gpu_index);
  const gpuAvailable = Boolean(
    effectiveConfig.gpu_index !== "" &&
    effectiveConfig.gpu_index != null &&
    Number.isInteger(gpuIndex) &&
    !gpuQuery.error &&
    !gpuQuery.data?.stale &&
    gpuQuery.data?.gpus?.some((gpu) => gpu.index === gpuIndex),
  );
  function close() {
    if (
      !(editor.dirty || apiKey || extra || clearKey) ||
      window.confirm("放弃未保存的接入配置？")
    )
      onClose();
  }
  function update(key: string, value: unknown) {
    setExtra((current) => {
      if (!current.trim()) return current;
      try {
        const parsed: unknown = JSON.parse(current);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          return current;
        }
        const next = { ...(parsed as Record<string, unknown>) };
        delete next[key];
        return Object.keys(next).length ? JSON.stringify(next, null, 2) : "";
      } catch {
        return current;
      }
    });
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
                config={effectiveConfig}
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
              disabled={
                save.isPending ||
                (!extraConfig.error &&
                  cudaNeedsGpu &&
                  (gpuQuery.isPending || gpuQuery.isFetching || !gpuAvailable))
              }
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

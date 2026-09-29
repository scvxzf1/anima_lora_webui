import { useQuery } from "@tanstack/react-query";
import { fetchGpus } from "../live-monitor/api";
import { captionProfileGpuKey } from "./gpu-query";

const categories = [
  ["copyright", "作品标签"],
  ["artist", "画师标签"],
  ["meta", "元数据标签"],
  ["model", "模型标签"],
  ["rating", "评级标签"],
  ["quality", "质量标签"],
] as const;

export function LocalProfileFields({
  provider,
  config,
  update,
}: {
  provider: string;
  config: Record<string, unknown>;
  update: (key: string, value: unknown) => void;
}) {
  const gpuQuery = useQuery({
    queryKey: captionProfileGpuKey,
    queryFn: ({ signal }) => fetchGpus(signal, true),
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    enabled: provider !== "openai_compatible" && config.device === "cuda",
  });
  const gpus =
    gpuQuery.error || gpuQuery.data?.stale ? [] : gpuQuery.data?.gpus || [];
  const selectedGpu = config.gpu_index;
  const selectedGpuIndex = Number(selectedGpu);
  const hasSelectedGpu =
    selectedGpu !== "" &&
    selectedGpu != null &&
    Number.isInteger(selectedGpuIndex) &&
    gpus.some((gpu) => gpu.index === selectedGpuIndex);

  return (
    <>
      <label>
        <span>模型资产 ID</span>
        <input
          value={String(config.asset_id || "")}
          onChange={(e) => update("asset_id", e.target.value)}
        />
      </label>
      <label>
        <span>执行设备</span>
        <select
          value={String(config.device || "auto")}
          onChange={(e) => {
            const device = e.target.value;
            if (device === "cuda") {
              update("device", device);
            } else {
              update("device", device);
              update("gpu_index", undefined);
            }
          }}
        >
          <option value="auto">自动</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </select>
      </label>
      {config.device === "cuda" && (
        <label>
          <span>GPU</span>
          <select
            aria-label="GPU"
            value={String(config.gpu_index ?? "")}
            onChange={(e) =>
              update(
                "gpu_index",
                e.target.value === "" ? "" : Number(e.target.value),
              )
            }
          >
            <option value="">选择可用 GPU</option>
            {config.gpu_index != null &&
              config.gpu_index !== "" &&
              !hasSelectedGpu && (
                <option value={String(config.gpu_index)}>
                  GPU {String(config.gpu_index)}（当前不可用）
                </option>
              )}
            {gpus.map((gpu) => (
              <option value={String(gpu.index)} key={gpu.index}>
                {String(gpu.label || gpu.name || `GPU ${gpu.index}`)}
              </option>
            ))}
          </select>
        </label>
      )}
      {config.device === "cuda" && (
        <div className="full-width" aria-live="polite">
          {gpuQuery.isPending ? (
            <small>正在读取 GPU 列表…</small>
          ) : gpuQuery.error ? (
            <small role="status">
              无法读取 GPU 列表，CUDA 不可用。{" "}
              <button type="button" onClick={() => gpuQuery.refetch()}>
                重试
              </button>
            </small>
          ) : gpuQuery.data?.stale || !gpus.length ? (
            <small role="status">
              {gpuQuery.data?.stale
                ? "GPU 列表已过期，CUDA 不可用。"
                : "未检测到可用 GPU，CUDA 不可用。"}{" "}
              <button type="button" onClick={() => gpuQuery.refetch()}>
                重新检测 GPU
              </button>
            </small>
          ) : !hasSelectedGpu ? (
            <small role="status">请选择当前可用的 GPU。</small>
          ) : null}
        </div>
      )}
      {[
        ["batch_size", "批次大小"],
        ["general_threshold", "通用阈值"],
        ["character_threshold", "角色阈值"],
      ].map(([key, label]) => (
        <label key={key}>
          <span>{label}</span>
          <input
            type="number"
            min={key === "batch_size" ? 1 : 0}
            max={
              key.includes("threshold")
                ? 1
                : key === "batch_size"
                  ? 64
                  : undefined
            }
            step={key.includes("threshold") ? "0.01" : "1"}
            value={String(config[key] ?? "")}
            onChange={(e) =>
              update(key, e.target.value === "" ? "" : Number(e.target.value))
            }
          />
        </label>
      ))}
      <label className="full-width">
        <span>排除标签</span>
        <textarea
          rows={3}
          value={
            Array.isArray(config.blacklist) ? config.blacklist.join("\n") : ""
          }
          onChange={(e) => update("blacklist", e.target.value.split("\n"))}
        />
      </label>
      {provider === "cltagger" &&
        categories.map(([key, label]) => (
          <label className="checkbox-row" key={key}>
            <input
              type="checkbox"
              checked={Boolean(config[`add_${key}_tag`] ?? key === "copyright")}
              onChange={(e) => update(`add_${key}_tag`, e.target.checked)}
            />
            <span>{label}</span>
          </label>
        ))}
    </>
  );
}

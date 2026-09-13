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
          onChange={(e) => update("device", e.target.value)}
        >
          <option value="auto">自动</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </select>
      </label>
      {[
        ["gpu_index", "GPU 序号"],
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

import type { WeightHeatmapData } from "./api";
import "./WeightHeatmap.css";

type Props = {
  data?: WeightHeatmapData;
  label: string;
  maxValue: number;
  sharedScale?: boolean;
  mappedLayers: number;
  totalLayers: number;
};

function formatNorm(value: number) {
  if (value === 0) return "—";
  if (value < 0.01) return value.toExponential(1);
  return value.toLocaleString("zh-CN", { maximumFractionDigits: value < 1 ? 3 : 2 });
}

export function WeightHeatmap({ data, label, maxValue, sharedScale = false, mappedLayers, totalLayers }: Props) {
  const cells = new Map<string, WeightHeatmapData["cells"][number]>(
    (data?.cells || []).map((cell) => [`${cell.block}|${cell.component}`, cell]),
  );
  const hasCells = Boolean(data?.blocks.length && data.components.length && maxValue > 0);

  return <div className="weight-heatmap">
    <div className="weight-heatmap-heading">
      <h3>{label}</h3>
      <div className="weight-heatmap-legend" aria-label="颜色从低到高表示静态权重范数">
        <span>低</span>
        {[0, 0.25, 0.5, 0.75, 1].map((level) => <i key={level} style={{ backgroundColor: `color-mix(in srgb, var(--primary) ${Math.round(10 + level * 48)}%, var(--surface))` }} />)}
        <span>高</span>
      </div>
    </div>
    <p className="tool-muted weight-heatmap-note">
      Block × 组件的静态 ΔW Frobenius 范数；颜色按{sharedScale ? "A/B 共同" : "本权重"}最大值归一化，非 prompt 激活。
      {totalLayers > 0 && ` 纳入 ${mappedLayers}/${totalLayers} 层。`}
    </p>
    {hasCells ? <div className="weight-heatmap-scroll" role="region" aria-label={`${label}热力图`} tabIndex={0}>
      <table className="weight-heatmap-table">
        <thead><tr><th scope="col">Block</th>{data!.components.map((component) => <th scope="col" key={component} title={component}>{component}</th>)}</tr></thead>
        <tbody>{data!.blocks.map((block, rowIndex) => <tr key={block}>
          <th scope="row">{block}</th>
          {data!.components.map((component, columnIndex) => {
            const value = data!.matrix[rowIndex]?.[columnIndex] || 0;
            const cell = cells.get(`${block}|${component}`);
            const intensity = Math.max(0, Math.min(1, value / maxValue));
            const detail = `Block ${block} / ${component}\n范数: ${formatNorm(value)}\n层数: ${cell?.layer_count || 0}\n最高层: ${cell?.top_layer || "—"}`;
            return <td key={component} className={value > 0 ? "weight-heatmap-value" : "weight-heatmap-zero"}
              title={detail} aria-label={detail.replaceAll("\n", "，")}
              style={value > 0 ? { backgroundColor: `color-mix(in srgb, var(--primary) ${Math.round(10 + intensity * 48)}%, var(--surface))` } : undefined}>
              {formatNorm(value)}
            </td>;
          })}
        </tr>)}</tbody>
      </table>
    </div> : <p className="weight-heatmap-empty">没有识别到带编号的 Block，无法生成热图；组件视图仍可查看全部已分析层。</p>}
  </div>;
}

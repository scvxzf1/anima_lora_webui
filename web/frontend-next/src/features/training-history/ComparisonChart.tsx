import { useMemo, useRef, useState } from "react";
import { comparisonLines, type ComparisonEntry } from "./comparisonData";
import { comparisonColors, useComparisonCanvas } from "./useComparisonCanvas";

export function ComparisonChart({ entries }: { entries: ComparisonEntry[] }) {
  const [overlap, setOverlap] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { lines, start, end, hasOverlap } = useMemo(() => comparisonLines(entries, overlap), [entries, overlap]);
  useComparisonCanvas(ref, lines, start, end);
  const plotted = lines.filter(({ data }) => data.length).length;
  return <section className="chart-section history-comparison-chart" aria-label="任务 Loss 对比">
    <div className="chart-heading">
      <h3>Loss / 训练步数</h3>
      <label className="checkbox-row">步数窗口<select aria-label="对比步数窗口" value={overlap ? "overlap" : "all"} onChange={(event) => setOverlap(event.target.value === "overlap")}>
        <option value="all">全部已读取步数</option><option value="overlap">共同步数范围</option>
      </select></label>
    </div>
    <p className="data-scope">{hasOverlap ? `步数 ${start} - ${end} · ${plotted} 条有效曲线 · 原始 Loss` : "没有共同的有效步数范围"}</p>
    {plotted ? <div ref={ref} className="metric-chart" role="img" aria-label={`Loss 对比，${plotted} 条曲线，共同坐标轴，步数 ${start} 至 ${end}`} /> : <p className="empty-state">暂无可比较的训练步数与 Loss</p>}
    <ul className="comparison-legend">{lines.map((line, index) => <li key={line.id}>
      <span className="comparison-line-key" style={{ borderColor: `var(${comparisonColors[index]})`, borderTopStyle: index % 2 ? "dashed" : "solid" }} />
      <span>{index + 1}. {line.name} · 当前范围 {line.data.length} 个有效点 / 已读取 {line.loaded} 点{line.total !== undefined && line.total > line.loaded ? ` / 共 ${line.total} 点（前段未读取）` : ""}{!line.data.length ? " · 无可绘制数据" : ""}</span>
    </li>)}</ul>
    <p className="data-scope">不同模型、数据集与损失配置的原始 Loss 不代表质量排名。</p>
  </section>;
}

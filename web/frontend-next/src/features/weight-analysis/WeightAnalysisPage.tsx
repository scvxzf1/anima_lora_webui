import { useMutation, useQuery } from "@tanstack/react-query";
import { Download, RefreshCw, Search } from "lucide-react";
import { useState } from "react";
import { QueryFeedback } from "../../components/QueryFeedback";
import { fetchWeightCandidates, inspectWeight, inspectWeightFile, type AnalysisRow, type WeightResult } from "./api";
import { WeightSource, type WeightSourceValue } from "./WeightSource";
import { WeightHeatmap } from "./WeightHeatmap";
import "../tools.css";
import "./WeightAnalysisPage.css";

const empty: WeightSourceValue = { path: "", file: null };
const fmt = (value?: number) => value === undefined ? "—" : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 4 });

export function WeightAnalysisPage() {
  const [first, setFirst] = useState<WeightSourceValue>(empty);
  const [second, setSecond] = useState<WeightSourceValue>(empty);
  const [compare, setCompare] = useState(false);
  const [result, setResult] = useState<WeightResult | null>(null);
  const [other, setOther] = useState<WeightResult | null>(null);
  const [tab, setTab] = useState<"heatmap" | "components" | "blocks" | "style" | "character">("heatmap");
  const candidates = useQuery({ queryKey: ["weight-analysis", "weights"], queryFn: ({ signal }) => fetchWeightCandidates(signal), retry: false });
  const inspect = useMutation({ mutationFn: async () => {
    const a = first.file ? await inspectWeightFile(first.file) : await inspectWeight(first.path.trim());
    const b = compare ? second.file ? await inspectWeightFile(second.file) : await inspectWeight(second.path.trim()) : null;
    return { a, b };
  }, onSuccess: ({ a, b }) => { setResult(a); setOther(b); }, retry: false });
  function chooseFile(file: File | undefined, target: "a" | "b") {
    if (!file) return;
    const next = { file, path: file.name };
    if (target === "a") setFirst(next); else setSecond(next);
    setResult(null); setOther(null);
  }
  function download() {
    if (!result) return;
    const blob = new Blob([JSON.stringify({ primary: result, comparison: other }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = "weight-analysis.json"; link.click(); URL.revokeObjectURL(url);
  }
  const rows: AnalysisRow[] = tab === "components" ? result?.component_summary || [] : tab === "blocks" ? result?.block_summary || [] : tab === "style" ? result?.style_top20 || [] : tab === "character" ? result?.character_top20 || [] : [];
  const sharedHeatmapMax = Math.max(result?.heatmap?.max_value || 0, other?.heatmap?.max_value || 0);
  return <main className="tool-page weight-analysis-page">
    <header className="tool-heading"><div><h1>权重分析</h1><p>静态读取 safetensors 权重</p></div><button type="button" onClick={() => candidates.refetch()} disabled={candidates.isFetching}><RefreshCw size={16} />刷新列表</button></header>
    <QueryFeedback query={candidates} label="权重列表" hasData={Boolean(candidates.data)} />
    <form className="tool-section" onSubmit={(event) => { event.preventDefault(); if (!inspect.isPending) inspect.mutate(); }}>
      <div className={`weight-workspace${compare ? " is-comparing" : ""}`}>
        <div className="weight-sources"><WeightSource label="权重 A" source={first} candidates={candidates.data?.weights || []} onPath={(path) => { setFirst({ path, file: null }); setResult(null); setOther(null); }} onFile={(file) => chooseFile(file, "a")} />
          {compare && <WeightSource label="权重 B" source={second} candidates={candidates.data?.weights || []} onPath={(path) => { setSecond({ path, file: null }); setResult(null); setOther(null); }} onFile={(file) => chooseFile(file, "b")} />}
          <div className="tool-row weight-actions"><label><input type="checkbox" checked={compare} onChange={(event) => { setCompare(event.target.checked); setOther(null); }} /> A/B 对比</label><button type="submit" disabled={inspect.isPending || !first.path.trim() || (compare && !second.path.trim())}><Search size={16} />{inspect.isPending ? "分析中" : "开始分析"}</button></div>
        </div>
        {!compare && <aside className="weight-recent"><h2>最近权重</h2>{(candidates.data?.weights || []).slice(0, 6).map((item) => <button type="button" key={item.abs_path || item.file} onClick={() => { setFirst({ path: item.abs_path || item.file, file: null }); setResult(null); setOther(null); }}><span>{item.name}</span><small>{item.mtime_text || item.file}</small></button>)}{candidates.data && !candidates.data.weights.length && <p className="tool-muted">暂无可用权重</p>}</aside>}
      </div>
      {inspect.error && <p role="alert" className="tool-error">分析失败：{inspect.error.message}</p>}
    </form>
    {result && <><section className="tool-section"><div className="tool-section-head"><h2>{result.file.name}</h2><button type="button" onClick={download}><Download size={15} />导出 JSON</button></div>
      {result.unsupported?.unsupported && <p role="alert" className="tool-error">{result.unsupported.reason || "此权重不支持 ΔW 分析"}</p>}
      <p className="tool-muted">{result.disclaimer}</p><div className="tool-grid"><Metric label="Adapter" value={result.adapter_type} /><Metric label="层数" value={fmt(result.summary.layer_count)} /><Metric label="Block" value={fmt(result.summary.block_count)} /><Metric label="总能量" value={fmt(result.summary.total_energy)} /></div>
      {other && <><h3>对比：{other.file.name}</h3>{other.unsupported?.unsupported ? <p className="tool-error">{other.unsupported.reason}</p> : <div className="tool-grid"><Metric label="B - A 层数" value={fmt((other.summary.layer_count || 0) - (result.summary.layer_count || 0))} /><Metric label="B - A 总能量" value={fmt((other.summary.total_energy || 0) - (result.summary.total_energy || 0))} /></div>}</>}
    </section>
      {other && !result.unsupported?.unsupported && !other.unsupported?.unsupported && <section className="tool-section"><h2>组件差值 · B - A</h2><div className="tool-table-wrap"><table className="tool-table"><thead><tr><th>组件</th><th>A 范数</th><th>B 范数</th><th>差值</th></tr></thead><tbody>{[...new Set([...result.component_summary, ...other.component_summary].map((row) => row.label || row.component || "其他"))].map((name) => { const a = result.component_summary.find((row) => (row.label || row.component || "其他") === name)?.fro_norm || 0; const b = other.component_summary.find((row) => (row.label || row.component || "其他") === name)?.fro_norm || 0; return <tr key={name}><td>{name}</td><td>{fmt(a)}</td><td>{fmt(b)}</td><td>{fmt(b - a)}</td></tr>; })}</tbody></table></div></section>}
      <section className="tool-section"><div className="tool-row weight-tabs" role="tablist" aria-label="分析视图">{([ ["heatmap", "热力图"], ["components", "组件"], ["blocks", "Block"], ["style", "风格候选"], ["character", "角色候选"] ] as const).map(([key, label]) => <button type="button" role="tab" aria-selected={tab === key} key={key} onClick={() => setTab(key)}>{label}</button>)}</div>
        {tab === "heatmap" ? <div className="weight-heatmap-view">
          <WeightHeatmap data={result.heatmap} label={other ? "权重 A" : "区块 × 组件"} maxValue={sharedHeatmapMax} sharedScale={Boolean(other)} mappedLayers={result.layers.filter((layer) => typeof layer.block === "number").length} totalLayers={result.layers.length} />
          {other && !other.unsupported?.unsupported && <WeightHeatmap data={other.heatmap} label="权重 B" maxValue={sharedHeatmapMax} sharedScale mappedLayers={other.layers.filter((layer) => typeof layer.block === "number").length} totalLayers={other.layers.length} />}
        </div> : <><div className="tool-table-wrap"><table className="tool-table"><thead><tr><th>名称</th><th>层数</th><th>范数</th><th>贡献</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${row.name || row.label}-${index}`}><td>{row.name || row.label || row.component || (row.block == null ? "其他" : String(row.block))}</td><td>{fmt(row.layer_count)}</td><td>{fmt(row.fro_norm)}</td><td>{fmt(row.contribution)}</td></tr>)}</tbody></table></div>
        {!rows.length && <p className="tool-muted">暂无可展示数据</p>}</>}
      </section></>}
  </main>;
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="tool-metric"><strong>{value}</strong><span>{label}</span></div>; }

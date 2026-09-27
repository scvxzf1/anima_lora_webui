import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, RefreshCw, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fetchMergedTrainingConfig, fetchTrainingConfigGroups, fetchTrainingPresets } from "../../api/trainingContext";
import { QueryFeedback } from "../../components/QueryFeedback";
import { deleteImages, fetchImages, fetchImageStatus, fetchImageWeights, imageKeys, startImage, stopImage } from "./api";
import "../tools.css";

type Draft = { prompt: string; negative_prompt: string; width: string; height: string; infer_steps: string; guidance_scale: string; seed: string; sampler: string; attn_mode: string; runtime_dtype: string; text_encoder_dtype: string; gpu_index: string; weight_path: string; lora_multiplier: string };
const initial: Draft = { prompt: "", negative_prompt: "", width: "1024", height: "1024", infer_steps: "28", guidance_scale: "4", seed: "", sampler: "euler", attn_mode: "flash", runtime_dtype: "bf16", text_encoder_dtype: "same", gpu_index: "", weight_path: "", lora_multiplier: "1" };

export function ImageTestPage() {
  const client = useQueryClient();
  const [filePath, setFilePath] = useState("");
  const [preset, setPreset] = useState("default");
  const [draft, setDraft] = useState(initial);
  const [days, setDays] = useState("7");
  const [notice, setNotice] = useState("");
  const wasRunning = useRef(false);
  const groups = useQuery({ queryKey: ["image-test", "configs"], queryFn: ({ signal }) => fetchTrainingConfigGroups(signal), retry: false });
  const presets = useQuery({ queryKey: ["image-test", "presets"], queryFn: ({ signal }) => fetchTrainingPresets(signal), retry: false });
  const files = groups.data?.flatMap((group) => group.files).filter((file) => file.trainable !== false) || [];
  const selected = files.find((item) => item.path === filePath);
  const config = useQuery({ queryKey: ["image-test", "config", filePath, preset], queryFn: ({ signal }) => fetchMergedTrainingConfig(selected!, preset, signal), enabled: Boolean(selected), retry: false });
  const family = String(config.data?.model_family || "anima").toLowerCase();
  useEffect(() => {
    const value = config.data;
    if (!value) return;
    const resolution = Number(value.resolution) > 0 ? String(value.resolution) : "1024";
    const nextFamily = String(value.model_family || "anima").toLowerCase();
    const sampler = String(value.sample_sampler || "euler").toLowerCase();
    const attention = String(value.attn_mode || "flash").toLowerCase();
    const precision = String(value.precision_preference || "bf16").toLowerCase();
    setDraft((current) => ({ ...current,
      width: resolution, height: resolution,
      infer_steps: String(value.sample_steps ?? value.infer_steps ?? 28),
      guidance_scale: String(value.guidance_scale ?? value.cfg_scale ?? 4),
      sampler: nextFamily === "krea2_raw" ? "euler" : ["euler", "er_sde", "lcm"].includes(sampler) ? sampler : "euler",
      attn_mode: nextFamily === "krea2_raw" ? (["flash", "torch"].includes(attention) ? attention : "torch") : (["flash", "torch", "sdpa", "sageattn", "flex", "xformers"].includes(attention) ? attention : "flash"),
      runtime_dtype: ["bf16", "fp16", "fp32"].includes(precision) ? precision : "bf16",
    }));
  }, [config.data]);
  const status = useQuery({ queryKey: imageKeys.status, queryFn: ({ signal }) => fetchImageStatus(signal), refetchInterval: (query) => query.state.data?.running ? 3000 : false, retry: false });
  useEffect(() => {
    if (!status.data) return;
    if (wasRunning.current && !status.data.running) void client.invalidateQueries({ queryKey: ["image-test", "images"] });
    wasRunning.current = status.data.running;
  }, [client, status.data]);
  const weights = useQuery({ queryKey: imageKeys.weights, queryFn: ({ signal }) => fetchImageWeights(signal), retry: false });
  const images = useQuery({ queryKey: imageKeys.images(days), queryFn: ({ signal }) => fetchImages(days, signal), retry: false });
  const gpus = useQuery({ queryKey: ["image-test", "gpus"], queryFn: ({ signal }) => import("../live-monitor/api").then((module) => module.fetchGpus(signal)), retry: false });
  function update(key: keyof Draft, value: string) { setDraft((current) => ({ ...current, [key]: value })); }
  async function refresh() { await Promise.all([status.refetch(), images.refetch(), weights.refetch()]); }
  const start = useMutation({ mutationFn: startImage, retry: false, onSuccess: async () => { setNotice("生图测试已启动"); await client.invalidateQueries({ queryKey: imageKeys.status }); } });
  const stop = useMutation({ mutationFn: stopImage, retry: false, onSuccess: async () => { setNotice("停止请求已发送"); await client.invalidateQueries({ queryKey: imageKeys.status }); } });
  const remove = useMutation({ mutationFn: deleteImages, retry: false, onSuccess: async (result) => { setNotice(`已删除 ${result.deleted_count} 张图片`); await Promise.all([client.invalidateQueries({ queryKey: imageKeys.images(days) }), client.invalidateQueries({ queryKey: imageKeys.status })]); } });
  function submit(event: React.FormEvent) {
    event.preventDefault(); setNotice("");
    if (!config.data || !selected || status.error || status.data?.running || !draft.prompt.trim()) return;
    start.mutate({ ...draft, flow_shift: family === "krea2_raw" ? "" : String(config.data.flow_shift ?? 1), sampler: family === "krea2_raw" ? "euler" : draft.sampler, attn_mode: family === "krea2_raw" && !["flash", "torch"].includes(draft.attn_mode) ? "flash" : draft.attn_mode, config: config.data });
  }
  return <main className="tool-page">
    <header className="tool-heading"><div><h1>生图测试</h1><p>独立推理任务</p></div><button type="button" onClick={refresh} disabled={status.isFetching || images.isFetching}><RefreshCw size={16} />刷新</button></header>
    <QueryFeedback query={status} label="生图状态" hasData={Boolean(status.data)} />
    {notice && <p role="status" className="tool-notice">{notice}</p>}
    {[start.error, stop.error, remove.error].filter(Boolean).map((error, index) => <p role="alert" className="tool-error" key={index}>{error!.message}</p>)}
    <section className="tool-section"><div className="tool-section-head"><h2>任务状态</h2><strong>{status.data?.status || "读取中"}</strong></div>
      {status.data?.error && <p role="alert" className="tool-error">{status.data.error}</p>}
      <div className="tool-grid"><div className="tool-metric"><strong>{status.data?.output_count ?? 0}</strong><span>输出图片</span></div><div className="tool-metric"><strong>{status.data?.started_at_text || "—"}</strong><span>开始时间</span></div><div className="tool-metric"><strong>{status.data?.output_dir || "—"}</strong><span>输出目录</span></div></div>
      {status.data?.logs?.length ? <details><summary>运行日志</summary><pre className="tool-log">{status.data.logs.join("\n")}</pre></details> : null}
    </section>
    <section className="tool-section"><h2>生成参数</h2><form onSubmit={submit}>
      <div className="tool-grid">
        <label className="tool-field">训练配置<select value={filePath} onChange={(event) => setFilePath(event.target.value)} required><option value="">选择已保存配置</option>{files.map((file) => <option key={file.path} value={file.path}>{file.label || file.path}</option>)}</select></label>
        <label className="tool-field">预设<select value={preset} onChange={(event) => setPreset(event.target.value)}>{(presets.data?.length ? presets.data : ["default"]).map((value) => <option key={value}>{value}</option>)}</select></label>
        <label className="tool-field tool-field-wide">正向提示词<textarea rows={3} value={draft.prompt} onChange={(event) => update("prompt", event.target.value)} required /></label>
        <label className="tool-field tool-field-wide">负向提示词<textarea rows={2} value={draft.negative_prompt} onChange={(event) => update("negative_prompt", event.target.value)} /></label>
        {([ ["width", "宽度"], ["height", "高度"], ["infer_steps", "采样步数"], ["guidance_scale", "CFG"], ["seed", "种子"], ["lora_multiplier", "LoRA 强度"] ] as const).map(([key, label]) => <label className="tool-field" key={key}>{label}<input type="number" min={key === "seed" ? undefined : 0} step={["guidance_scale", "lora_multiplier"].includes(key) ? "0.1" : "1"} value={draft[key]} onChange={(event) => update(key, event.target.value)} /></label>)}
        <label className="tool-field">采样器<select value={family === "krea2_raw" ? "euler" : draft.sampler} disabled={family === "krea2_raw"} onChange={(event) => update("sampler", event.target.value)}>{["euler", "er_sde", "lcm"].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label className="tool-field">注意力后端<select value={draft.attn_mode} onChange={(event) => update("attn_mode", event.target.value)}>{(family === "krea2_raw" ? ["flash", "torch"] : ["flash", "torch", "sdpa", "sageattn", "flex", "xformers"]).map((value) => <option key={value}>{value}</option>)}</select></label>
        <label className="tool-field">推理精度<select value={draft.runtime_dtype} onChange={(event) => update("runtime_dtype", event.target.value)}>{["bf16", "fp16", "fp32"].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label className="tool-field">文本编码器精度<select value={draft.text_encoder_dtype} onChange={(event) => update("text_encoder_dtype", event.target.value)}>{["same", "bf16", "fp16", "fp32"].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label className="tool-field">GPU<select value={draft.gpu_index} onChange={(event) => update("gpu_index", event.target.value)}><option value="">自动</option>{gpus.data?.gpus?.map((gpu, index) => <option key={index} value={String(gpu.index ?? index)}>{String(gpu.name || `GPU ${index}`)}</option>)}</select></label>
        <label className="tool-field tool-field-wide">LoRA 权重路径<input list="image-test-weights" value={draft.weight_path} onChange={(event) => update("weight_path", event.target.value)} placeholder="可选；从列表选择或输入路径" /><datalist id="image-test-weights">{weights.data?.weights?.map((weight) => <option value={weight.abs_path || weight.file} key={weight.file}>{weight.name}</option>)}</datalist></label>
      </div><div className="tool-row" style={{ marginTop: 16 }}><button type="submit" disabled={!config.data || Boolean(config.error) || Boolean(status.error) || status.data?.running || start.isPending}><Play size={16} />{start.isPending ? "启动中" : "开始生成"}</button><button type="button" disabled={!status.data?.running || stop.isPending} onClick={() => stop.mutate()}><Square size={16} />停止</button></div>
      {config.error && <p role="alert" className="tool-error">配置读取失败：{config.error.message}</p>}{groups.error && <p role="alert" className="tool-error">配置列表读取失败：{groups.error.message}</p>}
    </form></section>
    <section className="tool-section"><div className="tool-section-head"><h2>输出图片</h2><label className="tool-field">时间范围<select value={days} onChange={(event) => setDays(event.target.value)}><option value="7">近 7 天</option><option value="14">近 14 天</option><option value="30">近 30 天</option><option value="all">全部</option></select></label></div>
      <QueryFeedback query={images} label="输出图片" hasData={Boolean(images.data)} />{images.data && !images.data.images.length && <p className="tool-muted">{images.data.message || "暂无图片"}</p>}
      <div className="tool-gallery">{images.data?.images?.map((image) => <figure key={image.file}><a href={image.url} target="_blank" rel="noreferrer"><img src={image.url} alt={image.name} loading="lazy" /></a><figcaption>{image.name}<small>{image.mtime_text}</small><button type="button" title="删除图片" aria-label={`删除 ${image.name}`} disabled={remove.isPending} onClick={() => { if (window.confirm(`永久删除 ${image.name}？`)) remove.mutate([image.file]); }}><Trash2 size={14} /></button></figcaption></figure>)}</div>
    </section>
  </main>;
}

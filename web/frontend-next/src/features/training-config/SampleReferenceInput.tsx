import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, FolderOpen, ImagePlus, LoaderCircle, X } from "lucide-react";
import { importSampleReference, managedReferenceUrl } from "./sampleReferenceApi";
import "./SampleReferenceInput.css";

export function SampleReferenceInput({ value, disabled, onChange, onBusyChange, maxReferences = 4 }: {
  value: string[]; disabled: boolean; onChange: (paths: string[]) => void; onBusyChange?: (busy: boolean) => void; maxReferences?: number;
}) {
  const [path, setPath] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const request = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const paths = useRef(value);
  const pendingPath = useRef("");
  const input = useRef<HTMLInputElement>(null);
  const busyCallback = useRef(onBusyChange);
  const changeCallback = useRef(onChange);
  busyCallback.current = onBusyChange;
  changeCallback.current = onChange;

  function stop() {
    generation.current++;
    request.current?.abort(); request.current = null;
    setBusy(false); busyCallback.current?.(false);
  }
  useEffect(() => {
    if (JSON.stringify(paths.current) !== JSON.stringify(value)) {
      stop(); paths.current = value; pendingPath.current = ""; setPath(""); setError("");
    }
  }, [value]);
  useEffect(() => { if (disabled) stop(); }, [disabled]);
  useEffect(() => () => { generation.current++; request.current?.abort(); busyCallback.current?.(false); }, []);

  function commit(next: string[]) {
    paths.current = next;
    changeCallback.current(next);
  }
  async function load(sources: Array<File | string>) {
    if (disabled || !sources.length) return;
    if (paths.current.length + sources.length > maxReferences) {
      setError(`参考图最多 ${maxReferences} 张。`); return;
    }
    stop();
    const version = generation.current;
    const controller = new AbortController(); request.current = controller;
    setBusy(true); busyCallback.current?.(true); setError("");
    try {
      for (const source of sources) {
        const result = await importSampleReference(source, controller.signal);
        if (controller.signal.aborted || generation.current !== version) return;
        commit([...paths.current, result.reference_image]);
        if (typeof source === "string") { pendingPath.current = ""; setPath(""); }
      }
    } catch (reason) {
      if (!controller.signal.aborted && generation.current === version) setError(reason instanceof Error ? reason.message : "参考图载入失败。");
    } finally {
      if (generation.current === version) {
        request.current = null; setBusy(false); busyCallback.current?.(Boolean(pendingPath.current.trim()));
      }
    }
  }
  function update(next: string[]) {
    stop(); pendingPath.current = ""; setPath(""); setError(""); commit(next);
  }
  const full = value.length >= maxReferences;
  return <div className={`sample-reference-input${dragging ? " is-dragging" : ""}`} onDragOver={(event) => {
    event.preventDefault(); if (!disabled) setDragging(true);
  }} onDragLeave={() => setDragging(false)} onDrop={(event) => {
    event.preventDefault(); setDragging(false);
    if (!disabled) void load(Array.from(event.dataTransfer.files));
  }}>
    <span>参考图 ({value.length}/{maxReferences})</span>
    <small>缩放后参考图总像素上限 4 Mi。</small>
    <ol className="sample-reference-list">{value.map((reference, index) => <li key={`${reference}-${index}`}>
      {managedReferenceUrl(reference) ? <img src={managedReferenceUrl(reference)} alt={`参考图 ${index + 1}`} onError={() => setError(`参考图 ${index + 1} 预览加载失败。`)} /> : <span className="sample-reference-placeholder">{index + 1}</span>}
      <span className="sample-reference-name" title={reference}>{reference}</span>
      <button type="button" title="上移" aria-label={`上移参考图 ${index + 1}`} disabled={disabled || busy || index === 0} onClick={() => {
        const next = [...value]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; update(next);
      }}><ArrowUp size={16} /></button>
      <button type="button" title="下移" aria-label={`下移参考图 ${index + 1}`} disabled={disabled || busy || index === value.length - 1} onClick={() => {
        const next = [...value]; [next[index + 1], next[index]] = [next[index], next[index + 1]]; update(next);
      }}><ArrowDown size={16} /></button>
      <button type="button" title="移除" aria-label={`移除参考图 ${index + 1}`} disabled={disabled} onClick={() => update(value.filter((_, i) => i !== index))}><X size={16} /></button>
    </li>)}</ol>
    <div className="sample-reference-actions">
      <input ref={input} type="file" aria-label="选择参考图文件" accept="image/*" multiple disabled={disabled || busy || full} hidden onChange={(event) => {
        const files = Array.from(event.target.files || []); event.target.value = ""; void load(files);
      }} />
      <button type="button" disabled={disabled || busy || full} onClick={() => input.current?.click()}><ImagePlus size={16} />添加参考图</button>
      {busy && <span role="status"><LoaderCircle size={16} />正在载入</span>}
    </div>
    <div className="sample-reference-path">
      <input aria-label="参考图服务器绝对路径" placeholder="服务器绝对路径" value={path} disabled={disabled || busy || full} onChange={(event) => {
        pendingPath.current = event.target.value; setPath(event.target.value); setError(""); busyCallback.current?.(Boolean(event.target.value.trim()));
      }} />
      <button type="button" title="载入服务器参考图" aria-label="载入服务器参考图" disabled={disabled || busy || full || !path.trim()} onClick={() => void load([path])}><FolderOpen size={16} /></button>
    </div>
    {error && <p role="alert">{error}</p>}
  </div>;
}

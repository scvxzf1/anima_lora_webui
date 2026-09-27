import { ChevronDown, FileUp, Search, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { WeightCandidate } from "./api";

export type WeightSourceValue = { path: string; file: File | null };

export function WeightSource({ label, source, candidates, onPath, onFile }: {
  label: string;
  source: WeightSourceValue;
  candidates: WeightCandidate[];
  onPath: (path: string) => void;
  onFile: (file: File) => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState("");
  const picker = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const selected = candidates.find((item) => (item.abs_path || item.file) === source.path);
  const visible = candidates.filter((item) => `${item.name} ${item.file}`.toLowerCase().includes(search.toLowerCase())).slice(0, 60);

  useEffect(() => {
    if (!open) return;
    searchInput.current?.focus();
    function dismiss(event: PointerEvent) {
      if (event.target instanceof Node && !picker.current?.contains(event.target)) setOpen(false);
    }
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, [open]);

  function acceptFile(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".safetensors")) { setFileError("只支持 .safetensors 文件"); return; }
    setFileError("");
    onFile(file);
    setOpen(false);
  }

  return <div className="weight-source">
    <div className="weight-source-title"><strong>{label}</strong><span>{source.file ? "本地文件" : selected ? "已保存权重" : "未选择"}</span></div>
    <div className="weight-picker" ref={picker}>
      <button ref={trigger} type="button" className="weight-picker-trigger" aria-label={`${label}选择已有权重`} aria-expanded={open} aria-haspopup="dialog" onClick={() => setOpen(!open)}>
        <span>{selected?.name || "选择已有权重"}</span><ChevronDown size={16} />
      </button>
      {open && <div className="weight-picker-panel" role="dialog" aria-label={`${label}权重列表`} onKeyDown={(event) => { if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); } }}>
        <div className="weight-picker-search"><Search size={16} /><input ref={searchInput} aria-label={`搜索${label}权重`} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索文件名或路径" /></div>
        <div className="weight-picker-options">{visible.map((item) => <button type="button" key={item.abs_path || item.file} onClick={() => { onPath(item.abs_path || item.file); setOpen(false); setSearch(""); }}><span>{item.name}</span><small>{item.file}</small></button>)}{!visible.length && <p>{candidates.length ? "没有匹配的权重" : "暂无可选权重"}</p>}</div>
      </div>}
    </div>
    <div className="weight-source-path"><input aria-label={`${label}路径`} value={source.path} onChange={(event) => onPath(event.target.value)} placeholder="或输入服务端权重路径" />{source.path && <button type="button" className="weight-source-clear" title={`清除${label}`} aria-label={`清除${label}`} onClick={() => onPath("")}><X size={15} /></button>}</div>
    <div className={`weight-upload${dragging ? " is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]); }}>
      <input ref={input} className="weight-file-input" type="file" accept=".safetensors" aria-label={`上传${label}文件`} tabIndex={-1} onChange={(event) => { acceptFile(event.target.files?.[0]); event.target.value = ""; }} />
      <button type="button" onClick={() => input.current?.click()}><FileUp size={16} />选择本地文件</button><span>.safetensors</span>
    </div>
    {fileError && <p role="alert" className="tool-error">{fileError}</p>}
  </div>;
}

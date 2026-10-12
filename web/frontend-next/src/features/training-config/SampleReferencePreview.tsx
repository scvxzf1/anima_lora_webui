import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { trapDialogFocus } from "../dataset-editor/trapDialogFocus";
import { sampleReferencePreviewUrl } from "./sampleReferenceApi";
import "./SampleReferencePreview.css";

function ReferenceDialog({ path, label, thumbnail, onClose }: { path: string; label: string; thumbnail: HTMLImageElement | null; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const active = useRef(true);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const id = useId();
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !thumbnail?.complete || !thumbnail.naturalWidth) return;
    canvas.width = thumbnail.naturalWidth;
    canvas.height = thumbnail.naturalHeight;
    canvas.getContext("2d")?.drawImage(thumbnail, 0, 0);
  }, [thumbnail]);
  useEffect(() => {
    active.current = true;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    // Parent CommandDialog listens on document in the bubble phase. Capture
    // prevents one Escape or Tab from also affecting that editor dialog.
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault(); event.stopImmediatePropagation(); onClose();
      } else if (event.key === "Tab") {
        trapDialogFocus(event, ref.current); event.stopImmediatePropagation();
      }
    };
    document.addEventListener("keydown", keydown, true);
    return () => {
      active.current = false;
      document.removeEventListener("keydown", keydown, true);
      document.body.style.overflow = overflow;
      if (previous?.isConnected) previous.focus();
    };
  }, [onClose]);
  return createPortal(<div className="sample-reference-preview-backdrop" onClick={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <div className="sample-reference-preview-dialog" ref={ref} role="dialog" aria-modal="true" aria-labelledby={id}>
      <header><h2 id={id}>{label}</h2><button ref={closeRef} type="button" aria-label="关闭参考图预览" onClick={onClose}><X size={20} /></button></header>
      <div className="sample-reference-preview-stage" aria-busy={!ready && !failed}>
        {!ready && <canvas ref={canvasRef} aria-hidden="true" />}
        {!failed && <img className={ready ? "is-ready" : "is-loading"} src={sampleReferencePreviewUrl(path, false, true)} alt={`${label}大图`} decoding="async" onLoad={async (event) => {
          const image = event.currentTarget;
          try { await image.decode?.(); } catch { /* Loaded image remains usable when explicit decode is unavailable. */ }
          if (active.current) setReady(true);
        }} onError={() => setFailed(true)} />}
        {!ready && !failed && <p role="status">正在加载大图…</p>}
        {failed && <p role="alert">参考图预览加载失败，请检查服务器图片路径。</p>}
      </div>
      <p className="sample-reference-preview-path">{path}</p>
    </div>
  </div>, document.body);
}

export function SampleReferencePreview({ path, label }: { path: string; label: string }) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const thumbnail = useRef<HTMLImageElement>(null);
  const close = useRef(() => setOpen(false)).current;
  return <div className="sample-reference-thumbnail">
    <button type="button" aria-label={`展开${label}`} title={`展开${label}`} onClick={() => setOpen(true)}>
      {failed ? <span>预览失败</span> : <img ref={thumbnail} src={sampleReferencePreviewUrl(path, true)} alt={label} loading="lazy" decoding="async" onError={() => setFailed(true)} />}
    </button>
    {failed && <small role="status">{label}预览加载失败</small>}
    {open && <ReferenceDialog key={path} path={path} label={label} thumbnail={thumbnail.current} onClose={close} />}
  </div>;
}

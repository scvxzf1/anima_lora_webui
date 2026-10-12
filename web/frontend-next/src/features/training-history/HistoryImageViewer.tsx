import { Maximize, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { ResilientImage } from "../../components/ResilientImage";
import { useHistoryImageViewport } from "./useHistoryImageViewport";

export function HistoryImageViewer({ src, name }: { src: string; name: string }) {
  const viewport = useHistoryImageViewport();
  return <>
    <div className="history-image-toolbar" role="toolbar" aria-label="样张缩放工具">
      <button type="button" aria-label="缩小样张" title="缩小" disabled={!viewport.ready} onClick={() => viewport.zoom(.8)}><ZoomOut size={18} /></button>
      <output aria-label="样张缩放比例" aria-live="polite">{Math.round(viewport.view.scale * 100)}%</output>
      <button type="button" aria-label="放大样张" title="放大" disabled={!viewport.ready} onClick={() => viewport.zoom(1.25)}><ZoomIn size={18} /></button>
      <button type="button" disabled={!viewport.ready} onClick={viewport.fit}><Maximize size={16} />适应</button>
      <button type="button" disabled={!viewport.ready} onClick={viewport.reset}><RotateCcw size={16} />重置 100%</button>
      <span>滚轮缩放 · 拖动平移</span>
    </div>
    <div ref={viewport.host} className="history-image-viewport" data-dragging={viewport.dragging} {...viewport.pointerHandlers}>
      <ResilientImage src={src} alt={name} retry draggable={false}
        className={viewport.ready ? "history-image-zoomed" : "history-image-loading"}
        onLoad={(event) => viewport.onLoad(event.currentTarget)}
        style={viewport.ready ? { transform: `translate(${viewport.view.x}px, ${viewport.view.y}px) scale(${viewport.view.scale})` } : undefined} />
    </div>
  </>;
}

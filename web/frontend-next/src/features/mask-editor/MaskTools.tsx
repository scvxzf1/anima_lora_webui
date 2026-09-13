import { Brush, Eraser, Hand, Undo2, Redo2, Square, SquareDashed, Contrast, Layers, Image, Scan,
  ZoomIn, ZoomOut, Maximize, type LucideIcon } from 'lucide-react';
import type { MaskCanvas, Tool, View } from './maskCanvas';
import type { ViewportHandle } from './MaskViewport';

export function IconCommand({ icon: Icon, label, active, ...props }: {
  icon: LucideIcon; label: string; active?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" className="icon-button" title={label} aria-label={label}
    aria-pressed={active} {...props}><Icon size={18} /></button>;
}

export function MaskTools({ engine, tool, setTool, size, setSize, view, setView, opacity, setOpacity,
  disabled, changed, viewport, zoom }: {
  engine: MaskCanvas | null; tool: Tool; setTool: (tool: Tool) => void;
  size: number; setSize: (value: number) => void; view: View; setView: (view: View) => void;
  opacity: number; setOpacity: (value: number) => void; disabled: boolean; changed: () => void;
  viewport: React.RefObject<ViewportHandle | null>; zoom: number;
}) {
  const command = (action: () => void) => { action(); changed(); };
  return <aside className="mask-tools" aria-label="蒙版工具">
    <div className="mask-tool-group" role="group" aria-label="绘制工具">
      <IconCommand icon={Brush} label="画笔（白色参与训练）" active={tool === 'brush'} onClick={() => setTool('brush')} disabled={disabled} />
      <IconCommand icon={Eraser} label="橡皮擦（黑色忽略）" active={tool === 'eraser'} onClick={() => setTool('eraser')} disabled={disabled} />
      <IconCommand icon={Hand} label="平移" active={tool === 'pan'} onClick={() => setTool('pan')} />
    </div>
    <label>笔刷大小 <output>{size} px</output><input aria-label="笔刷大小" type="range" min="1" max="512" value={size} onChange={event => setSize(+event.target.value)} /></label>
    <div className="mask-tool-group">
      <IconCommand icon={Undo2} label="撤销" disabled={disabled || !engine?.canUndo} onClick={() => command(() => engine?.undo())} />
      <IconCommand icon={Redo2} label="重做" disabled={disabled || !engine?.canRedo} onClick={() => command(() => engine?.redo())} />
    </div>
    <div className="mask-tool-group">
      <IconCommand icon={Square} label="全选" disabled={disabled} onClick={() => command(() => engine?.fill(255))} />
      <IconCommand icon={SquareDashed} label="清空" disabled={disabled} onClick={() => command(() => engine?.fill(0))} />
      <IconCommand icon={Contrast} label="反选" disabled={disabled} onClick={() => command(() => engine?.invert())} />
    </div>
    <div className="mask-tool-group" role="group" aria-label="预览模式">
      <IconCommand icon={Layers} label="叠加预览" active={view === 'overlay'} onClick={() => setView('overlay')} />
      <IconCommand icon={Scan} label="黑白蒙版" active={view === 'mask'} onClick={() => setView('mask')} />
      <IconCommand icon={Image} label="原图预览" active={view === 'image'} onClick={() => setView('image')} />
    </div>
    <label>叠加透明度 <output>{Math.round(opacity * 100)}%</output><input aria-label="叠加透明度" type="range" min="0" max="1" step="0.05" value={opacity} onChange={event => setOpacity(+event.target.value)} /></label>
    <div className="mask-tool-group">
      <IconCommand icon={ZoomOut} label="缩小" onClick={() => viewport.current?.zoom(.8)} />
      <IconCommand icon={ZoomIn} label="放大" onClick={() => viewport.current?.zoom(1.25)} />
      <IconCommand icon={Maximize} label="适应画布" onClick={() => viewport.current?.fit()} />
      <output className="mask-zoom">{Math.round(zoom * 100)}%</output>
    </div>
  </aside>;
}

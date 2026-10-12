import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import { imagePoint, MaskCanvas, type Point, type Tool, type View } from './maskCanvas';

export type ViewportHandle = { fit: () => void; zoom: (factor: number) => void };
type Props = { engine: MaskCanvas; tool: Tool; size: number; opacity: number; view: View;
  disabled: boolean; onChange: () => void; onZoom: (zoom: number) => void };

export const MaskViewport = forwardRef<ViewportHandle, Props>(function MaskViewport(props, ref) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const state = useRef({ scale: 1, origin: { x: 0, y: 0 }, width: 1, height: 1,
    pointer: null as Point | null, drag: null as { id: number; last: Point; pan: boolean } | null });
  const frame = useRef<number | null>(null);
  const draw = useCallback(() => {
    frame.current = null;
    const node = canvas.current;
    if (!node) return;
    const s = state.current;
    const current = latest.current;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(s.width * dpr), height = Math.round(s.height * dpr);
    if (node.width !== width || node.height !== height) { node.width = width; node.height = height; }
    const ctx = node.getContext('2d')!;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, node.width, node.height);
    ctx.scale(dpr, dpr); ctx.translate(s.origin.x, s.origin.y); ctx.scale(s.scale, s.scale);
    current.engine.render(ctx, current.view, current.opacity);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (s.pointer && current.tool !== 'pan' && !current.disabled) {
      ctx.beginPath(); ctx.arc(s.pointer.x, s.pointer.y, current.size * s.scale / 2, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.strokeStyle = '#202125'; ctx.lineWidth = 1; ctx.stroke();
    }
  }, []);
  const redraw = useCallback(() => {
    if (frame.current === null) frame.current = requestAnimationFrame(draw);
  }, [draw]);
  function fit() {
    const s = state.current;
    s.scale = Math.min((s.width - 40) / props.engine.mask.width, (s.height - 40) / props.engine.mask.height, 1);
    s.scale = Math.max(.01, s.scale);
    s.origin = { x: (s.width - props.engine.mask.width * s.scale) / 2,
      y: (s.height - props.engine.mask.height * s.scale) / 2 };
    latest.current.onZoom(s.scale); redraw();
  }
  function zoom(factor: number, point?: Point) {
    const s = state.current;
    const anchor = point || { x: s.width / 2, y: s.height / 2 };
    const before = imagePoint(anchor, s.origin, s.scale);
    s.scale = Math.max(.01, Math.min(16, s.scale * factor));
    s.origin = { x: anchor.x - before.x * s.scale, y: anchor.y - before.y * s.scale };
    latest.current.onZoom(s.scale); redraw();
  }
  useImperativeHandle(ref, () => ({ fit, zoom }));
  useEffect(() => {
    const node = host.current!;
    const observer = new ResizeObserver(() => {
      state.current.width = node.clientWidth;
      state.current.height = node.clientHeight;
      fit();
    });
    observer.observe(node);
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      if (state.current.drag) return;
      const rect = node.getBoundingClientRect();
      zoom(Math.exp(-event.deltaY * .0015), { x: (event.clientX - rect.left) * node.clientWidth / rect.width,
        y: (event.clientY - rect.top) * node.clientHeight / rect.height });
    };
    node.addEventListener('wheel', wheel, { passive: false });
    return () => { observer.disconnect(); node.removeEventListener('wheel', wheel); };
  // The viewport resets only when a different image engine is installed.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.engine]);
  const version = props.engine.version;
  useEffect(() => { redraw(); }, [props.engine, version, props.view, props.opacity, props.tool, props.size, props.disabled, redraw]);
  useEffect(() => () => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
  }, []);
  function point(event: React.PointerEvent): Point {
    const node = host.current!, rect = node.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * node.clientWidth / rect.width,
      y: (event.clientY - rect.top) * node.clientHeight / rect.height };
  }
  function finish(event: React.PointerEvent) {
    const drag = state.current.drag;
    if (!drag || drag.id !== event.pointerId) return;
    if (!drag.pan) { props.engine.commit(); props.onChange(); }
    state.current.drag = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    redraw();
  }
  return <div ref={host} className="mask-viewport" data-tool={props.tool}>
    <canvas ref={canvas} aria-label="蒙版编辑画布" tabIndex={0}
      onPointerDown={event => {
        if (props.disabled || state.current.drag || ![0, 1].includes(event.button)) return;
        const p = point(event), s = state.current;
        const pan = props.tool === 'pan' || event.button === 1 || event.altKey;
        if (!pan && props.view === 'image') return;
        event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
        s.drag = { id: event.pointerId, last: p, pan }; s.pointer = p;
        if (!pan) { const at = imagePoint(p, s.origin, s.scale); props.engine.stroke(at, at, props.size, props.tool === 'eraser'); }
        redraw();
      }}
      onPointerMove={event => {
        const s = state.current, p = point(event), drag = s.drag;
        s.pointer = p;
        if (drag && drag.id === event.pointerId) {
          if (drag.pan) { s.origin.x += p.x - drag.last.x; s.origin.y += p.y - drag.last.y; }
          else props.engine.stroke(imagePoint(drag.last, s.origin, s.scale), imagePoint(p, s.origin, s.scale), props.size, props.tool === 'eraser');
          drag.last = p;
        }
        redraw();
      }} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}
      onPointerLeave={() => { state.current.pointer = null; redraw(); }} />
  </div>;
});

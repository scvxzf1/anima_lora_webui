import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";

type Point = { x: number; y: number };
type View = Point & { scale: number };

export function useHistoryImageViewport() {
  const host = useRef<HTMLDivElement>(null);
  const dimensions = useRef({ width: 0, height: 0 });
  const current = useRef<View>({ x: 0, y: 0, scale: 1 });
  const drag = useRef<{ id: number; point: Point } | null>(null);
  const [view, setView] = useState(current.current);
  const [ready, setReady] = useState(false);
  const [dragging, setDragging] = useState(false);
  const update = useCallback((value: View) => {
    current.current = value;
    setView(value);
  }, []);
  const center = useCallback((scale?: number) => {
    const node = host.current, size = dimensions.current;
    if (!node || !size.width || !size.height) return;
    const nextScale = scale ?? Math.max(.01, Math.min(node.clientWidth / size.width, node.clientHeight / size.height, 1));
    update({ scale: nextScale, x: (node.clientWidth - size.width * nextScale) / 2,
      y: (node.clientHeight - size.height * nextScale) / 2 });
  }, [update]);
  const zoom = useCallback((factor: number, anchor?: Point) => {
    const node = host.current, before = current.current;
    if (!node || !dimensions.current.width || drag.current) return;
    const point = anchor ?? { x: node.clientWidth / 2, y: node.clientHeight / 2 };
    const scale = Math.max(.01, Math.min(16, before.scale * factor));
    const ratio = scale / before.scale;
    update({ scale, x: point.x - (point.x - before.x) * ratio, y: point.y - (point.y - before.y) * ratio });
  }, [update]);
  const point = (event: { clientX: number; clientY: number }): Point => {
    const node = host.current!, rect = node.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * node.clientWidth / (rect.width || 1),
      y: (event.clientY - rect.top) * node.clientHeight / (rect.height || 1) };
  };
  useEffect(() => {
    const node = host.current!;
    const resize = () => center();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
    observer?.observe(node);
    window.addEventListener("resize", resize);
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? node.clientHeight : 1);
      zoom(Math.exp(-Math.max(-500, Math.min(500, delta)) * .0015), point(event));
    };
    node.addEventListener("wheel", wheel, { passive: false });
    return () => { observer?.disconnect(); window.removeEventListener("resize", resize); node.removeEventListener("wheel", wheel); };
  }, [center, zoom]);
  const finish = (event: PointerEvent<HTMLDivElement>) => {
    if (drag.current?.id !== event.pointerId) return;
    drag.current = null;
    setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return { host, view, ready, dragging, zoom, fit: () => center(), reset: () => center(1),
    onLoad: (image: HTMLImageElement) => {
      dimensions.current = { width: image.naturalWidth, height: image.naturalHeight };
      setReady(true);
      center();
    },
    pointerHandlers: {
      onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
        if (!ready || drag.current || ![0, 1].includes(event.button) || (event.target instanceof HTMLElement && event.target.closest("button"))) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { id: event.pointerId, point: point(event) };
        setDragging(true);
      },
      onPointerMove: (event: PointerEvent<HTMLDivElement>) => {
        const before = drag.current;
        if (!before || before.id !== event.pointerId) return;
        const next = point(event);
        update({ ...current.current, x: current.current.x + next.x - before.point.x, y: current.current.y + next.y - before.point.y });
        before.point = next;
      },
      onPointerUp: finish, onPointerCancel: finish, onLostPointerCapture: finish,
    },
  };
}

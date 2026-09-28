import { useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode, type RefObject } from "react";

type Axis = "side" | "top";
type Sizes = Record<Axis, number | null>;

const STORAGE_KEYS: Record<Axis, string> = {
  side: "dragon-next.training-library-side-percent",
  top: "dragon-next.training-library-top-percent",
};
const MIN_PERCENT = 10;
const MAX_PERCENT: Record<Axis, number> = { side: 42, top: 80 };

function clampPercent(axis: Axis, value: number) {
  return Math.max(MIN_PERCENT, Math.min(MAX_PERCENT[axis], Math.round(value)));
}

function readSize(axis: Axis) {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS[axis]);
    if (stored === null) return null;
    const value = Number(stored);
    return Number.isFinite(value) ? clampPercent(axis, value) : null;
  } catch {
    return null;
  }
}

function useMeasuredSize(layoutRef: RefObject<HTMLDivElement | null>) {
  const [measured, setMeasured] = useState<Sizes>({ side: null, top: null });
  useLayoutEffect(() => {
    const layout = layoutRef.current;
    const library = layout?.querySelector<HTMLElement>(".training-config-library");
    if (!layout || !library) return;
    const update = () => {
      const bounds = layout.getBoundingClientRect();
      const libraryBounds = library.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const next = {
        side: clampPercent("side", (libraryBounds.width / bounds.width) * 100),
        top: clampPercent("top", (libraryBounds.height / bounds.height) * 100),
      };
      setMeasured((current) => current.side === next.side && current.top === next.top ? current : next);
    };
    update();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(layout);
    observer?.observe(library);
    window.addEventListener("resize", update);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);
  return measured;
}

function useLibrarySizes(layoutRef: RefObject<HTMLDivElement | null>) {
  const [sizes, setSizes] = useState<Sizes>(() => ({ side: readSize("side"), top: readSize("top") }));
  const sizesRef = useRef<Sizes>(sizes);
  const measured = useMeasuredSize(layoutRef);

  function setSize(axis: Axis, value: number) {
    const next = { ...sizesRef.current, [axis]: clampPercent(axis, value) };
    sizesRef.current = next;
    setSizes(next);
  }

  function resizeFromPointer(axis: Axis, event: PointerEvent<HTMLDivElement>) {
    const rect = layoutRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return;
    setSize(axis, axis === "side"
      ? ((event.clientX - rect.left) / rect.width) * 100
      : ((event.clientY - rect.top) / rect.height) * 100);
  }

  function saveSize(axis: Axis) {
    const value = sizesRef.current[axis];
    if (value === null) return;
    try { localStorage.setItem(STORAGE_KEYS[axis], String(value)); }
    catch { /* Storage may be unavailable; the current layout remains usable. */ }
  }

  function resetSize(axis: Axis) {
    const next = { ...sizesRef.current, [axis]: null };
    sizesRef.current = next;
    setSizes(next);
    try { localStorage.removeItem(STORAGE_KEYS[axis]); }
    catch { /* Storage may be unavailable. */ }
  }

  function resizeFromKeyboard(axis: Axis, event: KeyboardEvent<HTMLDivElement>) {
    const decrement = axis === "side" ? "ArrowLeft" : "ArrowUp";
    const increment = axis === "side" ? "ArrowRight" : "ArrowDown";
    if (![decrement, increment, "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const layout = layoutRef.current?.getBoundingClientRect();
    const library = layoutRef.current?.querySelector<HTMLElement>(".training-config-library")?.getBoundingClientRect();
    if (!layout || !library) return;
    const extent = axis === "side" ? layout.width : layout.height;
    const libraryExtent = axis === "side" ? library.width : library.height;
    if (sizesRef.current[axis] === null && !extent) return;
    const current = sizesRef.current[axis] ?? Math.round((libraryExtent / extent) * 100);
    setSize(axis, event.key === "Home" ? MIN_PERCENT
      : event.key === "End" ? MAX_PERCENT[axis]
        : current + (event.key === increment ? 1 : -1) * (event.shiftKey ? 5 : 1));
    saveSize(axis);
  }

  return { sizes, measured, resizeFromPointer, saveSize, resetSize, resizeFromKeyboard };
}

type ResizeControls = ReturnType<typeof useLibrarySizes>;

function LibraryResizer({ axis, controls }: { axis: Axis; controls: ResizeControls }) {
  const { sizes, measured, resizeFromPointer, saveSize, resetSize, resizeFromKeyboard } = controls;
  return (
    <div
      className={`training-library-resizer training-library-resizer--${axis}`}
      role="separator"
      aria-label={axis === "side" ? "调整配置库宽度" : "调整配置库高度"}
      aria-orientation={axis === "side" ? "vertical" : "horizontal"}
      aria-valuemin={MIN_PERCENT}
      aria-valuemax={MAX_PERCENT[axis]}
      aria-valuenow={sizes[axis] ?? measured[axis] ?? undefined}
      tabIndex={0}
      title={axis === "side" ? "拖动调整宽度，双击恢复默认" : "拖动调整高度，双击恢复默认"}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        resizeFromPointer(axis, event);
      }}
      onPointerMove={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) resizeFromPointer(axis, event);
      }}
      onPointerUp={() => saveSize(axis)}
      onPointerCancel={() => saveSize(axis)}
      onDoubleClick={() => resetSize(axis)}
      onKeyDown={(event) => resizeFromKeyboard(axis, event)}
    />
  );
}

export function TrainingWorkspaceLayout({ children }: { children: ReactNode }) {
  const layoutRef = useRef<HTMLDivElement>(null);
  const controls = useLibrarySizes(layoutRef);
  const { sizes } = controls;

  const style = {
    "--training-library-side-size": sizes.side === null ? undefined : `${sizes.side}%`,
    "--training-library-top-size": sizes.top === null ? undefined : `${sizes.top}%`,
  } as CSSProperties;

  return (
    <div ref={layoutRef} className="training-workspace-layout" style={style}>
      {children}
      {(["side", "top"] as const).map((axis) => (
        <LibraryResizer key={axis} axis={axis} controls={controls} />
      ))}
    </div>
  );
}

export { STORAGE_KEYS as TRAINING_LIBRARY_SIZE_KEYS };

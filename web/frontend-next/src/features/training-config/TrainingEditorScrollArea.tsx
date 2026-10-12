import { useLayoutEffect, useRef, useState, type PointerEvent, type ReactNode } from "react";
import "./TrainingEditorScrollArea.css";

type Metrics = { top: number; viewport: number; content: number; track: number };
const emptyMetrics: Metrics = { top: 0, viewport: 0, content: 0, track: 0 };

export function TrainingEditorScrollArea({ children }: { children: ReactNode }) {
  const contentRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const dragOffset = useRef<number | null>(null);
  const [metrics, setMetrics] = useState<Metrics>(emptyMetrics);

  useLayoutEffect(() => {
    const content = contentRef.current;
    const track = trackRef.current;
    if (!content || !track) return;
    const update = () => setMetrics({
      top: content.scrollTop,
      viewport: content.clientHeight,
      content: content.scrollHeight,
      track: track.clientHeight,
    });
    if (typeof ResizeObserver === "undefined") {
      update();
      content.addEventListener("scroll", update, { passive: true });
      return () => content.removeEventListener("scroll", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(content);
    observer.observe(track);
    if (content.firstElementChild) observer.observe(content.firstElementChild);
    content.addEventListener("scroll", update, { passive: true });
    update();
    return () => {
      observer.disconnect();
      content.removeEventListener("scroll", update);
    };
  }, []);

  const maxScroll = Math.max(0, metrics.content - metrics.viewport);
  const thumbHeight = metrics.track && metrics.content
    ? Math.min(metrics.track, Math.max(28, metrics.track * metrics.viewport / metrics.content))
    : 0;
  const travel = Math.max(0, metrics.track - thumbHeight);
  const thumbTop = maxScroll ? metrics.top / maxScroll * travel : 0;

  const scrollFromPointer = (clientY: number, offset: number) => {
    const content = contentRef.current;
    const track = trackRef.current;
    if (!content || !track || !travel || !maxScroll) return;
    const position = clientY - track.getBoundingClientRect().top - offset;
    content.scrollTop = Math.max(0, Math.min(maxScroll, position / travel * maxScroll));
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !maxScroll) return;
    const thumb = event.currentTarget.querySelector<HTMLElement>(".training-editor-scroll-thumb");
    dragOffset.current = thumb?.contains(event.target as Node)
      ? event.clientY - thumb.getBoundingClientRect().top
      : thumbHeight / 2;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
    scrollFromPointer(event.clientY, dragOffset.current);
    event.preventDefault();
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragOffset.current !== null) scrollFromPointer(event.clientY, dragOffset.current);
  };

  const releasePointer = (event: PointerEvent<HTMLDivElement>) => {
    dragOffset.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div className="training-editor-scroll-area">
      <div className="training-edit-groups" id="training-edit-groups" ref={contentRef}>
        {children}
      </div>
      <div
        ref={trackRef}
        className="training-editor-scroll-track"
        role="scrollbar"
        aria-label="训练参数滚动条"
        aria-controls="training-edit-groups"
        aria-orientation="vertical"
        aria-valuemin={0}
        aria-valuemax={Math.ceil(maxScroll)}
        aria-valuenow={Math.round(metrics.top)}
        data-scrollable={maxScroll > 1}
        tabIndex={maxScroll > 1 ? 0 : -1}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={releasePointer}
        onPointerCancel={releasePointer}
        onLostPointerCapture={() => { dragOffset.current = null; }}
        onWheel={(event) => {
          if (!maxScroll) return;
          const unit = event.deltaMode === 1 ? 16
            : event.deltaMode === 2 ? metrics.viewport : 1;
          contentRef.current!.scrollTop += event.deltaY * unit;
          event.preventDefault();
        }}
        onKeyDown={(event) => {
          const content = contentRef.current;
          if (!content) return;
          const step = 40;
          const amount = {
            ArrowDown: step,
            ArrowUp: -step,
            PageDown: content.clientHeight,
            PageUp: -content.clientHeight,
            Home: -Infinity,
            End: Infinity,
          }[event.key];
          if (amount === undefined) return;
          event.preventDefault();
          content.scrollTop = amount === Infinity ? maxScroll : amount === -Infinity ? 0 : content.scrollTop + amount;
        }}
      >
        <div
          className="training-editor-scroll-thumb"
          style={{ height: thumbHeight, transform: `translateY(${thumbTop}px)` }}
        />
      </div>
    </div>
  );
}

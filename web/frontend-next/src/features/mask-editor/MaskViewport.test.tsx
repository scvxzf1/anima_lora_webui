import { act, fireEvent, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { StrictMode } from 'react';
import { MaskViewport } from './MaskViewport';
import type { MaskCanvas } from './maskCanvas';

const frames = new Map<number, FrameRequestCallback>();
let nextFrame = 0;
function flushFrame() {
  act(() => { for (const [id, callback] of frames) { frames.delete(id); callback(0); } });
}
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
beforeEach(() => {
  frames.clear(); nextFrame = 0;
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback);
    return nextFrame;
  }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => frames.delete(id)));
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(200);
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(200);
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    left: 0, top: 0, width: 200, height: 200,
  } as DOMRect);
  const context = Object.fromEntries(['setTransform', 'clearRect', 'scale', 'translate', 'beginPath', 'arc', 'stroke']
    .map(name => [name, vi.fn()])) as unknown as CanvasRenderingContext2D;
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
});

it('draws once per frame while retaining every pointer segment', () => {
  const engine = { mask: { width: 100, height: 100 }, stroke: vi.fn(), commit: vi.fn(), render: vi.fn() } as unknown as MaskCanvas;
  const onChange = vi.fn();
  const { getByLabelText, unmount } = render(<MaskViewport engine={engine} tool="brush" size={10}
    opacity={0.5} view="overlay" disabled={false} onChange={onChange} onZoom={vi.fn()} />);
  const canvas = getByLabelText('蒙版编辑画布');
  canvas.setPointerCapture = vi.fn();
  canvas.hasPointerCapture = vi.fn(() => false);
  flushFrame();
  expect(engine.render).toHaveBeenCalledTimes(1);

  fireEvent.pointerDown(canvas, { pointerId: 1, button: 0, clientX: 10, clientY: 10 });
  fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 20, clientY: 20 });
  fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 30, clientY: 30 });
  fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 40, clientY: 40 });
  expect(engine.stroke).toHaveBeenCalledTimes(4);
  expect(frames.size).toBe(1);
  flushFrame();
  expect(engine.render).toHaveBeenCalledTimes(2);
  fireEvent.pointerUp(canvas, { pointerId: 1 });
  expect(engine.commit).toHaveBeenCalledTimes(1);
  expect(onChange).toHaveBeenCalledTimes(1);
  expect(frames.size).toBe(1);
  unmount();
  expect(frames.size).toBe(0);
});

function viewport(engine: MaskCanvas) {
  return <MaskViewport engine={engine} tool="brush" size={10} opacity={0.5}
    view="overlay" disabled={false} onChange={vi.fn()} onZoom={vi.fn()} />;
}

it('redraws in-place mask edits and undo/redo without pointer movement', () => {
  const engine = { mask: { width: 100, height: 100 }, version: 0, render: vi.fn() };
  const { rerender } = render(viewport(engine as unknown as MaskCanvas));
  flushFrame();
  for (const version of [1, 0, 1, 2, 3]) {
    engine.version = version;
    rerender(viewport(engine as unknown as MaskCanvas));
    expect(frames.size).toBe(1);
    flushFrame();
  }
  expect(engine.render).toHaveBeenCalledTimes(6);
});

it('reschedules the initial draw after StrictMode effect cleanup', () => {
  const engine = { mask: { width: 100, height: 100 }, version: 0, render: vi.fn() } as unknown as MaskCanvas;
  const { unmount } = render(<StrictMode>{viewport(engine)}</StrictMode>);
  expect(frames.size).toBe(1);
  flushFrame();
  expect(engine.render).toHaveBeenCalledTimes(1);
  unmount();
  expect(frames.size).toBe(0);
});

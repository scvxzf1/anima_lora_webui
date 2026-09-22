import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { imagePoint, invertValues, MaskCanvas } from './maskCanvas';

let pixels = new WeakMap<HTMLCanvasElement, Uint8Array>();

function fakeContext(canvas: HTMLCanvasElement) {
  const pixel = () => {
    let value = pixels.get(canvas);
    if (!value) {
      value = new Uint8Array(canvas.width * canvas.height);
      pixels.set(canvas, value);
    }
    return value;
  };
  const state = {
    fillStyle: '#000000',
    lineWidth: 1,
    lineCap: 'round',
    lineJoin: 'round',
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    drawImage: vi.fn(),
    fillRect: vi.fn((x: number, y: number, width: number, height: number) => {
      const match = state.fillStyle.match(/^#([0-9a-f]{2})/i);
      const value = match ? Number.parseInt(match[1], 16) : 0;
      for (let row = y; row < y + height; row++) {
        for (let column = x; column < x + width; column++) pixel()[row * canvas.width + column] = value;
      }
    }),
    getImageData: vi.fn((_x: number, _y: number, width: number, height: number) => {
      const data = new Uint8ClampedArray(width * height * 4);
      pixel().forEach((value, index) => {
        data[index * 4] = value;
        data[index * 4 + 3] = 255;
      });
      return { data };
    }),
    createImageData: vi.fn((width: number, height: number) => ({
      data: new Uint8ClampedArray(width * height * 4),
    })),
    putImageData: vi.fn((data: { data: Uint8ClampedArray }) => {
      data.data.forEach((_value, index) => {
        if (index % 4 === 0) pixel()[index / 4] = data.data[index];
      });
    }),
  };
  return state as unknown as CanvasRenderingContext2D;
}

function makeEngine() {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
    return fakeContext(this);
  });
  const image = { naturalWidth: 1, naturalHeight: 1 } as HTMLImageElement;
  const initial = {} as HTMLImageElement;
  const engine = new MaskCanvas(image, initial);
  return engine;
}

describe('mask coordinates and pixels', () => {
  beforeEach(() => {
    pixels = new WeakMap<HTMLCanvasElement, Uint8Array>();
  });
  afterEach(() => vi.restoreAllMocks());

  it('maps zoomed and panned pointer coordinates to image pixels', () => {
    expect(imagePoint({ x: 230, y: 170 }, { x: 30, y: -30 }, 2)).toEqual({ x: 100, y: 100 });
  });
  it('inverts soft grayscale values without mutating the source', () => {
    const source = new Uint8Array([0, 64, 128, 255]);
    expect([...invertValues(source)]).toEqual([255, 191, 127, 0]);
    expect([...source]).toEqual([0, 64, 128, 255]);
  });

  it('tracks dirty state by saved pixels instead of retained history ids', () => {
    const engine = makeEngine();
    engine.fill(255);
    engine.markSaved(engine.version);
    for (let index = 0; index < 45; index++) engine.fill(index % 2 ? 0 : 255);
    expect(engine.dirty).toBe(false);
  });

  it('does not mark a changed save request as clean', () => {
    const engine = makeEngine();
    engine.fill(255);
    const version = engine.version;
    const snapshot = engine.snapshot();
    const context = engine.mask.getContext('2d')!;
    context.fillStyle = '#808080';
    context.fillRect(0, 0, 1, 1);
    engine.markSaved(version, snapshot);
    expect(engine.dirty).toBe(true);
  });
});

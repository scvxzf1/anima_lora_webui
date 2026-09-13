import { describe, expect, it } from 'vitest';
import { imagePoint, invertValues } from './maskCanvas';

describe('mask coordinates and pixels', () => {
  it('maps zoomed and panned pointer coordinates to image pixels', () => {
    expect(imagePoint({ x: 230, y: 170 }, { x: 30, y: -30 }, 2)).toEqual({ x: 100, y: 100 });
  });
  it('inverts soft grayscale values without mutating the source', () => {
    const source = new Uint8Array([0, 64, 128, 255]);
    expect([...invertValues(source)]).toEqual([255, 191, 127, 0]);
    expect([...source]).toEqual([0, 64, 128, 255]);
  });
});

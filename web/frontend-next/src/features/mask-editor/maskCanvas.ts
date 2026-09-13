export type Tool = 'brush' | 'eraser' | 'pan';
export type View = 'overlay' | 'mask' | 'image';
export type Point = { x: number; y: number };

export function imagePoint(point: Point, origin: Point, scale: number): Point {
  return { x: (point.x - origin.x) / scale, y: (point.y - origin.y) / scale };
}
export function invertValues(values: Uint8Array): Uint8Array {
  return values.map(value => 255 - value);
}
function context(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('浏览器无法创建画布');
  return ctx;
}
export async function loadBitmap(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

export class MaskCanvas {
  readonly mask = document.createElement('canvas');
  readonly overlay = document.createElement('canvas');
  private history: Uint8Array[] = [];
  private cursor = 0;
  private serial = 0;
  private ids = [0];
  private savedId = 0;
  private dirtyRect: { x: number; y: number; right: number; bottom: number } | null = null;
  constructor(readonly image: HTMLImageElement, initialMask: HTMLImageElement) {
    this.mask.width = this.overlay.width = image.naturalWidth;
    this.mask.height = this.overlay.height = image.naturalHeight;
    context(this.mask).drawImage(initialMask, 0, 0, this.mask.width, this.mask.height);
    this.invalidate();
    this.history = [this.values()];
  }
  get dirty() { return this.ids[this.cursor] !== this.savedId; }
  get canUndo() { return this.cursor > 0; }
  get canRedo() { return this.cursor < this.history.length - 1; }
  get version() { return this.ids[this.cursor]; }
  markSaved(version: number) { this.savedId = version; }
  private values() {
    const rgba = context(this.mask).getImageData(0, 0, this.mask.width, this.mask.height).data;
    const values = new Uint8Array(rgba.length / 4);
    for (let i = 0; i < values.length; i++) values[i] = rgba[i * 4];
    return values;
  }
  private invalidate(x = 0, y = 0, right = this.mask.width, bottom = this.mask.height) {
    const old = this.dirtyRect;
    this.dirtyRect = { x: Math.max(0, Math.floor(Math.min(x, old?.x ?? x))),
      y: Math.max(0, Math.floor(Math.min(y, old?.y ?? y))),
      right: Math.min(this.mask.width, Math.ceil(Math.max(right, old?.right ?? right))),
      bottom: Math.min(this.mask.height, Math.ceil(Math.max(bottom, old?.bottom ?? bottom))) };
  }
  private restore(values: Uint8Array) {
    const ctx = context(this.mask);
    const data = ctx.createImageData(this.mask.width, this.mask.height);
    values.forEach((value, i) => {
      data.data[i * 4] = data.data[i * 4 + 1] = data.data[i * 4 + 2] = value;
      data.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(data, 0, 0);
    this.invalidate();
  }
  commit() {
    const values = this.values();
    if (values.every((value, i) => value === this.history[this.cursor][i])) return;
    this.history.splice(this.cursor + 1);
    this.ids.splice(this.cursor + 1);
    this.history.push(values);
    this.ids.push(++this.serial);
    const limit = Math.max(2, Math.min(40, Math.floor(64 * 1024 * 1024 / values.length)));
    while (this.history.length > limit) { this.history.shift(); this.ids.shift(); }
    this.cursor = this.history.length - 1;
  }
  undo() { if (this.canUndo) this.restore(this.history[--this.cursor]); }
  redo() { if (this.canRedo) this.restore(this.history[++this.cursor]); }
  fill(value: number) {
    const ctx = context(this.mask);
    ctx.fillStyle = value ? '#ffffff' : '#000000';
    ctx.fillRect(0, 0, this.mask.width, this.mask.height);
    this.invalidate();
    this.commit();
  }
  invert() { this.restore(invertValues(this.values())); this.commit(); }
  stroke(from: Point, to: Point, size: number, erase: boolean) {
    const ctx = context(this.mask);
    ctx.strokeStyle = ctx.fillStyle = erase ? '#000000' : '#ffffff';
    ctx.lineWidth = size;
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
    ctx.beginPath(); ctx.arc(to.x, to.y, size / 2, 0, Math.PI * 2); ctx.fill();
    const radius = size / 2 + 2;
    this.invalidate(Math.min(from.x, to.x) - radius, Math.min(from.y, to.y) - radius,
      Math.max(from.x, to.x) + radius, Math.max(from.y, to.y) + radius);
  }
  render(ctx: CanvasRenderingContext2D, view: View, opacity: number) {
    ctx.drawImage(view === 'mask' ? this.mask : this.image, 0, 0);
    if (view !== 'overlay') return;
    const rect = this.dirtyRect;
    if (rect && rect.right > rect.x && rect.bottom > rect.y) {
      const data = context(this.mask).getImageData(rect.x, rect.y, rect.right - rect.x, rect.bottom - rect.y);
      for (let i = 0; i < data.data.length; i += 4) {
        const value = data.data[i];
        data.data[i] = 24; data.data[i + 1] = 190; data.data[i + 2] = 148; data.data[i + 3] = value;
      }
      context(this.overlay).putImageData(data, rect.x, rect.y);
    }
    this.dirtyRect = null;
    ctx.globalAlpha = opacity; ctx.drawImage(this.overlay, 0, 0); ctx.globalAlpha = 1;
  }
  toBlob(): Promise<Blob> {
    return new Promise((resolve, reject) => this.mask.toBlob(blob => {
      if (blob) resolve(blob); else reject(new Error('蒙版导出失败'));
    }, 'image/png'));
  }
}

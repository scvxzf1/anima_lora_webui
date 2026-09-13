import { act, cleanup, screen, waitFor, fireEvent } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import { DatasetCover } from './DatasetCover';
import { renderInApp, jsonResponse } from '../../test/renderInApp';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test('cover waits for visibility, caches result and renders failures', async () => {
  let observe!: (entries: { isIntersecting: boolean }[]) => void;
  vi.stubGlobal('IntersectionObserver', class {
    constructor(callback: typeof observe) { observe = callback; }
    observe() {}
    disconnect() {}
  });
  const fetch = vi.fn().mockResolvedValue(jsonResponse({ ok: true, image: null, reason: '目录不存在' }));
  vi.stubGlobal('fetch', fetch);
  const view = renderInApp(<DatasetCover file="configs/datasets/test.toml" />);
  expect(fetch).not.toHaveBeenCalled();
  act(() => observe([{ isIntersecting: true }]));
  await screen.findByText('无图像');
  expect(screen.getByLabelText('无图像：目录不存在')).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledTimes(1);
  view.unmount();
  view.client.clear();
});

test('decode failure uses no-image placeholder', async () => {
  vi.stubGlobal('IntersectionObserver', class {
    callback: (entries: { isIntersecting: boolean }[]) => void;
    constructor(callback: (entries: { isIntersecting: boolean }[]) => void) { this.callback = callback; }
    observe() { this.callback([{ isIntersecting: true }]); }
    disconnect() {}
  });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ ok: true, image: 'data:image/png;base64,broken', reason: '' })));
  const view = renderInApp(<DatasetCover file="configs/datasets/test.toml" />);
  await waitFor(() => expect(view.container.querySelector('img')).not.toBeNull());
  fireEvent.error(view.container.querySelector('img')!);
  await screen.findByText('无图像');
  expect(screen.getByLabelText('无图像：缩略图读取失败')).toBeInTheDocument();
  view.unmount();
  view.client.clear();
});

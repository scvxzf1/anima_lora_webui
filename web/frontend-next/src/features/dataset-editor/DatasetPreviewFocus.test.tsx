import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { DatasetPreviewDialog } from './DatasetPreviewDialog';
import type { DatasetPreviewResponse } from './types';

function page(offset: number, version: number): DatasetPreviewResponse {
  return {
    ok: true, file: 'alpha.toml', dataset_index: 1, dataset_label: 'Dataset',
    source: 'source', source_label: 'Source', directory: '/images', directory_exists: true,
    caption_extension: '.txt', prefer_json_caption: false, caption_source_mode: 'auto',
    caption_source_label: 'Auto', caption_summary: '', count: 1, total: 48, limit: 24,
    offset, returned: 1, next_offset: offset + 24, has_more_before: offset > 0,
    has_more_after: offset === 0, row: {}, settings: {}, message: '',
    images: [{ file: `/images/${offset}.png`, name: `${offset}.png`, url: `/image/${offset}?v=${version}`,
      caption: { ok: false, file: '', extension: '.txt', source_mode: 'auto', source_label: 'Auto',
        detected_mode: '', format_label: '', caption_count: 0, text: '', truncated: false, length: 0 } }],
  };
}

function mountPreview() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><DatasetPreviewDialog
    file="alpha.toml" datasetIndex={1} returnFocus={null} onClose={vi.fn()} embedded
  /></QueryClientProvider>);
  return () => { view.unmount(); client.clear(); };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.removeItem('dataset-preview-continuous');
});

it('refreshes the current page on return, coalesces focus signals and skips hidden events', async () => {
  let version = 0;
  let visible = 'visible';
  let complete: (() => void) | undefined;
  const offsets: number[] = [];
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visible as DocumentVisibilityState);
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), 'http://local').searchParams.get('offset') || 0);
    offsets.push(offset);
    if (version === 1) await new Promise<void>((resolve) => { complete = resolve; });
    return new Response(JSON.stringify(page(offset, version)));
  }));
  const dispose = mountPreview();
  await screen.findByRole('img', { name: '0.png' });
  fireEvent.click(screen.getByRole('button', { name: '下一页' }));
  await screen.findByRole('img', { name: '24.png' });
  const body = document.querySelector('.dataset-preview-body') as HTMLElement;
  body.scrollTop = 300;
  visible = 'hidden';
  fireEvent(document, new Event('visibilitychange'));
  fireEvent(window, new Event('focus'));
  expect(offsets).toEqual([0, 24]);
  version = 1;
  visible = 'visible';
  fireEvent(document, new Event('visibilitychange'));
  fireEvent(window, new Event('focus'));
  await waitFor(() => expect(complete).toBeDefined());
  fireEvent(window, new Event('focus'));
  expect(offsets).toEqual([0, 24, 24]);
  await act(async () => complete!());
  await waitFor(() => expect(screen.getByRole('img', { name: '24.png' })).toHaveAttribute('src', '/image/24?v=1'));
  expect(body.scrollTop).toBe(300);
  expect(screen.getByText('第 2 / 2 页 · 共 48 张')).toBeInTheDocument();
  fireEvent(window, new Event('focus'));
  await act(async () => {});
  expect(offsets).toEqual([0, 24, 24]);
  dispose();
});

it('replaces every loaded continuous page on window focus without losing the window or scroll', async () => {
  let version = 0;
  const offsets: number[] = [];
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), 'http://local').searchParams.get('offset') || 0);
    offsets.push(offset);
    return new Response(JSON.stringify(page(offset, version)));
  }));
  const dispose = mountPreview();
  await screen.findByRole('img', { name: '0.png' });
  fireEvent.click(screen.getByRole('button', { name: '连续滚动' }));
  const body = document.querySelector('.dataset-preview-body') as HTMLElement;
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  fireEvent.wheel(body, { deltaY: 100 });
  await screen.findByRole('img', { name: '24.png' });
  body.scrollTop = 350;
  version = 1;
  fireEvent(window, new Event('focus'));
  await waitFor(() => {
    expect(screen.getByRole('img', { name: '0.png' })).toHaveAttribute('src', '/image/0?v=1');
    expect(screen.getByRole('img', { name: '24.png' })).toHaveAttribute('src', '/image/24?v=1');
  });
  expect(offsets).toEqual([0, 24, 0, 24]);
  expect(body.scrollTop).toBe(350);
  expect(body.querySelectorAll('[data-preview-offset]')).toHaveLength(2);
  dispose();
});

it('keeps the selected large image open and replaces its versioned URL after return', async () => {
  let version = 0;
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(page(0, version)))));
  const dispose = mountPreview();
  await screen.findByRole('img', { name: '0.png' });
  fireEvent.click(screen.getByRole('button', { name: '查看大图 0.png' }));
  const viewer = screen.getByRole('dialog');
  version = 1;
  fireEvent(window, new Event('focus'));
  await waitFor(() => expect(viewer.querySelector('img')).toHaveAttribute('src', '/image/0?v=1'));
  expect(viewer).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '关闭大图' })).toBeInTheDocument();
  dispose();
});

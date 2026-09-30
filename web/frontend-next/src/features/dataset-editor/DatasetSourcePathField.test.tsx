import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, expect, it, vi } from 'vitest';

import { DatasetSourcePathField } from './DatasetSourcePathField';
import { emptyDatasetForm, type DatasetFormValues } from './datasetForm';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function response(payload: unknown): Response {
  return new Response(JSON.stringify(payload), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function Harness({ initial = '', path = 'datasets.0.source_dir' }: { initial?: string; path?: 'datasets.0.source_dir' | 'datasets.1.source_dir' }) {
  const values = emptyDatasetForm();
  values.datasets[0].source_dir = initial;
  values.datasets.push({ ...values.datasets[0], source_dir: initial });
  const form = useForm<DatasetFormValues>({ defaultValues: values });
  return <DatasetSourcePathField form={form} path={path} label="原始图片目录" />;
}

it('checks initial and changed draft paths against inspect without saving the configuration', async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({ source_exists: true, source_is_dir: true, source_image_count: 12 }));
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();
  render(<Harness initial="old" />);
  await screen.findByText('检测到 12 张图片');
  await user.clear(screen.getByRole('textbox', { name: '原始图片目录' }));
  await user.type(screen.getByRole('textbox', { name: '原始图片目录' }), 'new-draft');
  await screen.findByText('检测到 12 张图片', {}, { timeout: 2500 });
  const urls = fetchMock.mock.calls.map(([input]) => String(input));
  expect(urls.some((url) => url.includes('source_image_dir=old&inspect=1'))).toBe(true);
  expect(urls.some((url) => url.includes('source_image_dir=new-draft&inspect=1'))).toBe(true);
  expect(fetchMock.mock.calls.every(([input, init]) => String(input).includes('inspect=1') && !init?.method)).toBe(true);
});

it.each([
  [{ source_exists: false }, '路径不存在'],
  [{ source_exists: true, source_is_dir: false }, '路径不是目录'],
  [{ source_exists: true, source_is_dir: true, source_image_count: 0 }, '目录存在，未检测到图片'],
  [{ source_exists: true, source_is_dir: true, source_inspection_error: 'permission denied' }, '目录无法完整读取'],
  [{}, '重启服务后检测'],
])('renders inspection result %#', async (payload, expected) => {
  vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response(payload)));
  render(<Harness initial="folder" />);
  expect(await screen.findByText(expected)).toBeInTheDocument();
});

it('does not let a late response for an older draft replace the current status', async () => {
  let finishOld: ((value: Response) => void) | undefined;
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (String(input).includes('older')) return new Promise<Response>((resolve) => { finishOld = resolve; });
    return Promise.resolve(response({ source_exists: true, source_is_dir: true, source_image_count: 3 }));
  });
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();
  render(<Harness initial="older" />);
  await waitFor(() => expect(finishOld).toBeDefined());
  await user.clear(screen.getByRole('textbox', { name: '原始图片目录' }));
  await user.type(screen.getByRole('textbox', { name: '原始图片目录' }), 'newer');
  expect(await screen.findByText('检测到 3 张图片', {}, { timeout: 2500 })).toBeInTheDocument();
  finishOld?.(response({ source_exists: false }));
  await waitFor(() => expect(screen.getByText('检测到 3 张图片')).toBeInTheDocument());
  expect(screen.queryByText('路径不存在')).not.toBeInTheDocument();
});

it('shows server inspection errors and tolerates missing inspection fields', async () => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => (
    String(input).includes('legacy')
      ? response({})
      : response({ source_exists: true, source_is_dir: true, source_inspection_error: 'io error' })
  )));
  const { rerender } = render(<Harness initial="folder" />);
  expect(await screen.findByText('目录无法完整读取')).toBeInTheDocument();
  rerender(<Harness key="legacy" initial="legacy" />);
  expect(await screen.findByText('重启服务后检测')).toBeInTheDocument();
});

it('copies the draft path and ignores directory picker cancellation', async () => {
  const writeText = vi.fn(async (_text: string) => undefined);
  vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({})));
  const picker = vi.fn().mockRejectedValue(new DOMException('cancelled', 'AbortError'));
  Object.defineProperty(window, 'showDirectoryPicker', { configurable: true, value: picker });
  const user = userEvent.setup();
  Object.defineProperty(navigator, 'clipboard', { configurable: true, get: () => ({ writeText }) });
  render(<Harness initial="draft/path" />);
  await user.click(screen.getByRole('button', { name: '复制目录路径' }));
  await waitFor(() => expect(writeText).toHaveBeenCalledWith('draft/path'));
  await user.click(screen.getByRole('button', { name: '选择本机文件夹名' }));
  expect(picker).toHaveBeenCalledWith({ mode: 'read' });
  expect(screen.getByRole('textbox', { name: '原始图片目录' })).toHaveValue('draft/path');
});

it('rechecks when the field path changes but its value stays the same', async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({ source_exists: true, source_is_dir: true, source_image_count: 2 }));
  vi.stubGlobal('fetch', fetchMock);
  const { rerender } = render(<Harness initial="same" />);
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  rerender(<Harness initial="same" path="datasets.1.source_dir" />);
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
});

it('refresh cancels the pending debounce and sends exactly one inspection request', async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({ source_exists: true, source_is_dir: true, source_image_count: 4 }));
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();
  render(<Harness initial="folder" />);
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  const input = screen.getByRole('textbox', { name: '原始图片目录' });
  await user.clear(input);
  await user.type(input, 'changed');
  await user.click(screen.getByRole('button', { name: '重新检查目录' }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  await new Promise((resolve) => window.setTimeout(resolve, 500));
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it('reports clipboard failures without throwing', async () => {
  vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({})));
  const user = userEvent.setup();
  const writeText = vi.fn().mockRejectedValue(new Error('denied'));
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
  render(<Harness initial="draft/path" />);
  await screen.findByText('重启服务后检测');
  await user.click(screen.getByRole('button', { name: '复制目录路径' }));
  expect(writeText).toHaveBeenCalledWith('draft/path');
  expect(await screen.findByText('无法访问剪贴板，请手动复制路径')).toBeInTheDocument();
});

it('cleans up the fallback folder input when the picker is cancelled', async () => {
  Object.defineProperty(window, 'showDirectoryPicker', { configurable: true, value: undefined });
  vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response({})));
  const user = userEvent.setup();
  render(<Harness initial="draft/path" />);
  await user.click(screen.getByRole('button', { name: '选择本机文件夹名' }));
  const picker = document.querySelector('input[type="file"][webkitdirectory]');
  expect(picker).not.toBeNull();
  fireEvent(picker!, new Event('cancel'));
  await waitFor(() => expect(document.querySelector('input[type="file"][webkitdirectory]')).toBeNull());
});

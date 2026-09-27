import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider, useLocation } from 'react-router-dom';
import { afterEach, expect, it, vi } from 'vitest';

import { DatasetImageWorkspacePage } from './DatasetImageWorkspacePage';

vi.mock('./DatasetPreviewDialog', () => ({
  DatasetPreviewDialog: ({ file, datasetIndex, embedded }: { file: string; datasetIndex: number; embedded: boolean }) => (
    <div data-testid="preview-panel">{file}:{datasetIndex}:{String(embedded)}</div>
  ),
}));

vi.mock('../mask-editor/MaskEditorPage', () => ({
  MaskEditorPage: ({ embedded }: { embedded: boolean }) => <div data-testid="mask-panel">mask:{String(embedded)}</div>,
}));

vi.mock('../captioning/CaptioningPage', () => ({
  CaptioningPage: ({ embeddedBasePath }: { embeddedBasePath: string }) => {
    const location = useLocation();
    return <div data-testid="tagging-panel" data-base={embeddedBasePath}>{location.pathname}</div>;
  },
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it('switches the three feature views while keeping the dataset and subset in the URL', async () => {
  const file = 'configs/datasets/alpha.toml';
  const fetchMock = vi.fn(async () => new Response(JSON.stringify({
    ok: true,
    file,
    name: 'alpha',
    readonly: false,
    datasets: [{ source_dir: 'image_dataset/alpha' }, { source_dir: 'image_dataset/alpha-2' }],
  }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  const query = new URLSearchParams({ dataset: file, subset: '1' }).toString();
  const returnTo = `/datasets?${new URLSearchParams({ dataset: file })}`;
  const router = createMemoryRouter([
    { path: '/datasets', element: <main>dataset page</main> },
    { path: '/datasets/workspace/*', element: <DatasetImageWorkspacePage /> },
  ], {
    initialEntries: [returnTo, { pathname: '/datasets/workspace/preview', search: `?${query}`, state: { returnTo } }],
    initialIndex: 1,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const user = userEvent.setup();
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  expect(await screen.findByRole('heading', { name: '图片工作台' })).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText('alpha · 子集 2')).toBeInTheDocument());
  expect(screen.getByRole('tab', { name: '预览' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByTestId('preview-panel')).toHaveTextContent(`${file}:1:true`);

  const previewTab = screen.getByRole('tab', { name: '预览' });
  previewTab.focus();
  await user.keyboard('{ArrowRight}');
  expect(await screen.findByTestId('mask-panel')).toHaveTextContent('mask:true');
  await waitFor(() => expect(router.state.location.search).toContain('subset=1'));
  expect(router.state.location.pathname).toBe('/datasets/workspace/masks');
  expect(screen.getByRole('tab', { name: '编辑蒙版' })).toHaveAttribute('aria-selected', 'true');

  await user.click(screen.getByRole('tab', { name: '打标' }));
  expect(await screen.findByTestId('tagging-panel')).toHaveAttribute('data-base', '/datasets/workspace/tagging');
  expect(router.state.location.pathname).toBe('/datasets/workspace/tagging');
  expect(router.state.location.search).toContain('dataset=configs%2Fdatasets%2Falpha.toml');
  expect(router.state.location.search).toContain('subset=1');

  await user.click(screen.getByRole('button', { name: '返回数据集' }));
  await waitFor(() => expect(router.state.location.pathname).toBe('/datasets'));
});

it('ignores an invalid return location instead of navigating to unrelated history', async () => {
  const file = 'configs/datasets/alpha.toml';
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
    ok: true, file, name: 'alpha', datasets: [{ source_dir: 'image_dataset/alpha' }],
  }), { status: 200 })));
  const router = createMemoryRouter([
    { path: '/captioning', element: <main>unrelated page</main> },
    { path: '/datasets', element: <main>dataset page</main> },
    { path: '/datasets/workspace/*', element: <DatasetImageWorkspacePage /> },
  ], {
    initialEntries: ['/captioning', {
      pathname: '/datasets/workspace/preview',
      search: `?${new URLSearchParams({ dataset: file })}`,
      state: { returnTo: '/datasets-other' },
    }],
    initialIndex: 1,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const user = userEvent.setup();
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await user.click(await screen.findByRole('button', { name: '返回数据集' }));
  await waitFor(() => expect(router.state.location.pathname).toBe('/datasets'));
  expect(router.state.location.search).toContain('dataset=configs%2Fdatasets%2Falpha.toml');
});

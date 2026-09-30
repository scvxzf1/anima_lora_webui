import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DatasetGroupExport } from './DatasetGroupExport';

function renderExport({ empty = false } = {}) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <DatasetGroupExport groupId="system/角色 +" label="角色" empty={empty} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('DatasetGroupExport', () => {
  it('requests an encoded dataset group export and downloads the archive', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['zip']) });
    vi.stubGlobal('fetch', fetchMock);
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:dataset-group');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    renderExport();

    fireEvent.click(screen.getByRole('button', { name: '导出分组 角色' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/file-groups/system%2F%E8%A7%92%E8%89%B2%20%2B/export?kind=dataset',
    ));
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob)));
    expect((click.mock.instances[0] as HTMLAnchorElement).download).toBe('dataset-group.zip');
  });

  it('shows a group-local error and prevents duplicate pending requests', async () => {
    let release: (value: unknown) => void = () => undefined;
    const fetchMock = vi.fn(() => new Promise((resolve) => { release = resolve; }));
    vi.stubGlobal('fetch', fetchMock);
    renderExport();
    const button = screen.getByRole('button', { name: '导出分组 角色' });

    fireEvent.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    fireEvent.click(button);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => release({ ok: false, status: 409, json: async () => ({ error: '压缩失败' }) }));
    expect(await screen.findByRole('alert')).toHaveTextContent('压缩失败');
    expect(button).toBeEnabled();
  });

  it('disables export for empty groups', () => {
    renderExport({ empty: true });
    expect(screen.getByRole('button', { name: '导出分组 角色' })).toBeDisabled();
  });
});

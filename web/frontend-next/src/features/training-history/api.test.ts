import { afterEach, describe, expect, it, vi } from 'vitest';

import { batchUpdateHistoryTasks, fetchHistoryTaskDetail, fetchHistoryTasks } from './api';

describe('training history API', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads history tasks including archived records', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, tasks: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await fetchHistoryTasks(200);

    expect(fetchMock).toHaveBeenCalledWith('/api/training/history?limit=200&include_archived=1', expect.objectContaining({ signal: undefined }));
  });

  it('encodes task ids when reading history details', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, task: {} }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await fetchHistoryTaskDetail('abc/123');

    expect(fetchMock).toHaveBeenCalledWith('/api/training/history/abc%2F123', expect.objectContaining({ signal: undefined }));
  });

  it('batches archive and delete actions with the required payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, message: 'ok' }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await batchUpdateHistoryTasks({ action: 'archive', task_ids: ['a', 'b'] });

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/training/history/batch',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ action: 'archive', task_ids: ['a', 'b'] }),
      }),
    );
  });
});

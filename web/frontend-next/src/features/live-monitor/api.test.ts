import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchTrainingLogs, fetchTrainingStatus, stopTraining } from './api';

describe('live monitor API', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads training status and logs with the expected query contract', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith('/api/training/logs')) {
        return new Response(JSON.stringify({ records: [] }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchTrainingStatus();
    await fetchTrainingLogs(300);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/training/status', expect.objectContaining({ signal: undefined }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/training/logs?limit=300', expect.objectContaining({ signal: undefined }));
  });

  it('stops training through POST', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, message: '训练已停止' }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await stopTraining();

    expect(fetchMock).toHaveBeenCalledWith('/api/training/stop', expect.objectContaining({ method: 'POST' }));
  });
});

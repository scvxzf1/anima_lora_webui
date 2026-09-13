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

    await stopTraining('task-a');

    expect(fetchMock).toHaveBeenCalledWith('/api/training/stop', expect.objectContaining({ method: 'POST', body: JSON.stringify({ task_id: 'task-a' }) }));
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('Content-Type')).toBe('application/json');
  });

  it('does not send an unguarded stop for an unknown task', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(() => stopTraining(' ')).toThrow('任务身份未确认');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

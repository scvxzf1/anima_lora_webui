import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  cancelQueueItem,
  fetchQueueSnapshot,
  moveQueueItem,
  retryQueueItem,
  saveQueueSettings,
} from './api';

describe('training queue API', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads the queue snapshot from the training queue endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, items: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await fetchQueueSnapshot();

    expect(fetchMock).toHaveBeenCalledWith('/api/training/queue', expect.objectContaining({ signal: undefined }));
  });

  it('persists queue policy settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await saveQueueSettings({ paused: true, failure_policy: 'pause', auto_retry: true, max_attempts: 3, retry_backoff_sec: 10 });

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/training/queue/settings',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ paused: true, failure_policy: 'pause', auto_retry: true, max_attempts: 3, retry_backoff_sec: 10 }),
      }),
    );
  });

  it('moves a queued item with an explicit direction', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await moveQueueItem('queue-item-1', 'up');

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/training/queue/queue-item-1/move',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ direction: 'up' }) }),
    );
  });

  it('retries and cancels queue items through the item endpoints', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await retryQueueItem('queue-item-2');
    await cancelQueueItem('queue-item-2', true);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/training/queue/queue-item-2/retry', expect.objectContaining({ method: 'POST' }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/training/queue/queue-item-2', expect.objectContaining({
      method: 'DELETE',
      body: JSON.stringify({ delete_runtime: true }),
    }));
  });
});

import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, apiRequest } from './client';

describe('apiRequest', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns a successful JSON envelope', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: true, value: 3 }), { status: 200 }),
      ),
    );

    await expect(apiRequest<{ value: number }>('/api/example')).resolves.toMatchObject({
      value: 3,
    });
  });

  it('rejects ok=false business responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: false, error: 'invalid config' }), {
          status: 200,
        }),
      ),
    );

    await expect(apiRequest('/api/example')).rejects.toEqual(
      expect.objectContaining<ApiError>({
        name: 'ApiError',
        message: 'invalid config',
        status: 200,
        payload: {
          ok: false,
          error: 'invalid config',
        },
      }),
    );
  });
});

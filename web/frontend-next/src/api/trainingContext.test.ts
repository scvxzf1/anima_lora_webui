import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchTrainingPresets } from './trainingContext';

describe('training context API', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads presets from the backend items envelope', async () => {
    const presets = ['balanced_16g', 'debug', 'default'];
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, items: presets }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchTrainingPresets()).resolves.toEqual(presets);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/presets',
      expect.objectContaining({ signal: undefined }),
    );
  });
});

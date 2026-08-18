import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  applyDatasetPreset,
  createDatasetGroup,
  deleteDatasetGroup,
  deleteDatasetPreset,
  fetchDatasetPreset,
  fetchDatasetPresetImages,
  importDatasetPreset,
  placeDatasetGroup,
  placeDatasetPreset,
  renameDatasetGroup,
  saveDatasetPreset,
  saveDatasetPresetAs,
} from './api';

const writePayload = {
  datasets: [{ source_dir: 'image_dataset/characters', num_repeats: 4, is_reg: false }],
  defaults: { resolution: 1024, batch_size: 1 },
  stage_schedule_enabled: false,
  stage_schedule: [],
};

describe('dataset editor API', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('applies a saved dataset preset to an explicit training config', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, message: '已应用数据集预设' }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await applyDatasetPreset(
      'configs/datasets/角色 A.toml',
      'configs/imported/train.toml',
    );

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/dataset-presets/apply',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          dataset_file: 'configs/datasets/角色 A.toml',
          train_file: 'configs/imported/train.toml',
        }),
      }),
    );
  });

  it('encodes dataset preset paths when reading details', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ok: true,
          file: 'configs/datasets/角色 A.toml',
          name: '角色 A',
          content: '',
          datasets: [],
          defaults: {},
          readonly: false,
          summary: {},
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    await fetchDatasetPreset('configs/datasets/角色 A.toml');

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/dataset-presets/read?file=configs%2Fdatasets%2F%E8%A7%92%E8%89%B2+A.toml',
      expect.objectContaining({ signal: undefined }),
    );
  });

  it('creates dataset groups with the dataset kind contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ok: true,
          message: '分组已创建',
          group: { id: 'characters', label: '角色', files: [] },
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    await createDatasetGroup('角色');

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/file-groups',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ label: '角色', kind: 'dataset' }),
      }),
    );
  });

  it('requests source images with the saved preset and preview limit contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, images: [], count: 0, total: 0 }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await fetchDatasetPresetImages('configs/datasets/角色 A.toml', 2);

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/dataset-presets/images?file=configs%2Fdatasets%2F%E8%A7%92%E8%89%B2+A.toml&dataset_index=2&source=source&limit=120',
      expect.objectContaining({ signal: undefined }),
    );
  });

  it('renames and deletes dataset groups through encoded group URLs', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => (
      new Response(JSON.stringify({ ok: true, message: '已完成', group: { id: '角色 A' } }), { status: 200 })
    ));
    vi.stubGlobal('fetch', fetchMock);

    await renameDatasetGroup('角色 A', '人物');
    await deleteDatasetGroup('角色 A');

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/config/file-groups/%E8%A7%92%E8%89%B2%20A', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ label: '人物' }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/config/file-groups/%E8%A7%92%E8%89%B2%20A',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });

  it('places groups and presets through the shared ordered-library endpoint', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => (
      new Response(JSON.stringify({ ok: true, message: '位置已更新' }), { status: 200 })
    ));
    vi.stubGlobal('fetch', fetchMock);

    await placeDatasetGroup('characters', 2);
    await placeDatasetPreset(
      'configs/datasets/alpha.toml',
      'styles',
      ['configs/datasets/beta.toml', 'configs/datasets/alpha.toml'],
    );

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/config/file-groups/place', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ target: 'group', group: 'characters', scope: 'dataset', index: 2 }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/config/file-groups/place', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        target: 'file',
        file: 'configs/datasets/alpha.toml',
        group: 'styles',
        order: ['configs/datasets/beta.toml', 'configs/datasets/alpha.toml'],
      }),
    }));
  });

  it('imports dataset presets as JSON with overwrite disabled by default', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, message: '已导入', file: 'configs/datasets/characters.toml' }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await importDatasetPreset('characters', '[[datasets]]\nsource_dir = "image_dataset/characters"\n');

    expect(fetchMock).toHaveBeenCalledWith('/api/config/dataset-presets/import', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        name: 'characters',
        content: '[[datasets]]\nsource_dir = "image_dataset/characters"\n',
        overwrite: false,
      }),
    }));
  });

  it('saves new and existing presets with an explicit overwrite contract', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => (
      new Response(JSON.stringify({ ok: true, message: '已保存', file: 'configs/datasets/characters.toml' }), { status: 200 })
    ));
    vi.stubGlobal('fetch', fetchMock);

    await saveDatasetPreset('configs/datasets/characters.toml', writePayload, false);
    await saveDatasetPreset('configs/datasets/characters.toml', writePayload, true);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/config/dataset-presets', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ file: 'configs/datasets/characters.toml', ...writePayload, overwrite: false }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/config/dataset-presets', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ file: 'configs/datasets/characters.toml', ...writePayload, overwrite: true }),
    }));
  });

  it('uses save-as for copies and encodes the path when deleting a preset', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => (
      new Response(JSON.stringify({ ok: true, message: '已完成', file: 'configs/datasets/角色 A.toml' }), { status: 200 })
    ));
    vi.stubGlobal('fetch', fetchMock);

    await saveDatasetPresetAs('角色 A', writePayload);
    await deleteDatasetPreset('configs/datasets/角色 A.toml');

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/config/dataset-presets/save-as', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ name: '角色 A', ...writePayload }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/config/dataset-presets?file=configs%2Fdatasets%2F%E8%A7%92%E8%89%B2+A.toml',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});

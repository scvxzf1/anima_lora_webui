import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DatasetWorkspace } from './DatasetWorkspace';

type PresetRecord = ReturnType<typeof presetPayload>;
type GroupRecord = {
  id: string;
  label: string;
  kind: 'dataset';
  renamable: boolean;
  deletable: boolean;
  movable: boolean;
  filePaths: string[];
};

function presetPayload(file: string, readonly = false) {
  const name = file.split('/').pop()?.replace(/\.toml$/i, '') || 'dataset';
  return {
    ok: true as const,
    file,
    name,
    content: '',
    readonly,
    defaults: {
      resolution: 1024,
      batch_size: 1,
      enable_bucket: true,
      prior_loss_weight: 1,
    },
    summary: {
      dataset_count: 1,
      train_dataset_count: 1,
      reg_dataset_count: 0,
      repeat_total: name === 'alpha' ? 5 : 3,
      resolution: 1024,
      batch_size: 1,
    },
    datasets: [{
      source_dir: `image_dataset/${name}`,
      image_dir: `post_image_dataset/${name}`,
      cache_dir: `post_image_dataset/${name}_cache`,
      num_repeats: name === 'alpha' ? 5 : 3,
      is_reg: false,
      settings: {},
    }],
    stage_schedule_enabled: false,
    stage_schedule: [] as Array<Record<string, unknown>>,
  };
}

function previewPayload(file: string, datasetIndex = 0) {
  const name = file.split('/').pop()?.replace(/\.toml$/i, '') || 'dataset';
  return {
    ok: true as const,
    file,
    dataset_index: datasetIndex,
    dataset_label: `第 ${datasetIndex + 1} 组数据集`,
    source: 'source' as const,
    source_label: '原始图目录',
    directory: `image_dataset/${name}`,
    directory_exists: true,
    caption_extension: '.txt',
    prefer_json_caption: false,
    caption_source_mode: 'auto',
    caption_source_label: '自动识别',
    caption_summary: '1 张图片识别到标注',
    count: 1,
    total: 1,
    limit: 120,
    row: {
      source_dir: `image_dataset/${name}`,
      image_dir: `post_image_dataset/${name}`,
      num_repeats: 5,
      recursive: true,
    },
    settings: {
      resolution: 1024,
      enable_bucket: true,
      min_bucket_reso: 256,
      max_bucket_reso: 1024,
      bucket_reso_steps: 64,
      validation_split: 0.1,
      validation_split_num: 0,
    },
    message: '',
    images: [{
      file: `image_dataset/${name}/${name}-01.png`,
      name: `${name}-01.png`,
      url: `/api/config/dataset-presets/image?file=${encodeURIComponent(file)}&dataset_index=${datasetIndex}&source=source&image=${name}-01.png`,
      mtime: 1_700_000_000,
      mtime_text: '2026-08-15 12:00',
      size_bytes: 2048,
      width: 1024,
      height: 1024,
      total_pixels: 1_048_576,
      caption: {
        ok: true,
        file: `image_dataset/${name}/${name}-01.txt`,
        extension: '.txt',
        source_mode: 'auto',
        source_label: '自动识别',
        detected_mode: 'txt',
        format_label: 'TXT',
        caption_count: 1,
        text: `${name} caption`,
        truncated: false,
        length: `${name} caption`.length,
      },
    }],
  };
}

function createFetchMock() {
  const presets = new Map<string, PresetRecord>([
    ['configs/datasets/alpha.toml', presetPayload('configs/datasets/alpha.toml')],
    ['configs/datasets/beta.toml', presetPayload('configs/datasets/beta.toml', true)],
  ]);
  let groups: GroupRecord[] = [{
    id: 'characters',
    label: '角色',
    kind: 'dataset',
    renamable: true,
    deletable: true,
    movable: true,
    filePaths: ['configs/datasets/alpha.toml', 'configs/datasets/beta.toml'],
  }, {
    id: 'concepts',
    label: '概念',
    kind: 'dataset',
    renamable: true,
    deletable: true,
    movable: true,
    filePaths: [],
  }];

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method || 'GET';

    if (url === '/api/config/file-groups?kind=training' && method === 'GET') {
      return jsonResponse([{
        id: 'imported',
        label: '导入配置',
        kind: 'training',
        methods_subdir: 'imported',
        files: [{
          path: 'configs/imported/train.toml',
          label: 'train.toml',
          filename: 'train.toml',
          method: 'lora',
          methods_subdir: 'imported',
          trainable: true,
          locked: false,
        }],
      }]);
    }

    if (url === '/api/presets' && method === 'GET') {
      return jsonResponse(['default', 'low_vram']);
    }

    if (url.startsWith('/api/config/merged?') && method === 'GET') {
      return jsonResponse({ max_train_steps: 1200 });
    }

    if (url === '/api/config/dataset-presets/apply' && method === 'POST') {
      const body = JSON.parse(String(init?.body || '{}'));
      return jsonResponse({
        ok: true,
        message: '已应用数据集预设',
        dataset_config: body.dataset_file,
        datasets: presets.get(body.dataset_file)?.datasets || [],
        defaults: presets.get(body.dataset_file)?.defaults || {},
        train_content: `dataset_config = "${body.dataset_file}"`,
        changed: ['dataset_config'],
        values: { dataset_config: body.dataset_file },
        summary: presets.get(body.dataset_file)?.summary || {},
      });
    }

    if (url === '/api/config/dataset-presets' && method === 'GET') {
      const items = [...presets.values()].map((preset) => ({
        path: preset.file,
        label: `${preset.name}.toml`,
        readonly: preset.readonly,
        summary: preset.summary,
      }));
      const itemByPath = new Map(items.map((item) => [item.path, item]));
      return jsonResponse({
        ok: true,
        presets: items,
        groups: groups.map((group) => ({
          ...group,
          files: group.filePaths.map((path) => itemByPath.get(path)).filter(Boolean),
        })),
      });
    }

    if (url.startsWith('/api/config/dataset-presets/read?')) {
      const file = new URL(url, 'http://localhost').searchParams.get('file') || '';
      const preset = presets.get(file);
      return preset ? jsonResponse(preset) : jsonResponse({ ok: false, error: 'not found' }, 404);
    }

    if (url.startsWith('/api/config/dataset-presets/images?')) {
      const params = new URL(url, 'http://localhost').searchParams;
      const file = params.get('file') || '';
      const datasetIndex = Number(params.get('dataset_index') || 0);
      return presets.has(file)
        ? jsonResponse(previewPayload(file, datasetIndex))
        : jsonResponse({ ok: false, error: 'not found' }, 404);
    }

    if (url === '/api/config/dataset-presets' && method === 'PUT') {
      const body = JSON.parse(String(init?.body || '{}'));
      const preset = mutationPreset(body.file, body);
      presets.set(body.file, presetPayloadFromMutation(preset));
      return jsonResponse(preset);
    }

    if (url === '/api/config/dataset-presets/save-as' && method === 'POST') {
      const body = JSON.parse(String(init?.body || '{}'));
      const stem = String(body.name).replace(/\.toml$/i, '');
      const file = `configs/datasets/${stem}.toml`;
      if (presets.has(file)) return jsonResponse({ ok: false, error: '数据集预设已存在' }, 400);
      const preset = mutationPreset(file, body);
      presets.set(file, presetPayloadFromMutation(preset));
      return jsonResponse(preset);
    }

    if (url === '/api/config/dataset-presets/import' && method === 'POST') {
      const body = JSON.parse(String(init?.body || '{}'));
      const stem = String(body.name).replace(/\.toml$/i, '');
      const file = `configs/datasets/${stem}.toml`;
      if (presets.has(file)) return jsonResponse({ ok: false, error: '数据集预设已存在' }, 400);
      const preset = { ...presetPayload(file), content: String(body.content || '') };
      presets.set(file, preset);
      return jsonResponse({ ...preset, message: '数据集预设已导入' });
    }

    if (url.startsWith('/api/config/dataset-presets?') && method === 'DELETE') {
      const file = new URL(url, 'http://localhost').searchParams.get('file') || '';
      presets.delete(file);
      groups.forEach((group) => {
        group.filePaths = group.filePaths.filter((path) => path !== file);
      });
      return jsonResponse({ ok: true, message: '数据集预设已删除', file });
    }

    if (url === '/api/config/file-groups' && method === 'POST') {
      const body = JSON.parse(String(init?.body || '{}'));
      groups.push({
        id: 'styles',
        label: body.label,
        kind: 'dataset',
        renamable: true,
        deletable: true,
        movable: true,
        filePaths: [],
      });
      return jsonResponse({
        ok: true,
        message: '分组已创建',
        group: { id: 'styles', label: body.label, files: [] },
      });
    }

    if (url.startsWith('/api/config/file-groups/') && method === 'PATCH') {
      const groupId = decodeURIComponent(url.split('/').pop() || '');
      const body = JSON.parse(String(init?.body || '{}'));
      const group = groups.find((item) => item.id === groupId);
      if (!group) return jsonResponse({ ok: false, error: '未知分组' }, 400);
      group.label = body.label;
      return jsonResponse({ ok: true, message: '分组已重命名', group: { ...group, files: [] } });
    }

    if (url.startsWith('/api/config/file-groups/') && method === 'DELETE') {
      const groupId = decodeURIComponent(url.split('/').pop() || '');
      const deleted = groups.find((item) => item.id === groupId);
      groups = groups.filter((item) => item.id !== groupId);
      groups.push({
        id: 'unfiled_datasets',
        label: '未分组',
        kind: 'dataset',
        renamable: false,
        deletable: false,
        movable: true,
        filePaths: deleted?.filePaths ?? [],
      });
      return jsonResponse({ ok: true, message: '分组已删除，TOML 文件已保留在其他可见分组中' });
    }

    if (url === '/api/config/file-groups/place' && method === 'POST') {
      const body = JSON.parse(String(init?.body || '{}'));
      if (body.target === 'group') {
        const oldIndex = groups.findIndex((group) => group.id === body.group);
        const [moving] = groups.splice(oldIndex, 1);
        groups.splice(Math.max(0, Math.min(Number(body.index), groups.length)), 0, moving);
        return jsonResponse({ ok: true, message: '分组位置已更新', group: { ...moving, files: [] } });
      }
      if (body.target === 'file') {
        groups.forEach((group) => {
          group.filePaths = group.filePaths.filter((path) => path !== body.file);
        });
        const target = groups.find((group) => group.id === body.group);
        if (!target) return jsonResponse({ ok: false, error: '目标分组不存在' }, 400);
        target.filePaths = Array.isArray(body.order) ? body.order : [...target.filePaths, body.file];
        return jsonResponse({ ok: true, message: '配置位置已更新', group: { ...target, files: [] } });
      }
    }

    return jsonResponse({ ok: false, error: `unhandled ${method} ${url}` }, 500);
  });

  return { fetchMock, presets };
}

function mutationPreset(file: string, body: Record<string, unknown>) {
  const datasets = body.datasets as PresetRecord['datasets'];
  const defaults = body.defaults as PresetRecord['defaults'];
  return {
    ok: true as const,
    message: `已保存数据集预设 ${file.split('/').pop()}`,
    file,
    content: '',
    datasets,
    defaults,
    summary: {
      dataset_count: datasets.length,
      train_dataset_count: datasets.filter((row) => !row.is_reg).length,
      reg_dataset_count: datasets.filter((row) => row.is_reg).length,
      repeat_total: datasets.reduce((total, row) => total + Number(row.num_repeats || 1), 0),
      resolution: Number(defaults.resolution || 1024),
      batch_size: Number(defaults.batch_size || 1),
    },
    stage_schedule_enabled: Boolean(body.stage_schedule_enabled),
    stage_schedule: Array.isArray(body.stage_schedule)
      ? body.stage_schedule as Array<Record<string, unknown>>
      : [],
  };
}

function presetPayloadFromMutation(preset: ReturnType<typeof mutationPreset>): PresetRecord {
  return {
    ...preset,
    name: preset.file.split('/').pop()?.replace(/\.toml$/i, '') || 'dataset',
    readonly: false,
  };
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status });
}

function renderWorkspace() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const router = createMemoryRouter([
    { path: '/datasets', element: <DatasetWorkspace /> },
    { path: '/other', element: <main>其他页面</main> },
  ], { initialEntries: ['/datasets'] });
  return {
    router,
    ...render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
    ),
  };
}

describe('DatasetWorkspace', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('renders grouped presets, filters them, and loads the selected editor', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();

    expect(await screen.findByRole('heading', { name: 'alpha.toml' })).toBeInTheDocument();
    expect(screen.getByLabelText('原始图片目录')).toHaveValue('image_dataset/alpha');
    await user.type(screen.getByRole('searchbox', { name: '搜索预设' }), 'beta');
    expect(screen.queryByRole('button', { name: /alpha.toml/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /beta.toml/ }));

    expect(await screen.findByRole('heading', { name: 'beta.toml' })).toBeInTheDocument();
    expect(screen.getByLabelText('原始图片目录')).toHaveValue('image_dataset/beta');
    expect(screen.getByRole('button', { name: '保存' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '复制' })).toBeEnabled();
  });

  it('blocks browser unload and SPA navigation while the dataset draft is dirty', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const user = userEvent.setup();
    const { router } = renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });

    const cleanUnload = new Event('beforeunload', { cancelable: true });
    expect(window.dispatchEvent(cleanUnload)).toBe(true);
    expect(cleanUnload.defaultPrevented).toBe(false);

    await user.type(screen.getByLabelText('原始图片目录'), '-dirty');
    const dirtyUnload = new Event('beforeunload', { cancelable: true });
    expect(window.dispatchEvent(dirtyUnload)).toBe(false);
    expect(dirtyUnload.defaultPrevented).toBe(true);

    await act(async () => {
      await router.navigate('/other');
    });
    await waitFor(() => expect(confirm).toHaveBeenCalledWith(
      '当前数据集有未保存修改，离开会丢失这些修改。是否继续？',
    ));
    expect(router.state.location.pathname).toBe('/datasets');

    confirm.mockReturnValue(true);
    await act(async () => {
      await router.navigate('/other');
    });
    expect(await screen.findByText('其他页面')).toBeInTheDocument();
  });

  it('closes basic dialogs with Escape and restores focus to their trigger', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });

    const newButton = screen.getByRole('button', { name: '新建' });
    await user.click(newButton);
    const nameDialog = screen.getByRole('dialog', { name: '新建数据集预设' });
    expect(within(nameDialog).getByRole('textbox', { name: '预设名称' })).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', { name: '新建数据集预设' })).not.toBeInTheDocument();
    expect(newButton).toHaveFocus();

    const deleteButton = screen.getByRole('button', { name: '删除' });
    await user.click(deleteButton);
    const deleteDialog = screen.getByRole('alertdialog', { name: '删除数据集预设' });
    expect(within(deleteDialog).getByRole('button', { name: '关闭删除确认' })).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('alertdialog', { name: '删除数据集预设' })).not.toBeInTheDocument();
    expect(deleteButton).toHaveFocus();
  });

  it('applies only the saved preset to the selected training config after confirmation', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });

    expect(screen.getByLabelText('当前训练配置')).toHaveValue('configs/imported/train.toml');
    expect(screen.getByLabelText('当前硬件预设')).toHaveValue('default');
    expect(screen.getByText('1200 steps')).toBeInTheDocument();

    const applyButton = screen.getByRole('button', { name: '应用到当前训练配置' });
    await user.click(applyButton);
    const dialog = screen.getByRole('alertdialog', { name: '应用数据集到训练配置' });
    expect(within(dialog).getByText('configs/datasets/alpha.toml')).toBeInTheDocument();
    expect(within(dialog).getByText('configs/imported/train.toml')).toBeInTheDocument();
    await user.click(within(dialog).getByRole('button', { name: '确认应用' }));

    expect(await screen.findByText('已应用数据集预设')).toBeInTheDocument();
    const body = requestBody(fetchMock, '/api/config/dataset-presets/apply', 'POST');
    expect(body).toEqual({
      dataset_file: 'configs/datasets/alpha.toml',
      train_file: 'configs/imported/train.toml',
    });

    await user.type(screen.getByLabelText('原始图片目录'), '-dirty');
    expect(applyButton).toBeDisabled();
    expect(applyButton).toHaveAttribute('title', '请先保存当前数据集修改');
  });

  it('edits and saves stage scheduling without any legacy bridge', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });

    await user.click(screen.getByRole('checkbox', { name: '已关闭' }));
    const configureButton = screen.getByRole('button', { name: '配置阶段' });
    await user.click(configureButton);
    const dialog = screen.getByRole('dialog', { name: '配置分阶段调度' });
    await user.click(within(dialog).getByRole('button', { name: '三段模板' }));
    expect(within(dialog).getAllByRole('textbox', { name: /阶段 \d 名称/ })).toHaveLength(3);
    await user.click(within(dialog).getByRole('button', { name: '应用到当前预设' }));
    expect(configureButton).toHaveFocus();

    await user.click(screen.getByRole('button', { name: '保存' }));
    await screen.findByText('已保存数据集预设 alpha.toml');
    const body = requestBody(fetchMock, '/api/config/dataset-presets', 'PUT');
    expect(body.stage_schedule_enabled).toBe(true);
    expect(body.stage_schedule).toHaveLength(3);
    expect(body.stage_schedule[0]).toMatchObject({ subset_index: 0, start_pct: 0 });
    expect(body.stage_schedule[2]).toMatchObject({ subset_index: 0, end_pct: 1 });
  });

  it('creates a dataset group and refreshes the library', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.type(screen.getByRole('textbox', { name: '新分组' }), '风格');
    await user.click(screen.getByRole('button', { name: '创建' }));

    expect(await screen.findByText('分组已创建')).toHaveAttribute('role', 'status');
    expect(requestBody(fetchMock, '/api/config/file-groups')).toEqual({ label: '风格', kind: 'dataset' });
  });

  it('reorders dataset groups with a dataset-scoped place request', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '下移分组 角色' }));

    await waitFor(() => expect(requestBody(fetchMock, '/api/config/file-groups/place')).toEqual({
      target: 'group',
      group: 'characters',
      scope: 'dataset',
      index: 1,
    }));
  });

  it('reorders presets using the full target-group order', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '下移预设 alpha.toml' }));

    await waitFor(() => expect(requestBody(fetchMock, '/api/config/file-groups/place')).toEqual({
      target: 'file',
      file: 'configs/datasets/alpha.toml',
      group: 'characters',
      order: ['configs/datasets/beta.toml', 'configs/datasets/alpha.toml'],
    }));
  });

  it('starts and cancels dnd-kit keyboard sorting without mutating order', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    const handle = screen.getByRole('button', { name: '拖动排序预设 alpha.toml' });
    handle.focus();
    await user.keyboard('[Space]');
    await waitFor(() => expect(handle.closest('.dataset-preset-row')).toHaveAttribute('data-dragging', 'true'));
    await user.keyboard('[Escape]');

    await waitFor(() => expect(handle.closest('.dataset-preset-row')).toHaveAttribute('data-dragging', 'false'));
    expect(requestIndex(fetchMock, '/api/config/file-groups/place', 'POST')).toBe(-1);
  });

  it('moves editable presets across dataset groups', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.selectOptions(screen.getByRole('combobox', { name: '移动 alpha.toml 到分组' }), 'concepts');

    await waitFor(() => expect(requestBody(fetchMock, '/api/config/file-groups/place')).toEqual({
      target: 'file',
      file: 'configs/datasets/alpha.toml',
      group: 'concepts',
      order: ['configs/datasets/alpha.toml'],
    }));
  });

  it('renames a user-managed dataset group', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '重命名分组 角色' }));
    await user.clear(screen.getByRole('textbox', { name: '分组名称' }));
    await user.type(screen.getByRole('textbox', { name: '分组名称' }), '人物');
    await user.click(screen.getByRole('button', { name: '保存名称' }));

    expect(await screen.findByRole('heading', { name: '人物' })).toBeInTheDocument();
    expect(requestBody(fetchMock, '/api/config/file-groups/characters', 'PATCH')).toEqual({ label: '人物' });
  });

  it('deletes only group metadata and keeps presets visible', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '删除分组 角色' }));
    expect(screen.getByRole('alertdialog')).toHaveTextContent('TOML 预设文件不会被删除');
    await user.click(screen.getByRole('button', { name: '删除分组' }));

    expect(await screen.findByRole('heading', { name: '未分组' })).toBeInTheDocument();
    const libraryPath = screen.getAllByText('configs/datasets/alpha.toml')
      .find((element) => element.classList.contains('dataset-preset-path'));
    expect(libraryPath?.closest('button')).toHaveClass('dataset-preset');
    expect(presets.has('configs/datasets/alpha.toml')).toBe(true);
  });

  it('imports TOML without overwrite and selects the imported preset', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    const file = new File(['[[datasets]]\nsource_dir = "image_dataset/imported"\n'], 'source.toml', {
      type: 'text/plain',
    });
    await user.upload(screen.getByLabelText('选择要导入的预设'), file);
    await user.clear(await screen.findByRole('textbox', { name: '预设名称' }));
    await user.type(screen.getByRole('textbox', { name: '预设名称' }), 'imported');
    await user.click(screen.getByRole('button', { name: '导入预设' }));

    expect(await screen.findByRole('heading', { name: 'imported.toml' })).toBeInTheDocument();
    expect(presets.has('configs/datasets/imported.toml')).toBe(true);
    expect(requestBody(fetchMock, '/api/config/dataset-presets/import')).toEqual({
      name: 'imported',
      content: '[[datasets]]\nsource_dir = "image_dataset/imported"\n',
      overwrite: false,
    });
  });

  it('exports the selected preset using its raw TOML and basename', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:dataset-export');
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '导出' }));

    await waitFor(() => expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob)));
    const anchor = click.mock.instances[0] as HTMLAnchorElement;
    expect(anchor.download).toBe('alpha.toml');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:dataset-export');
  });

  it('overwrites an existing preset with edited form values', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    const source = await screen.findByLabelText('原始图片目录');
    await user.clear(source);
    await user.type(source, 'image_dataset/alpha-updated');
    await user.click(screen.getByRole('button', { name: '保存' }));

    await screen.findByText('已保存数据集预设 alpha.toml');
    const body = requestBody(fetchMock, '/api/config/dataset-presets', 'PUT');
    expect(body.overwrite).toBe(true);
    expect(body.file).toBe('configs/datasets/alpha.toml');
    expect(body.datasets[0].source_dir).toBe('image_dataset/alpha-updated');
  });

  it('previews saved subset images, captions, refresh, fullscreen, and focus restoration', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, 'writeText');
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    const previewButton = screen.getByRole('button', { name: '预览子集 1 图片和标注' });
    await user.click(previewButton);

    const dialog = await screen.findByRole('dialog', { name: '子集 1 图片与标注' });
    expect(within(dialog).getByText(/原始图目录 · image_dataset\/alpha · 1\/1 张/)).toBeInTheDocument();
    expect(within(dialog).getAllByText('自动识别')).not.toHaveLength(0);
    const thumbnail = within(dialog).getByRole('img', { name: 'alpha-01.png' });
    expect(thumbnail).toHaveAttribute('loading', 'lazy');
    expect(within(dialog).getByRole('button', { name: '关闭图片预览' })).toHaveFocus();
    const refreshButton = within(dialog).getByRole('button', { name: '刷新' });
    const cardCopyButton = within(dialog).getByRole('button', { name: '复制 alpha-01.png 的标注' });
    cardCopyButton.focus();
    await user.tab();
    expect(refreshButton).toHaveFocus();
    await user.tab({ shift: true });
    expect(cardCopyButton).toHaveFocus();
    await user.click(cardCopyButton);
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('alpha caption'));
    const previewCard = thumbnail.closest('.dataset-preview-card');
    fireEvent.error(thumbnail);
    expect(previewCard).toHaveAttribute('data-image-error', 'true');

    await user.click(within(dialog).getByRole('button', { name: '查看大图 alpha-01.png' }));
    const viewer = await screen.findByRole('dialog', { name: 'alpha-01.png' });
    expect(within(viewer).getByRole('button', { name: '关闭大图' })).toHaveFocus();
    await user.tab({ shift: true });
    expect(within(viewer).getByRole('button', { name: '复制' })).toHaveFocus();
    fireEvent.error(within(viewer).getByRole('img', { name: 'alpha-01.png' }));
    expect(within(viewer).getByRole('alert')).toHaveTextContent('图片加载失败');
    await user.click(within(viewer).getByRole('button', { name: '重试' }));
    expect(within(viewer).getByRole('img', { name: 'alpha-01.png' })).toBeInTheDocument();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', { name: 'alpha-01.png' })).not.toBeInTheDocument();
    expect(screen.getByRole('dialog', { name: '子集 1 图片与标注' })).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: '刷新' }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(
      ([input]) => String(input).startsWith('/api/config/dataset-presets/images?'),
    )).toHaveLength(2));
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', { name: '子集 1 图片与标注' })).not.toBeInTheDocument();
    expect(previewButton).toHaveFocus();
  });

  it('disables stale preview for drafts and dirty edits, but allows saved readonly presets', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    const source = screen.getByLabelText('原始图片目录');
    await user.type(source, '-dirty');
    expect(screen.getByRole('button', { name: '预览子集 1 图片和标注' })).toBeDisabled();

    vi.spyOn(window, 'confirm').mockReturnValue(true);
    await user.click(screen.getByRole('button', { name: /beta.toml/ }));
    expect(await screen.findByRole('heading', { name: 'beta.toml' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '预览子集 1 图片和标注' })).toBeEnabled();
  });

  it('saves the complete defaults and advanced subset field contract', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    const defaults = screen.getByRole('group', { name: '默认训练参数' });
    await user.clear(within(defaults).getByLabelText('保留 Token 数'));
    await user.type(within(defaults).getByLabelText('保留 Token 数'), '5');
    await user.clear(within(defaults).getByLabelText('标注扩展名'));
    await user.type(within(defaults).getByLabelText('标注扩展名'), '.caption');
    await user.selectOptions(within(defaults).getByLabelText('标注来源'), 'captions_json');

    const subset = screen.getByRole('group', { name: '子集 1' });
    await user.click(within(subset).getByText('高级规则'));
    await user.clear(within(subset).getByLabelText('路径筛选'));
    await user.type(within(subset).getByLabelText('路径筛选'), 'character_*');
    await user.click(within(subset).getByLabelText('NL/Tag 标签混合'));
    await user.clear(within(subset).getByLabelText('Tag 占比'));
    await user.type(within(subset).getByLabelText('Tag 占比'), '0.6');
    await user.click(within(subset).getByLabelText('触发词图像复制'));
    await user.type(within(subset).getByLabelText('触发词'), 'alpha_token');
    await user.clear(within(subset).getByLabelText('复制循环次数'));
    await user.type(within(subset).getByLabelText('复制循环次数'), '3');
    await user.click(screen.getByRole('button', { name: '保存' }));

    await screen.findByText('已保存数据集预设 alpha.toml');
    const body = requestBody(fetchMock, '/api/config/dataset-presets', 'PUT');
    expect(body.defaults).toMatchObject({
      keep_tokens: 5,
      caption_extension: '.caption',
      caption_source_mode: 'captions_json',
    });
    expect(body.datasets[0]).toMatchObject({
      recursive: true,
      path_pattern: 'character_*',
      nl_tag_mix: { enabled: true, tag_ratio: 0.6 },
      trigger_clone: { enabled: true, prompt: 'alpha_token', num_repeats: 3 },
    });
    expect(body.datasets[0].settings).toMatchObject({
      resolution: 1024,
      min_bucket_reso: 256,
      max_bucket_reso: 1024,
      validation_seed: 42,
    });
  });

  it('adds and reorders subsets while keeping stable row values', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '添加子集' }));
    let subsets = screen.getAllByRole('group', { name: /子集 \d/ });
    await user.type(within(subsets[1]).getByLabelText('原始图片目录'), 'image_dataset/second');
    await user.click(within(subsets[1]).getByRole('button', { name: '上移子集 2' }));
    subsets = screen.getAllByRole('group', { name: /子集 \d/ });

    expect(within(subsets[0]).getByLabelText('原始图片目录')).toHaveValue('image_dataset/second');
    await user.click(screen.getByRole('button', { name: '保存' }));
    await waitFor(() => expect(requestBody(fetchMock, '/api/config/dataset-presets', 'PUT').datasets[0].source_dir)
      .toBe('image_dataset/second'));
  });

  it('copies experimental rules to selected subset scopes', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '添加子集' }));
    const subsets = screen.getAllByRole('group', { name: /子集 \d/ });
    await user.type(within(subsets[1]).getByLabelText('原始图片目录'), 'image_dataset/second');
    await user.click(within(subsets[0]).getByText('高级规则'));
    await user.click(within(subsets[0]).getByLabelText('NL/Tag 标签混合'));
    await user.click(within(subsets[0]).getByLabelText('触发词图像复制'));
    await user.type(within(subsets[0]).getByLabelText('触发词'), 'shared_token');
    await user.click(within(subsets[0]).getByLabelText('子集 2'));
    await user.click(within(subsets[0]).getByRole('button', { name: '应用到所选子集' }));
    await user.click(screen.getByRole('button', { name: '保存' }));

    await waitFor(() => expect(requestBody(fetchMock, '/api/config/dataset-presets', 'PUT').datasets[1]).toMatchObject({
      nl_tag_mix: { enabled: true, tag_ratio: 0.7 },
      trigger_clone: { enabled: true, prompt: 'shared_token', num_repeats: 1 },
    }));
  });

  it('creates and saves a new preset with overwrite disabled', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '新建' }));
    await user.clear(screen.getByRole('textbox', { name: '预设名称' }));
    await user.type(screen.getByRole('textbox', { name: '预设名称' }), 'gamma');
    await user.click(screen.getByRole('button', { name: '创建草稿' }));

    expect(screen.getByRole('heading', { name: 'gamma.toml' })).toBeInTheDocument();
    const source = await screen.findByLabelText('原始图片目录');
    await user.type(source, 'image_dataset/gamma');
    await user.click(screen.getByRole('button', { name: '保存' }));
    await waitFor(() => expect(presets.has('configs/datasets/gamma.toml')).toBe(true));
    const body = requestBody(fetchMock, '/api/config/dataset-presets', 'PUT', -1);
    expect(body.overwrite).toBe(false);
    expect(body.file).toBe('configs/datasets/gamma.toml');
  });

  it('protects dirty edits when switching presets', async () => {
    const { fetchMock } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const user = userEvent.setup();
    renderWorkspace();
    const source = await screen.findByLabelText('原始图片目录');
    await user.type(source, '-dirty');
    await user.click(screen.getByRole('button', { name: /beta.toml/ }));

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('切换预设会丢弃'));
    expect(screen.getByRole('heading', { name: 'alpha.toml' })).toBeInTheDocument();
  });

  it('copies a readonly preset through save-as', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: /beta.toml/ }));
    await screen.findByRole('heading', { name: 'beta.toml' });
    await user.click(screen.getByRole('button', { name: '复制' }));
    await user.click(screen.getByRole('button', { name: '复制预设' }));

    await waitFor(() => expect(presets.has('configs/datasets/beta_copy.toml')).toBe(true));
    expect(screen.getByRole('heading', { name: 'beta_copy.toml' })).toBeInTheDocument();
  });

  it('renames by saving the new preset before deleting the old TOML', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '重命名' }));
    await user.clear(screen.getByRole('textbox', { name: '预设名称' }));
    await user.type(screen.getByRole('textbox', { name: '预设名称' }), 'alpha-renamed');
    await user.click(screen.getByRole('button', { name: '确认重命名' }));

    await waitFor(() => expect(presets.has('configs/datasets/alpha-renamed.toml')).toBe(true));
    await waitFor(() => expect(
      requestIndex(fetchMock, '/api/config/dataset-presets?file=configs%2Fdatasets%2Falpha.toml', 'DELETE'),
    ).toBeGreaterThanOrEqual(0));
    await waitFor(() => expect(presets.has('configs/datasets/alpha.toml')).toBe(false));
    expect(requestIndex(fetchMock, '/api/config/dataset-presets/save-as', 'POST')).toBeLessThan(
      requestIndex(fetchMock, '/api/config/dataset-presets?file=configs%2Fdatasets%2Falpha.toml', 'DELETE'),
    );
  });

  it('deletes only after the explicit dangerous-action dialog', async () => {
    const { fetchMock, presets } = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await screen.findByRole('heading', { name: 'alpha.toml' });
    await user.click(screen.getByRole('button', { name: '删除' }));
    expect(screen.getByRole('alertdialog')).toHaveTextContent('不删除图片、缩放图或缓存目录');
    await user.click(screen.getByRole('button', { name: '删除预设' }));

    await waitFor(() => expect(presets.has('configs/datasets/alpha.toml')).toBe(false));
    expect(await screen.findByRole('heading', { name: 'beta.toml' })).toBeInTheDocument();
  });
});

function requestBody(fetchMock: ReturnType<typeof vi.fn>, url: string, method = 'POST', occurrence = 0) {
  const calls = fetchMock.mock.calls.filter(
    ([input, init]) => String(input) === url && ((init as RequestInit | undefined)?.method || 'GET') === method,
  );
  const call = occurrence < 0 ? calls.at(occurrence) : calls[occurrence];
  return JSON.parse(String((call?.[1] as RequestInit | undefined)?.body || '{}'));
}

function requestIndex(fetchMock: ReturnType<typeof vi.fn>, url: string, method: string) {
  return fetchMock.mock.calls.findIndex(
    ([input, init]) => String(input) === url && (init as RequestInit | undefined)?.method === method,
  );
}

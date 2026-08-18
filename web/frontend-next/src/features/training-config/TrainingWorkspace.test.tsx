import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TrainingWorkspace } from './TrainingWorkspace';

function renderWorkspace() {
  const router = createMemoryRouter([
    { path: '/training', element: <TrainingWorkspace /> },
    { path: '/other', element: <main>其他页面</main> },
  ], { initialEntries: ['/training'] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return {
    router,
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  };
}

function createFetchMock() {
  let content = 'output_name = "dragon-run"\nmax_train_steps = 1600\n';
  let merged = {
    output_name: 'dragon-run',
    model_family: 'krea2_raw',
    max_train_steps: 1600,
    train_batch_size: 1,
    dataset_config: 'configs/datasets/alpha.toml',
    gradient_checkpointing: true,
  } as Record<string, unknown>;
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method || 'GET';
    if (url === '/api/config/file-groups?kind=training') return jsonResponse([{
      id: 'imported',
      label: '导入配置',
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
    if (url === '/api/presets') return jsonResponse(['default', 'low_vram']);
    if (url.startsWith('/api/config/merged?')) {
      const preset = new URL(url, 'http://localhost').searchParams.get('preset');
      return jsonResponse({ ...merged, max_train_steps: preset === 'low_vram' ? 800 : merged.max_train_steps });
    }
    if (url.startsWith('/api/config/raw?') && method === 'GET') return jsonResponse({
      file: 'configs/imported/train.toml',
      content,
      meta: { path: 'configs/imported/train.toml', label: 'train.toml', method: 'lora', methods_subdir: 'imported', locked: false },
    });
    if (url === '/api/config/raw/patch-preview' && method === 'POST') {
      const body = requestJson(init);
      return jsonResponse(patchResult(body.file, body.values));
    }
    if (url === '/api/config/raw' && method === 'PATCH') {
      const body = requestJson(init);
      const result = patchResult(body.file, body.values);
      content = result.content;
      merged = { ...merged, ...body.values };
      return jsonResponse(result);
    }
    if (url === '/api/config/raw/save-as' && method === 'POST') {
      const body = requestJson(init);
      return jsonResponse({ ok: true, file: body.file, message: '保存成功', warnings: [] });
    }
    if (url === '/api/training/preflight' && method === 'POST') return jsonResponse({
      ok: false,
      variant: 'lora',
      preset: 'default',
      methods_subdir: 'imported',
      summary: { errors: 1, warnings: 0, checks: 2 },
      checks: [
        { level: 'error', key: 'qwen3', message: 'Qwen3 文本编码器 不存在', path: 'models/qwen.safetensors' },
        { level: 'ok', key: 'source_image_dir', message: '源图像目录 存在', path: 'image_dataset' },
      ],
      errors: [{ level: 'error', key: 'qwen3', message: 'Qwen3 文本编码器 不存在' }],
      warnings: [],
    });
    return new Response(JSON.stringify({ ok: false, error: `unhandled ${method} ${url}` }), { status: 500 });
  });
  return fetchMock;
}

function patchResult(file: string, values: Record<string, unknown>) {
  const changed = Object.keys(values);
  const lines = Object.entries(values).map(([key, value]) => (
    typeof value === 'string' ? `${key} = ${JSON.stringify(value)}` : `${key} = ${String(value)}`
  ));
  return { ok: true, file, message: '保存成功', content: `${lines.join('\n')}\n`, changed, warnings: [] };
}

function requestJson(init?: RequestInit) {
  return JSON.parse(String(init?.body || '{}'));
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status });
}

describe('TrainingWorkspace', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('shows source-aware merged values without exposing a start action', async () => {
    vi.stubGlobal('fetch', createFetchMock());
    renderWorkspace();

    expect(screen.getByRole('heading', { name: '训练配置' })).toBeInTheDocument();
    expect(await screen.findByDisplayValue('dragon-run')).toBeInTheDocument();
    expect(screen.getByDisplayValue('configs/datasets/alpha.toml')).toBeInTheDocument();
    expect(screen.getAllByText('1 当前文件')).toHaveLength(2);
    expect(screen.getAllByText('继承/预设').length).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: /启动/ })).not.toBeInTheDocument();
  });

  it('previews and saves only changed fields, then runs structured preflight', async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    const outputName = await screen.findByLabelText('输出名称');
    await waitFor(() => expect(outputName).toBeEnabled());

    await user.clear(outputName);
    await user.type(outputName, 'dragon-edited');
    expect(screen.getByText('有未保存修改')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '运行预检测' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: '预览变更' }));
    expect(await screen.findByText('1 项')).toBeInTheDocument();
    expect(screen.getByText('output_name', { selector: '.training-patch-preview code' })).toBeInTheDocument();
    expect(requestBody(fetchMock, '/api/config/raw/patch-preview', 'POST').values).toEqual({ output_name: 'dragon-edited' });

    await user.click(screen.getByRole('button', { name: '保存配置' }));
    expect(await screen.findByText('保存成功')).toBeInTheDocument();
    expect(requestBody(fetchMock, '/api/config/raw', 'PATCH').values).toEqual({ output_name: 'dragon-edited' });
    await waitFor(() => expect(screen.getByRole('button', { name: '运行预检测' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '运行预检测' }));
    expect(await screen.findByText('需要处理')).toBeInTheDocument();
    expect(screen.getByText('Qwen3 文本编码器 不存在')).toBeInTheDocument();
    expect(screen.getByText('源图像目录 存在')).toBeInTheDocument();
  });

  it('guards dirty preset switching, browser unload, and SPA navigation', async () => {
    vi.stubGlobal('fetch', createFetchMock());
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const user = userEvent.setup();
    const { router } = renderWorkspace();
    const outputName = await screen.findByLabelText('输出名称');
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, '-dirty');

    const unload = new Event('beforeunload', { cancelable: true });
    expect(window.dispatchEvent(unload)).toBe(false);
    await user.selectOptions(screen.getByLabelText('当前硬件预设'), 'low_vram');
    expect(confirm).toHaveBeenCalledWith('当前训练配置有未保存修改。切换硬件预设会丢失这些修改，是否继续？');
    expect(screen.getByLabelText('当前硬件预设')).toHaveValue('default');

    await act(async () => { await router.navigate('/other'); });
    expect(confirm).toHaveBeenCalledWith('当前训练配置有未保存修改，离开会丢失这些修改。是否继续？');
    expect(router.state.location.pathname).toBe('/training');
  });

  it('saves a patched copy into imported configs without overwriting', async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    const outputName = await screen.findByLabelText('输出名称');
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.clear(outputName);
    await user.type(outputName, 'copy-value');
    await user.click(screen.getByRole('button', { name: '另存配置' }));
    const dialog = screen.getByRole('dialog', { name: '另存训练配置' });
    await user.clear(within(dialog).getByRole('textbox', { name: '配置名称' }));
    await user.type(within(dialog).getByRole('textbox', { name: '配置名称' }), 'dragon copy');
    await user.click(within(dialog).getByRole('button', { name: '确认另存' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/config/raw/save-as',
      expect.objectContaining({ method: 'POST' }),
    ));
    const body = requestBody(fetchMock, '/api/config/raw/save-as', 'POST');
    expect(body.file).toBe('configs/imported/dragon_copy.toml');
    expect(body.content).toContain('output_name = "copy-value"');
  });
});

function requestBody(fetchMock: ReturnType<typeof vi.fn>, url: string, method: string) {
  const call = fetchMock.mock.calls.find(([input, init]) => String(input) === url && (init?.method || 'GET') === method);
  if (!call) throw new Error(`missing ${method} ${url}`);
  return requestJson(call[1]);
}

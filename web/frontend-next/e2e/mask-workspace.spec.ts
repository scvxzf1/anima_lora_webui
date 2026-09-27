import { expect, test, type Page } from '@playwright/test';
import { mockWorkspace } from './fixtures';

const datasetFile = 'configs/datasets/studio.toml';
const imageFile = 'images/studio/photo.png';
const otherImageFile = 'images/studio/other.png';
const secondSubsetImageFile = 'images/studio-extra/cover.png';

const subsetImages = [
  [
    { file: imageFile, name: 'photo.png' },
    { file: otherImageFile, name: 'other.png' },
  ],
  [{ file: secondSubsetImageFile, name: 'cover.png' }],
];

type SaveAttempt = {
  ifMatch: string | undefined;
  contentType: string | undefined;
  bodyBytes: number;
};

type ImageRead = {
  datasetIndex: string | null;
  image: string | null;
};

type MaskWorkspaceOptions = {
  maskImageCount?: number;
  conflictFirstSave?: boolean;
  holdFirstApply?: boolean;
  conflictFirstApply?: boolean;
  firstApplyFailure?: 'conflict' | 'server' | 'offline';
  failPageReadUntilRetry?: boolean;
  failPresetReadUntilRetry?: boolean;
};

async function setupMaskWorkspace(page: Page, options: MaskWorkspaceOptions = {}) {
  const workspace = await mockWorkspace(page);
  const images = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#4c7b78';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const imageUrl = canvas.toDataURL('image/png');
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    return { imageUrl, maskUrl: canvas.toDataURL('image/png') };
  });
  const saves: SaveAttempt[] = [];
  const applyAttempts: Array<string | undefined> = [];
  const applyPayloads: Array<{ indices?: number[] }> = [];
  const imageReads: ImageRead[] = [];
  const pageReads: Array<{ datasetIndex: string | null; offset: string | null }> = [];
  const presetReads: string[] = [];
  let pageReadAvailable = !options.failPageReadUntilRetry;
  let presetReadAvailable = !options.failPresetReadUntilRetry;
  let announceApplyStarted = () => {};
  let releaseApplyResponse = () => {};
  const applyStarted = new Promise<void>(resolve => { announceApplyStarted = resolve; });
  const applyResponseGate = new Promise<void>(resolve => { releaseApplyResponse = resolve; });
  const reply = (route: Parameters<Parameters<Page['route']>[1]>[0], data: unknown, status = 200) =>
    route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(data),
    });
  const maskItemsFor = (datasetIndex: number) => {
    const selectedImages = subsetImages[datasetIndex] || [];
    if (datasetIndex !== 0 || !options.maskImageCount) return selectedImages;
    return Array.from({ length: options.maskImageCount }, (_, index) => ({
      file: `images/studio/image-${index}.png`,
      name: `image-${index}.png`,
    }));
  };

  await page.route(
    url => url.pathname.startsWith('/api/config/dataset-masks') ||
      url.pathname === '/api/config/dataset-presets/read' ||
      url.pathname === '/api/config/dataset-presets/images',
    async route => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      if (url.pathname === '/api/config/dataset-presets/read' && method === 'GET') {
        presetReads.push(url.searchParams.get('file') || '');
        if (!presetReadAvailable) {
          return reply(route, { ok: false, error: 'preset unavailable' }, 503);
        }
        return reply(route, {
          ok: true,
          file: datasetFile,
          name: 'studio',
          content: '',
          datasets: [
            { source_dir: 'images/studio', image_dir: 'cache/studio', num_repeats: 1, settings: {} },
            { source_dir: 'images/studio-extra', image_dir: 'cache/studio-extra', num_repeats: 1, settings: {} },
          ],
          defaults: { resolution: 1024, batch_size: 1 },
          readonly: false,
          summary: {},
        });
      }
      if (url.pathname === '/api/config/dataset-presets/images' && method === 'GET') {
        const datasetIndex = Number(url.searchParams.get('dataset_index') || 0);
        const selectedImages = subsetImages[datasetIndex] || [];
        return reply(route, {
          ok: true,
          file: datasetFile,
          dataset_index: datasetIndex,
          directory: datasetIndex === 0 ? 'images/studio' : 'images/studio-extra',
          row: { source_dir: datasetIndex === 0 ? 'images/studio' : 'images/studio-extra', num_repeats: 1 },
          settings: { resolution: 1024, enable_bucket: true },
          caption_source_label: 'TXT',
          caption_summary: '',
          total: selectedImages.length,
          images: selectedImages.map(item => ({
            ...item,
            url: images.imageUrl,
            thumbnail_url: images.imageUrl,
            caption: { text: '' },
          })),
        });
      }
      if (url.pathname === '/api/config/dataset-masks/image' && method === 'PUT') {
        const headers = route.request().headers();
        saves.push({
          ifMatch: headers['if-match'],
          contentType: headers['content-type'],
          bodyBytes: route.request().postDataBuffer()?.byteLength ?? 0,
        });
        if (options.conflictFirstSave && saves.length === 1) {
          return reply(route, { ok: false, error: 'mask revision conflict' }, 409);
        }
        return reply(route, { ok: true, revision: 'mask-rev-2' });
      }
      if (url.pathname === '/api/config/dataset-masks/apply' && method === 'POST') {
        const ifMatch = route.request().headers()['if-match'];
        applyAttempts.push(ifMatch);
        applyPayloads.push(route.request().postDataJSON() as { indices?: number[] });
        if (options.holdFirstApply && applyAttempts.length === 1) {
          announceApplyStarted();
          await applyResponseGate;
        }
        if (applyAttempts.length === 1 && options.firstApplyFailure === 'offline') {
          return route.abort();
        }
        if (applyAttempts.length === 1 && options.firstApplyFailure === 'server') {
          return reply(route, { ok: false, error: 'mask apply unavailable' }, 500);
        }
        if (options.conflictFirstApply !== false && applyAttempts.length === 1) {
          return reply(route, { ok: false, error: 'mask config revision conflict' }, 409);
        }
        return reply(route, { ok: true, message: 'masks applied' });
      }
      if (url.pathname === '/api/config/dataset-masks/image' && method === 'GET') {
        const datasetIndex = url.searchParams.get('dataset_index');
        const image = url.searchParams.get('image');
        imageReads.push({ datasetIndex, image });
        const selectedImage = maskItemsFor(Number(datasetIndex)).find(item => item.file === image);
        if (!selectedImage) return reply(route, { ok: false, error: 'mask image not found' }, 404);
        return reply(route, {
          ok: true,
          revision: 'mask-rev-1',
          width: 128,
          height: 128,
          image_url: images.imageUrl,
          mask_url: images.maskUrl,
          has_mask: false,
          readonly: false,
          basis: 'training',
          mask_dir: 'masks/studio',
          mask_file: `masks/studio/${selectedImage.name}`,
        });
      }
      if (url.pathname === '/api/config/dataset-masks' && method === 'GET') {
        pageReads.push({
          datasetIndex: url.searchParams.get('dataset_index'),
          offset: url.searchParams.get('offset'),
        });
        if (!pageReadAvailable) {
          return reply(route, { ok: false, error: 'mask page unavailable' }, 503);
        }
        const datasetIndex = Number(url.searchParams.get('dataset_index') || 0);
        const allImages = maskItemsFor(datasetIndex);
        const offset = Math.max(0, Number(url.searchParams.get('offset') || 0));
        const selectedImages = allImages.slice(offset, offset + 48);
        return reply(route, {
          ok: true,
          file: datasetFile,
          dataset_index: datasetIndex,
          dataset_label: datasetIndex === 0 ? 'Studio' : 'Studio Extra',
          source: 'source',
          source_label: '源图',
          directory: datasetIndex === 0 ? 'images/studio' : 'images/studio-extra',
          directory_exists: true,
          caption_extension: '.txt',
          prefer_json_caption: false,
          caption_source_mode: 'txt',
          caption_source_label: 'TXT',
          caption_summary: '',
          count: selectedImages.length,
          total: allImages.length,
          limit: 48,
          images: selectedImages.map(item => ({
            ...item,
            url: images.imageUrl,
            thumbnail_url: images.imageUrl,
            has_mask: false,
            caption: {
              ok: true,
              file: item.file.replace(/\.png$/, '.txt'),
              extension: '.txt',
              source_mode: 'txt',
              source_label: 'TXT',
              detected_mode: 'txt',
              format_label: 'TXT',
              caption_count: 0,
              text: '',
              truncated: false,
              length: 0,
            },
          })),
          row: { source_dir: datasetIndex === 0 ? 'images/studio' : 'images/studio-extra', mask_mode: 'external', mask_dir: 'masks/studio' },
          settings: {},
          message: '',
          mask_dir: 'masks/studio',
          mask_mode: 'external',
          config_revision: 'config-rev-1',
          readonly: false,
          offset,
          next_offset: offset + selectedImages.length,
          has_more_after: offset + selectedImages.length < allImages.length,
        });
      }
      return route.fallback();
    },
  );

  return {
    ...workspace,
    saves,
    applyAttempts,
    applyPayloads,
    imageReads,
    pageReads,
    presetReads,
    allowPageRead: () => { pageReadAvailable = true; },
    allowPresetRead: () => { presetReadAvailable = true; },
    applyStarted,
    releaseApplyResponse,
  };
}

async function openMaskWorkspace(page: Page) {
  await page.goto(`/next/datasets/workspace/masks?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);
  await expect(page.getByRole('heading', { name: '蒙版编辑' })).toBeVisible();
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute('aria-busy', 'false');
  return page.locator('canvas[aria-label="蒙版编辑画布"]');
}

test('dataset image preview renders the complete mock response without runtime errors', async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));

  await page.goto(`/next/datasets/workspace/preview?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);
  await expect(page.getByRole('heading', { name: '图片工作台' })).toBeVisible();
  await expect(page.locator('.dataset-preview-card')).toHaveCount(3);
  await expect(page.locator('.dataset-preview-details dd').nth(2)).toHaveText('images/studio');
  expect(pageErrors).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('dataset image preview exposes its pending state as a status', async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let announceStarted = () => {};
  let releaseResponse = () => {};
  const started = new Promise<void>(resolve => { announceStarted = resolve; });
  const responseGate = new Promise<void>(resolve => { releaseResponse = resolve; });
  await page.route(url => url.pathname === '/api/config/dataset-presets/images', async route => {
    announceStarted();
    await responseGate;
    return route.fallback();
  });

  await page.goto(`/next/datasets/workspace/preview?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);
  await started;
  await expect(page.getByRole('status')).toHaveText('正在读取图片与标注');

  releaseResponse();
  await expect(page.locator('.dataset-preview-card')).toHaveCount(3);
  await expect(page.getByRole('status')).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('dataset image preview read errors recover only after an explicit retry', async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let allowRead = false;
  let reads = 0;
  await page.route(url => url.pathname === '/api/config/dataset-presets/images', async route => {
    reads += 1;
    if (!allowRead) return route.fulfill({ status: 503, json: { ok: false, error: 'preview unavailable' } });
    return route.fallback();
  });

  await page.goto(`/next/datasets/workspace/preview?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);
  await expect(page.getByRole('heading', { name: '图片工作台' })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('无法读取数据集预览');
  await expect(page.getByRole('alert')).toContainText('preview unavailable');
  await expect(page.getByRole('button', { name: '重试', exact: true })).toBeEnabled();
  await expect(page.locator('.dataset-preview-card')).toHaveCount(0);
  const failedReadCount = reads;
  expect(failedReadCount).toBeGreaterThan(0);

  allowRead = true;
  await page.getByRole('button', { name: '重试', exact: true }).click();
  await expect(page.locator('.dataset-preview-card')).toHaveCount(3);
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(reads).toBe(failedReadCount + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

async function drawMaskStroke(page: Page, canvas: ReturnType<Page['locator']>) {
  const bounds = await canvas.boundingBox();
  if (!bounds) throw new Error('蒙版画布没有可见边界');
  const x = bounds.x + bounds.width / 2;
  const y = bounds.y + bounds.height / 2;
  await page.mouse.move(x - 24, y);
  await page.mouse.down();
  await page.mouse.move(x + 24, y, { steps: 3 });
  await page.mouse.up();
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');
}

test('workspace deep links without a dataset show empty states and a recovery path', async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const contextReads: string[] = [];
  await page.route(
    url => url.pathname === '/api/config/dataset-presets/read' || url.pathname.startsWith('/api/config/dataset-masks'),
    async route => {
      contextReads.push(`${route.request().method()} ${new URL(route.request().url()).pathname}`);
      return route.fallback();
    },
  );

  await page.goto('/next/datasets/workspace/preview');
  await expect(page.getByRole('heading', { name: '图片工作台' })).toBeVisible();
  await expect(page.locator('.dataset-image-workspace-context')).toContainText('未选择数据集 · 子集 1');
  await expect(page.locator('.dataset-empty')).toHaveText('未选择数据集');
  expect(contextReads).toEqual([]);

  await page.goto('/next/datasets/workspace/masks');
  await expect(page.getByRole('heading', { name: '蒙版编辑' })).toBeVisible();
  await expect(page.locator('.mask-empty')).toHaveText('未选择数据集');
  expect(contextReads).toEqual([]);
  await page.getByRole('button', { name: '返回数据集' }).click();
  await expect(page).toHaveURL(/\/next\/datasets\?dataset=$/);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('mask page read error retries into an editable workspace', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { failPageReadUntilRetry: true });
  await page.goto(`/next/datasets/workspace/masks?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);

  await expect(page.getByRole('heading', { name: '蒙版编辑' })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('mask page unavailable');
  await expect(page.getByRole('button', { name: '重试加载' })).toBeVisible();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toHaveCount(0);
  const failedReadCount = mocks.pageReads.length;
  expect(failedReadCount).toBeGreaterThan(0);

  mocks.allowPageRead();
  await page.getByRole('button', { name: '重试加载' }).click();
  const canvas = page.locator('canvas[aria-label="蒙版编辑画布"]');
  await expect(canvas).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.mask-filename')).toHaveText('photo.png');
  expect(mocks.pageReads).toHaveLength(failedReadCount + 1);
  expect(mocks.imageReads).toContainEqual({ datasetIndex: '0', image: imageFile });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('mask workspace keeps the canvas and commands unavailable while reads are pending', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page);
  let announcePageStarted = () => {};
  let releasePageResponse = () => {};
  let announceImageStarted = () => {};
  let releaseImageResponse = () => {};
  const pageStarted = new Promise<void>(resolve => { announcePageStarted = resolve; });
  const pageGate = new Promise<void>(resolve => { releasePageResponse = resolve; });
  const imageStarted = new Promise<void>(resolve => { announceImageStarted = resolve; });
  const imageGate = new Promise<void>(resolve => { releaseImageResponse = resolve; });
  await page.route(url => url.pathname === '/api/config/dataset-masks', async route => {
    if (route.request().method() !== 'GET') return route.fallback();
    announcePageStarted();
    await pageGate;
    return route.fallback();
  });
  await page.route(url => url.pathname === '/api/config/dataset-masks/image', async route => {
    if (route.request().method() !== 'GET') return route.fallback();
    announceImageStarted();
    await imageGate;
    return route.fallback();
  });

  await page.goto(`/next/datasets/workspace/masks?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);
  await expect(page.getByRole('heading', { name: '蒙版编辑' })).toBeVisible();
  await pageStarted;
  await expect(page.locator('.mask-empty')).toHaveText('正在加载图片');
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: '应用到子集' })).toBeDisabled();

  releasePageResponse();
  await imageStarted;
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute('aria-busy', 'true');
  await expect(page.locator('.mask-empty')).toHaveText('正在加载图片');
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toHaveCount(0);

  releaseImageResponse();
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('preset read error can be explicitly retried without losing the mask page', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { failPresetReadUntilRetry: true });
  await page.goto(`/next/datasets/workspace/masks?${new URLSearchParams({
    dataset: datasetFile,
    subset: '0',
  })}`);

  const canvas = page.locator('canvas[aria-label="蒙版编辑画布"]');
  await expect(page.getByRole('heading', { name: '蒙版编辑' })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('preset unavailable');
  await expect(page.getByRole('button', { name: '重试加载' })).toBeVisible();
  await expect(canvas).toBeVisible();
  await expect(page.locator('.mask-filename')).toHaveText('photo.png');
  const failedReadCount = mocks.presetReads.length;
  expect(failedReadCount).toBeGreaterThan(0);

  mocks.allowPresetRead();
  await page.getByRole('button', { name: '重试加载' }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(canvas).toBeVisible();
  expect(mocks.presetReads).toHaveLength(failedReadCount + 1);
  expect(mocks.presetReads.every(file => file === datasetFile)).toBe(true);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('mask pagination requests a nonzero offset and keeps the selected image identity', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { maskImageCount: 60 });
  await openMaskWorkspace(page);

  await expect(page.getByRole('button', { name: 'image-1.png' })).toBeVisible();
  await page.getByRole('button', { name: '下一页图片' }).click();
  await expect(page.getByRole('button', { name: 'image-48.png' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'image-1.png' })).toHaveCount(0);
  expect(mocks.pageReads).toContainEqual({ datasetIndex: '0', offset: '48' });

  await page.getByRole('button', { name: 'image-48.png' }).click();
  await expect(page.locator('.mask-filename')).toHaveText('image-48.png');
  expect(mocks.imageReads).toContainEqual({ datasetIndex: '0', image: 'images/studio/image-48.png' });

  await page.getByRole('button', { name: '上一页图片' }).click();
  await expect(page.getByRole('button', { name: 'image-1.png' })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('dirty mask workspace confirms before switching views', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page);
  const canvas = await openMaskWorkspace(page);
  await drawMaskStroke(page, canvas);

  const taggingTab = page.getByRole('tab', { name: '打标' });
  await taggingTab.click();
  const discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await expect(discard).toBeVisible();
  await expect(discard).toContainText('离开页面将丢弃这些修改。');
  await discard.getByRole('button', { name: '继续编辑' }).click();
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/masks\?/);
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');

  await taggingTab.click();
  await page.getByRole('dialog', { name: '放弃未保存修改？' })
    .getByRole('button', { name: '放弃修改并继续' }).click();
  await expect(page.locator('[data-view="tagging"]')).toBeVisible();
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/tagging\?/);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('dirty embedded mask confirms before returning to datasets', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page);
  const canvas = await openMaskWorkspace(page);
  await drawMaskStroke(page, canvas);

  const back = page.getByRole('button', { name: '返回数据集' });
  await back.click();
  const discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await expect(discard).toBeVisible();
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/masks\?/);
  await discard.getByRole('button', { name: '继续编辑' }).click();
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/masks\?/);

  await back.click();
  await page.getByRole('dialog', { name: '放弃未保存修改？' })
    .getByRole('button', { name: '放弃修改并继续' }).click();
  await expect(page).toHaveURL(/\/next\/datasets\?dataset=/);
  expect(mocks.saves).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('dirty mask confirms before switching images or subsets', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page);
  const canvas = await openMaskWorkspace(page);
  await drawMaskStroke(page, canvas);

  const otherImage = page.getByRole('button', { name: 'other.png' });
  await otherImage.click();
  let discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await expect(discard).toBeVisible();
  await discard.getByRole('button', { name: '继续编辑' }).click();
  await expect(page.getByRole('button', { name: 'photo.png' })).toHaveAttribute('aria-current', 'true');
  await expect(otherImage).not.toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.mask-filename')).toHaveText('photo.png');
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');

  await otherImage.click();
  discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await discard.getByRole('button', { name: '放弃修改并继续' }).click();
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute('aria-busy', 'false');
  await expect(otherImage).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.mask-filename')).toHaveText('other.png');
  expect(mocks.imageReads).toContainEqual({ datasetIndex: '0', image: otherImageFile });

  await drawMaskStroke(page, canvas);
  const subset = page.getByLabel('蒙版子集');
  await subset.selectOption('1');
  discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await expect(discard).toBeVisible();
  await discard.getByRole('button', { name: '继续编辑' }).click();
  await expect(subset).toHaveValue('0');
  await expect(page.locator('.mask-filename')).toHaveText('other.png');
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');

  await subset.selectOption('1');
  discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await discard.getByRole('button', { name: '放弃修改并继续' }).click();
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute('aria-busy', 'false');
  await expect(subset).toHaveValue('1');
  await expect(page.locator('.mask-filename')).toHaveText('cover.png');
  await expect(page.getByRole('button', { name: 'cover.png' })).toHaveAttribute('aria-current', 'true');
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/masks\?.*subset=1/);
  await page.getByRole('tab', { name: '预览' }).click();
  await expect(page.locator('.dataset-image-workspace-context')).toContainText('子集 2');
  await page.getByRole('tab', { name: '打标' }).click();
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/tagging\?.*subset=1/);
  await page.getByRole('tab', { name: '编辑蒙版' }).click();
  await expect(subset).toHaveValue('1');
  expect(mocks.imageReads).toContainEqual({ datasetIndex: '1', image: secondSubsetImageFile });
  expect(mocks.saves).toEqual([]);
  expect(mocks.applyAttempts).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('mask save conflict keeps the drawing and retries with its original revision', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { conflictFirstSave: true });
  const canvas = await openMaskWorkspace(page);
  await drawMaskStroke(page, canvas);

  const save = page.getByRole('button', { name: '保存', exact: true });
  await save.click();
  await expect(page.getByRole('alert')).toContainText('mask revision conflict');
  await expect(page.locator('.mask-save-state')).toHaveText('未保存');
  await expect(save).toBeEnabled();

  await save.click();
  await expect(page.locator('.mask-save-state')).toHaveText('蒙版已保存');
  await expect(save).toBeDisabled();
  expect(mocks.saves).toHaveLength(2);
  expect(mocks.saves.map(attempt => attempt.ifMatch)).toEqual(['mask-rev-1', 'mask-rev-1']);
  expect(mocks.saves.every(attempt => attempt.contentType === 'image/png')).toBe(true);
  expect(mocks.saves.every(attempt => attempt.bodyBytes > 0)).toBe(true);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('applying masks locks while pending and allows retry after a conflict', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { holdFirstApply: true });
  const canvas = await openMaskWorkspace(page);
  await drawMaskStroke(page, canvas);
  await page.getByRole('button', { name: '保存', exact: true }).click();
  await expect(page.locator('.mask-save-state')).toHaveText('蒙版已保存');

  await page.getByRole('button', { name: '应用到子集' }).click();
  const dialog = page.getByRole('dialog', { name: '应用蒙版到此子集？' });
  const confirm = dialog.getByRole('button', { name: '确认应用' });
  await confirm.click();
  await mocks.applyStarted;
  await expect(confirm).toBeDisabled();
  await expect(dialog.getByRole('button', { name: '取消' })).toBeDisabled();
  expect(mocks.applyAttempts).toHaveLength(1);

  mocks.releaseApplyResponse();
  await expect(dialog.getByRole('alert')).toContainText('mask config revision conflict');
  await expect(confirm).toBeEnabled();
  await confirm.click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText('已应用，后续启动的训练生效', { exact: true })).toBeVisible();
  expect(mocks.applyAttempts).toEqual(['config-rev-1', 'config-rev-1']);
  expect(mocks.applyPayloads).toEqual([{ indices: [0] }, { indices: [0] }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test('applies the current mask directory only to selected subsets', async ({ page }) => {
  const mocks = await setupMaskWorkspace(page, { conflictFirstApply: false });
  await openMaskWorkspace(page);
  await page.getByRole('button', { name: '应用到子集' }).click();
  const dialog = page.getByRole('dialog', { name: '应用蒙版到此子集？' });
  await dialog.getByRole('checkbox', { name: /2\. images\/studio-extra/ }).check();
  await dialog.getByRole('button', { name: '确认应用到 2 个子集' }).click();
  await expect(dialog).not.toBeVisible();
  expect(mocks.applyAttempts).toEqual(['config-rev-1']);
  expect(mocks.applyPayloads).toEqual([{ indices: [0, 1] }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const failure of ['server', 'offline'] as const) {
  test(`mask apply ${failure} failure stays open until explicit retry`, async ({ page }) => {
    const mocks = await setupMaskWorkspace(page, { holdFirstApply: true, firstApplyFailure: failure });
    const canvas = await openMaskWorkspace(page);
    await drawMaskStroke(page, canvas);
    await page.getByRole('button', { name: '保存', exact: true }).click();
    await expect(page.locator('.mask-save-state')).toHaveText('蒙版已保存');

    await page.getByRole('button', { name: '应用到子集' }).click();
    const dialog = page.getByRole('dialog', { name: '应用蒙版到此子集？' });
    const confirm = dialog.getByRole('button', { name: '确认应用' });
    await confirm.click();
    await mocks.applyStarted;
    await expect(confirm).toBeDisabled();
    await expect(dialog.getByRole('button', { name: '取消' })).toBeDisabled();
    expect(mocks.applyAttempts).toHaveLength(1);

    mocks.releaseApplyResponse();
    await expect(dialog.getByRole('alert')).toContainText(
      failure === 'offline' ? '连接中断，操作结果尚未确认' : 'mask apply unavailable',
    );
    await expect(confirm).toBeEnabled();
    expect(mocks.applyAttempts).toHaveLength(1);

    await confirm.click();
    await expect(dialog).not.toBeVisible();
    await expect(page.getByText('已应用，后续启动的训练生效', { exact: true })).toBeVisible();
    expect(mocks.applyAttempts).toEqual(['config-rev-1', 'config-rev-1']);
    expect(mocks.saves).toHaveLength(1);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

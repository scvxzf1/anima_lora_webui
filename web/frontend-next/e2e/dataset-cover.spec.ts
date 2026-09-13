import { test, expect } from '@playwright/test';
import { mockWorkspace } from './fixtures';

test('dataset cover renders square and explains missing images', async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  let requests = 0;
  await page.route('**/api/config/dataset-presets/cover?*', route => {
    requests++;
    return route.fulfill({ json: { ok: true, image: null, reason: '图片目录不存在或无法访问' } });
  });
  await page.goto('/next/datasets');
  const cover = page.locator('.dataset-cover').first();
  await expect(cover).toContainText('无图像');
  expect(requests).toBe(1);
  const box = await cover.boundingBox();
  expect(box!.width).toBe(box!.height);
  await cover.hover();
  await expect(page.getByRole('tooltip')).toBeVisible();
  await page.screenshot({ path: info.outputPath('cover-missing-desktop.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: info.outputPath('cover-missing-mobile.png') });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(mocks.writes).toEqual([]);
});

test('dataset cover displays an image without loading the full gallery', async ({ page }, info) => {
  await mockWorkspace(page);
  const paths: string[] = [];
  page.on('request', request => paths.push(new URL(request.url()).pathname));
  await page.goto('/next/datasets');
  const image = page.locator('.dataset-cover img').first();
  await expect(image).toBeVisible();
  expect(await image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  expect(paths).not.toContain('/api/config/dataset-presets/images');
  await page.screenshot({ path: info.outputPath('cover-ready-desktop.png') });
});

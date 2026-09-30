import { expect, test } from '@playwright/test';

import { mockWorkspace } from './fixtures';

for (const width of [1440, 390]) {
  test(`dataset source path actions stay within ${width}px viewport`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await mockWorkspace(page);
    const inspected: string[] = [];
    await page.route('**/api/config/data-dirs/suggest?*', async route => {
      inspected.push(route.request().url());
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ source_exists: true, source_is_dir: true, source_image_count: 8 }),
      });
    });
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: { writeText: async () => undefined },
      });
      Object.defineProperty(window, 'showDirectoryPicker', {
        configurable: true,
        value: async () => ({ name: 'picked-folder' }),
      });
    });

    await page.goto('/next/datasets');
    const source = page.getByLabel('原始图片目录', { exact: true });
    await expect(source).toHaveValue('images/studio');
    await expect(page.getByText('检测到 8 张图片').first()).toBeVisible();
    await source.fill('images/new-draft');
    await expect(page.getByText('检测到 8 张图片').first()).toBeVisible();
    await page.getByRole('button', { name: '选择本机文件夹名' }).first().click();
    await expect(source).toHaveValue('picked-folder');
    await page.getByRole('button', { name: '复制目录路径' }).first().click();
    await expect(page.getByText('目录路径已复制').first()).toBeVisible();
    await page.screenshot({ path: info.outputPath(`dataset-source-path-${width}.png`) });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect(inspected.some(url => url.includes('images%2Fnew-draft'))).toBe(true);
  });
}

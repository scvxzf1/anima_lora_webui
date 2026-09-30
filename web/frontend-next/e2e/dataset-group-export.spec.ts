import { expect, test } from '@playwright/test';
import { mockWorkspace } from './fixtures';

test('dataset group export downloads the dataset archive without layout overflow', async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  const groupId = 'system/角色 +';
  const exports: string[] = [];
  await page.route((url) => url.pathname === '/api/config/dataset-presets' && !url.search, (route) =>
    route.fulfill({ json: {
      ok: true,
      presets: [{
        path: 'configs/datasets/readonly.toml', filename: 'readonly.toml', label: 'Readonly',
        readonly: true, summary: { dataset_count: 1, repeat_total: 1 },
      }],
      groups: [{
        id: groupId, label: '系统角色', kind: 'dataset', locked: true, system: true,
        files: [{
          path: 'configs/datasets/readonly.toml', filename: 'readonly.toml', label: 'Readonly',
          readonly: true, summary: { dataset_count: 1, repeat_total: 1 },
        }],
      }],
    } }),
  );
  await page.route((url) => url.pathname.includes('/api/config/file-groups/') && url.pathname.endsWith('/export'), (route) => {
    exports.push(route.request().url());
    return route.fulfill({ status: 200, contentType: 'application/zip', body: 'mock zip' });
  });

  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/next/datasets');
    const button = page.getByRole('button', { name: '导出分组 系统角色' });
    await expect(button).toBeVisible();
    await expect(button).toBeEnabled();
    const downloadPromise = page.waitForEvent('download');
    await button.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('dataset-group.zip');
    await expect.poll(() => exports.length).toBeGreaterThan(0);
    const requestUrl = new URL(exports.at(-1)!);
    expect(requestUrl.pathname).toContain(encodeURIComponent(groupId));
    expect(requestUrl.searchParams.get('kind')).toBe('dataset');
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`dataset-group-export-${width}.png`), fullPage: true });
  }

  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

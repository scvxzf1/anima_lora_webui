import { test, expect } from '@playwright/test';
import { mockWorkspace } from './fixtures';

for (const viewport of [{ width: 1911, height: 1446 }, { width: 1280, height: 720 }, { width: 390, height: 844 }]) {
  test(`dataset panes scroll independently at ${viewport.width}`, async ({ page }, info) => {
    const mocks = await mockWorkspace(page);
    await page.route('**/api/config/dataset-presets', route => {
      const presets = Array.from({ length: 20 }, (_, index) => ({
        path: index === 0 ? 'configs/datasets/studio.toml' : `configs/datasets/scroll-${index}.toml`, label: `Scroll ${index}`, summary: {},
      }));
      return route.fulfill({ json: { ok: true, presets, groups: [{ id: 'scroll', label: 'Scroll', files: presets }] } });
    });
    await page.setViewportSize(viewport);
    await page.goto('/next/datasets');
    await expect(page.getByRole('button', { name: '添加子集', exact: true })).toBeVisible();
    await page.getByText('高级规则', { exact: true }).click();
    const library = page.locator('.dataset-library');
    const detail = page.locator('.dataset-detail');
    const dimensions = await page.evaluate(() => ({
      height: document.documentElement.scrollHeight, width: document.documentElement.scrollWidth,
      viewportHeight: innerHeight, viewportWidth: innerWidth,
    }));
    expect(dimensions.height).toBeLessThanOrEqual(dimensions.viewportHeight + 1);
    expect(dimensions.width).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
    for (const pane of [library, detail]) {
      expect(await pane.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true);
      expect(await pane.evaluate(el => getComputedStyle(el).overflowY)).toBe('auto');
    }
    const detailBefore = await detail.evaluate(el => el.scrollTop);
    await library.hover();
    await page.mouse.wheel(0, 400);
    await expect.poll(() => library.evaluate(el => el.scrollTop)).toBeGreaterThan(0);
    expect(await detail.evaluate(el => el.scrollTop)).toBe(detailBefore);
    const libraryBefore = await library.evaluate(el => el.scrollTop);
    await detail.hover();
    await page.mouse.wheel(0, 400);
    await expect.poll(() => detail.evaluate(el => el.scrollTop)).toBeGreaterThan(detailBefore);
    expect(await library.evaluate(el => el.scrollTop)).toBe(libraryBefore);
    expect(await page.evaluate(() => scrollY)).toBe(0);
    await page.screenshot({ path: info.outputPath('independent-scroll.png') });
    expect(mocks.writes).toEqual([]);
  });
}

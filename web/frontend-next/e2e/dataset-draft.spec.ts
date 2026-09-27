import { test, expect } from '@playwright/test';
import { mockWorkspace } from './fixtures';

for (const width of [1440, 390]) {
  test(`dataset dirty save protects image workspace entry ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    let nativeDialogs = 0;
    page.on('dialog', async dialog => { nativeDialogs++; await dialog.dismiss(); });
    await page.goto('/next/datasets');
    const source = page.getByLabel('原始图片目录', { exact: true });
    const save = page.getByRole('button', { name: '保存', exact: true });
    await expect(source).toHaveValue('images/studio');
    await expect(save).toBeDisabled();
    await source.fill('images/changed');
    await expect(save).toBeEnabled();
    await source.fill('images/studio');
    await expect(save).toBeDisabled();
    await source.fill('images/changed');
    await save.click();
    await expect(page.getByRole('alert')).toContainText('Fixture blocked command');
    await expect(page.getByRole('alert')).toContainText('请核对后使用“重新读取”');
    await expect(save).toBeDisabled();
    const reload = page.getByRole('button', { name: '重新读取', exact: true });
    await reload.click();
    const dialog = page.getByRole('dialog', { name: '放弃未保存修改？' });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('button', { name: '继续编辑' })).toBeFocused();
    await page.screenshot({ path: info.outputPath('dataset-discard.png') });
    expect(await dialog.evaluate(node => {
      const box = node.getBoundingClientRect();
      return box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    await page.keyboard.press('Escape');
    await expect(reload).toBeFocused();
    await expect(source).toHaveValue('images/changed');
    const openWorkbench = page.getByRole('button', {
      name: '打开子集 1 图片工作台',
    });
    await expect(openWorkbench).toBeDisabled();
    await expect(openWorkbench).toHaveAttribute(
      'title',
      '请先保存当前预设，并确保没有未保存修改',
    );
    await expect(source).toHaveValue('images/changed');
    await expect(page).toHaveURL(/\/next\/datasets$/);
    expect(nativeDialogs).toBe(0);
    expect(mocks.writes).toHaveLength(1);
  });
}

test('dataset save conflict preserves edits until reload supplies a fresh revision', async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const saveAttempts: Array<Record<string, unknown>> = [];
  let conflictObserved = false;
  await page.route(url => url.pathname === '/api/config/dataset-presets/read', async route => {
    if (!conflictObserved) return route.fallback();
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        file: 'configs/datasets/studio.toml',
        name: 'studio',
        content: 'source_dir = "images/server-version"',
        revision: 'revision-2',
        datasets: [{ source_dir: 'images/server-version', image_dir: 'cache/studio', num_repeats: 1, settings: {} }],
        defaults: { resolution: 1024, batch_size: 1 },
        readonly: false,
        summary: { dataset_count: 1, repeat_total: 1 },
      }),
    });
  });
  await page.route('**/api/config/dataset-presets', async route => {
    if (route.request().method() !== 'PUT') return route.fallback();
    const attempt = route.request().postDataJSON() as Record<string, unknown>;
    saveAttempts.push(attempt);
    if (saveAttempts.length === 1) {
      conflictObserved = true;
      return route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({ ok: false, error: 'dataset revision conflict' }),
      });
    }
    const payload = attempt as {
      file: string;
      datasets: unknown[];
      defaults: Record<string, unknown>;
    };
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        file: payload.file,
        message: 'saved with fresh revision',
        content: '',
        revision: 'revision-3',
        datasets: payload.datasets,
        defaults: payload.defaults,
        summary: { dataset_count: 1, repeat_total: 1 },
      }),
    });
  });

  await page.goto('/next/datasets');
  const source = page.getByLabel('原始图片目录', { exact: true });
  const save = page.getByRole('button', { name: '保存', exact: true });
  await source.fill('images/changed');
  await save.click();

  await expect(page.getByRole('alert')).toContainText('dataset revision conflict');
  await expect(page.getByRole('alert')).toContainText('请核对后使用“重新读取”');
  await expect(source).toHaveValue('images/changed');
  await expect(page.locator('.dataset-detail-status')).toContainText('有未保存修改');
  await expect(save).toBeDisabled();

  const reload = page.getByRole('button', { name: '重新读取', exact: true });
  await reload.click();
  const discard = page.getByRole('dialog', { name: '放弃未保存修改？' });
  await discard.getByRole('button', { name: '继续编辑' }).click();
  await expect(source).toHaveValue('images/changed');
  await expect(save).toBeDisabled();

  await reload.click();
  await page.getByRole('dialog', { name: '放弃未保存修改？' })
    .getByRole('button', { name: '放弃修改并继续' }).click();
  await expect(source).toHaveValue('images/server-version');
  await expect(save).toBeDisabled();
  await expect(page.getByRole('alert')).toHaveCount(0);

  await source.fill('images/after-reload');
  await expect(save).toBeEnabled();
  await save.click();
  await expect(page.getByText('saved with fresh revision', { exact: true })).toBeVisible();
  expect(saveAttempts).toHaveLength(2);
  expect(saveAttempts[0]).toMatchObject({ datasets: [{ source_dir: 'images/changed' }] });
  expect(saveAttempts[1]).toMatchObject({
    revision: 'revision-2',
    datasets: [{ source_dir: 'images/after-reload' }],
  });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

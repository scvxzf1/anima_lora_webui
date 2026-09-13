import { test, expect } from '@playwright/test';
import { mockWorkspace } from './fixtures';

for (const width of [1440, 390]) {
  test(`dataset dirty save and internal discard dialog ${width}`, async ({ page }, info) => {
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
    await expect(save).toBeEnabled();
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
    await page.getByRole('link', { name: '打开此数据集的打标工作台' }).click();
    await expect(dialog).toBeVisible();
    await dialog.getByRole('button', { name: '继续编辑' }).click();
    await expect(page).toHaveURL(/\/next\/datasets$/);
    await page.getByRole('link', { name: '打开此数据集的打标工作台' }).click();
    await dialog.getByRole('button', { name: '放弃修改并继续' }).click();
    await expect(page).toHaveURL(/\/next\/captioning/);
    expect(nativeDialogs).toBe(0);
    expect(mocks.writes).toHaveLength(1);
  });
}

import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

test("training group export downloads the server archive and reports failures", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let fail = false;
  await page.route("**/api/config/file-groups/imported/export?kind=training", (route) =>
    fail
      ? route.fulfill({ status: 400, json: { ok: false, error: "分组文件不可读取" } })
      : route.fulfill({
          contentType: "application/zip",
          headers: { "Content-Disposition": "attachment; filename*=UTF-8''training-group.zip" },
          body: "PK",
        }),
  );
  await page.route("**/api/config/file-groups?kind=training", (route) => route.fulfill({ json: [
    { id: "imported", label: "Studio presets", files: [configFile] },
    { id: "readonly", label: "只读分组", readonly: true, files: [{ ...configFile, path: "configs/gui-methods/locked.toml", readonly: true }] },
    { id: "empty", label: "空分组", files: [] },
  ] }));
  await page.goto("/next/training");
  await expect(page.getByRole("switch", { name: "详细管理" })).not.toBeChecked();
  await expect(page.getByRole("button", { name: "导出分组 只读分组" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "导出分组 空分组" })).toBeDisabled();
  const exportButton = page.getByRole("button", { name: "导出分组 Studio presets" });
  const [download] = await Promise.all([page.waitForEvent("download"), exportButton.click()]);
  expect(download.suggestedFilename()).toBe("training-group.zip");
  fail = true;
  await exportButton.click();
  await expect(page.getByRole("alert")).toContainText("分组文件不可读取");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training library management is opt-in and persists", async ({ page }) => {
  await mockWorkspace(page);
  await page.goto("/next/training");
  const library = page.getByRole("complementary", { name: "训练配置库" });
  const toggle = library.getByRole("switch", { name: "详细管理" });
  await expect(toggle).not.toBeChecked();
  await expect(library.getByRole("button", { name: "新建分组" })).toBeVisible();
  await expect(library.locator(".training-library-item").first()).toBeVisible();
  await expect(library.getByRole("button", { name: "拖动排序 Studio portrait" })).toHaveCount(0);
  await expect(library.getByRole("button", { name: "重命名当前分组" })).toHaveCount(0);

  await library.getByText("详细管理", { exact: true }).click();
  await expect(toggle).toBeChecked();
  await expect(library.getByRole("button", { name: "拖动排序 Studio portrait" })).toBeVisible();
  await page.reload();
  await expect(library.getByRole("switch", { name: "详细管理" })).toBeChecked();
  await library.getByText("详细管理", { exact: true }).click();
  await expect(library.getByRole("button", { name: "拖动排序 Studio portrait" })).toHaveCount(0);
});

test("a short click selects while a 200ms hold drags with management hidden", async ({ page }) => {
  await mockWorkspace(page);
  const second = { ...configFile, path: "configs/imported/second.toml", label: "Second" };
  let files = [configFile, second];
  const writes: unknown[] = [];
  await page.route("**/api/config/file-groups?kind=training", (route) =>
    route.fulfill({ json: [{ id: "imported", label: "导入配置", files }] }),
  );
  await page.route("**/api/config/file-groups/place", async (route) => {
    writes.push(route.request().postDataJSON());
    files = [second, configFile];
    await route.fulfill({ json: { ok: true } });
  });
  await page.goto("/next/training");
  const selected = page.locator(".training-library-item", { hasText: "Studio portrait" });
  const source = page.locator(".training-library-item", { hasText: "Second" });
  const target = page.locator(".training-library-row", { hasText: "Studio portrait" });
  await expect(page.getByRole("switch", { name: "详细管理" })).not.toBeChecked();
  await expect(source).toHaveAttribute("data-hold-drag", "true");
  await source.click();
  await expect(source).toHaveAttribute("data-selected", "true");
  await selected.click();
  await expect(selected).toHaveAttribute("data-selected", "true");
  await expect(page.locator(".training-library-drag-overlay")).toHaveCount(0);
  expect(writes).toEqual([]);

  const start = (await source.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(100);
  await expect(page.locator(".training-library-drag-overlay")).toHaveCount(0);
  await page.waitForTimeout(120);
  await expect(page.locator(".training-library-drag-overlay")).toBeVisible();
  const end = (await target.boundingBox())!;
  await page.mouse.move(end.x + end.width / 2, end.y + end.height * 0.25, { steps: 12 });
  await expect(target).toHaveAttribute("data-drop-position", "before");
  await page.mouse.up();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toEqual({ target: "file", file: second.path, group: "imported", index: 0 });
  await expect(selected).toHaveAttribute("data-selected", "true");
});

test("library collapse, pointer ordering and clean save state", async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockWorkspace(page);
  const second = {
    ...configFile,
    path: "configs/imported/second.toml",
    label: "Second",
  };
  let files = [configFile, second];
  const writes: unknown[] = [];
  await page.route("**/api/config/file-groups?kind=training", (route) =>
    route.fulfill({
      json: [
        { id: "imported", label: "导入配置", files },
        {
          id: "gui_methods",
          label: "可训练方法变体",
          files: [
            {
              ...configFile,
              path: "configs/gui-methods/lora.toml",
              label: "LoRA variant",
              locked: true,
            },
          ],
        },
      ],
    }),
  );
  await page.route("**/api/config/file-groups/place", async (route) => {
    writes.push(route.request().postDataJSON());
    files = [second, configFile];
    await route.fulfill({ json: { ok: true } });
  });
  await page.goto("/next/training");
  await page.getByText("详细管理", { exact: true }).click();
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  const save = page.getByRole("button", { name: "保存配置", exact: true });
  const output = page.getByRole("textbox", { name: "输出名称", exact: true });
  await expect(output).toHaveValue("studio-portrait");
  await expect(save).toBeDisabled();
  expect(await save.evaluate((node) => getComputedStyle(node).backgroundColor)).not.toBe(
    await page.getByRole("button", { name: "立即启动", exact: true }).evaluate((node) => getComputedStyle(node).backgroundColor),
  );
  await output.fill("changed");
  await expect(save).toBeEnabled();
  await output.fill("studio-portrait");
  await expect(save).toBeDisabled();
  const group = page.locator(".training-library-group-toggle", { hasText: "可训练方法变体" });
  await expect(group).toHaveAttribute("aria-expanded", "false");
  await group.click();
  await expect(
    page.locator(".training-library-item", { hasText: "LoRA variant" }),
  ).toBeVisible();
  await group.click();
  const handle = page.getByRole("button", { name: "拖动排序 Studio portrait" });
  const target = page.getByRole("button", { name: "拖动排序 Second" });
  const start = (await handle.boundingBox())!;
  const targetRow = page.locator(".training-library-row").filter({ has: target });
  const end = (await targetRow.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(end.x + end.width / 2, end.y + end.height * 0.75, {
    steps: 10,
  });
  await expect(targetRow).toHaveAttribute("data-drop-position", "after");
  await page.mouse.up();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toEqual({
    target: "file",
    file: configFile.path,
    group: "imported",
    index: 1,
  });
  await expect(page.locator(".training-library-item strong").first()).toHaveText(
    "Second",
  );
  await expect(save).toBeDisabled();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: info.outputPath("library-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("complementary", { name: "训练配置库" })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: info.outputPath("library-mobile.png") });
});

test("keyboard sorting failure retains server order", async ({ page }) => {
  await mockWorkspace(page);
  await page.route("**/api/config/file-groups?kind=training", (route) =>
    route.fulfill({
      json: [
        {
          id: "imported",
          label: "导入配置",
          files: [
            configFile,
            {
              ...configFile,
              path: "configs/imported/second.toml",
              label: "Second",
            },
          ],
        },
      ],
    }),
  );
  let requests = 0;
  await page.route("**/api/config/file-groups/place", (route) => {
    requests++;
    return route.fulfill({ status: 400, json: { error: "排序失败" } });
  });
  await page.goto("/next/training");
  await page.getByText("详细管理", { exact: true }).click();
  const handle = page.getByRole("button", { name: "拖动排序 Studio portrait" });
  await handle.focus();
  await page.keyboard.press("Space");
  await expect(handle).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("status")).toContainText("over droppable area configs/imported/second.toml");
  await page.keyboard.press("Space");
  await expect(page.getByRole("alert")).toHaveText("排序失败");
  expect(requests).toBe(1);
  await expect(page.locator(".training-library-item strong").first()).toHaveText(
    "Studio portrait",
  );
});

test("keyboard sorting persists a downward move", async ({ page }) => {
  await mockWorkspace(page);
  const second = { ...configFile, path: "configs/imported/second.toml", label: "Second" };
  let files = [configFile, second];
  const requests: unknown[] = [];
  await page.route("**/api/config/file-groups?kind=training", (route) => route.fulfill({
    json: [{ id: "imported", label: "导入配置", files }],
  }));
  await page.route("**/api/config/file-groups/place", async (route) => {
    requests.push(route.request().postDataJSON());
    files = [second, configFile];
    await route.fulfill({ json: { ok: true } });
  });
  await page.goto("/next/training");
  await page.getByText("详细管理", { exact: true }).click();
  const handle = page.getByRole("button", { name: "拖动排序 Studio portrait" });
  await handle.focus();
  await page.keyboard.press("Space");
  await expect(handle).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("status")).toContainText("over droppable area configs/imported/second.toml");
  await page.keyboard.press("Space");
  await expect(page.locator(".training-library-item strong").first()).toHaveText("Second");
  expect(requests).toEqual([{ target: "file", file: configFile.path, group: "imported", index: 1 }]);
});

import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

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
  const group = page.getByRole("button", { name: /可训练方法变体/ });
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

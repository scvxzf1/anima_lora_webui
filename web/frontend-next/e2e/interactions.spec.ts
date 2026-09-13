import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const width of [1440, 390]) {
  test(`command dialogs preserve focus and do not submit ${width}`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    await page.goto("/next/training");
    const saveAs = page.getByRole("button", { name: "另存配置", exact: true });
    await saveAs.click();
    await expect(
      page.getByRole("dialog", { name: "另存训练配置" }),
    ).toBeVisible();
    await expect(page.getByLabel("配置名称", { exact: true })).toBeFocused();
    await page.screenshot({ path: info.outputPath("save-as.png") });
    await page.keyboard.press("Escape");
    await expect(saveAs).toBeFocused();
    await page.getByRole("button", { name: "立即启动", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "确认启动训练" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText("Fixture ready")).toBeVisible();
    await expect(
      dialog.getByRole("button", { name: "确认启动", exact: true }),
    ).toBeDisabled();
    await page.keyboard.press("Shift+Tab");
    expect(
      await dialog.evaluate((node) => node.contains(document.activeElement)),
    ).toBe(true);
    await page.screenshot({ path: info.outputPath("launch.png") });
    await page.keyboard.press("Escape");
    expect(mocks.writes.map((entry) => entry.path)).toEqual([
      "/api/training/preflight",
    ]);
    expect(mocks.unhandled).toEqual([]);
  });

  test(`caption providers assets sources and history ${width}`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    for (const path of ["providers", "prompts", "assets"]) {
      await page.goto(`/next/captioning/${path}`);
      await expect(
        page.getByRole("heading", { name: "打标工作台" }),
      ).toBeVisible();
      if (path === "providers") {
        await page.getByRole("button", { name: "编辑", exact: true }).click();
        await expect(page.getByRole("dialog")).toBeVisible();
        await expect(page.locator('input[type="password"]')).toHaveValue("");
      }
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        )
        .toBe(true);
      await page.screenshot({ path: info.outputPath(`caption-${path}.png`) });
      if (path === "providers") await page.keyboard.press("Escape");
    }
    await page.goto("/next/captioning");
    await page
      .getByRole("combobox", { name: "数据集预设", exact: true })
      .selectOption("configs/datasets/studio.toml");
    await page.getByRole("button", { name: "扫描图片", exact: true }).click();
    await expect(page.locator(".caption-image-grid img").first()).toBeVisible();
    await page.screenshot({ path: info.outputPath("caption-source.png") });
    await page.goto("/next/history/fixture-run?view=artifacts");
    await expect(
      page.getByText("model.safetensors", { exact: true }),
    ).toBeVisible();
    await expect(page.locator("img").first()).toBeVisible();
    const status = page.locator(".history-detail-state");
    await expect(status).toHaveText("完成");
    expect(
      await status.evaluate((node) => {
        const range = document.createRange();
        range.selectNodeContents(node);
        return range.getClientRects().length;
      }),
    ).toBe(1);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({ path: info.outputPath("history-artifacts.png") });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("saved display preferences and zoom apply without double scaling", async ({
  page,
}, info) => {
  await mockWorkspace(page);
  await page.route("**/api/settings/global", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        ui_scale: 150,
        ui_scale_config: 200,
        dragon_motion_enabled: false,
        dragon_config_help_always_visible: true,
        dragon_config_tags_always_visible: true,
      }),
    }),
  );
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/next/training");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByLabel("输出名称", { exact: true })).toHaveValue(
    "studio-portrait",
  );
  await expect(page.locator("html")).toHaveAttribute("data-next-motion", "off");
  await expect(page.locator(".training-field-help").first()).toBeVisible();
  const zooms = await page.evaluate(() =>
    [".next-layout", ".next-content"].map((selector) =>
      Number(getComputedStyle(document.querySelector(selector)!).zoom),
    ),
  );
  expect(zooms[0] * zooms[1]).toBeCloseTo(2);
  await page.screenshot({ path: info.outputPath("training-ui-scale-200.png") });
  await page.getByRole("button", { name: "另存配置", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByLabel("配置名称", { exact: true })).toBeVisible();
  await expect(dialog.getByRole("button", { name: "确认另存" })).toBeVisible();
  await page.screenshot({ path: info.outputPath("dialog-ui-scale-200.png") });
});

test("effective 200 percent browser layout remains accessible", async ({
  browser,
  baseURL,
}, info) => {
  const context = await browser.newContext({
    viewport: { width: 720, height: 450 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  await mockWorkspace(page);
  await page.goto(`${baseURL}/next/training`);
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByLabel("输出名称", { exact: true })).toHaveValue(
    "studio-portrait",
  );
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  await page.screenshot({
    path: info.outputPath("training-effective-browser-zoom-200.png"),
  });
  await context.close();
});

test("local tag controls and dictionary translation keep writes explicit", async ({
  page,
}, info) => {
  const mocks = await mockWorkspace(page);
  await page.route("**/api/captioning/profiles", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        active_profile_id: "local",
        provider_types: [{ id: "cltagger", label: "CLTagger", kind: "local" }],
        profiles: [
          {
            id: "local",
            name: "Local fixture",
            provider: "cltagger",
            kind: "local",
            available: true,
            status: "ready",
            config: {
              asset_id: "fixture",
              general_threshold: 0.35,
              character_threshold: 0.6,
              blacklist: ["text"],
              add_copyright_tag: true,
            },
          },
        ],
      }),
    }),
  );
  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "排除标签", exact: true }),
  ).toHaveValue("text");
  await expect(page.getByRole("checkbox", { name: "作品标签" })).toBeChecked();
  await expect(
    page.getByRole("checkbox", { name: "画师标签" }),
  ).not.toBeChecked();
  await page.screenshot({ path: info.outputPath("local-profile.png") });
  await page.keyboard.press("Escape");
  await page.route("**/api/captioning/translate-tags", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        translations: ["几何构图"],
        matched: 1,
        total: 1,
      }),
    }),
  );
  await page.goto("/next/captioning?job=caption-1");
  const original = await page.getByLabel("候选标注").inputValue();
  await page.getByText("标签翻译", { exact: true }).click();
  await page.getByRole("button", { name: "查询词典" }).click();
  await expect(page.getByText("几何构图", { exact: true })).toBeVisible();
  await expect(page.getByLabel("候选标注")).toHaveValue(original);
  await page.screenshot({ path: info.outputPath("caption-translation.png") });
  await page.getByRole("button", { name: "采用翻译候选" }).click();
  await expect(page.getByLabel("候选标注")).toHaveValue("几何构图");
  expect(mocks.writes).toEqual([]);
});

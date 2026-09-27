import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("mobile navigation follows route and browser history", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");

  const toggle = page.getByRole("button", { name: "切换导航" });
  const nav = page.getByRole("navigation", { name: "主导航" });
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await nav.getByRole("link", { name: "数据集蓝图" }).click();
  await expect(page).toHaveURL(/\/next\/datasets$/);
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect(nav.getByRole("link", { name: "数据集蓝图" })).toHaveAttribute(
    "aria-current",
    "page",
  );

  await nav.getByRole("link", { name: "训练队列" }).click();
  await expect(page).toHaveURL(/\/next\/queue$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/next\/datasets$/);
  await page.goForward();
  await expect(page).toHaveURL(/\/next\/queue$/);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("theme toggle persists across navigation and reload", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await page.getByRole("button", { name: "浅色外观" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(
    await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-theme")),
  ).toBe("light");
  await page
    .getByRole("navigation", { name: "主导航" })
    .getByRole("link", { name: "数据集蓝图" })
    .click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  await page.getByRole("button", { name: "深色外观" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("desktop sidebar resizes, respects its minimum, and persists", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/settings");

  const sidebar = page.locator(".app-navigation");
  const resizer = page.getByRole("separator", { name: "调整侧边栏宽度" });
  const initial = await sidebar.boundingBox();
  expect(initial?.width).toBeCloseTo(204, 0);

  const handle = await resizer.boundingBox();
  expect(handle).not.toBeNull();
  await page.mouse.move(handle!.x + handle!.width / 2, handle!.y + 100);
  await page.mouse.down();
  await page.mouse.move(8, handle!.y + 100, { steps: 5 });
  await page.mouse.up();
  expect((await sidebar.boundingBox())?.width).toBeCloseTo(128, 0);
  expect(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-sidebar-width"))).toBe("128");

  await resizer.focus();
  await page.keyboard.press("ArrowRight");
  expect((await sidebar.boundingBox())?.width).toBeCloseTo(138, 0);
  await page.reload();
  expect((await sidebar.boundingBox())?.width).toBeCloseTo(138, 0);

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(resizer).toBeHidden();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("navigation switches between sidebar and top bar without losing width", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/settings");
  const layout = page.locator(".next-layout");
  const navigation = page.locator(".app-navigation");
  const content = page.locator(".next-content");

  await page.evaluate(() => localStorage.setItem("dragon-next-ui-v1-sidebar-width", "260"));
  await page.reload();
  await page.getByRole("button", { name: "切换为顶部栏" }).click();
  await expect(layout).toHaveAttribute("data-navigation-layout", "top");
  await expect(page.getByRole("separator", { name: "调整侧边栏宽度" })).toBeHidden();
  expect((await navigation.boundingBox())!.y).toBeLessThan((await content.boundingBox())!.y);
  expect((await navigation.boundingBox())!.width).toBeCloseTo(1280, 0);

  await page.getByRole("navigation", { name: "主导航" }).getByRole("link", { name: "训练配置" }).click();
  await expect(page).toHaveURL(/\/next\/training$/);
  await expect(layout).toHaveAttribute("data-navigation-layout", "top");
  expect((await content.boundingBox())!.y).toBeGreaterThan((await navigation.boundingBox())!.y);
  await page.reload();
  await expect(layout).toHaveAttribute("data-navigation-layout", "top");
  await page.getByRole("button", { name: "切换为侧边栏" }).click();
  await expect(layout).toHaveAttribute("data-navigation-layout", "sidebar");
  expect((await navigation.boundingBox())!.width).toBeCloseTo(260, 0);
  expect(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-navigation-layout"))).toBe("sidebar");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("wide top bar stays on one row and preserves an unsaved settings field", async ({ page }) => {
  await page.setViewportSize({ width: 1285, height: 900 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/settings");
  const output = page.getByLabel("训练输出目录");
  await output.fill("output/unsaved-navigation-layout");
  await page.getByRole("button", { name: "切换为顶部栏" }).click();

  const navigation = page.locator(".app-navigation");
  const links = page.getByRole("navigation", { name: "主导航" });
  expect((await navigation.boundingBox())!.height).toBeLessThan(65);
  expect(await links.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await expect(output).toHaveValue("output/unsaved-navigation-layout");
  await navigation.locator("summary").click();
  await expect(navigation.getByRole("link", { name: "旧版界面" })).toBeVisible();

  await page.setViewportSize({ width: 1024, height: 900 });
  expect((await navigation.boundingBox())!.height).toBeGreaterThan(65);
  await expect(output).toHaveValue("output/unsaved-navigation-layout");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("narrow sidebar keeps icon navigation usable and can reset its width", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/settings");
  const sidebar = page.locator(".app-navigation");
  const resizer = page.getByRole("separator", { name: "调整侧边栏宽度" });
  await resizer.focus();
  await page.keyboard.press("Home");
  expect((await sidebar.boundingBox())!.width).toBeCloseTo(128, 0);
  await expect(page.getByRole("navigation", { name: "主导航" }).getByRole("link", { name: "训练配置" })).toBeVisible();
  await expect(sidebar.locator(".app-links a span").first()).toBeHidden();
  await sidebar.locator("summary").click();
  await expect(sidebar.getByRole("link", { name: "生图测试" })).toBeVisible();
  await expect(sidebar.getByRole("button", { name: "浅色外观" })).toBeVisible();
  await expect(sidebar.getByRole("button", { name: "切换为顶部栏" })).toBeVisible();

  await page.setViewportSize({ width: 810, height: 720 });
  await resizer.focus();
  await page.keyboard.press("Home");
  expect((await sidebar.boundingBox())!.width).toBeCloseTo(81, 0);
  await expect(sidebar.locator(".nav-legacy-link span")).toBeHidden();
  await expect(sidebar.getByRole("link", { name: "旧版界面" })).toBeVisible();
  await expect(sidebar.getByRole("link", { name: "生图测试" })).toBeVisible();

  await page.setViewportSize({ width: 1280, height: 720 });
  await resizer.dblclick({ position: { x: 5, y: 100 } });
  expect((await sidebar.boundingBox())!.width).toBeCloseTo(204, 0);
  expect(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-sidebar-width"))).toBeNull();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("reduced motion follows the system preference", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/next/training");
  await expect(page.locator("html")).toHaveAttribute("data-next-motion", "off");

  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.locator("html")).toHaveAttribute("data-next-motion", "on");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("skip link receives keyboard focus and route error can recover", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");
  await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();
  await page.evaluate(() =>
    (document.activeElement as HTMLElement | null)?.blur(),
  );
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: "跳到工作区" });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/next\/training#workspace$/);
  await expect(page.locator("#workspace")).toBeFocused();
  await page.goBack();
  await expect(page).toHaveURL(/\/next\/training$/);
  await page.goForward();
  await expect(page).toHaveURL(/\/next\/training#workspace$/);

  await page.goto("/next/unknown-workspace");
  const routeError = page.getByRole("alert");
  await expect(
    routeError.getByRole("heading", { name: "工作区未能加载" }),
  ).toBeVisible();
  await routeError.getByRole("link", { name: "训练配置" }).click();
  await expect(page).toHaveURL(/\/next\/training$/);
  await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("skip-link history keeps the dirty training guard active", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/queue");
  await page
    .getByRole("navigation", { name: "主导航" })
    .getByRole("link", { name: "训练配置" })
    .click();
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();

  const output = page.getByRole("textbox", { name: "输出名称", exact: true });
  await output.fill("unsaved-skip-link");
  const prompts: string[] = [];
  page.on("dialog", async (dialog) => {
    prompts.push(dialog.message());
    await dialog.dismiss();
  });

  const skipLink = page.getByRole("link", { name: "跳到工作区" });
  await skipLink.focus();
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/next\/training#workspace$/);
  await page.evaluate(() => window.history.back());
  await expect(page).toHaveURL(/\/next\/training$/);
  await page.evaluate(() => window.history.back());

  await expect.poll(() => prompts.length).toBeGreaterThan(0);
  await expect(page).toHaveURL(/\/next\/training$/);
  await expect(output).toHaveValue("unsaved-skip-link");
  expect(prompts).toContain(
    "当前训练配置有未保存修改，离开会丢失这些修改。是否继续？",
  );
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("skip-link history keeps the dirty dataset guard active", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/queue");
  await page
    .getByRole("navigation", { name: "主导航" })
    .getByRole("link", { name: "数据集蓝图" })
    .click();

  const source = page.getByLabel("原始图片目录", { exact: true });
  await source.fill("images/skip-link-draft");
  const skipLink = page.getByRole("link", { name: "跳到工作区" });
  await skipLink.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/next\/datasets#workspace$/);
  await expect(page.locator("#workspace")).toBeFocused();

  await page.evaluate(() => window.history.back());
  await expect(page).toHaveURL(/\/next\/datasets$/);
  await page.evaluate(() => window.history.back());
  const discard = page.getByRole("dialog", { name: "放弃未保存修改？" });
  await expect(discard).toBeVisible();
  await discard.getByRole("button", { name: "继续编辑" }).click();

  await expect(page).toHaveURL(/\/next\/datasets$/);
  await expect(source).toHaveValue("images/skip-link-draft");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("lazy workspace failure shows a recoverable route error", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let failedImports = 0;
  await page.route("**/src/features/settings/SettingsPage.tsx*", (route) => {
    failedImports += 1;
    return route.abort();
  });

  await page.goto("/next/settings");
  const error = page.getByRole("alert");
  await expect(
    error.getByRole("heading", { name: "工作区未能加载" }),
  ).toBeVisible();
  expect(failedImports).toBeGreaterThan(0);
  await error.getByRole("link", { name: "训练配置" }).click();
  await expect(page).toHaveURL(/\/next\/training$/);
  await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("lazy workspace loading is announced as a polite status", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");

  let announceImport = () => {};
  let releaseImport = () => {};
  const importStarted = new Promise<void>((resolve) => { announceImport = resolve; });
  const importGate = new Promise<void>((resolve) => { releaseImport = resolve; });
  await page.route("**/src/features/settings/SettingsPage.tsx*", async (route) => {
    announceImport();
    await importGate;
    await route.continue();
  });

  await page.getByRole("navigation", { name: "主导航" }).getByRole("link", { name: "全局设置" }).click();
  await importStarted;
  const loading = page.locator(".route-loading");
  const status = loading.getByRole("status");
  await expect(loading).toHaveAttribute("aria-busy", "true");
  await expect(status).toHaveAttribute("aria-live", "polite");
  await expect(status).toHaveText("正在加载工作区");

  releaseImport();
  await expect(page.getByRole("heading", { name: "全局设置" })).toBeVisible();
  await expect(loading).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

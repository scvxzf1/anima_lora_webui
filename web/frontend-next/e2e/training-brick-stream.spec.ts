import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace, configFile } from "./fixtures";

const stages = ["input", "method", "training", "resources"];
const titles = ["输入准备", "方法配置", "训练计划", "设备与性能"];
async function open(page: Page) {
  const mocks = await mockWorkspace(page);
  await page.route("**/api/config/model-blocks?**", (route) =>
    route.fulfill({ json: { block_count: 56, max_swap: 54 } }),
  );
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue(
    configFile.path,
  );
  return mocks;
}
async function noOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".training-config-page")
      .evaluate((node) => node.scrollWidth <= node.clientWidth + 1),
  ).toBe(true);
}

for (const [width, theme, screenshot] of [
  [1440, "light", "desktop"],
  [1050, "light", "narrow"],
  [390, "light", "mobile"],
  [1440, "dark", "dark"],
] as const) {
  test(`brick stream layout, navigation and scroll ${width} ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.addInitScript(
      (value) => localStorage.setItem("dragon-next-ui-v1-theme", value),
      theme,
    );
    const mocks = await open(page);
    await expect(
      page.getByRole("button", { name: "分组视图", exact: true }),
    ).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("button", { name: "积木流视图", exact: true }).click();
    const sections = page.locator("[data-training-stage]");
    expect(
      await sections.evaluateAll((nodes) =>
        nodes.map((node) => (node as HTMLElement).dataset.trainingStage),
      ),
    ).toEqual(stages);
    expect(
      await sections.evaluateAll((nodes) =>
        nodes.every((node) => !(node as HTMLElement).hidden),
      ),
    ).toBe(true);
    const ids = await page
      .locator('[id^="training-field-"]')
      .evaluateAll((nodes) => nodes.map((node) => node.id));
    expect(new Set(ids).size).toBe(ids.length);
    const nav = page.getByRole("navigation", { name: "快速定位配置阶段" });
    for (const [index, title] of titles.entries()) {
      await nav.getByRole("button", { name: title, exact: true }).click();
      await expect(
        nav.getByRole("button", { name: title, exact: true }),
      ).toHaveAttribute("aria-current", "location");
      await expect(
        page.locator(`#training-stage-${stages[index]} > header`),
      ).toBeInViewport();
    }
    const scrollRoot = page.locator(
      width <= 800 ? ".training-config-page" : ".training-edit-groups",
    );
    // Simulate a user scroll without going through the stage navigation handler.
    await scrollRoot.evaluate((root) => {
      const section = document.querySelector<HTMLElement>(
        "#training-stage-training",
      )!;
      root.scrollTop +=
        section.getBoundingClientRect().top -
        root.getBoundingClientRect().top -
        (innerWidth <= 800 ? 88 : 0);
    });
    await expect(
      nav.getByRole("button", { name: "训练计划", exact: true }),
    ).toHaveAttribute("aria-current", "location");
    await nav.getByRole("button", { name: "输入准备", exact: true }).click();
    await noOverflow(page);
    const grid = page.locator(
      "#resource-fields-input-models .training-edit-fields",
    );
    const columns = await grid.evaluate(
      (node) => getComputedStyle(node).gridTemplateColumns.split(" ").length,
    );
    expect(columns).toBe(12);
    const spans = await grid.locator(".training-field-card").evaluateAll((nodes) =>
      nodes.map((node) => getComputedStyle(node).gridColumnStart),
    );
    if (width === 390) expect(spans.every((span) => span === "span 12")).toBe(true);
    if (width > 800) {
      const track = page.getByRole("scrollbar", { name: "训练参数滚动条" });
      await expect(track).toBeVisible();
      await track.focus();
      await page.keyboard.press("End");
      await expect
        .poll(() => scrollRoot.evaluate((node) => node.scrollTop))
        .toBeGreaterThan(100);
      expect(
        await page
          .locator(".training-config-page")
          .evaluate((node) => node.scrollTop),
      ).toBe(0);
      await page.keyboard.press("Home");
    } else {
      await expect(
        page.getByRole("scrollbar", { name: "训练参数滚动条" }),
      ).toBeHidden();
      expect(
        await page
          .locator(".training-edit-groups")
          .evaluate((node) => node.scrollTop),
      ).toBe(0);
      await expect
        .poll(() => scrollRoot.evaluate((node) => node.scrollTop))
        .toBeGreaterThan(0);
    }
    if (width > 800) {
      await nav.getByRole("button", { name: "方法配置", exact: true }).click();
      await expect(
        page.locator("#training-stage-method > header"),
      ).toBeInViewport();
      await page
        .locator("#resource-heading-method-capacity")
        .scrollIntoViewIfNeeded();
    }
    await page.screenshot({
      path: `/tmp/training-stream-e2e/${screenshot}-stream.png`,
    });
    await page.reload();
    await expect(
      page.getByRole("button", { name: "积木流视图", exact: true }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("switching preserves draft, filters, collapse and blur-only edits", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const mocks = await mockWorkspace(page);
  await page.route("**/api/config/merged?**", (route) =>
    route.fulfill({
      json: {
        model_family: "krea2_raw",
        output_name: "studio-portrait",
        network_dim: 32,
        pretrained_model_name_or_path:
          "models/diffusion_models/krea2_raw.safetensors",
        blocks_to_swap: 20,
      },
    }),
  );
  await page.route("**/api/config/model-blocks?**", (route) =>
    route.fulfill({
      json: {
        ok: true,
        block_count: 56,
        max_swap: 54,
      },
    }),
  );
  await page.route("**/api/training/preflight", (route) =>
    route.fulfill({
      json: {
        ok: true,
        summary: { errors: 0, warnings: 1, checks: 1 },
        checks: [
          {
            level: "warning",
            key: "base_compute",
            message: "Stream fixture warning",
          },
        ],
      },
    }),
  );
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  const output = page.getByRole("textbox", { name: "输出名称", exact: true });
  await output.fill("stream-draft");
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await expect(output).toHaveValue("stream-draft");
  const identity = page.locator("#resource-heading-training-identity");
  await identity.click();
  await expect(identity).toHaveAttribute("aria-expanded", "false");
  await page.getByRole("button", { name: "分组视图", exact: true }).click();
  await expect(identity).toHaveAttribute("aria-expanded", "false");
  await page.getByLabel("搜索参数").fill("output_name");
  await page.getByLabel("参数视图").selectOption("changed");
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await expect(page.getByLabel("搜索参数")).toHaveValue("output_name");
  await expect(page.getByLabel("参数视图")).toHaveValue("changed");
  await expect(output).toHaveValue("stream-draft");
  await page.getByLabel("搜索参数").fill("");
  await page.getByLabel("参数视图").selectOption("all");
  await page
    .getByRole("navigation", { name: "快速定位配置阶段" })
    .getByRole("button", { name: /设备与性能/ })
    .click();
  await page.locator("#resource-heading-resources-residency").click();
  const blockSwap = page.getByRole("spinbutton", {
    name: "Block swap 数量",
    exact: true,
  });
  await blockSwap.fill("22");
  await page.getByRole("button", { name: "分组视图", exact: true }).click();
  await expect(blockSwap).toHaveValue("22");
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await expect(blockSwap).toHaveValue("22");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("preflight report and locating persist across view switches", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const mocks = await open(page);
  await page.route("**/api/training/preflight", (route) =>
    route.fulfill({
      json: {
        ok: true,
        summary: { errors: 0, warnings: 1, checks: 1 },
        checks: [
          {
            level: "warning",
            key: "base_compute",
            message: "Stream fixture warning",
          },
        ],
      },
    }),
  );
  await page.getByRole("button", { name: "运行预检测", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText(
    "Stream fixture warning",
  );
  await page.keyboard.press("Escape");
  const requests: string[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/"))
      requests.push(request.url());
  });
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await page.getByRole("button", { name: "分组视图", exact: true }).click();
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  expect(requests).toEqual([]);
  await page.getByRole("button", { name: "预览与预检", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText(
    "Stream fixture warning",
  );
  await page
    .getByRole("button", { name: "定位 base_compute", exact: true })
    .click();
  await expect(page.locator("#training-field-base_compute")).toBeInViewport();
  await expect(page.getByLabel("搜索参数")).toHaveValue("base_compute");
  expect(mocks.unhandled).toEqual([]);
});

test("dataset input keeps focus when switching with the mouse", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await open(page);
  const dataset = page.getByRole("textbox", {
    name: "数据集配置",
    exact: true,
  });
  await dataset.focus();
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await page.screenshot({
    path: "/tmp/training-stream-e2e/dataset-focus-failure.png",
  });
  await expect(dataset).toBeFocused();
});

test("filtered stream keeps the requested last stage active", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await open(page);
  await page.getByRole("button", { name: "积木流视图", exact: true }).click();
  await page.getByLabel("搜索参数").fill("network_dim");
  const resources = page
    .getByRole("navigation", { name: "快速定位配置阶段" })
    .getByRole("button", { name: /设备与性能/ });
  await resources.click();
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  await page.screenshot({
    path: "/tmp/training-stream-e2e/filtered-last-stage.png",
  });
  await expect(resources).toHaveAttribute("aria-current", "location");
});

for (const mode of ["group", "stream"] as const) {
  test(`saves equivalent payload and reloads in ${mode} view`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockWorkspace(page);
    let savedValues: Record<string, unknown> | undefined;
    let revision = 1;
    await page.route("**/api/config/merged?**", (route) =>
      route.fulfill({
        json: {
          model_family: "krea2_raw",
          output_name: savedValues?.output_name || "studio-portrait",
          network_dim: 32,
          pretrained_model_name_or_path:
            "models/diffusion_models/krea2_raw.safetensors",
        },
      }),
    );
    await page.route("**/api/config/raw", async (route) => {
      const request = route.request();
      if (request.method() === "GET") {
        return route.fulfill({
          json: {
            file: configFile.path,
            content: `output_name = "${savedValues?.output_name || "studio-portrait"}"\nnetwork_dim = ${savedValues?.network_dim || 32}\n`,
            revision: `fixture-${revision}`,
            meta: configFile,
          },
        });
      }
      if (request.method() !== "PATCH") return route.continue();
      const body = request.postDataJSON() as { values?: Record<string, unknown> };
      savedValues = body.values || {};
      revision += 1;
      return route.fulfill({
        json: {
          ok: true,
          file: configFile.path,
          message: "Fixture saved",
          content: `output_name = "${savedValues.output_name}"\nnetwork_dim = ${savedValues.network_dim}\n`,
          revision: `fixture-${revision}`,
          changed: Object.keys(savedValues),
          warnings: [],
        },
      });
    });
    await page.goto("/next/training");
    await expect(page.getByText("已同步", { exact: true })).toBeVisible();
    if (mode === "stream") {
      await page.getByRole("button", { name: "积木流视图", exact: true }).click();
    } else {
      await page.getByRole("tab", { name: "训练计划", exact: true }).click();
    }
    const output = page.getByRole("textbox", { name: "输出名称", exact: true });
    await output.fill(`equivalent-${mode}`);
    await page.getByRole("button", { name: "保存配置", exact: true }).click();
    await expect(page.getByText("Fixture saved", { exact: true })).toBeVisible();
    expect(savedValues).toEqual({ output_name: `equivalent-${mode}` });
    await page.reload();
    await expect(page.getByText("已同步", { exact: true })).toBeVisible();
    if (mode === "group") {
      await page.getByRole("tab", { name: "训练计划", exact: true }).click();
    }
    await expect(page.getByRole("textbox", { name: "输出名称", exact: true })).toHaveValue(
      `equivalent-${mode}`,
    );
  });
}

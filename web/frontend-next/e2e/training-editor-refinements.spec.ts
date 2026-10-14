import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const shots = "/tmp/training-editor-refinements";
async function open(page: Page, qwenEdit = false) {
  const mocks = await mockWorkspace(page);
  if (qwenEdit) {
    await page.route("**/api/config/merged?**", async (route) => route.fulfill({
      json: {
        model_family: "qwen_image_2_1", qwen_image_2_1_task: "edit",
        pretrained_model_name_or_path: "models/diffusion_models/qwen_edit.safetensors",
        qwen3: "models/text_encoders/qwen.safetensors", vae: "models/vae/qwen.safetensors",
        dataset_config: "configs/datasets/studio.toml", network_dim: 32,
        output_name: "qwen-edit", max_train_steps: 1600, learning_rate: 0.00002,
      },
    }));
  }
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  await expect(
    page.locator(".training-model-path-control input").first(),
  ).toHaveValue(qwenEdit ? "models/diffusion_models/qwen_edit.safetensors" : "models/diffusion_models/krea2_raw.safetensors");
  return mocks;
}
async function fits(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  for (const selector of [
    ".training-config-page",
    ".training-edit-card",
    ".training-model-path-control",
  ]) {
    const overflow = await page.locator(selector).evaluateAll((nodes) =>
      nodes
        .filter(
          (node) =>
            node.getBoundingClientRect().width > 0 &&
            node.scrollWidth > node.clientWidth + 1,
        )
        .map((node) => ({
          width: node.clientWidth,
          scroll: node.scrollWidth,
          children: [...node.querySelectorAll("*")]
            .filter(
              (child) =>
                child.getBoundingClientRect().width > 0 &&
                child.scrollWidth > child.clientWidth + 1,
            )
            .slice(0, 5)
            .map((child) => ({
              class: child.className,
              width: child.clientWidth,
              scroll: child.scrollWidth,
            })),
        })),
    );
    expect(overflow, selector).toEqual([]);
  }
}

for (const [width, height] of [[390, 1000], [768, 1000], [1080, 1000], [1440, 1000], [1920, 1080], [2560, 1440]])
  test(`responsive model and cache cards use available width in both views ${width}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height });
    const qwenEdit = width >= 1920;
    const mocks = await open(page, qwenEdit);
    {
      for (const view of ["分组视图", "积木流视图"]) {
        await page.getByRole("button", { name: view, exact: true }).click();
        if (
          await page
            .getByRole("button", { name: "展开配置库", exact: true })
            .isVisible()
        )
          await page
            .getByRole("button", { name: "展开配置库", exact: true })
            .click();
        for (const library of width <= 800
          ? ["auto", "hidden"]
          : ["side", "top", "hidden"]) {
          if (library === "hidden")
            await page
              .getByRole("button", { name: "收起配置库", exact: true })
              .click();
          else if (library !== "auto")
            await page
              .getByRole("button", {
                name: library === "side" ? "侧边排列" : "顶部排列",
                exact: true,
              })
              .click();
          for (const group of ["models", "cache"]) {
            if (view === "分组视图")
              await page
                .getByRole("tab", { name: "输入准备", exact: true })
                .click();
            const resource = page.locator(
              `.training-resource-group[data-group="${group}"]`,
            );
            if (
              (await resource
                .getByRole("button")
                .first()
                .getAttribute("aria-expanded")) === "false"
            )
              await resource.getByRole("button").first().click();
            const fields = resource.locator(".training-edit-fields");
            const geometry = await fields.evaluate((node) => ({
              width: node.parentElement!.getBoundingClientRect().width,
              columns:
                getComputedStyle(node).gridTemplateColumns.split(" ").length,
            }));
            const expected =
              group === "models" && qwenEdit && geometry.width >= 1760
                ? 5
                : group === "models" && qwenEdit && geometry.width >= 980
                  ? 6
                  : geometry.width >= 1280
                    ? 4
                    : geometry.width >= 980
                  ? 3
                  : geometry.width >= 620
                    ? 2
                    : 1;
            expect(
              geometry.columns,
              `${width} ${view} ${library} ${group} width=${geometry.width}`,
            ).toBe(expected);
            await fits(page);
            if (width >= 1920) {
              const fill = await page.locator(".training-edit-card").evaluateAll((cards) =>
                cards.map((card) => ({
                  width: card.getBoundingClientRect().width,
                  available: card.parentElement!.clientWidth,
                })),
              );
              for (const card of fill) expect(card.width / card.available).toBeGreaterThan(0.98);
              const stage = await page.locator(".training-stage-nav").boundingBox();
              expect(fill[0].width / stage!.width).toBeGreaterThan(0.97);
              if (group === "models") {
                const bricks = fields.locator(":scope > label");
                await expect(bricks).toHaveCount(5);
                await expect(page.locator("#training-control-qwen_image_2_1_task")).toHaveValue("edit");
                const boxes = await bricks.evaluateAll((nodes) => nodes.map((node) => {
                  const box = node.getBoundingClientRect();
                  return { x: box.x, y: box.y, right: box.right };
                }));
                const edges = await fields.evaluate((node) => {
                  const rect = node.getBoundingClientRect();
                  const style = getComputedStyle(node);
                  return { left: rect.left + parseFloat(style.paddingLeft), right: rect.right - parseFloat(style.paddingRight) };
                });
                await test.info().attach(`${width}-${view}-${library}-model-geometry`, {
                  body: JSON.stringify({ viewport: { width, height }, view, library, cards: fill, stageWidth: stage!.width, grid: geometry, edges, bricks: boxes }, null, 2),
                  contentType: "application/json",
                });
                // Each occupied row reaches both grid edges, including Qwen's final two fields.
                for (const y of new Set(boxes.map((box) => box.y))) {
                  const row = boxes.filter((box) => box.y === y);
                  expect(row[0].x).toBeCloseTo(edges.left, 0);
                  expect(row.at(-1)!.right).toBeCloseTo(edges.right, 0);
                }
              }
            }
            if (
              library === "hidden" &&
              view === "积木流视图" &&
              ((width >= 1920 && group === "models") ||
                (width === 1080 && group === "cache") ||
                (width === 390 && group === "models"))
            ) {
              await resource.scrollIntoViewIfNeeded();
              await page.screenshot({
                path: `${shots}/${width >= 1920 ? `${width}-models` : width === 1080 ? "cache-laptop" : "mobile"}.png`,
              });
            }
          }
          if (library === "hidden")
            await page
              .getByRole("button", { name: "展开配置库", exact: true })
              .click();
        }
      }
    }
    await page.getByRole("button", { name: "分组视图", exact: true }).click();
    await page.getByRole("tab", { name: "输入准备", exact: true }).click();
    const control = page.locator(".training-model-path-control").first();
    const input = control.locator("input");
    const long = `models/${"long-directory/".repeat(45)}checkpoint.safetensors`;
    await input.fill(long);
    await expect(input).toHaveValue(long);
    expect((await input.boundingBox())!.width).toBeLessThan(600);
    await expect(control.getByRole("button")).toBeVisible();
    const bounds = await control.evaluate((node) => {
      const box = node.getBoundingClientRect();
      const button = node.querySelector("button")!.getBoundingClientRect();
      return button.right <= box.right + 1;
    });
    expect(bounds).toBe(true);
    await fits(page);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });

for (const theme of ["light", "dark"])
  test(`start command has readable primary contrast and preserves dirty labels ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 1000 });
    await page.addInitScript(
      (value) => localStorage.setItem("dragon-next-ui-v1-theme", value),
      theme,
    );
    const mocks = await open(page);
    const start = page.getByRole("button", { name: "立即启动", exact: true });
    await expect(start).toBeEnabled();
    await expect(start).toHaveCSS("background-color", "rgb(29, 78, 216)");
    const paint = await start.evaluate((node) => {
      const style = getComputedStyle(node);
      const lum = (color: string) => {
        const values = color
          .match(/[\d.]+/g)!
          .slice(0, 3)
          .map(Number)
          .map((n) => {
            n /= 255;
            return n <= 0.04045 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4;
          });
        return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722;
      };
      const a = lum(style.color),
        b = lum(style.backgroundColor);
      return {
        background: style.backgroundColor,
        color: style.color,
        ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05),
        readable: node.scrollWidth <= node.clientWidth + 1,
      };
    });
    expect(paint.background).toBe("rgb(29, 78, 216)");
    expect(paint.color).toBe("rgb(255, 255, 255)");
    expect(paint.ratio).toBeGreaterThanOrEqual(4.5);
    expect(paint.readable).toBe(true);
    expect(
      await page
        .getByRole("button", { name: "加入队列", exact: true })
        .evaluate((node) => node.scrollWidth <= node.clientWidth + 1),
    ).toBe(true);
    await expect(
      page.getByRole("button", { name: "保存配置", exact: true }),
    ).toBeDisabled();
    expect(
      await page
        .getByRole("button", { name: "保存配置", exact: true })
        .evaluate((node) => getComputedStyle(node).backgroundColor),
    ).not.toBe(paint.background);
    expect(
      await page
        .getByRole("button", { name: "加入队列", exact: true })
        .evaluate((node) => getComputedStyle(node).backgroundColor),
    ).not.toBe(paint.background);
    for (const name of ["TOML", "恢复页面默认值"])
      expect(
        await page
          .getByRole("button", { name, exact: true })
          .evaluate((node) => getComputedStyle(node).backgroundColor),
      ).toBe("rgba(0, 0, 0, 0)");
    await page
      .locator(".training-model-path-control input")
      .first()
      .fill("models/dirty.safetensors");
    await expect(
      page.getByRole("button", { name: "保存并启动", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "保存并入队", exact: true }),
    ).toBeVisible();
    await fits(page);
    if (theme === "dark")
      await page.screenshot({ path: `${shots}/dark-primary.png` });
    expect(mocks.writes).toEqual([]);
  });

test("server model picker GET, cancel, pagination, file, Diffusers, fallback and focus", async ({
  page,
}) => {
  await page.setViewportSize({ width: 768, height: 1000 });
  const mocks = await mockWorkspace(page);
  const requests: string[] = [];
  let failure = false;
  const longName = `${"very-long-model-name-".repeat(15)}.safetensors`;
  await page.route("**/api/settings/model-files?**", async (route) => {
    expect(route.request().method()).toBe("GET");
    const url = new URL(route.request().url());
    requests.push(url.search);
    if (failure)
      return route.fulfill({ status: 404, json: { error: "unavailable" } });
    const directory = url.searchParams.get("directory") || "/models";
    const offset = Number(url.searchParams.get("offset"));
    return route.fulfill({
      json: {
        ok: true,
        roots: [{ name: "Models", path: "/models" }],
        directory,
        parent: directory === "/models" ? null : "/models",
        entries:
          directory !== "/models"
            ? []
            : offset
              ? [
                  {
                    name: "second.safetensors",
                    path: "/models/second.safetensors",
                    kind: "file",
                  },
                ]
              : [
                  {
                    name: "Diffusers",
                    path: "/models/Diffusers",
                    kind: "directory",
                  },
                  { name: longName, path: `/models/${longName}`, kind: "file" },
                ],
        offset,
        next_offset: directory === "/models" && offset === 0 ? 100 : null,
        total: 101,
      },
    });
  });
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  await expect(
    page.locator(".training-model-path-control input").first(),
  ).toHaveValue("models/diffusion_models/krea2_raw.safetensors");
  const control = page.locator(".training-model-path-control").first();
  const input = control.locator("input");
  const browse = control.getByRole("button");
  const before = await input.inputValue();
  expect(requests).toHaveLength(0);
  await browse.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByTitle(`/models/${longName}`, { exact: true }),
  ).toBeVisible();
  await expect(
    dialog.getByRole("button", { name: "使用所选文件", exact: true }),
  ).toBeDisabled();
  await dialog.getByTitle(`/models/${longName}`, { exact: true }).click();
  expect(
    await dialog.evaluate((node) => node.scrollWidth <= node.clientWidth + 1),
  ).toBe(true);
  await page.screenshot({ path: `${shots}/model-browser.png` });
  await dialog.getByRole("button", { name: "下一页", exact: true }).click();
  await expect(
    dialog.getByTitle("/models/second.safetensors", { exact: true }),
  ).toBeVisible();
  await expect(
    dialog.getByRole("button", { name: "使用所选文件", exact: true }),
  ).toBeDisabled();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(browse).toBeFocused();
  await expect(input).toHaveValue(before);
  await browse.click();
  await dialog.getByTitle(`/models/${longName}`, { exact: true }).click();
  await dialog
    .getByRole("button", { name: "使用所选文件", exact: true })
    .click();
  await expect(input).toHaveValue(`/models/${longName}`);
  await expect(browse).toBeFocused();
  await browse.click();
  await dialog.getByTitle("/models/Diffusers", { exact: true }).click();
  await expect(dialog.getByLabel("当前目录")).toHaveText("/models/Diffusers");
  await dialog
    .getByRole("button", { name: "返回上级目录", exact: true })
    .click();
  await expect(dialog.getByLabel("当前目录")).toHaveText("/models");
  await dialog.getByTitle("/models/Diffusers", { exact: true }).click();
  await dialog
    .getByRole("button", { name: "选择当前目录", exact: true })
    .click();
  await expect(input).toHaveValue("/models/Diffusers");
  failure = true;
  await browse.click();
  await expect(dialog.getByRole("alert")).toContainText("目录浏览暂不可用");
  await expect(
    dialog.getByRole("button", { name: "选择当前目录", exact: true }),
  ).toBeDisabled();
  failure = false;
  await dialog
    .getByRole("button", { name: "重试读取目录", exact: true })
    .click();
  await expect(
    dialog.getByTitle("/models/Diffusers", { exact: true }),
  ).toBeVisible();
  await dialog.getByRole("button", { name: "取消", exact: true }).click();
  await expect(input).toHaveValue("/models/Diffusers");
  await expect(browse).toBeFocused();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("regular groups automatically pack wide fields without row gaps", async ({ page }) => {
  const mocks = await open(page);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  const group = page.locator('.training-resource-group[data-group="optimizer"]');
  const toggle = group.getByRole("button").first();
  if ((await toggle.getAttribute("aria-expanded")) === "false") await toggle.click();
  const fields = group.locator(".training-edit-fields");
  await expect(fields).toHaveAttribute("data-field-count", "5");
  await expect(fields).toHaveAttribute("data-auto-columns", "3");
  const rows = await fields.evaluate((node) => {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    const left = rect.left + parseFloat(style.paddingLeft);
    const right = rect.right - parseFloat(style.paddingRight);
    const boxes = [...node.children].map((child) => {
      const box = child.getBoundingClientRect();
      return { x: box.x, y: box.y, right: box.right };
    });
    return [...new Set(boxes.map((box) => box.y))].map((y) => {
      const row = boxes.filter((box) => box.y === y);
      return { left: row[0].x, right: row.at(-1)!.right, expectedLeft: left, expectedRight: right };
    });
  });
  expect(rows).toHaveLength(2);
  for (const row of rows) {
    expect(row.left).toBeCloseTo(row.expectedLeft, 0);
    expect(row.right).toBeCloseTo(row.expectedRight, 0);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await fields.evaluate((node) => getComputedStyle(node).gridTemplateColumns.trim().split(/\s+/).length),
  ).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

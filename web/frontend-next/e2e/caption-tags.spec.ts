import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function setup(page: Page, firstCaption = "red hair, mid, red hair") {
  const mocks = await mockWorkspace(page);
  const calls: { method: string; path: string; body: unknown }[] = [];
  const job = {
    id: "caption-1",
    state: "done",
    profile_name: "Tag editor",
    profile_id: "provider-a",
    settings: { provider: "cltagger" },
    total: 2,
    completed: 2,
    failed: 0,
    items: [0, 1].map((index) => ({
      id: `tag-${index}`,
      name: `tag-${index}.png`,
      file: `tag-${index}.png`,
      state: "ready",
      caption: "Original",
      proposed_caption: index === 0 ? firstCaption : "blue sky",
      url: `/api/config/dataset-presets/image?image=${index}`,
    })),
  };
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/jobs/caption-1"),
    (route) => {
      const path = new URL(route.request().url()).pathname;
      const method = route.request().method();
      if (method === "PATCH") {
        const body = route.request().postDataJSON() as {
          proposed_caption: string;
        };
        calls.push({ method, path, body });
        const id = path.split("/").at(-1);
        job.items.find((item) => item.id === id)!.proposed_caption =
          body.proposed_caption;
      }
      if (path.endsWith("/logs")) return route.fulfill({ json: { lines: [] } });
      if (method === "POST" && path.endsWith("/commit")) {
        calls.push({ method, path, body: route.request().postDataJSON() });
        return route.fulfill({
          json: { written: 0, conflicts: 0, skipped: 0, errors: [], job },
        });
      }
      return route.fulfill({ json: { ok: true, job } });
    },
  );
  return { calls, mocks };
}

test("switching an untouched multiline caption to tags and back does not PATCH", async ({ page }) => {
  const rawCaption = "red hair, mid\nred hair";
  const { calls, mocks } = await setup(page, rawCaption);
  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("button", { name: "标签", exact: true }).last().click();
  await page.getByRole("textbox", { name: "编辑标签 red hair", exact: true }).first().focus();
  await page.getByRole("button", { name: "原始文本", exact: true }).last().click();
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true }).last()).toHaveValue(rawCaption);
  await expect(page.getByRole("button", { name: "保存候选", exact: true })).toBeDisabled();
  expect(calls.filter((call) => call.method === "PATCH")).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const width of [390, 1280]) {
  test(`tag editing preserves keystrokes, duplicate identity, item isolation, and PATCH-only save at ${width}px`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 844 });
    const { calls, mocks } = await setup(page);
    await page.goto("/next/captioning?job=caption-1");
    await page
      .getByRole("button", { name: "标签", exact: true })
      .last()
      .click();

    const duplicates = page.getByRole("textbox", {
      name: "编辑标签 red hair",
      exact: true,
    });
    await expect(duplicates).toHaveCount(2);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: info.outputPath("caption-tags.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "上移 red hair", exact: true })
      .last()
      .click();
    await duplicates.nth(0).fill("");
    await duplicates.nth(0).pressSequentially("long multi word tag");
    await expect(duplicates.nth(0)).toBeFocused();
    await expect(duplicates.nth(0)).toHaveValue("long multi word tag");
    await duplicates.nth(0).press("Enter");
    await page.getByRole("button", { name: "保存候选", exact: true }).click();
    await expect
      .poll(() => calls.filter((call) => call.method === "PATCH").length)
      .toBe(1);
    expect(calls[0]).toMatchObject({
      method: "PATCH",
      body: { proposed_caption: "long multi word tag, red hair, mid" },
    });

    await page
      .getByRole("button", { name: "原始文本", exact: true })
      .last()
      .click();
    await page
      .getByRole("textbox", { name: "候选标注", exact: true })
      .fill("unsaved raw draft");
    if (width < 650)
      await page
        .getByRole("button", { name: "候选列表 (2)", exact: true })
        .click();
    await page
      .getByRole("button", { name: "tag-1.png ready", exact: true })
      .click();
    await expect(
      page.getByRole("textbox", { name: "候选标注", exact: true }).last(),
    ).toHaveValue("blue sky");
    await page
      .getByRole("button", { name: "标签", exact: true })
      .last()
      .click();
    const add = page.getByRole("textbox", { name: "添加标签", exact: true });
    await add.fill("unsaved added tag");
    const existing = page.getByRole("textbox", {
      name: "编辑标签 blue sky",
      exact: true,
    });
    await existing.fill("blue cloud");
    await existing.press("Enter");
    await expect(add).toHaveValue("unsaved added tag");
    if (width < 650)
      await page
        .getByRole("button", { name: "候选列表 (2)", exact: true })
        .click();
    await page.getByRole("button", { name: /^tag-0\.png ready/ }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "当前编辑" }),
    ).toContainText("tag-0.png");
    await expect(
      page.getByRole("textbox", { name: "添加标签", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "原始文本", exact: true }).last(),
    ).toHaveAttribute("aria-pressed", "true");
    await expect(
      page.getByRole("textbox", { name: "候选标注", exact: true }).last(),
    ).toHaveValue("unsaved raw draft");
    await expect
      .poll(() =>
        calls.some(
          (call) => call.method === "POST" && call.path.endsWith("/commit"),
        ),
      )
      .toBe(false);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("pointer drag reorders tag draft without writing until Save", async ({
  page,
}) => {
  const { calls, mocks } = await setup(page);
  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("button", { name: "标签", exact: true }).last().click();
  const source = page.getByRole("button", { name: "拖动重排 mid" });
  const target = page
    .getByRole("textbox", { name: "编辑标签 red hair" })
    .first();
  const from = await source.boundingBox();
  const to = await target.boundingBox();
  expect(from).not.toBeNull();
  expect(to).not.toBeNull();
  await page.mouse.move(from!.x + from!.width / 2, from!.y + from!.height / 2);
  await page.mouse.down();
  await page.mouse.move(to!.x + to!.width / 2, to!.y + to!.height / 2, {
    steps: 12,
  });
  await page.mouse.up();
  await expect(page.locator(".caption-tag-row input").first()).toHaveValue(
    "mid",
  );
  expect(calls).toEqual([]);
  await page.getByRole("button", { name: "保存候选", exact: true }).click();
  await expect.poll(() => calls.length).toBe(1);
  expect(calls[0]).toMatchObject({
    method: "PATCH",
    body: { proposed_caption: "mid, red hair, red hair" },
  });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

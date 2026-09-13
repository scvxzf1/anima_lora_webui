import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function captionFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const writes: { method: string; path: string; body: unknown }[] = [];
  const job = {
    id: "caption-1", state: "done", profile_name: "Studio captions", total: 55, completed: 55, failed: 0,
    items: Array.from({ length: 55 }, (_, i) => ({
      id: `image-${i}`, name: `studio-${i}.png`, file: `studio-${i}.png`, state: "ready",
      caption: "Original caption", proposed_caption: `Candidate ${i}`,
      url: `/api/config/dataset-presets/image?image=${i}`,
    })),
  };
  await page.route((url) => url.pathname.startsWith("/api/captioning/jobs/caption-1"), (route) => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    if (method !== "GET") {
      const body = route.request().postDataJSON();
      writes.push({ method, path, body });
      if (method === "PATCH") {
        const id = path.split("/").at(-1);
        const item = job.items.find((item) => item.id === id)!;
        item.proposed_caption = body.proposed_caption;
      } else return route.fulfill({ status: 409, json: { error: "TXT changed externally" } });
    }
    return route.fulfill({ json: { ok: true, job } });
  });
  return { ...mocks, commandWrites: writes };
}

test("caption drafts retain object identity across pages and block silent navigation loss", async ({ page }) => {
  const mocks = await captionFixture(page);
  await page.goto("/next/captioning?job=caption-1");
  const editor = page.getByRole("textbox", { name: "候选标注", exact: true });
  await editor.fill("Draft zero");
  await page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true }).check();
  await page.getByRole("button", { name: "下一页候选" }).click();
  await expect(page.locator(".caption-current-object")).toContainText("studio-0.png · 第 1 页");
  await expect(editor).toHaveValue("Draft zero");
  await expect(page.locator(".caption-review-context")).toContainText("其他页 1 项");
  await page.getByRole("button", { name: "studio-50.png ready", exact: true }).click();
  await editor.fill("Draft fifty");
  await page.getByRole("checkbox", { name: "选择 studio-50.png", exact: true }).check();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 2 项");
  await expect(page.getByRole("button", { name: "写回选中 (2)", exact: true })).toBeDisabled();
  await page.locator(".caption-draft-list summary").click();
  await page.locator(".caption-draft-list").getByRole("button", { name: "studio-0.png · 第 1 页" }).click();
  await expect(editor).toHaveValue("Draft zero");
  await expect(page.getByRole("button", { name: "上一页候选" })).toBeDisabled();

  let prompt = "";
  page.once("dialog", async (dialog) => { prompt = dialog.message(); await dialog.dismiss(); });
  await page.getByRole("link", { name: "接入预设", exact: true }).click();
  await expect(editor).toHaveValue("Draft zero");
  await expect.poll(() => prompt).toContain("离开会丢失");
  await page.getByRole("button", { name: "保存候选", exact: true }).click();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 1 项");
  await page.locator(".caption-draft-list").getByRole("button", { name: "studio-50.png · 第 2 页" }).click();
  await expect(editor).toHaveValue("Draft fifty");
  await page.getByRole("button", { name: "保存候选", exact: true }).click();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 0 项");
  page.once("dialog", async (dialog) => { prompt = dialog.message(); await dialog.accept(); });
  await page.getByRole("button", { name: "写回选中 (2)", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("TXT changed externally");
  expect(prompt).toContain("2 项已保存候选");
  expect(prompt).toContain("其他页");
  expect(mocks.commandWrites.map((write) => write.method)).toEqual(["PATCH", "PATCH", "POST"]);
  expect(mocks.commandWrites.at(-1)?.body).toEqual({ item_ids: ["image-0", "image-50"] });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`caption focused workspace ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await captionFixture(page);
      await page.addInitScript((theme) => localStorage.setItem("dragon-next-ui-v1-theme", theme), theme);
      await page.goto("/next/captioning?job=caption-1");
      await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 0");
      await expect.poll(() => page.locator(".caption-original").evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBe(480);
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      if (viewport.width < 650) {
        await expect(page.locator(".review-items")).not.toBeVisible();
        const imageTop = await page.locator(".caption-original").evaluate((node) => node.getBoundingClientRect().top);
        expect(imageTop).toBeLessThan(viewport.height);
        await page.getByRole("button", { name: "候选列表 (55)", exact: true }).click();
        await expect(page.locator(".review-items")).toBeVisible();
        await page.getByRole("button", { name: "下一页候选" }).click();
        await page.getByRole("button", { name: "studio-50.png ready", exact: true }).click();
        await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 50");
      }
      await page.screenshot({ path: info.outputPath("caption-editor.png"), fullPage: true });
      expect(mocks.commandWrites).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}

test("missing image remains distinct from editable text and retry recovers the preview", async ({ page }) => {
  const mocks = await captionFixture(page);
  let missing = true;
  await page.route((url) => url.pathname === "/api/config/dataset-presets/image", (route) => missing
    ? route.fulfill({ status: 404, body: "missing image" }) : route.fallback());
  await page.goto("/next/captioning?job=caption-1");
  await expect(page.locator(".caption-image-error")).toContainText("无法读取图片：studio-0.png");
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 0");
  missing = false;
  await page.getByRole("button", { name: "重试图片" }).click();
  await expect.poll(() => page.locator(".caption-original").evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBe(480);
  expect(mocks.commandWrites).toEqual([]); expect(mocks.writes).toEqual([]);
});

test("switching caption jobs requires explicit draft discard and isolates item state", async ({ page }) => {
  const mocks = await captionFixture(page);
  const next = { id: "caption-2", profile_name: "Second job", state: "done", total: 1, completed: 1, failed: 0,
    items: [{ id: "image-0", name: "second.png", state: "ready", caption: "", proposed_caption: "Second candidate", url: "" }] };
  await page.route((url) => url.pathname === "/api/captioning/jobs", (route) => route.fulfill({ json: { jobs: [next] } }));
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-2", (route) => route.fulfill({ json: { job: next } }));
  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("textbox", { name: "候选标注", exact: true }).fill("Unsaved first job");
  await Promise.all([
    page.waitForEvent("dialog").then((dialog) => dialog.dismiss()),
    page.getByRole("button", { name: /Second job/ }).click(),
  ]);
  await expect(page).toHaveURL(/job=caption-1/);
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Unsaved first job");
  await Promise.all([
    page.waitForEvent("dialog").then((dialog) => dialog.accept()),
    page.getByRole("button", { name: /Second job/ }).click(),
  ]);
  await expect(page).toHaveURL(/job=caption-2/);
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Second candidate");
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 0 项");
  await expect(page.locator(".caption-image-error")).toContainText("未保存可用图片地址");
  expect(mocks.commandWrites).toEqual([]); expect(mocks.writes).toEqual([]);
});

import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("training editor uses an independent custom scrollbar on desktop and page scrolling on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 800 });
  await mockWorkspace(page);
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();

  const groups = page.locator(".training-edit-groups");
  const track = page.getByRole("scrollbar", { name: "训练参数滚动条" });
  await page.getByRole("tab", { name: "设备与性能" }).click();
  while (await groups.evaluate((node) => node.scrollHeight - node.clientHeight) < 300) {
    const collapsed = page.locator('#training-stage-resources .training-resource-group button[aria-expanded="false"]').first();
    if (await collapsed.count() === 0) break;
    await collapsed.click();
  }
  await expect.poll(() => track.getAttribute("aria-valuemax")).not.toBe("0");
  const initial = await groups.evaluate((node) => ({
    top: node.scrollTop,
    max: node.scrollHeight - node.clientHeight,
    nativeWidth: getComputedStyle(node).scrollbarWidth,
    webkitDisplay: getComputedStyle(node, "::-webkit-scrollbar").display,
  }));
  expect(initial.nativeWidth).toBe("none");
  expect(initial.webkitDisplay).toBe("none");
  expect(initial.max).toBeGreaterThan(100);

  await groups.hover();
  await page.mouse.wheel(0, 450);
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBeGreaterThan(100);
  expect(await page.locator(".training-config-page").evaluate((node) => node.scrollTop)).toBe(0);
  expect(await page.evaluate(() => window.scrollY)).toBe(0);

  const trackBox = (await track.boundingBox())!;
  await page.mouse.click(trackBox.x + trackBox.width / 2, trackBox.y + trackBox.height * 0.85);
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBeGreaterThan(initial.max * 0.5);
  const thumb = page.locator(".training-editor-scroll-thumb");
  const thumbBox = (await thumb.boundingBox())!;
  await page.mouse.move(thumbBox.x + thumbBox.width / 2, thumbBox.y + thumbBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(trackBox.x + trackBox.width / 2, trackBox.y + 3, { steps: 5 });
  await page.mouse.up();
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBeLessThan(initial.max * 0.2);

  await track.focus();
  await page.keyboard.press("End");
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBeGreaterThan(initial.max * 0.9);
  await page.keyboard.press("Home");
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBe(0);
  await page.keyboard.press("PageDown");
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);

  const maxBefore = Number(await track.getAttribute("aria-valuemax"));
  const expanded = page.locator("#training-stage-resources .training-resource-group").first().locator("h3 button");
  await expanded.click();
  await expect(expanded).toHaveAttribute("aria-expanded", "false");
  await expect.poll(async () => Number(await track.getAttribute("aria-valuemax"))).toBeLessThan(maxBefore);
  await expanded.click();
  await expect.poll(async () => Number(await track.getAttribute("aria-valuemax"))).toBeGreaterThan(maxBefore - 2);

  await page.getByRole("tab", { name: "训练计划" }).click();
  await expect.poll(() => groups.evaluate((node) => node.scrollTop)).toBe(0);

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(track).toBeHidden();
  expect(await groups.evaluate((node) => getComputedStyle(node).overflowY)).toBe("visible");
  const pageScroll = page.locator(".training-config-page");
  await groups.hover();
  await page.mouse.wheel(0, 500);
  await expect.poll(() => pageScroll.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
  expect(await groups.evaluate((node) => node.scrollTop)).toBe(0);
});

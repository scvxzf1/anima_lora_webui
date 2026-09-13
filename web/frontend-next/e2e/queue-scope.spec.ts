import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("bulk queue command fixes the confirmed revision and requires refresh after a race", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let revision = "snapshot-1";
  const items = [{ id: "queued-1", state: "queued", variant: "First" }];
  const snapshot = () => ({ revision, paused: true, status: "idle", items, summary: { total: items.length, queued: items.filter((item) => item.state === "queued").length } });
  const requests: unknown[] = [];
  await page.route((url) => url.pathname === "/api/training/queue", (route) => route.fulfill({ json: snapshot() }));
  await page.route((url) => url.pathname === "/api/training/queue/cancel-waiting", (route) => {
    const body = route.request().postDataJSON(); requests.push(body);
    if (body.expected_revision !== revision) return route.fulfill({ status: 409, json: { error: "队列已发生变化，本次操作未执行。请刷新队列并重新确认范围。" } });
    items.forEach((item) => { item.state = "canceled"; }); revision = "snapshot-3";
    return route.fulfill({ json: { ...snapshot(), message: "已取消 2 个等待任务" } });
  });
  await page.goto("/next/queue");
  await expect(page.locator(".queue-card")).toHaveCount(1);
  // A server-side enqueue between the visible snapshot and command acceptance.
  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toContain("当前快照涉及 1 条队列记录");
    items.push({ id: "queued-2", state: "queued", variant: "New" }); revision = "snapshot-2";
    await dialog.accept();
  });
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("本次操作未执行");
  expect(requests).toEqual([{ expected_revision: "snapshot-1" }]);
  expect(items.every((item) => item.state === "queued")).toBe(true);
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.locator(".queue-card")).toHaveCount(2);
  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toContain("当前快照涉及 2 条队列记录");
    await dialog.accept();
  });
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("已取消 2 个等待任务");
  expect(requests).toEqual([{ expected_revision: "snapshot-1" }, { expected_revision: "snapshot-2" }]);
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("bulk operations do not fall back to unguarded requests when the revision is absent", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/queue");
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("缺少队列快照版本");
  expect(mocks.writes).toEqual([]);
});

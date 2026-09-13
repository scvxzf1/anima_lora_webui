import { expect, test, type WebSocketRoute } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("large history stays paged and comparisons only read snapshots", async ({
  page,
}, info) => {
  const mocks = await mockWorkspace(page);
  await page.route(
    (url) => url.pathname === "/api/training/history",
    (route) => {
      const limit =
        Number(new URL(route.request().url()).searchParams.get("limit")) || 200;
      const cursor = Number(new URL(route.request().url()).searchParams.get("cursor")) || 0;
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          total: 10000,
          next_cursor: cursor + limit < 10000 ? cursor + limit : null,
          tasks: Array.from({ length: Math.min(10000 - cursor, limit) }, (_, index) => {
            const i = cursor + index;
            return ({
            id: `run-${i}`,
            name: `Run ${i}`,
            group: "Portrait studies",
            history_group_key: `config-${i % 2}`,
            history_source_config_file: `config-${i % 2}.toml`,
            state: "idle",
            job: "training",
          }); }),
        }),
      });
    },
  );
  await page.route(/\/api\/training\/history\/run-\d+$/, (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        task: {
          name: new URL(route.request().url()).pathname.split("/").at(-1),
          job: "training",
          state: "idle",
        },
        metrics: Array.from({ length: 20 }, (_, step) => ({
          step,
          loss: 1 / (step + 1),
        })),
      }),
    }),
  );
  await page.goto("/next/history?collection=Portrait%20studies&layout=list");
  await expect(page.locator(".history-card")).toHaveCount(100);
  await expect(page.getByRole("button", { name: "重命名集合" })).toBeDisabled();
  await page.getByRole("button", { name: "载入更多记录" }).click();
  await expect(page.locator(".history-card")).toHaveCount(100);
  await page.getByRole("checkbox", { name: "选择 Run 0", exact: true }).check();
  await page.getByRole("checkbox", { name: "选择 Run 1", exact: true }).check();
  await page
    .getByRole("button", { name: "对比记录 (2-4)", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "历史任务对比" }),
  ).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(1);
  await page.screenshot({ path: info.outputPath("history-comparison.png") });
  await page.keyboard.press("Escape");
  await page
    .getByRole("button", { name: "config-0.toml", exact: true })
    .click();
  await expect(page).toHaveURL(/config=config-0/);
  await expect(
    page.getByRole("button", { name: "下移配置分组" }),
  ).toBeEnabled();
  expect(mocks.writes).toEqual([]);
});

test("large logs metrics image pages and reconnect stay bounded", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const sockets: WebSocketRoute[] = [];
  const active = new Set<WebSocketRoute>();
  await page.routeWebSocket("**/ws/training", (socket) => {
    sockets.push(socket);
    active.add(socket);
    socket.onClose(() => active.delete(socket));
  });
  await page.route((url) => url.pathname === "/api/training/metrics", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(
        Array.from({ length: 50000 }, (_, step) => ({
          step,
          loss: 1 / (step + 1),
        })),
      ),
    }),
  );
  await page.route("**/api/training/logs?*", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        records: Array.from({ length: 50000 }, (_, id) => ({
          id,
          line: `Fixture log ${id}`,
        })),
      }),
    }),
  );
  await page.goto("/next/monitor");
  await expect(
    page.getByRole("img", { name: /Loss 趋势，2000 个点/ }),
  ).toBeVisible();
  await expect(page.getByRole("log")).toContainText("Fixture log 49999");
  expect(
    (await page.getByRole("log").innerText()).split("\n").length,
  ).toBeLessThanOrEqual(300);
  await expect.poll(() => active.size).toBe(1);
  const previousConnections = sockets.length;
  const connected = [...active][0];
  await connected.close();
  active.delete(connected);
  await expect.poll(() => sockets.length).toBe(previousConnections + 1);
  await expect(page.getByText("实时连接已建立")).toBeVisible();
  for (let i = 0; i < 20; i++) {
    await page
      .getByRole("navigation", { name: "主导航" })
      .getByRole("link", { name: "训练队列" })
      .click();
    await expect(
      page.getByRole("heading", { name: "训练队列", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("navigation", { name: "主导航" })
      .getByRole("link", { name: "当前监控" })
      .click();
    await expect(
      page.getByRole("heading", { name: "当前监控", exact: true }),
    ).toBeVisible();
  }
  await expect.poll(() => active.size).toBe(1);
  await page.route("**/api/config/dataset-presets/images?*", (route) => {
    const start =
      Number(new URL(route.request().url()).searchParams.get("offset")) || 0;
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        total: 1000,
        images: Array.from({ length: 60 }, (_, i) => ({
          file: `image-${start + i}.png`,
          name: `image-${start + i}.png`,
          url: `/api/config/dataset-presets/image?image=${start + i}`,
          caption: { text: "fixture" },
        })),
      }),
    });
  });
  await page.goto("/next/captioning");
  await page
    .getByRole("combobox", { name: "数据集预设", exact: true })
    .selectOption("configs/datasets/studio.toml");
  await page.getByRole("button", { name: "扫描图片", exact: true }).click();
  await expect(page.locator(".caption-image-grid img")).toHaveCount(60);
  await page.getByRole("button", { name: "下一页图片", exact: true }).click();
  await expect(page.locator(".caption-image-grid img")).toHaveCount(60);
  expect(mocks.writes).toEqual([]);
});

import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";

const target = process.env.DRAGON_VERIFY_URL;
if (
  !target ||
  !/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(target)
) {
  throw new Error("Set DRAGON_VERIFY_URL to the loopback backend origin");
}
const output =
  process.env.DRAGON_VERIFY_OUTPUT || "/tmp/dragon-next-production-check";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: "chrome" });
const viewports = [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
  { width: 360, height: 800 },
];
const report = {
  generatedAt: new Date().toISOString(),
  target,
  browser: { name: "Chrome", version: browser.version() },
  viewports,
  staticPackage: null,
  staticAssets: [],
  pages: [],
  scaleChecks: [],
  themeChecks: [],
  errors: [],
  blockedCommands: [],
  firstRouteGzipBytes: 0,
};
try {
  const page = await browser.newPage({ viewport: viewports[0] });
  let scaleOverride = null;
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    if (!["GET", "HEAD", "OPTIONS"].includes(request.method())) {
      report.blockedCommands.push(new URL(route.request().url()).pathname);
      return route.abort();
    }
    const url = new URL(request.url());
    if (
      request.method() === "GET" &&
      url.pathname === "/api/settings/global" &&
      scaleOverride !== null
    ) {
      const response = await route.fetch();
      const settings = await response.json();
      return route.fulfill({
        response,
        json: {
          ...settings,
          ui_scale: scaleOverride,
          ui_scale_config: scaleOverride,
        },
      });
    }
    return route.continue();
  });
  page.on("pageerror", (error) => report.errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") report.errors.push(message.text());
  });
  const bundles = new Map();
  let collect = true;
  const responses = [];
  const assetResponses = [];
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (path.startsWith("/static/dragon-next/") && response.status() >= 400)
      report.errors.push(`${response.status()} ${path}`);
    if (path.startsWith("/static/dragon-next/assets/")) {
      assetResponses.push(
        response.body().then((body) => ({
          path,
          status: response.status(),
          bytes: body.length,
          sha256: createHash("sha256").update(body).digest("hex"),
        })),
      );
    }
    if (collect && path.endsWith(".js"))
      responses.push(
        response
          .body()
          .then((body) => bundles.set(path, gzipSync(body).length)),
      );
  });
  const entryResponse = await page.request.get(
    `${target}/static/dragon-next/index.html`,
  );
  const entryBody = await entryResponse.body();
  const entryHtml = entryBody.toString("utf8");
  const entryAssets = [
    ...entryHtml.matchAll(/(?:src|href)="([^"]+\/assets\/[^"]+)"/g),
  ].map((match) => match[1]);
  report.staticPackage = {
    status: entryResponse.status(),
    etag: entryResponse.headers().etag || null,
    lastModified: entryResponse.headers()["last-modified"] || null,
    entrySha256: createHash("sha256").update(entryBody).digest("hex"),
    assets: entryAssets,
  };
  if (entryResponse.status() !== 200)
    report.errors.push(`Static entry: ${entryResponse.status()}`);

  const routes = [
    ["training", "训练配置"],
    ["datasets", "数据集蓝图"],
    ["queue", "训练队列"],
    ["monitor", "当前监控"],
    ["history", "历史任务"],
    ["models", "模型配置"],
    ["captioning", "打标工作台"],
    ["settings", "全局设置"],
  ];
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    for (const [route, title] of routes) {
      const start = performance.now();
      const response = await page.goto(`${target}/next/${route}`);
      const heading = page.getByRole("heading", { name: title, exact: true });
      await heading.waitFor();
      if (route === "training") {
        await page.getByRole("tab", { name: "训练计划", exact: true }).click();
        await page
          .getByRole("textbox", { name: "输出名称", exact: true })
          .waitFor();
        if (collect) {
          await Promise.all(responses);
          collect = false;
          report.firstRouteGzipBytes = [...bundles.values()].reduce(
            (sum, size) => sum + size,
            0,
          );
        }
      }
      report.pages.push({
        route,
        title: await heading.innerText(),
        viewport,
        status: response.status(),
        readyMs: Math.round(performance.now() - start),
        overflow: await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        ),
      });
      await page.screenshot({
        path: `${output}/${route}-${viewport.width}x${viewport.height}.png`,
      });
    }
  }
  report.staticAssets = [
    ...new Map(
      (await Promise.all(assetResponses)).map((asset) => [asset.path, asset]),
    ).values(),
  ].sort((a, b) => a.path.localeCompare(b.path));

  await page.setViewportSize({ width: 1280, height: 720 });
  for (const scale of [125, 150, 200]) {
    scaleOverride = scale;
    const response = await page.goto(`${target}/next/training`);
    await page
      .getByRole("heading", { name: "训练配置", exact: true })
      .waitFor();
    await page.getByRole("tab", { name: "训练计划", exact: true }).click();
    await page
      .getByRole("textbox", { name: "输出名称", exact: true })
      .waitFor();
    const measurements = await page.evaluate(() => {
      const layout = document.querySelector(".next-layout");
      const content = document.querySelector(".next-content");
      return {
        layoutZoom: Number(getComputedStyle(layout).zoom),
        contentZoom: Number(getComputedStyle(content).zoom),
        overflow: document.documentElement.scrollWidth > innerWidth,
      };
    });
    report.scaleChecks.push({
      scale,
      status: response.status(),
      ...measurements,
    });
    await page.screenshot({ path: `${output}/training-scale-${scale}.png` });
  }
  scaleOverride = null;

  await page.goto(`${target}/next/training`);
  await page.getByRole("heading", { name: "训练配置", exact: true }).waitFor();
  const defaultTheme = await page.evaluate(
    () => document.documentElement.dataset.theme,
  );
  await page.getByRole("button", { name: "浅色外观", exact: true }).click();
  await page.waitForFunction(
    () => document.documentElement.dataset.theme === "light",
  );
  report.themeChecks.push({ defaultTheme, testedTheme: "light" });
  await page.screenshot({ path: `${output}/training-light-1280x720.png` });

  for (const mode of ["dragon", "classic"]) {
    const response = await page.request.get(`${target}/?ui=${mode}`);
    if (response.status() !== 200)
      report.errors.push(`Legacy ${mode}: ${response.status()}`);
  }
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (
    report.errors.length ||
    report.blockedCommands.length ||
    report.pages.length !== routes.length * viewports.length ||
    report.pages.some((page) => page.overflow || page.status !== 200) ||
    report.staticAssets.some((asset) => asset.status !== 200) ||
    report.scaleChecks.some(
      (check) => check.overflow || check.status !== 200,
    ) ||
    report.themeChecks.some(
      (check) => check.defaultTheme !== "dark" || check.testedTheme !== "light",
    )
  )
    process.exitCode = 1;
} finally {
  await browser.close();
}

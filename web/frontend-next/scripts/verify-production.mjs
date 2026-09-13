import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
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
const report = {
  pages: [],
  errors: [],
  blockedCommands: [],
  firstRouteGzipBytes: 0,
};
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
  });
  await page.route("**/api/**", (route) => {
    if (!["GET", "HEAD", "OPTIONS"].includes(route.request().method())) {
      report.blockedCommands.push(new URL(route.request().url()).pathname);
      return route.abort();
    }
    return route.continue();
  });
  page.on("pageerror", (error) => report.errors.push(error.message));
  const bundles = new Map();
  let collect = true;
  const responses = [];
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (path.startsWith("/static/dragon-next/") && response.status() >= 400)
      report.errors.push(`${response.status()} ${path}`);
    if (collect && path.endsWith(".js"))
      responses.push(
        response
          .body()
          .then((body) => bundles.set(path, gzipSync(body).length)),
      );
  });
  for (const [route, title] of [
    ["training", "训练配置"],
    ["datasets", "数据集蓝图"],
    ["queue", "训练队列"],
    ["monitor", "当前监控"],
    ["history", "历史任务"],
    ["models", "模型配置"],
    ["captioning", "打标工作台"],
    ["settings", "全局设置"],
  ]) {
    const start = performance.now();
    const response = await page.goto(`${target}/next/${route}`);
    await page.getByRole("heading", { name: title, exact: true }).waitFor();
    if (route === "training") {
      await page
        .getByRole("textbox", { name: "输出名称", exact: true })
        .waitFor();
      await Promise.all(responses);
      collect = false;
      report.firstRouteGzipBytes = [...bundles.values()].reduce(
        (sum, size) => sum + size,
        0,
      );
    }
    report.pages.push({
      route,
      status: response.status(),
      readyMs: Math.round(performance.now() - start),
      overflow: await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
    });
    await page.screenshot({ path: `${output}/${route}.png` });
  }
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
    report.pages.some((page) => page.overflow || page.status !== 200)
  )
    process.exitCode = 1;
} finally {
  await browser.close();
}

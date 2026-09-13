import { expect, test, type Page } from "@playwright/test";
import { parse, stringify } from "smol-toml";
import { configFile, mockWorkspace } from "./fixtures";

async function fixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const second = { ...configFile, path: "configs/imported/second.toml", label: "Second" };
  const files = [configFile, second];
  const values: Record<string, Record<string, unknown>> = Object.fromEntries(files.map((file, index) => [file.path, {
    output_name: index ? "second" : "studio-portrait", model_family: "krea2_raw", network_dim: 32, max_train_steps: 1600,
  }]));
  let saveFails = true;
  let preflightFails = true;
  const commands: { path: string; body: Record<string, unknown> }[] = [];
  await page.route((url) => url.pathname === "/api/config/file-groups", (route) => route.fulfill({ json: [{ id: "imported", label: "Studio", files }] }));
  await page.route((url) => ["/api/config/raw", "/api/config/merged"].includes(url.pathname), (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "PATCH") {
      const body = route.request().postDataJSON(); commands.push({ path: url.pathname, body });
      if (saveFails) return route.fulfill({ status: 409, json: { error: "disk readonly" } });
      values[body.file] = { ...values[body.file], ...body.values };
      return route.fulfill({ json: { ok: true, content: stringify(values[body.file]), message: "保存成功" } });
    }
    const path = url.searchParams.get("file") || url.searchParams.get("config_file") || configFile.path;
    return route.fulfill({ json: url.pathname.endsWith("merged") ? values[path] : { file: path, content: stringify(values[path]), meta: files.find((file) => file.path === path) } });
  });
  await page.route((url) => ["/api/config/raw/patch-preview", "/api/config/raw/save-as", "/api/training/preflight", "/api/training/queue"].includes(url.pathname), (route) => {
    if (route.request().method() === "GET") return route.fallback();
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON(); commands.push({ path, body });
    if (path.endsWith("patch-preview")) return route.fulfill({ json: { ok: true, content: stringify({ ...values[body.file], ...body.values }) } });
    if (path.endsWith("save-as")) {
      values[body.file] = parse(body.content);
      files.push({ ...configFile, path: body.file, label: "Copied draft" });
      return route.fulfill({ json: { ok: true, file: body.file, message: "已另存" } });
    }
    if (path.endsWith("preflight")) return route.fulfill({ json: { ok: !preflightFails, summary: { errors: preflightFails ? 1 : 0, warnings: 0, checks: 1 }, checks: [{ level: preflightFails ? "error" : "ok", key: "model", message: preflightFails ? "模型文件缺失" : "Fixture ready" }] } });
    return route.fulfill({ json: { ok: true, message: "已冻结配置并暂停入队" } });
  });
  return { ...mocks, commands, values, saveOk: () => { saveFails = false; }, preflightOk: () => { preflightFails = false; } };
}

for (const width of [1440, 390]) {
  test(`training draft, copy and guarded queue workflow ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await fixture(page);
    await page.goto("/next/training");
    await page.getByRole("tab", { name: "训练计划", exact: true }).click();
    const output = page.getByRole("textbox", { name: "输出名称", exact: true });
    await expect(output).toHaveValue("studio-portrait");
    await output.fill("draft-first");
    page.once("dialog", (dialog) => dialog.dismiss());
    await page.getByLabel("当前训练配置", { exact: true }).selectOption("configs/imported/second.toml");
    await expect(output).toHaveValue("draft-first");
    await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue(configFile.path);
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByLabel("当前训练配置", { exact: true }).selectOption("configs/imported/second.toml");
    await expect(output).toHaveValue("second");
    await output.fill("preserved-copy");
    await page.getByRole("button", { name: "保存并入队", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("disk readonly");
    await expect(output).toHaveValue("preserved-copy");
    expect(mocks.commands.some(({ path }) => path.startsWith("/api/training/"))).toBe(false);
    await page.getByRole("button", { name: "另存配置", exact: true }).click();
    await page.getByRole("textbox", { name: "配置名称", exact: true }).fill("draft copy");
    await page.getByRole("button", { name: "确认另存", exact: true }).click();
    await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue("configs/imported/draft_copy.toml");
    await expect(page.getByText("disk readonly", { exact: true })).toHaveCount(0);
    await expect(output).toHaveValue("preserved-copy");
    expect(mocks.values["configs/imported/second.toml"].output_name).toBe("second");
    await page.getByRole("button", { name: "加入队列", exact: true }).click();
    await expect(page.getByRole("dialog")).toContainText("模型文件缺失");
    await page.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
    await expect(page.getByRole("button", { name: "确认入队", exact: true })).toBeDisabled();
    await page.keyboard.press("Escape");
    mocks.preflightOk();
    await page.getByRole("button", { name: "加入队列", exact: true }).click();
    await expect(page.getByRole("dialog")).toContainText("Fixture ready");
    await page.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
    await page.getByRole("button", { name: "确认入队", exact: true }).click();
    await expect(page.getByRole("dialog")).toContainText("已冻结配置并暂停入队");
    const launches = mocks.commands.filter(({ path }) => path === "/api/training/queue");
    expect(launches).toHaveLength(1);
    expect(launches[0].body).toMatchObject({ config_file: "configs/imported/draft_copy.toml", start_paused: true, confirmed: true, confirm_preprocess: true });
    await page.keyboard.press("Escape");
    await page.evaluate(() => scrollTo(0, 0));
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath("training-workspace.png") });
    expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
  });
}

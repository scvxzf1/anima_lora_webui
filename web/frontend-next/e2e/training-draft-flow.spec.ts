import { expect, test, type Page, type Route } from "@playwright/test";
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
    if (route.request().method() === "PUT") {
      const body = route.request().postDataJSON(); commands.push({ path: url.pathname, body });
      values[body.file] = parse(body.content);
      return route.fulfill({ json: { ok: true, file: body.file, content: body.content, message: "保存成功" } });
    }
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

async function revisionConflictFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const patchBodies: Record<string, unknown>[] = [];
  let revision = "server-v1";
  let merged = {
    model_family: "krea2_raw",
    pretrained_model_name_or_path: "models/diffusion_models/krea2_raw.safetensors",
    qwen3: "models/text_encoders/qwen3vl.safetensors",
    vae: "models/vae/qwen.safetensors",
    output_name: "studio-portrait",
    network_dim: 32,
    network_alpha: 32,
    train_batch_size: 1,
    gradient_accumulation_steps: 4,
    learning_rate: 0.00002,
    max_train_steps: 1600,
    dataset_config: "configs/datasets/studio.toml",
    gradient_checkpointing: true,
    base_compute: "nf4",
  };
  const reply = (route: Route, data: unknown, status = 200) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(data),
    });

  await page.route(
    (url) => ["/api/config/raw", "/api/config/merged"].includes(url.pathname),
    async (route) => {
      const url = new URL(route.request().url());
      const path = url.pathname;
      if (route.request().method() === "PATCH") {
        const body = route.request().postDataJSON() as Record<string, unknown>;
        patchBodies.push(body);
        if (patchBodies.length === 1) {
          revision = "server-v2";
          merged = { ...merged, output_name: "server-revision" };
          return reply(route, { ok: false, error: "训练配置已在其他位置修改" }, 409);
        }
        if (body.revision !== revision)
          return reply(route, { ok: false, error: "stale revision" }, 409);
        merged = { ...merged, ...(body.values as Record<string, unknown>) };
        revision = "server-v3";
        return reply(route, {
          ok: true,
          file: configFile.path,
          message: "训练配置已保存",
          content: stringify(merged),
          revision,
          changed: Object.keys(body.values as Record<string, unknown>),
          warnings: [],
        });
      }
      if (path === "/api/config/raw") {
        return reply(route, {
          file: configFile.path,
          content: stringify(merged),
          revision,
          meta: configFile,
        });
      }
      return reply(route, merged);
    },
  );
  return { ...mocks, patchBodies };
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
    await page.getByLabel("当前训练配置", { exact: true }).selectOption("configs/imported/second.toml");
    const switchDialog = page.getByRole("dialog", { name: "放弃未保存修改？" });
    await expect(switchDialog).toBeVisible();
    await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue(configFile.path);
    await switchDialog.getByRole("button", { name: "取消", exact: true }).click();
    await expect(output).toHaveValue("draft-first");
    await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue(configFile.path);
    await page.getByLabel("当前训练配置", { exact: true }).selectOption("configs/imported/second.toml");
    await expect(switchDialog).toBeVisible();
    await switchDialog.getByRole("button", { name: "切换并放弃修改", exact: true }).click();
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

test("training enqueue holds pending state and never retries a lost submission", async ({ page }) => {
  const mocks = await fixture(page);
  mocks.preflightOk();
  const calls: { method: string; path: string; body: unknown }[] = [];
  let releaseQueue: () => void = () => {};
  const queueGate = new Promise<void>((resolve) => { releaseQueue = resolve; });
  await page.route((url) => url.pathname === "/api/training/queue" && url.search === "", async (route) => {
    calls.push({ method: route.request().method(), path: new URL(route.request().url()).pathname, body: route.request().postDataJSON() });
    await queueGate;
    return route.abort();
  });

  await page.goto("/next/training");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await page.getByRole("button", { name: "加入队列", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "确认加入队列" });
  await expect(dialog).toContainText("Fixture ready");
  await dialog.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
  const submit = dialog.getByRole("button", { name: "确认入队", exact: true });
  await submit.click();

  const pending = dialog.getByRole("button", { name: "正在提交", exact: true });
  await expect(pending).toBeDisabled();
  await expect.poll(() => calls.length).toBe(1);
  expect(calls[0]).toMatchObject({
    method: "POST",
    path: "/api/training/queue",
    body: { start_paused: true, confirmed: true, confirm_preprocess: true },
  });
  releaseQueue();

  await expect(dialog.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(dialog.getByRole("button", { name: "确认入队", exact: true })).toBeDisabled();
  await expect(dialog.getByText("已冻结配置并暂停入队", { exact: true })).toHaveCount(0);
  expect(calls).toHaveLength(1);
  expect(mocks.commands.filter(({ path }) => path === "/api/training/queue")).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training configuration read failure can recover on explicit retry", async ({ page }) => {
  const mocks = await fixture(page);
  let attempts = 0;
  let recover = false;
  await page.route((url) => url.pathname === "/api/config/file-groups", async (route) => {
    attempts += 1;
    if (!recover) {
      return route.fulfill({ status: 503, json: { error: "training config unavailable" } });
    }
    return route.fallback();
  });

  await page.goto("/next/training");
  await expect(page.getByRole("alert")).toContainText("training config unavailable");
  const retry = page.getByRole("button", { name: "重试读取", exact: true });
  await expect(retry).toBeEnabled();
  const attemptsBeforeRetry = attempts;
  recover = true;

  await retry.click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  expect(attempts).toBe(attemptsBeforeRetry + 1);
  expect(mocks.commands).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training capability catalog failure blocks execution and recovers on retry", async ({ page }) => {
  const mocks = await fixture(page);
  let recover = false;
  let attempts = 0;
  await page.route((url) => url.pathname === "/api/config/model-families", async (route) => {
    attempts += 1;
    if (!recover) return route.fulfill({ status: 503, json: { error: "capabilities unavailable" } });
    return route.fallback();
  });

  await page.goto("/next/training");
  const error = page.getByRole("alert").filter({ hasText: "模型能力目录不可用" });
  await expect(error).toContainText("capabilities unavailable");
  const start = page.getByRole("button", { name: "立即启动", exact: true });
  const enqueue = page.getByRole("button", { name: "加入队列", exact: true });
  await expect(start).toBeDisabled();
  await expect(enqueue).toBeDisabled();

  const attemptsBeforeRetry = attempts;
  recover = true;
  await error.getByRole("button", { name: "重试", exact: true }).click();
  await expect(error).toHaveCount(0);
  await expect(start).toBeEnabled();
  await expect(enqueue).toBeEnabled();
  expect(attempts).toBe(attemptsBeforeRetry + 1);
  expect(mocks.commands).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training configuration 409 preserves the draft until an explicit reload adopts the server revision", async ({ page }) => {
  const mocks = await revisionConflictFixture(page);
  await page.goto("/next/training");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();

  const output = page.getByRole("textbox", { name: "输出名称", exact: true });
  await expect(output).toHaveValue("studio-portrait");
  await output.fill("local-draft");
  await page.getByRole("button", { name: "保存配置", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("训练配置已在其他位置修改");
  await expect(output).toHaveValue("local-draft");
  await expect(page.getByRole("button", { name: "保存配置", exact: true })).toBeDisabled();
  expect(mocks.patchBodies).toHaveLength(1);
  expect(mocks.patchBodies[0]).toMatchObject({ revision: "server-v1", values: { output_name: "local-draft" } });

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "重新加载配置", exact: true }).click();
  await expect(output).toHaveValue("server-revision");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  expect(mocks.patchBodies).toHaveLength(1);

  await output.fill("after-reload");
  await page.getByRole("button", { name: "保存配置", exact: true }).click();
  await expect(page.getByText("训练配置已保存", { exact: true })).toBeVisible();
  await expect.poll(() => mocks.patchBodies.length).toBe(2);
  expect(mocks.patchBodies[1]).toMatchObject({ revision: "server-v2", values: { output_name: "after-reload" } });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training workspace success renders the editable config and command surface", async ({ page }) => {
  const mocks = await fixture(page);
  await page.goto("/next/training");

  await expect(page.getByRole("heading", { name: "训练配置", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "训练设备快捷选择" })).toBeVisible();
  await expect(page.getByLabel("当前训练配置", { exact: true })).toHaveValue(configFile.path);
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeEnabled();
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training workspace empty config library disables execution safely", async ({ page }) => {
  const mocks = await fixture(page);
  await page.route(url => url.pathname === "/api/config/file-groups", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    return route.fulfill({ json: [] });
  });

  await page.goto("/next/training");
  await expect(page.getByRole("heading", { name: "训练配置", exact: true })).toBeVisible();
  await expect(page.getByLabel("当前训练配置", { exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training workspace loading disables context controls before fixture release", async ({ page }) => {
  const mocks = await fixture(page);
  let announceStarted = () => {};
  let releaseResponse = () => {};
  const started = new Promise<void>(resolve => { announceStarted = resolve; });
  const responseGate = new Promise<void>(resolve => { releaseResponse = resolve; });
  await page.route(url => url.pathname === "/api/config/file-groups", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    announceStarted();
    await responseGate;
    return route.fallback();
  });

  await page.goto("/next/training");
  await started;
  await expect(page.getByText("正在同步训练上下文", { exact: true })).toBeVisible();
  await expect(page.getByLabel("当前训练配置", { exact: true })).toBeDisabled();
  await expect(page.locator(".training-config-source")).toHaveAttribute("aria-busy", "true");
  releaseResponse();
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training optimizer and learning-rate choices can be saved from the editor", async ({ page }) => {
  const mocks = await fixture(page);
  await page.goto("/next/training");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await page.getByRole("button", { name: /优化器与学习率/ }).click();

  const optimizer = page.getByLabel("优化器", { exact: true });
  const optimizerOptions = await optimizer.locator("option").allTextContents();
  expect(optimizerOptions).toContain("Automagic");
  await page.getByRole("button", { name: "查看优化器帮助", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Automagic 属于实验优化器");
  await page.getByRole("button", { name: "关闭字段说明", exact: true }).click();
  await optimizer.selectOption("Automagic");

  const scheduler = page.getByLabel("学习率调度器", { exact: true });
  const schedulerOptions = await scheduler.locator("option").allTextContents();
  expect(schedulerOptions).toContain("constant_with_warmup");
  await page.getByRole("button", { name: "查看学习率调度器帮助", exact: true }).click();
  const schedulerHelp = page.getByRole("dialog");
  await expect(schedulerHelp).toContainText("constant_with_warmup 表示先线性热身再固定");
  await expect(schedulerHelp).toContainText("lr_warmup_steps");
  await page.getByRole("button", { name: "关闭字段说明", exact: true }).click();
  await scheduler.selectOption("constant_with_warmup");

  mocks.saveOk();
  await page.getByRole("button", { name: "保存配置", exact: true }).click();
  await expect(page.getByText("保存成功", { exact: true })).toBeVisible();
  const save = mocks.commands.find(({ path }) => path === "/api/config/raw");
  expect(save?.body).toMatchObject({
    values: { optimizer_type: "Automagic", lr_scheduler: "constant_with_warmup" },
  });

  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training TOML editor saves the selected configuration", async ({ page }) => {
  const mocks = await fixture(page);
  await page.goto("/next/training");
  await page.getByRole("button", { name: "TOML", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "TOML 配置" });
  await expect(dialog).toBeVisible();
  const editor = dialog.getByRole("textbox", { name: "TOML", exact: true });
  await expect(editor).toContainText('output_name = "studio-portrait"');
  const updated = (await editor.inputValue()).replace('output_name = "studio-portrait"', 'output_name = "edited-in-toml"');
  await editor.fill(updated);
  await dialog.getByRole("button", { name: "保存 TOML", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  const save = mocks.commands.find(({ path, body }) => path === "/api/config/raw" && "content" in body);
  expect(save?.body).toMatchObject({ file: configFile.path, content: updated });
  expect(mocks.values[configFile.path].output_name).toBe("edited-in-toml");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "输出名称", exact: true })).toHaveValue("edited-in-toml");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

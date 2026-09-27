import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

type Preset = {
  file: string;
  revision: number;
  datasets: Record<string, unknown>[];
  defaults: Record<string, unknown>;
};

async function datasetPresetFixture(page: Page, baseFile = "configs/datasets/studio.toml") {
  const mocks = await mockWorkspace(page);
  const base: Preset = {
    file: baseFile,
    revision: 1,
    datasets: [{ source_dir: "images/studio", image_dir: "cache/studio", num_repeats: 1, settings: {} }],
    defaults: { resolution: 1024, batch_size: 1 },
  };
  const presets = new Map<string, Preset>([[base.file, base]]);
  const behaviors = new Map<string, "conflict" | "offline-once" | "slow-once" | "unknown-once">();
  const deleteBehaviors = new Map<string, "unknown-after-commit-once">();
  const saveAsCalls: { name: string; body: Record<string, unknown> }[] = [];
  const deleteCalls: string[] = [];
  let releaseSaveAs: () => void = () => {};
  const saveAsGate = new Promise<void>((resolve) => { releaseSaveAs = resolve; });
  let releaseDelete: () => void = () => {};
  const deleteGate = new Promise<void>((resolve) => { releaseDelete = resolve; });
  const summary = (preset: Preset) => ({
    path: preset.file,
    filename: preset.file.split("/").at(-1),
    label: `${preset.file.split("/").at(-1)?.replace(/\.toml$/, "")} / 1 subset`,
    summary: { dataset_count: preset.datasets.length, repeat_total: 1 },
  });
  const response = (preset: Preset, message = "") => ({
    ok: true,
    file: preset.file,
    name: preset.file.split("/").at(-1)?.replace(/\.toml$/, ""),
    content: "# in-memory fixture\n",
    revision: `revision-${preset.revision}`,
    datasets: preset.datasets,
    defaults: preset.defaults,
    stage_schedule_enabled: false,
    stage_schedule: [],
    summary: { dataset_count: preset.datasets.length, repeat_total: 1 },
    message,
  });
  const makePreset = (file: string, body: Record<string, unknown>): Preset => ({
    file,
    revision: 1,
    datasets: body.datasets as Record<string, unknown>[],
    defaults: body.defaults as Record<string, unknown>,
  });

  await page.route((url) =>
    url.pathname === "/api/config/dataset-presets" ||
    url.pathname.startsWith("/api/config/dataset-presets/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    const reply = (body: unknown, status = 200) => route.fulfill({ status, json: body });
    if (path === "/api/config/dataset-presets" && method === "GET") {
      const files = [...presets.values()].map(summary);
      return reply({
        ok: true,
        presets: files,
        groups: files.length ? [{ id: "studio", label: "User presets", files }] : [],
      });
    }
    if (path === "/api/config/dataset-presets/read" && method === "GET") {
      const preset = presets.get(url.searchParams.get("file") || "");
      return preset
        ? reply({ ...response(preset), readonly: false })
        : reply({ ok: false, error: "预设不存在" }, 404);
    }
    if (path === "/api/config/dataset-presets/save-as" && method === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      const name = String(body.name || "dataset").replace(/\.toml$/i, "");
      const file = `configs/datasets/${name}.toml`;
      saveAsCalls.push({ name, body });
      const behavior = behaviors.get(name);
      const attempt = saveAsCalls.filter((call) => call.name === name).length;
      if (behavior === "slow-once" && attempt === 1) await saveAsGate;
      if (behavior === "offline-once" && attempt === 1) return route.abort();
      if (behavior === "conflict") {
        if (!presets.has(file)) {
          const competitor = makePreset(file, {
            datasets: [{ source_dir: "images/external", image_dir: "cache/external", num_repeats: 1, settings: {} }],
            defaults: { resolution: 512 },
          });
          presets.set(file, competitor);
        }
        return reply({ ok: false, error: "目标预设已存在" }, 409);
      }
      if (presets.has(file)) return reply({ ok: false, error: "目标预设已存在" }, 409);
      const preset = makePreset(file, body);
      presets.set(file, preset);
      if (behavior === "unknown-once" && attempt === 1) return route.abort();
      return reply({ ...response(preset, "数据集预设已另存"), readonly: false });
    }
    if (path === "/api/config/dataset-presets" && method === "DELETE") {
      const file = url.searchParams.get("file") || "";
      deleteCalls.push(file);
      const attempt = deleteCalls.filter((call) => call === file).length;
      if (deleteBehaviors.get(file) === "unknown-after-commit-once" && attempt === 1) {
        await deleteGate;
        presets.delete(file);
        return route.abort();
      }
      presets.delete(file);
      return reply({ ok: true, file, message: "数据集预设已删除" });
    }
    return route.fallback();
  });
  return { mocks, presets, behaviors, deleteBehaviors, saveAsCalls, deleteCalls, releaseDelete, releaseSaveAs };
}

test("dataset preset CRUD serializes duplicate submits and preserves state after delete conflict", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const presets = new Map<string, Preset>([["configs/datasets/studio.toml", {
    file: "configs/datasets/studio.toml",
    revision: 1,
    datasets: [{ source_dir: "images/studio", image_dir: "cache/studio", num_repeats: 1, settings: {} }],
    defaults: { resolution: 1024, batch_size: 1 },
  }]]);
  const writes: { method: string; path: string; body: unknown }[] = [];
  const deleteAttempts = new Map<string, number>();
  let copySaveAsCalls = 0;
  let releaseCopy: () => void = () => {};
  const copyGate = new Promise<void>((resolve) => { releaseCopy = resolve; });

  const summary = (preset: Preset) => ({
    path: preset.file,
    filename: preset.file.split("/").at(-1),
    label: `${preset.file.split("/").at(-1)?.replace(/\.toml$/, "")} / 1 subset`,
    summary: { dataset_count: preset.datasets.length, repeat_total: 1 },
  });
  const mutationResponse = (preset: Preset, message: string) => ({
    ok: true,
    file: preset.file,
    name: preset.file.split("/").at(-1)?.replace(/\.toml$/, ""),
    content: "# in-memory fixture\n",
    revision: `revision-${preset.revision}`,
    datasets: preset.datasets,
    defaults: preset.defaults,
    stage_schedule_enabled: false,
    stage_schedule: [],
    summary: { dataset_count: preset.datasets.length, repeat_total: 1 },
    message,
  });
  const library = () => {
    const files = [...presets.values()].map(summary);
    return {
      ok: true,
      presets: files,
      groups: files.length ? [{ id: "studio", label: "User presets", files }] : [],
    };
  };
  const readPreset = (preset: Preset) => ({
    ...mutationResponse(preset, ""),
    readonly: false,
  });

  await page.route((url) => url.pathname.startsWith("/api/config/dataset-presets"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    const path = url.pathname;
    const reply = (body: unknown, status = 200) => route.fulfill({ status, json: body });

    if (path === "/api/config/dataset-presets" && method === "GET") return reply(library());
    if (path === "/api/config/dataset-presets/read" && method === "GET") {
      const preset = presets.get(url.searchParams.get("file") || "");
      return preset
        ? reply(readPreset(preset))
        : reply({ ok: false, error: "预设不存在" }, 404);
    }
    if (path === "/api/config/dataset-presets" && method === "PUT") {
      const body = request.postDataJSON() as Record<string, unknown>;
      writes.push({ method, path, body });
      const preset: Preset = {
        file: String(body.file),
        revision: 1,
        datasets: body.datasets as Record<string, unknown>[],
        defaults: body.defaults as Record<string, unknown>,
      };
      presets.set(preset.file, preset);
      return reply(mutationResponse(preset, "数据集预设已保存"));
    }
    if (path === "/api/config/dataset-presets/save-as" && method === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      writes.push({ method, path, body });
      const name = String(body.name || "dataset").replace(/\.toml$/i, "");
      if (name === "copy-next") {
        copySaveAsCalls += 1;
        if (copySaveAsCalls === 1) await copyGate;
      }
      const file = `configs/datasets/${name}.toml`;
      if (name === "race-copy") {
        presets.set(file, {
          file,
          revision: 1,
          datasets: body.datasets as Record<string, unknown>[],
          defaults: body.defaults as Record<string, unknown>,
        });
      }
      if (presets.has(file)) return reply({ ok: false, error: "目标预设已存在" }, 409);
      const preset: Preset = {
        file,
        revision: 1,
        datasets: body.datasets as Record<string, unknown>[],
        defaults: body.defaults as Record<string, unknown>,
      };
      presets.set(file, preset);
      return reply(mutationResponse(preset, "数据集预设已另存"));
    }
    if (path === "/api/config/dataset-presets" && method === "DELETE") {
      const file = url.searchParams.get("file") || "";
      writes.push({ method, path: `${path}?file=${encodeURIComponent(file)}`, body: null });
      const attempts = (deleteAttempts.get(file) || 0) + 1;
      deleteAttempts.set(file, attempts);
      if (file.endsWith("/copy-next.toml") && attempts === 1) {
        return reply({ ok: false, error: "旧预设删除冲突" }, 409);
      }
      if (file.endsWith("/renamed-copy.toml") && attempts === 1) {
        return reply({ ok: false, error: "服务器版本冲突" }, 409);
      }
      presets.delete(file);
      return reply({ ok: true, file, message: "数据集预设已删除" });
    }
    return route.fallback();
  });

  await page.goto("/next/datasets");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");

  await page.getByRole("button", { name: "新建", exact: true }).click();
  const createDialog = page.getByRole("dialog", { name: "新建数据集预设" });
  await createDialog.getByLabel("预设名称").fill("draft-next");
  await createDialog.getByRole("button", { name: "创建草稿", exact: true }).click();
  await expect(createDialog).toHaveCount(0);
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/draft-next");
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("数据集预设已保存", { exact: true })).toBeVisible();
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("draft-next.toml");
  expect(writes.filter((write) => write.method === "PUT")).toHaveLength(1);

  await page.getByLabel("原始图片目录", { exact: true }).fill("images/local-edit");
  await page.getByRole("button", { name: "另存", exact: true }).click();
  const conflictDialog = page.getByRole("dialog", { name: "另存数据集预设" });
  await conflictDialog.getByLabel("预设名称").fill("race-copy");
  await conflictDialog.getByRole("button", { name: "另存预设", exact: true }).click();
  await expect(conflictDialog).toHaveCount(0);
  await expect(page.getByRole("alert")).toContainText("目标预设已存在");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("draft-next.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/local-edit");
  await expect(page.locator(".dataset-detail-status")).toContainText("有未保存修改");
  await expect(page.getByRole("button", { name: "另存", exact: true })).toBeEnabled();

  await page.getByRole("button", { name: "另存", exact: true }).click();
  const saveAsDialog = page.getByRole("dialog", { name: "另存数据集预设" });
  await saveAsDialog.getByLabel("预设名称").fill("copy-next");
  const saveAsButton = saveAsDialog.getByRole("button", { name: "另存预设", exact: true });
  await saveAsButton.evaluate((button) => {
    const click = new MouseEvent("click", { bubbles: true, cancelable: true });
    button.dispatchEvent(click);
    button.dispatchEvent(click);
  });
  await expect.poll(() => copySaveAsCalls).toBe(1);
  releaseCopy();
  await expect(saveAsDialog).toHaveCount(0);
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("copy-next.toml");
  const saveAsWrites = writes.filter((write) => write.path.endsWith("/save-as"));
  expect(saveAsWrites).toHaveLength(2);
  expect(saveAsWrites[0].body).toMatchObject({ name: "race-copy", datasets: [{ source_dir: "images/local-edit" }] });
  expect(saveAsWrites[1].body).toMatchObject({ name: "copy-next", datasets: [{ source_dir: "images/local-edit" }] });

  await page.getByRole("button", { name: "重命名", exact: true }).click();
  const renameDialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await renameDialog.getByLabel("预设名称").fill("renamed-copy");
  await renameDialog.getByRole("button", { name: "确认重命名", exact: true }).click();
  await expect(renameDialog).toHaveCount(0);
  await expect(page.getByRole("alert")).toContainText("新预设已保存，但旧预设删除失败：旧预设删除冲突");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("renamed-copy.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/local-edit");
  await expect(page.locator(".dataset-detail-status")).toContainText("已同步");
  expect(presets.has("configs/datasets/copy-next.toml")).toBe(true);
  expect(presets.has("configs/datasets/renamed-copy.toml")).toBe(true);

  await page.getByRole("button", { name: "删除", exact: true }).click();
  const deleteDialog = page.getByRole("alertdialog", { name: "删除数据集预设" });
  await deleteDialog.getByRole("button", { name: "删除预设", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("服务器版本冲突");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("renamed-copy.toml");
  await expect(page.getByRole("button", { name: "删除", exact: true })).toBeEnabled();

  await page.getByRole("button", { name: "删除", exact: true }).click();
  await page.getByRole("alertdialog", { name: "删除数据集预设" })
    .getByRole("button", { name: "删除预设", exact: true }).click();
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  expect(deleteAttempts.get("configs/datasets/copy-next.toml")).toBe(1);
  expect(deleteAttempts.get("configs/datasets/renamed-copy.toml")).toBe(2);
  expect(writes.filter((write) => write.method === "DELETE")).toHaveLength(3);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset rename keeps the old preset when creating the new preset conflicts", async ({ page }) => {
  const { mocks, presets, behaviors, deleteCalls } = await datasetPresetFixture(page);
  behaviors.set("taken-name", "conflict");
  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/rename-draft");

  await page.getByRole("button", { name: "重命名", exact: true }).click();
  const conflictDialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await conflictDialog.getByLabel("预设名称").fill("taken-name");
  await conflictDialog.getByRole("button", { name: "确认重命名", exact: true }).click();

  await expect(conflictDialog).toHaveCount(0);
  await expect(page.getByRole("alert")).toContainText("目标预设已存在");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/rename-draft");
  await expect(page.locator(".dataset-detail-status")).toContainText("有未保存修改");
  expect(presets.get("configs/datasets/studio.toml")?.datasets[0].source_dir).toBe("images/studio");
  expect(presets.get("configs/datasets/taken-name.toml")?.datasets[0].source_dir).toBe("images/external");
  expect(deleteCalls).toEqual([]);

  await page.getByRole("button", { name: "重命名", exact: true }).click();
  const retryDialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await retryDialog.getByLabel("预设名称").fill("renamed-after-conflict");
  await retryDialog.getByRole("button", { name: "确认重命名", exact: true }).click();
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("renamed-after-conflict.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/rename-draft");
  expect(presets.has("configs/datasets/studio.toml")).toBe(false);
  expect(deleteCalls).toEqual(["configs/datasets/studio.toml"]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset rename does not delete the old preset when the create response is unknown", async ({ page }) => {
  const { mocks, presets, behaviors, saveAsCalls, deleteCalls } = await datasetPresetFixture(page);
  behaviors.set("unknown-rename", "unknown-once");
  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/rename-unknown");

  await page.getByRole("button", { name: "重命名", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await dialog.getByLabel("预设名称").fill("unknown-rename");
  await dialog.getByRole("button", { name: "确认重命名", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/rename-unknown");
  await expect(page.locator(".dataset-detail-status")).toContainText("有未保存修改");
  expect(saveAsCalls.map((call) => call.name)).toEqual(["unknown-rename"]);
  expect(presets.has("configs/datasets/studio.toml")).toBe(true);
  expect(presets.has("configs/datasets/unknown-rename.toml")).toBe(true);
  expect(deleteCalls).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset save-as reports an offline result, preserves the draft, and allows explicit retry", async ({ page }) => {
  const { mocks, presets, behaviors, saveAsCalls } = await datasetPresetFixture(page);
  behaviors.set("offline-copy", "offline-once");
  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/offline-draft");

  await page.getByRole("button", { name: "另存", exact: true }).click();
  let dialog = page.getByRole("dialog", { name: "另存数据集预设" });
  await dialog.getByLabel("预设名称").fill("offline-copy");
  await dialog.getByRole("button", { name: "另存预设", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/offline-draft");
  await expect(page.locator(".dataset-detail-status")).toContainText("有未保存修改");
  await expect(page.getByRole("button", { name: "另存", exact: true })).toBeEnabled();
  expect(saveAsCalls.map((call) => call.name)).toEqual(["offline-copy"]);
  expect(presets.has("configs/datasets/offline-copy.toml")).toBe(false);

  await page.getByRole("button", { name: "另存", exact: true }).click();
  dialog = page.getByRole("dialog", { name: "另存数据集预设" });
  await dialog.getByLabel("预设名称").fill("offline-copy");
  await dialog.getByRole("button", { name: "另存预设", exact: true }).click();

  await expect(page.locator(".dataset-detail-header h2")).toHaveText("offline-copy.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/offline-draft");
  expect(saveAsCalls.map((call) => call.name)).toEqual(["offline-copy", "offline-copy"]);
  expect(presets.get("configs/datasets/offline-copy.toml")?.datasets[0].source_dir).toBe("images/offline-draft");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset rename reconciles an old-file delete after its committed response is lost", async ({ page }) => {
  const { mocks, presets, deleteBehaviors, saveAsCalls, deleteCalls, releaseDelete } = await datasetPresetFixture(page);
  const oldFile = "configs/datasets/studio.toml";
  const nextFile = "configs/datasets/renamed-after-unknown-delete.toml";
  deleteBehaviors.set(oldFile, "unknown-after-commit-once");

  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/rename-unknown-delete");
  await page.getByRole("button", { name: "重命名", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await dialog.getByLabel("预设名称").fill("renamed-after-unknown-delete");
  await dialog.getByRole("button", { name: "确认重命名", exact: true }).click();

  await expect.poll(() => deleteCalls.length).toBe(1);
  await expect(page.getByRole("button", { name: "重命名", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "保存", exact: true })).toBeDisabled();
  await expect(page.getByLabel("原始图片目录", { exact: true })).toBeDisabled();
  expect(presets.has(oldFile)).toBe(true);
  expect(presets.has(nextFile)).toBe(true);

  releaseDelete();
  await expect(dialog).toHaveCount(0);
  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("renamed-after-unknown-delete.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/rename-unknown-delete");
  await expect(page.locator(".dataset-detail-status")).toContainText("已同步");
  const presetList = page.locator(".dataset-preset-list");
  await expect(presetList.locator(".dataset-preset-title")).toHaveText(["renamed-after-unknown-delete / 1 subset"]);
  expect(presets.has(oldFile)).toBe(false);
  expect(presets.has(nextFile)).toBe(true);
  expect(saveAsCalls.map((call) => call.name)).toEqual(["renamed-after-unknown-delete"]);
  expect(deleteCalls).toEqual([oldFile]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset rename preserves the old preset and dirty draft after an offline create, then retries", async ({ page }) => {
  const { mocks, presets, behaviors, saveAsCalls, deleteCalls } = await datasetPresetFixture(page);
  const oldFile = "configs/datasets/studio.toml";
  const nextFile = "configs/datasets/offline-rename.toml";
  behaviors.set("offline-rename", "offline-once");

  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/offline-rename");
  await page.getByRole("button", { name: "重命名", exact: true }).click();
  let dialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await dialog.getByLabel("预设名称").fill("offline-rename");
  await dialog.getByRole("button", { name: "确认重命名", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/offline-rename");
  await expect(page.locator(".dataset-detail-status")).toContainText("有未保存修改");
  expect(presets.has(oldFile)).toBe(true);
  expect(presets.has(nextFile)).toBe(false);
  expect(saveAsCalls.map((call) => call.name)).toEqual(["offline-rename"]);
  expect(deleteCalls).toEqual([]);

  await page.getByRole("button", { name: "重命名", exact: true }).click();
  dialog = page.getByRole("dialog", { name: "重命名数据集预设" });
  await dialog.getByLabel("预设名称").fill("offline-rename");
  await dialog.getByRole("button", { name: "确认重命名", exact: true }).click();

  await expect(page.locator(".dataset-detail-header h2")).toHaveText("offline-rename.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/offline-rename");
  await expect(page.locator(".dataset-detail-status")).toContainText("已同步");
  expect(presets.has(oldFile)).toBe(false);
  expect(presets.has(nextFile)).toBe(true);
  expect(saveAsCalls.map((call) => call.name)).toEqual(["offline-rename", "offline-rename"]);
  expect(deleteCalls).toEqual([oldFile]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset save-as locks editor controls while its request is pending", async ({ page }) => {
  const { mocks, presets, behaviors, saveAsCalls, deleteCalls, releaseSaveAs } = await datasetPresetFixture(page);
  const oldFile = "configs/datasets/studio.toml";
  const nextFile = "configs/datasets/slow-save-as.toml";
  behaviors.set("slow-save-as", "slow-once");

  await page.goto("/next/datasets");
  await page.getByLabel("原始图片目录", { exact: true }).fill("images/slow-save-as");
  await page.getByRole("button", { name: "另存", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "另存数据集预设" });
  await dialog.getByLabel("预设名称").fill("slow-save-as");
  await dialog.getByRole("button", { name: "另存预设", exact: true }).click();

  await expect.poll(() => saveAsCalls.length).toBe(1);
  await expect(dialog).toHaveCount(0);
  await expect(page.getByRole("button", { name: "保存", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "另存", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "重命名", exact: true })).toBeDisabled();
  await expect(page.getByLabel("原始图片目录", { exact: true })).toBeDisabled();
  expect(presets.has(oldFile)).toBe(true);
  expect(presets.has(nextFile)).toBe(false);

  releaseSaveAs();
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("slow-save-as.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/slow-save-as");
  await expect(page.locator(".dataset-detail-status")).toContainText("已同步");
  expect(presets.has(oldFile)).toBe(true);
  expect(presets.has(nextFile)).toBe(true);
  expect(saveAsCalls.map((call) => call.name)).toEqual(["slow-save-as"]);
  expect(deleteCalls).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("long dataset names and paths remain within narrow and desktop viewports", async ({ page }) => {
  const longName = `dataset-${"a".repeat(96)}-${"训练".repeat(8)}`;
  const longFile = `configs/datasets/${longName}.toml`;
  await datasetPresetFixture(page, longFile);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/next/datasets");

  await expect(page.locator(".dataset-detail-header h2")).toHaveText(`${longName}.toml`);
  await expect(page.locator(".dataset-detail-header p").filter({ hasText: longFile })).toHaveText(longFile);
  await expect(page.locator(".dataset-preset-path")).toHaveText(longFile);

  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    const layout = await page.evaluate(() => {
      const path = document.querySelector<HTMLElement>(".dataset-preset-path");
      const detailPath = document.querySelector<HTMLElement>(".dataset-detail-header p:not(.eyebrow)");
      if (!path || !detailPath) return null;
      return {
        viewport: window.innerWidth,
        documentWidth: document.documentElement.scrollWidth,
        pathRight: path.getBoundingClientRect().right,
        pathScrollWidth: path.scrollWidth,
        pathClientWidth: path.clientWidth,
        pathOverflow: getComputedStyle(path).textOverflow,
        detailPathRight: detailPath.getBoundingClientRect().right,
      };
    });

    expect(layout).not.toBeNull();
    expect(layout!.documentWidth).toBeLessThanOrEqual(layout!.viewport);
    expect(layout!.pathRight).toBeLessThanOrEqual(layout!.viewport);
    expect(layout!.detailPathRight).toBeLessThanOrEqual(layout!.viewport);
    expect(layout!.pathScrollWidth).toBeGreaterThan(layout!.pathClientWidth);
    expect(layout!.pathOverflow).toBe("ellipsis");
  }
});

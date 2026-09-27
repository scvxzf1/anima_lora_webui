import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const importedFile = "configs/datasets/roundtrip.toml";
const importedContent = 'source_dir = "images/roundtrip"\nimage_dir = "cache/roundtrip"\n';

test("dataset import selects the returned preset and exports its exact TOML content", async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  const basePreset = {
    path: "configs/datasets/studio.toml",
    filename: "studio.toml",
    label: "Studio / 1 subset",
    summary: { dataset_count: 1, repeat_total: 1 },
  };
  const importedPreset = {
    path: importedFile,
    filename: "roundtrip.toml",
    label: "roundtrip / 1 subset",
    summary: { dataset_count: 1, repeat_total: 1 },
  };
  let imported = false;
  let releaseImport: () => void = () => {};
  const importGate = new Promise<void>((resolve) => { releaseImport = resolve; });
  const imports: Record<string, unknown>[] = [];
  const reads: string[] = [];
  const library = () => ({
    ok: true,
    presets: imported ? [basePreset, importedPreset] : [basePreset],
    groups: [
      { id: "studio", label: "Studio", files: [basePreset] },
      ...(imported ? [{ id: "imported", label: "Imported", files: [importedPreset] }] : []),
    ],
  });

  await page.route((url) => url.pathname === "/api/config/dataset-presets" && url.search === "", (route) =>
    route.fulfill({ json: library() }),
  );
  await page.route((url) => url.pathname === "/api/config/dataset-presets/import", async (route) => {
    imports.push(route.request().postDataJSON() as Record<string, unknown>);
    await importGate;
    imported = true;
    return route.fulfill({ json: {
      ok: true,
      file: importedFile,
      message: "数据集预设已导入",
      content: importedContent,
      revision: "roundtrip-revision-1",
      datasets: [{ source_dir: "images/roundtrip", image_dir: "cache/roundtrip", num_repeats: 1, settings: {} }],
      defaults: { resolution: 1024, batch_size: 1 },
      summary: { dataset_count: 1, repeat_total: 1 },
    } });
  });
  await page.route((url) => url.pathname === "/api/config/dataset-presets/read", (route) => {
    const file = new URL(route.request().url()).searchParams.get("file") || "";
    reads.push(file);
    if (file !== importedFile) return route.fallback();
    return route.fulfill({ json: {
      ok: true,
      file: importedFile,
      name: "roundtrip",
      content: importedContent,
      revision: "roundtrip-revision-1",
      datasets: [{ source_dir: "images/roundtrip", image_dir: "cache/roundtrip", num_repeats: 1, settings: {} }],
      defaults: { resolution: 1024, batch_size: 1 },
      readonly: false,
      summary: { dataset_count: 1, repeat_total: 1 },
    } });
  });

  await page.goto("/next/datasets");
  await page.getByLabel("选择要导入的预设").setInputFiles({
    name: "roundtrip.toml",
    mimeType: "text/plain",
    buffer: Buffer.from(importedContent),
  });
  const dialog = page.getByRole("dialog", { name: "导入数据集预设" });
  await expect(dialog.getByLabel("预设名称")).toHaveValue("roundtrip");
  const importButton = dialog.getByRole("button", { name: "导入预设", exact: true });
  await importButton.click();
  await expect(importButton).toBeDisabled();
  await expect.poll(() => imports.length).toBe(1);
  expect(imports[0]).toEqual({ name: "roundtrip", content: importedContent, overwrite: false });

  releaseImport();
  await expect(dialog).toHaveCount(0);
  await expect(page.getByText("数据集预设已导入", { exact: true })).toBeVisible();
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("roundtrip.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/roundtrip");
  expect(reads).toContain(importedFile);

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出", exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("roundtrip.toml");
  const destination = info.outputPath(download.suggestedFilename());
  await download.saveAs(destination);
  expect(await readFile(destination, "utf8")).toBe(importedContent);
  expect(reads.filter((file) => file === importedFile).length).toBeGreaterThanOrEqual(2);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset import 409 keeps the source and selection until an explicit retry succeeds", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const imports: Record<string, unknown>[] = [];
  let releaseImport: () => void = () => {};
  const importGate = new Promise<void>((resolve) => { releaseImport = resolve; });
  let imported = false;
  await page.route((url) => url.pathname === "/api/config/dataset-presets" && !url.search, (route) =>
    route.fulfill({ json: {
      ok: true,
      presets: [
        { path: "configs/datasets/studio.toml", filename: "studio.toml", label: "Studio", summary: { dataset_count: 1, repeat_total: 1 } },
        ...(imported ? [{ path: importedFile, filename: "roundtrip.toml", label: "roundtrip", summary: { dataset_count: 1, repeat_total: 1 } }] : []),
      ],
      groups: [],
    } }),
  );
  await page.route((url) => url.pathname === "/api/config/dataset-presets/import", async (route) => {
    imports.push(route.request().postDataJSON() as Record<string, unknown>);
    if (imports.length === 1) {
      await importGate;
      return route.fulfill({ status: 409, json: { ok: false, error: "预设名称已存在" } });
    }
    imported = true;
    return route.fulfill({ json: { ok: true, file: importedFile, message: "数据集预设已导入" } });
  });
  await page.route((url) => url.pathname === "/api/config/dataset-presets/read" && url.searchParams.get("file") === importedFile,
    (route) => route.fulfill({ json: {
      ok: true, file: importedFile, name: "roundtrip", content: importedContent, revision: "import-rev-1",
      datasets: [{ source_dir: "images/roundtrip", image_dir: "cache/roundtrip", num_repeats: 1, settings: {} }],
      defaults: { resolution: 1024, batch_size: 1 }, readonly: false,
      summary: { dataset_count: 1, repeat_total: 1 },
    } }),
  );

  await page.goto("/next/datasets");
  await page.getByLabel("选择要导入的预设").setInputFiles({
    name: "roundtrip.toml", mimeType: "text/plain", buffer: Buffer.from(importedContent),
  });
  const dialog = page.getByRole("dialog", { name: "导入数据集预设" });
  const submit = dialog.getByRole("button", { name: "导入预设", exact: true });
  await submit.click();
  await expect(submit).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "取消" })).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "关闭" })).toBeDisabled();
  await expect(dialog.getByLabel("预设名称")).toBeDisabled();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
  await expect.poll(() => imports.length).toBe(1);
  releaseImport();
  await expect(dialog.getByRole("alert")).toContainText("预设名称已存在");
  await expect(dialog.getByLabel("预设名称")).toHaveValue("roundtrip");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(submit).toBeEnabled();
  await page.waitForTimeout(600);
  expect(imports).toHaveLength(1);

  await submit.click();
  await expect.poll(() => imports.length).toBe(2);
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("roundtrip.toml");
  expect(imports[0]).toEqual(imports[1]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset import lost response requires a library check instead of blind resubmission", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let imported = false;
  let importCount = 0;
  let libraryReads = 0;
  await page.route((url) => url.pathname === "/api/config/dataset-presets" && !url.search, (route) => {
    libraryReads += 1;
    return route.fulfill({ json: {
      ok: true,
      presets: [
        { path: "configs/datasets/studio.toml", filename: "studio.toml", label: "Studio", summary: { dataset_count: 1, repeat_total: 1 } },
        ...(imported ? [{ path: importedFile, filename: "roundtrip.toml", label: "roundtrip", summary: { dataset_count: 1, repeat_total: 1 } }] : []),
      ],
      groups: imported ? [{ id: "imported", label: "Imported", files: [
        { path: importedFile, filename: "roundtrip.toml", label: "roundtrip", summary: { dataset_count: 1, repeat_total: 1 } },
      ] }] : [],
    } });
  });
  await page.route((url) => url.pathname === "/api/config/dataset-presets/import", (route) => {
    importCount += 1;
    imported = true;
    return route.abort("failed");
  });

  await page.goto("/next/datasets");
  await page.getByLabel("选择要导入的预设").setInputFiles({
    name: "roundtrip.toml", mimeType: "text/plain", buffer: Buffer.from(importedContent),
  });
  const dialog = page.getByRole("dialog", { name: "导入数据集预设" });
  await dialog.getByRole("button", { name: "导入预设", exact: true }).click();
  await expect(dialog.getByRole("alert")).toContainText("结果尚未确认");
  await expect(dialog.getByRole("button", { name: "导入预设", exact: true })).toBeDisabled();
  await expect(dialog.getByLabel("预设名称")).toBeDisabled();
  await expect(dialog.getByLabel("预设名称")).toHaveValue("roundtrip");
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await page.waitForTimeout(600);
  expect(importCount).toBe(1);

  const previousReads = libraryReads;
  await dialog.getByRole("button", { name: "关闭并刷新预设库核对" }).click();
  await expect(dialog).toHaveCount(0);
  await expect.poll(() => libraryReads).toBeGreaterThan(previousReads);
  await expect(page.getByRole("button", { name: /数据集封面 roundtrip configs\/datasets\/roundtrip\.toml/ })).toBeVisible();
  expect(importCount).toBe(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dirty dataset edits require discard confirmation before opening the import picker", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/datasets");
  const source = page.getByLabel("原始图片目录", { exact: true });
  await source.fill("images/local-draft");

  const firstChooser = page.waitForEvent("filechooser", { timeout: 500 }).then(() => true).catch(() => false);
  await page.getByRole("button", { name: "导入", exact: true }).click();
  const discard = page.getByRole("dialog", { name: "放弃未保存修改？" });
  await expect(discard).toBeVisible();
  await discard.getByRole("button", { name: "继续编辑" }).click();
  expect(await firstChooser).toBe(false);
  await expect(source).toHaveValue("images/local-draft");
  await expect(page.getByRole("dialog", { name: "导入数据集预设" })).toHaveCount(0);

  const chooserPromise = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "导入", exact: true }).click();
  await page.getByRole("dialog", { name: "放弃未保存修改？" })
    .getByRole("button", { name: "放弃修改并继续" }).click();
  const chooser = await chooserPromise;
  await chooser.setFiles({
    name: "roundtrip.toml",
    mimeType: "text/plain",
    buffer: Buffer.from(importedContent),
  });
  const importDialog = page.getByRole("dialog", { name: "导入数据集预设" });
  await expect(importDialog.getByLabel("预设名称")).toHaveValue("roundtrip");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("saved dataset preset enters and returns from the image workspace", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/datasets");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/studio");

  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await expect(page).toHaveURL(/\/next\/datasets\/workspace\/preview\?/);
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await expect(page.locator(".dataset-preview-card")).toHaveCount(3);

  await page.getByRole("button", { name: "返回数据集", exact: true }).click();
  await expect(page).toHaveURL(/\/next\/datasets\?dataset=/);
  await expect(page.locator(".dataset-detail-header h2")).toHaveText("studio.toml");
  await expect(page.getByLabel("原始图片目录", { exact: true })).toHaveValue("images/studio");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

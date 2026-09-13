import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

for (const width of [1440, 390]) {
  test(`library names use in-page dialogs ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await mockWorkspace(page);
    let file = { ...configFile };
    let groupLabel = "Studio";
    const extra: { id: string; label: string; files: (typeof file)[] }[] = [];
    const writes: { url: string; body: Record<string, string> }[] = [];
    await page.route("**/api/config/file-groups?kind=training", (route) =>
      route.fulfill({
        json: [{ id: "studio", label: groupLabel, files: [file] }, ...extra],
      }),
    );
    await page.route("**/api/config/file-groups", async (route) => {
      const body = route.request().postDataJSON();
      writes.push({ url: "create", body });
      if (body.label === "Taken")
        return route.fulfill({ status: 400, json: { error: "名称已存在" } });
      extra.push({ id: "new", label: body.label, files: [] });
      await route.fulfill({ json: { ok: true } });
    });
    await page.route("**/api/config/file-groups/studio", async (route) => {
      const body = route.request().postDataJSON();
      writes.push({ url: "group", body });
      groupLabel = body.label;
      await route.fulfill({ json: { ok: true } });
    });
    await page.route("**/api/config/raw/rename", async (route) => {
      const body = route.request().postDataJSON();
      writes.push({ url: "file", body });
      file = {
        ...file,
        path: body.target,
        filename: "renamed.toml",
        label: "renamed.toml",
      };
      await route.fulfill({ json: { ok: true, file: file.path } });
    });
    const nativeDialogs: string[] = [];
    page.on("dialog", async (dialog) => {
      nativeDialogs.push(dialog.type());
      await dialog.dismiss();
    });
    await page.goto("/next/training");
    await page.getByRole("tab", { name: "训练计划", exact: true }).click();
    await expect(
      page.getByRole("textbox", { name: "输出名称", exact: true }),
    ).toHaveValue("studio-portrait");
    if (width === 390)
      await page
        .getByRole("button", { name: "展开配置库", exact: true })
        .click();
    const create = page.getByRole("button", { name: "新建分组", exact: true });
    await create.click();
    let dialog = page.getByRole("dialog", { name: "新建分组", exact: true });
    await expect(dialog.getByLabel("分组名称")).toBeFocused();
    await expect(
      dialog.getByRole("button", { name: "确认", exact: true }),
    ).toBeDisabled();
    await page.keyboard.press("Escape");
    await expect(create).toBeFocused();
    expect(writes).toHaveLength(0);
    await create.click();
    await dialog.getByLabel("分组名称").fill("Taken");
    await dialog.getByRole("button", { name: "确认", exact: true }).click();
    await expect(dialog.getByRole("alert")).toHaveText("名称已存在");
    await dialog.getByLabel("分组名称").fill("Empty new group");
    await dialog.getByRole("button", { name: "确认", exact: true }).click();
    await expect(dialog).not.toBeVisible();
    await expect(
      page.locator(".training-library-group-toggle", {
        hasText: "Empty new group",
      }),
    ).toBeVisible();
    await page.locator(".training-library-group").filter({ hasText: "Studio" })
      .getByRole("button", { name: "重命名当前分组", exact: true })
      .click();
    dialog = page.getByRole("dialog", { name: "重命名分组", exact: true });
    await expect(dialog.getByLabel("分组名称")).toHaveValue("Studio");
    await dialog.getByLabel("分组名称").fill("New studio");
    await dialog.getByRole("button", { name: "确认", exact: true }).click();
    await expect(dialog).not.toBeVisible();
    await expect(
      page.locator(".training-library-group-toggle", { hasText: "New studio" }),
    ).toBeVisible();
    await page.getByRole("button", { name: `重命名 ${configFile.filename}`, exact: true }).click();
    dialog = page.getByRole("dialog", { name: "重命名文件", exact: true });
    await dialog.getByLabel("文件名称").fill("../bad");
    await expect(
      dialog.getByRole("button", { name: "确认", exact: true }),
    ).toBeDisabled();
    await dialog.getByLabel("文件名称").fill("renamed");
    await page.screenshot({ path: info.outputPath("rename-dialog.png") });
    await dialog.getByRole("button", { name: "确认", exact: true }).click();
    await expect(dialog).not.toBeVisible();
    await expect(
      page.getByRole("combobox", { name: "当前训练配置", exact: true }),
    ).toHaveValue("configs/imported/renamed.toml");
    expect(writes.at(-1)).toEqual({
      url: "file",
      body: {
        source: configFile.path,
        target: "configs/imported/renamed.toml",
      },
    });
    expect(nativeDialogs).toEqual([]);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });
}

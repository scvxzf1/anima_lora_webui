import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

const riskLabel = "我已了解以上风险，仍要继续";
const risk = {
  level: "error",
  key: "mixed_nf4_compile",
  message: "新 NF4 权重存在编译梯度风险",
  overridable: true,
  risk_code: "mixed_nf4_compile",
};
const riskReport = {
  ok: false,
  can_override: true,
  variant: "lora",
  preset: "default",
  methods_subdir: "imported",
  summary: { errors: 1, warnings: 0, checks: 1 },
  checks: [risk],
  errors: [risk],
  warnings: [],
};

for (const mode of ["start", "queue"] as const) {
  test(`NF4 ${mode} requires both confirmations and sends explicit override`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    const requests: Record<string, unknown>[] = [];
    await page.route("**/api/training/preflight", (route) =>
      route.fulfill({ status: 409, json: riskReport }),
    );
    await page.route(`**/api/training/${mode === "start" ? "start" : "queue"}`, (route) => {
      requests.push(route.request().postDataJSON());
      return route.fulfill({ json: { ok: true, message: "Fixture accepted" } });
    });
    await page.goto("/next/training");
    await page.getByRole("button", { name: mode === "start" ? "立即启动" : "加入队列", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: mode === "start" ? "确认启动训练" : "确认加入队列" });
    const ordinary = dialog.getByRole("checkbox", { name: /确认以上检查结果/ });
    const acknowledgement = dialog.getByRole("checkbox", { name: riskLabel, exact: true });
    const submit = dialog.getByRole("button", { name: mode === "start" ? "无视风险继续启动" : "无视风险继续入队", exact: true });
    await expect(dialog).toContainText(risk.message);
    await expect(ordinary).not.toBeChecked();
    await expect(acknowledgement).not.toBeChecked();
    await expect(submit).toBeDisabled();
    await ordinary.check();
    await expect(submit).toBeDisabled();
    expect(requests).toEqual([]);
    await acknowledgement.check();
    await expect(submit).toBeEnabled();
    await submit.click();
    await expect(dialog.getByRole("status")).toContainText("Fixture accepted");
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      config_file: configFile.path,
      confirmed: true,
      confirm_preprocess: true,
      ignore_risks: true,
      ...(mode === "queue" ? { start_paused: true } : {}),
    });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

for (const kind of ["hard-error", "legacy"] as const) {
  test(`${kind} preflight cannot be overridden`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    const report = kind === "legacy"
      ? { ...riskReport, can_override: undefined }
      : {
          ...riskReport,
          checks: [{ level: "error", key: "missing_model", message: "模型文件不存在" }],
          errors: [{ level: "error", key: "missing_model", message: "模型文件不存在" }],
        };
    await page.route("**/api/training/preflight", (route) =>
      route.fulfill({ status: 409, json: report }),
    );
    await page.goto("/next/training");
    await page.getByRole("button", { name: "立即启动", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "确认启动训练" });
    await expect(dialog).toContainText(report.errors[0].message);
    await expect(dialog.getByRole("checkbox", { name: riskLabel })).toHaveCount(0);
    await dialog.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
    await expect(dialog.getByRole("button", { name: "确认启动", exact: true })).toBeDisabled();
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

for (const failure of ["network", "500"] as const) {
  test(`NF4 override ${failure} response remains unknown without retry`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    let calls = 0;
    await page.route("**/api/training/preflight", (route) => route.fulfill({ json: riskReport }));
    await page.route("**/api/training/start", (route) => {
      calls += 1;
      return failure === "network" ? route.abort() : route.fulfill({ status: 500, json: { error: "Fixture failure" } });
    });
    await page.goto("/next/training");
    await page.getByRole("button", { name: "立即启动", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "确认启动训练" });
    const submit = dialog.getByRole("button", { name: "无视风险继续启动", exact: true });
    await dialog.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
    await dialog.getByRole("checkbox", { name: riskLabel }).check();
    await submit.click();
    await expect(dialog.getByRole("alert")).toContainText("启动结果可能未知");
    await dialog.getByRole("checkbox", { name: /确认以上检查结果/ }).check();
    await dialog.getByRole("checkbox", { name: riskLabel }).check();
    await expect(submit).toBeDisabled();
    expect(calls).toBe(1);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

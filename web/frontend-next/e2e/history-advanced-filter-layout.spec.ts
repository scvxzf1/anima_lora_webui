import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const variant = "variant-with-a-long-but-valid-name-that-is-not-on-the-first-page";

test("history advanced filter keeps a URL-only long variant contained at desktop and mobile widths", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/history", (route) =>
    route.fulfill({
      json: {
        ok: true,
        total: 1,
        tasks: [{
          id: "layout-fixture",
          name: "History filter layout fixture",
          job: "training",
          state: "idle",
          archived: false,
          model_family: "krea2_raw",
          training_variant: "different-variant-on-page-one",
          history_source_config_file: "configs/imported/layout-fixture.toml",
          run_dir: "output/runs/layout-fixture",
          started_at_text: "2026-09-30 10:00",
          metric_count: 1,
          log_count: 1,
        }],
      },
    }),
  );

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(`/next/history?variant=${encodeURIComponent(variant)}`);
    const advanced = page.locator(".history-advanced-filter");
    await advanced.locator("summary").click();
    const variantSelect = page.getByRole("combobox", { name: "训练变体" });
    await expect(variantSelect).toHaveValue(variant);
    await expect(variantSelect.locator("option:checked")).toContainText("当前条件");

    const layout = await page.evaluate(() => {
      const controls = [...document.querySelectorAll<HTMLElement>(".history-advanced-grid label")];
      const boxes = controls.map((label) => {
        const labelBox = label.getBoundingClientRect();
        const controlBox = label.querySelector("select")!.getBoundingClientRect();
        return {
          label: { left: labelBox.left, right: labelBox.right, top: labelBox.top, bottom: labelBox.bottom },
          control: { left: controlBox.left, right: controlBox.right, top: controlBox.top, bottom: controlBox.bottom },
        };
      });
      return { scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth, boxes };
    });
    expect(layout.scrollWidth).toBeLessThanOrEqual(layout.innerWidth);
    for (const { label, control } of layout.boxes) {
      expect(control.left).toBeGreaterThanOrEqual(label.left);
      expect(control.right).toBeLessThanOrEqual(label.right + 1);
      expect(control.top).toBeGreaterThanOrEqual(label.top);
      expect(control.bottom).toBeLessThanOrEqual(label.bottom + 1);
    }
    await page.screenshot({ path: `test-results/history-filter-${width}.png`, fullPage: true });
  }

  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

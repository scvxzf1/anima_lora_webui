import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const file = "configs/datasets/studio.toml";

for (const width of [1440, 390]) {
  test(`dataset preview pages remain usable at ${width}px`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    const offsets: number[] = [];
    await page.route(
      (url) => url.pathname === "/api/config/dataset-presets/images",
      (route) => {
        const url = new URL(route.request().url());
        const offset = Number(url.searchParams.get("offset") || 0);
        const limit = Number(url.searchParams.get("limit"));
        offsets.push(offset);
        expect(limit).toBe(24);
        expect(url.searchParams.get("file")).toBe(file);
        const count = Math.min(limit, 50 - offset);
        return route.fulfill({
          json: {
            ok: true,
            file,
            dataset_index: 0,
            dataset_label: "Studio",
            source: "source",
            source_label: "原始图目录",
            directory: "images/studio",
            directory_exists: true,
            caption_extension: ".txt",
            prefer_json_caption: false,
            caption_source_mode: "txt",
            caption_source_label: "TXT",
            caption_summary: "",
            count,
            total: 50,
            limit,
            offset,
            returned: count,
            next_offset: offset + count,
            has_more_before: offset > 0,
            has_more_after: offset + count < 50,
            images: Array.from({ length: count }, (_, index) => {
              const name = `image-${offset + index}.png`;
              return {
                file: `images/studio/${name}`,
                name,
                url: `/api/config/dataset-presets/image?image=${offset + index}`,
                thumbnail_url: `/api/config/dataset-presets/image?image=${offset + index}`,
                caption: {
                  ok: false,
                  file: "",
                  extension: ".txt",
                  source_mode: "txt",
                  source_label: "TXT",
                  detected_mode: "",
                  format_label: "",
                  caption_count: 0,
                  text: "",
                  truncated: false,
                  length: 0,
                },
              };
            }),
            row: { source_dir: "images/studio", num_repeats: 1 },
            settings: { resolution: 1024 },
            message: "",
          },
        });
      },
    );

    const query = new URLSearchParams({ dataset: file, subset: "0" });
    await page.goto(`/next/datasets/workspace/preview?${query}`);
    const panel = page.getByRole("region", { name: "子集 1 图片与标注" });
    await expect(panel.getByRole("img", { name: "image-0.png" })).toBeVisible();
    await expect(panel.getByRole("button", { name: "上一页" })).toBeDisabled();
    await panel.getByRole("button", { name: "下一页" }).click();
    await expect(
      panel.getByRole("img", { name: "image-24.png" }),
    ).toBeVisible();
    await expect(panel.getByRole("img", { name: "image-0.png" })).toHaveCount(
      0,
    );
    await panel.getByRole("spinbutton", { name: "跳转页码" }).fill("3");
    await panel.getByRole("button", { name: "跳转" }).click();
    await expect(
      panel.getByRole("img", { name: "image-48.png" }),
    ).toBeVisible();
    await expect(panel.getByText("第 3 / 3 页 · 共 50 张")).toBeVisible();
    await expect(panel.getByRole("button", { name: "下一页" })).toBeDisabled();
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    await page.screenshot({
      path: info.outputPath(`dataset-preview-page-${width}.png`),
      fullPage: true,
    });
    expect([...new Set(offsets)]).toEqual([0, 24, 48]);
    expect(offsets.length).toBeLessThan(5);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

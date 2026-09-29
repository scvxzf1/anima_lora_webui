import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { DatasetPreviewDialog } from "./DatasetPreviewDialog";
import type { DatasetPreviewResponse } from "./types";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function previewPage(
  offset: number,
  total = 50,
  file = "alpha.toml",
): DatasetPreviewResponse {
  const count = Math.min(24, Math.max(0, total - offset));
  return {
    ok: true,
    file,
    dataset_index: 0,
    dataset_label: "第 1 组数据集",
    source: "source",
    source_label: "原始图目录",
    directory: "image_dataset/alpha",
    directory_exists: true,
    caption_extension: ".txt",
    prefer_json_caption: false,
    caption_source_mode: "auto",
    caption_source_label: "自动识别",
    caption_summary: "",
    count,
    total,
    limit: 24,
    offset,
    returned: count,
    next_offset: Math.min(total, offset + count),
    has_more_before: offset > 0,
    has_more_after: offset + count < total,
    row: { source_dir: "image_dataset/alpha", num_repeats: 1 },
    settings: {},
    message: "",
    images: Array.from({ length: count }, (_, index) => {
      const name = `image-${offset + index}.png`;
      return {
        file: `image_dataset/alpha/${name}`,
        name,
        url: `/api/image/${name}`,
        caption: {
          ok: false,
          file: "",
          extension: ".txt",
          source_mode: "auto",
          source_label: "自动识别",
          detected_mode: "",
          format_label: "",
          caption_count: 0,
          text: "",
          truncated: false,
          length: 0,
        },
      };
    }),
  };
}

function renderPreview(file = "alpha.toml") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={client}>
      <DatasetPreviewDialog
        file={file}
        datasetIndex={0}
        returnFocus={null}
        onClose={vi.fn()}
        embedded
      />
    </QueryClientProvider>,
  );
  return { client, view };
}

it("navigates bounded server pages and a direct page jump", async () => {
  const offsets: number[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const offset = Number(
        new URL(String(input), "http://local").searchParams.get("offset") || 0,
      );
      offsets.push(offset);
      return new Response(JSON.stringify(previewPage(offset)), { status: 200 });
    }),
  );
  const { client, view } = renderPreview();
  const panel = await screen.findByRole("region", {
    name: "子集 1 图片与标注",
  });

  await within(panel).findByRole("img", { name: "image-0.png" });
  expect(within(panel).getByRole("button", { name: "上一页" })).toBeDisabled();
  expect(within(panel).getByRole("button", { name: "下一页" })).toBeEnabled();
  expect(within(panel).getByText("第 1 / 3 页 · 共 50 张")).toBeInTheDocument();
  const body = panel.querySelector(".dataset-preview-body") as HTMLElement;
  body.scrollTop = 500;
  fireEvent.click(within(panel).getByRole("button", { name: "下一页" }));
  expect(body.scrollTop).toBe(0);
  await within(panel).findByRole("img", { name: "image-24.png" });
  expect(
    within(panel).queryByRole("img", { name: "image-0.png" }),
  ).not.toBeInTheDocument();
  fireEvent.change(
    within(panel).getByRole("spinbutton", { name: "跳转页码" }),
    { target: { value: "3" } },
  );
  fireEvent.click(within(panel).getByRole("button", { name: "跳转" }));
  await within(panel).findByRole("img", { name: "image-48.png" });
  expect(within(panel).getAllByRole("img")).toHaveLength(2);
  expect(within(panel).getByRole("button", { name: "下一页" })).toBeDisabled();
  fireEvent.click(within(panel).getByRole("button", { name: "上一页" }));
  await within(panel).findByRole("img", { name: "image-24.png" });
  expect(offsets).toEqual([0, 24, 48, 24]);
  await waitFor(() =>
    expect(
      client.getQueryCache().findAll({
        queryKey: ["datasets", "preview", "alpha.toml", 0],
      }),
    ).toHaveLength(1),
  );
  view.unmount();
  client.clear();
});

it("keeps the latest page and file when an older page response arrives late", async () => {
  let resolveMiddle: ((response: Response) => void) | undefined;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      const offset = Number(url.searchParams.get("offset") || 0);
      const file = url.searchParams.get("file") || "alpha.toml";
      if (offset === 24)
        return new Promise<Response>((resolve) => {
          resolveMiddle = resolve;
        });
      return Promise.resolve(
        new Response(JSON.stringify(previewPage(offset, 50, file)), {
          status: 200,
        }),
      );
    }),
  );
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "下一页" }));
  await waitFor(() => expect(resolveMiddle).toBeDefined());
  expect(
    screen.queryByRole("img", { name: "image-0.png" }),
  ).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("spinbutton", { name: "跳转页码" }), {
    target: { value: "3" },
  });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  await screen.findByRole("img", { name: "image-48.png" });
  resolveMiddle!(
    new Response(JSON.stringify(previewPage(24)), { status: 200 }),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "image-48.png" }),
    ).toBeInTheDocument(),
  );
  expect(
    screen.queryByRole("img", { name: "image-24.png" }),
  ).not.toBeInTheDocument();

  view.rerender(
    <QueryClientProvider client={client}>
      <DatasetPreviewDialog
        file="beta.toml"
        datasetIndex={0}
        returnFocus={null}
        onClose={vi.fn()}
        embedded
      />
    </QueryClientProvider>,
  );
  await screen.findByText("beta.toml", { selector: "dd" });
  expect(
    screen.queryByRole("img", { name: "image-48.png" }),
  ).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("does not offer navigation for an old response without pagination metadata", async () => {
  const {
    offset: _offset,
    returned: _returned,
    next_offset: _next,
    has_more_before: _before,
    has_more_after: _after,
    ...legacy
  } = previewPage(0);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(legacy), { status: 200 })),
  );
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByText(/分页接口尚未生效/)).toBeInTheDocument();
  expect(
    screen.queryByRole("navigation", { name: "图片分页" }),
  ).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("returns to a valid page when the dataset shrinks between requests", async () => {
  const offsets: number[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const offset = Number(
        new URL(String(input), "http://local").searchParams.get("offset") || 0,
      );
      offsets.push(offset);
      const result =
        offset === 48
          ? previewPage(10, 10)
          : previewPage(0, offsets.length === 1 ? 50 : 10);
      return new Response(JSON.stringify(result), { status: 200 });
    }),
  );
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.change(screen.getByRole("spinbutton", { name: "跳转页码" }), {
    target: { value: "3" },
  });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  await waitFor(() => expect(offsets).toEqual([0, 48, 0]));
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getAllByRole("img")).toHaveLength(10);
  expect(
    screen.queryByRole("navigation", { name: "图片分页" }),
  ).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("reports a non-clamped page mismatch and offers a recovery path", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const offset = Number(
        new URL(String(input), "http://local").searchParams.get("offset") || 0,
      );
      expect([0, 24]).toContain(offset);
      return new Response(JSON.stringify(previewPage(0)), { status: 200 });
    }),
  );
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "下一页" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "分页结果与请求位置不一致",
  );
  expect(
    screen.queryByRole("img", { name: "image-0.png" }),
  ).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "返回第一页" }));
  await screen.findByRole("img", { name: "image-0.png" });
  view.unmount();
  client.clear();
});

it("restores the preview scroll position and trigger focus after closing an image", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify(previewPage(0, 1)), { status: 200 }),
    ),
  );
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  body.scrollTop = 240;
  const trigger = screen.getByRole("button", { name: "查看大图 image-0.png" });
  fireEvent.click(trigger);
  expect(
    await screen.findByRole("dialog", { name: "image-0.png" }),
  ).toBeInTheDocument();
  fireEvent.keyDown(window, { key: "Escape" });
  await waitFor(() =>
    expect(
      screen.queryByRole("dialog", { name: "image-0.png" }),
    ).not.toBeInTheDocument(),
  );
  expect(body.scrollTop).toBe(240);
  expect(trigger).toHaveFocus();
  view.unmount();
  client.clear();
});

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { DatasetPreviewDialog } from "./DatasetPreviewDialog";
import type { DatasetPreviewResponse } from "./types";

const preferenceKeys = ["dataset-preview-info-expanded", "dataset-preview-continuous"];

beforeEach(() => preferenceKeys.forEach((key) => localStorage.removeItem(key)));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  preferenceKeys.forEach((key) => localStorage.removeItem(key));
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

it("toggles the info rail and loads adjacent pages only while continuous scrolling is enabled", async () => {
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    offsets.push(offset);
    return new Response(JSON.stringify(previewPage(offset, 72)), { status: 200 });
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  const rail = screen.getByText("数据集文件").closest("aside");
  const collapse = screen.getByRole("button", { name: "折叠信息栏" });
  expect(collapse).toHaveAttribute("aria-expanded", "true");
  fireEvent.click(collapse);
  expect(rail).not.toBeInTheDocument();
  expect(document.querySelector(".dataset-preview-layout")).toHaveAttribute("data-info-expanded", "false");
  fireEvent.click(screen.getByRole("button", { name: "展开信息栏" }));
  expect(screen.getByText("数据集文件")).toBeInTheDocument();

  const toggle = screen.getByRole("button", { name: "连续滚动" });
  expect(toggle).toHaveAttribute("aria-pressed", "false");
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute("aria-pressed", "true");
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 3000 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  body.scrollTop = 2400;
  fireEvent.scroll(body);
  await screen.findByRole("img", { name: "image-24.png" });
  expect(offsets).toEqual([0, 24]);
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute("aria-pressed", "false");
  expect(body.querySelector('[data-preview-offset="24"]')).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("prepends a previous page without moving the reading position", async () => {
  let resolvePrevious: ((response: Response) => void) | undefined;
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    offsets.push(offset);
    if (offset === 24) return new Promise<Response>((resolve) => { resolvePrevious = resolve; });
    return Promise.resolve(new Response(JSON.stringify(previewPage(offset)), { status: 200 }));
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.change(screen.getByRole("spinbutton", { name: "跳转页码" }), { target: { value: "3" } });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  await screen.findByRole("img", { name: "image-48.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 3000 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  const page = await waitFor(() => {
    const element = body.querySelector<HTMLElement>('[data-preview-offset="48"]');
    expect(element).toBeInTheDocument();
    return element!;
  });
  vi.spyOn(page, "getBoundingClientRect").mockImplementation(() => {
    const top = body.querySelector('[data-preview-offset="24"]') ? 300 : 100;
    return { top, bottom: top + 400 } as DOMRect;
  });
  body.scrollTop = 500;
  fireEvent.scroll(body);
  body.scrollTop = 100;
  fireEvent.scroll(body);
  await waitFor(() => expect(resolvePrevious).toBeDefined());
  resolvePrevious!(new Response(JSON.stringify(previewPage(24)), { status: 200 }));
  await screen.findByRole("img", { name: "image-24.png" });
  expect(body.scrollTop).toBe(300);
  expect(offsets).toEqual([0, 48, 24]);
  view.unmount();
  client.clear();
});

it("waits for an explicit retry after a scroll fetch fails and resets on dataset change", async () => {
  let attempts = 0;
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://local");
    const offset = Number(url.searchParams.get("offset") || 0);
    if (offset === 24 && ++attempts === 1) return Promise.reject(new Error("network down"));
    return Promise.resolve(new Response(JSON.stringify(previewPage(offset, 50, url.searchParams.get("file") || "alpha.toml")), { status: 200 }));
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 3000 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  body.scrollTop = 2400;
  fireEvent.scroll(body);
  expect(await screen.findByRole("alert")).toHaveTextContent("network down");
  fireEvent.scroll(body);
  expect(attempts).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "重试下一页" }));
  await screen.findByRole("img", { name: "image-24.png" });
  expect(attempts).toBe(2);
  fireEvent.click(screen.getByRole("button", { name: "折叠信息栏" }));
  view.rerender(<QueryClientProvider client={client}><DatasetPreviewDialog file="beta.toml" datasetIndex={0} returnFocus={null} onClose={vi.fn()} embedded /></QueryClientProvider>);
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) =>
    new URL(String(input), "http://local").searchParams.get("file") === "beta.toml",
  )).toBe(true));
  expect(await screen.findByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "展开信息栏" })).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByRole("img", { name: "image-24.png" })).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("persists both preview preferences across remounts", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(previewPage(0, 1)), { status: 200 })));
  const first = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "false");
  expect(screen.getByRole("button", { name: "折叠信息栏" })).toHaveAttribute("aria-expanded", "true");
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  fireEvent.click(screen.getByRole("button", { name: "折叠信息栏" }));
  expect(localStorage.getItem(preferenceKeys[0])).toBe("false");
  expect(localStorage.getItem(preferenceKeys[1])).toBe("true");
  first.view.unmount();
  first.client.clear();

  const second = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "展开信息栏" })).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  fireEvent.click(screen.getByRole("button", { name: "展开信息栏" }));
  second.view.unmount();
  second.client.clear();

  const third = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "false");
  expect(screen.getByRole("button", { name: "折叠信息栏" })).toHaveAttribute("aria-expanded", "true");
  third.view.unmount();
  third.client.clear();
});

it("falls back to defaults with malformed browser storage", async () => {
  localStorage.setItem(preferenceKeys[0], "invalid");
  localStorage.setItem(preferenceKeys[1], "invalid");
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(previewPage(0, 1)), { status: 200 })));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "false");
  expect(screen.getByRole("button", { name: "折叠信息栏" })).toHaveAttribute("aria-expanded", "true");
  view.unmount();
  client.clear();
});

it("keeps toggles usable when browser storage is denied", async () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("denied"); });
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("denied"); });
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(previewPage(0, 1)), { status: 200 })));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "false");
  expect(screen.getByRole("button", { name: "折叠信息栏" })).toHaveAttribute("aria-expanded", "true");
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  fireEvent.click(screen.getByRole("button", { name: "折叠信息栏" }));
  expect(screen.getByRole("button", { name: "连续滚动" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "展开信息栏" })).toHaveAttribute("aria-expanded", "false");
  view.unmount();
  client.clear();
});

it("loads before a short final page and from the top of a directly opened middle page", async () => {
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    offsets.push(offset);
    return new Response(JSON.stringify(previewPage(offset)), { status: 200 });
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.change(screen.getByRole("spinbutton", { name: "跳转页码" }), { target: { value: "3" } });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  await screen.findByRole("img", { name: "image-48.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 500 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="48"]')).toBeInTheDocument());
  fireEvent.wheel(body, { deltaY: -100 });
  await screen.findByRole("img", { name: "image-24.png" });
  expect(offsets).toEqual([0, 48, 24]);
  view.unmount();
  client.clear();
});

it("aborts an adjacent request on unmount and ignores its late response", async () => {
  let resolvePrevious: ((response: Response) => void) | undefined;
  let adjacentSignal: AbortSignal | undefined;
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    if (offset === 24) {
      adjacentSignal = init?.signal as AbortSignal;
      return new Promise<Response>((resolve) => { resolvePrevious = resolve; });
    }
    return Promise.resolve(new Response(JSON.stringify(previewPage(offset)), { status: 200 }));
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.change(screen.getByRole("spinbutton", { name: "跳转页码" }), { target: { value: "3" } });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  await screen.findByRole("img", { name: "image-48.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  await waitFor(() => expect(body.querySelector('[data-preview-offset="48"]')).toBeInTheDocument());
  fireEvent.wheel(body, { deltaY: -100 });
  await waitFor(() => expect(resolvePrevious).toBeDefined());
  expect(adjacentSignal?.aborted).toBe(false);
  view.unmount();
  expect(adjacentSignal?.aborted).toBe(true);
  resolvePrevious!(new Response(JSON.stringify(previewPage(24)), { status: 200 }));
  client.clear();
});

it("preserves the reading window on base refetch and refreshes the visible page", async () => {
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    offsets.push(offset);
    return new Response(JSON.stringify(previewPage(offset, 72)), { status: 200 });
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 3000 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  body.scrollTop = 2400;
  fireEvent.scroll(body);
  await screen.findByRole("img", { name: "image-24.png" });
  const first = body.querySelector<HTMLElement>('[data-preview-offset="0"]')!;
  const second = body.querySelector<HTMLElement>('[data-preview-offset="24"]')!;
  vi.spyOn(first, "getBoundingClientRect").mockReturnValue({ top: -900, bottom: -100 } as DOMRect);
  vi.spyOn(second, "getBoundingClientRect").mockReturnValue({ top: 100, bottom: 900 } as DOMRect);
  fireEvent.scroll(body);
  await waitFor(() => expect(screen.getByText("第 2 / 3 页 · 共 72 张")).toBeInTheDocument());
  await client.refetchQueries({ queryKey: ["datasets", "preview", "alpha.toml", 0, 0] });
  expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument();
  expect(body.querySelector('[data-preview-offset="24"]')).toBeInTheDocument();
  expect(screen.getByText("第 2 / 3 页 · 共 72 张")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(offsets.slice(-2)).toEqual([0, 24]));
  await screen.findByRole("img", { name: "image-24.png" });
  view.unmount();
  client.clear();
});

it("keeps at most five adjacent pages while advancing the reading anchor", async () => {
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    offsets.push(offset);
    return new Response(JSON.stringify(previewPage(offset, 168)), { status: 200 });
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 500 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  for (const offset of [24, 48, 72, 96, 120, 144]) {
    fireEvent.wheel(body, { deltaY: 100 });
    await waitFor(() => expect(body.querySelector(`[data-preview-offset="${offset}"]`)).toBeInTheDocument());
  }
  expect(body.querySelectorAll("[data-preview-offset]")).toHaveLength(5);
  expect(body.querySelector('[data-preview-offset="0"]')).not.toBeInTheDocument();
  expect(body.querySelector('[data-preview-offset="48"]')).toBeInTheDocument();
  expect(offsets).toEqual([0, 24, 48, 72, 96, 120, 144]);
  expect(screen.getByText("第 5 / 7 页 · 共 168 张")).toBeInTheDocument();
  view.unmount();
  client.clear();
});

it("cancels an old adjacent page when the dataset total changes", async () => {
  let total = 72;
  let resolveAdjacent: ((response: Response) => void) | undefined;
  let adjacentSignal: AbortSignal | undefined;
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const offset = Number(new URL(String(input), "http://local").searchParams.get("offset") || 0);
    if (offset === 24) {
      adjacentSignal = init?.signal as AbortSignal;
      return new Promise<Response>((resolve) => { resolveAdjacent = resolve; });
    }
    return Promise.resolve(new Response(JSON.stringify(previewPage(0, total)), { status: 200 }));
  }));
  const { client, view } = renderPreview();
  await screen.findByRole("img", { name: "image-0.png" });
  fireEvent.click(screen.getByRole("button", { name: "连续滚动" }));
  const body = document.querySelector(".dataset-preview-body") as HTMLElement;
  Object.defineProperty(body, "scrollHeight", { configurable: true, value: 500 });
  Object.defineProperty(body, "clientHeight", { configurable: true, value: 500 });
  await waitFor(() => expect(body.querySelector('[data-preview-offset="0"]')).toBeInTheDocument());
  fireEvent.wheel(body, { deltaY: 100 });
  await waitFor(() => expect(resolveAdjacent).toBeDefined());
  total = 10;
  await client.refetchQueries({ queryKey: ["datasets", "preview", "alpha.toml", 0, 0] });
  await waitFor(() => expect(adjacentSignal?.aborted).toBe(true));
  resolveAdjacent!(new Response(JSON.stringify(previewPage(24, 72)), { status: 200 }));
  await waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(10));
  expect(body.querySelector('[data-preview-offset="24"]')).not.toBeInTheDocument();
  view.unmount();
  client.clear();
});

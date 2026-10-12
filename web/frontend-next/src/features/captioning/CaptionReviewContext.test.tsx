import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { CaptionReviewContext } from "./CaptionReviewContext";

it("reports filtered pages while retaining drafts and selections outside the filter", () => {
  const items = Array.from({ length: 65 }, (_, id) => ({ id: String(id), name: `${id}.png` }));
  const props = { items, visibleItems: [items[64]], current: items[64], page: 0,
    drafts: { "0": "red", "64": "blue" }, checked: ["0", "64"], onLocate: vi.fn(),
    mobileView: "editor" as const, onView: vi.fn() };
  const view = render(<CaptionReviewContext {...props} />);
  expect(screen.getByText("当前编辑：64.png · 第 1 页")).toBeInTheDocument();
  expect(screen.getByText("已勾选 2 项（当前页外 1 项） · 未保存草稿 2 项")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "定位当前候选" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "候选列表 (1/65)" })).toBeInTheDocument();
  view.rerender(<CaptionReviewContext {...props} current={items[0]} />);
  expect(screen.getByText("当前编辑：0.png · 不在当前筛选中")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "定位当前候选" })).toBeInTheDocument();
});

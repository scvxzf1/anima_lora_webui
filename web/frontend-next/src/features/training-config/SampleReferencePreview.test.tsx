import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SampleReferencePreview } from "./SampleReferencePreview";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("requests display size only on opening and waits for async decode", async () => {
  render(<SampleReferencePreview path="/reference.png" label="参考图 1" />);
  const thumbnail = screen.getByRole("img");
  expect(thumbnail.getAttribute("src")).toContain("thumbnail=1");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "展开参考图 1" }));
  const image = screen.getByRole("img", { name: "参考图 1大图" });
  expect(image.getAttribute("src")).toContain("display=1");
  expect(screen.getByRole("status")).toHaveTextContent("正在加载大图");
  let decoded!: () => void;
  Object.defineProperty(image, "decode", { value: () => new Promise<void>((resolve) => { decoded = resolve; }) });
  fireEvent.load(image);
  expect(image).toHaveClass("is-loading");
  await act(async () => decoded());
  expect(image).toHaveClass("is-ready");
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "关闭参考图预览" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("draws the already loaded thumbnail without another thumbnail image request", () => {
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
  render(<SampleReferencePreview path="/reference.png" label="参考图 1" />);
  const thumbnail = screen.getByRole("img");
  Object.defineProperties(thumbnail, { complete: { value: true }, naturalWidth: { value: 160 }, naturalHeight: { value: 80 } });
  fireEvent.click(screen.getByRole("button", { name: "展开参考图 1" }));
  expect(drawImage).toHaveBeenCalledWith(thumbnail, 0, 0);
  expect(screen.getAllByRole("img")).toHaveLength(2);
  fireEvent.error(screen.getByRole("img", { name: "参考图 1大图" }));
  expect(screen.getByRole("alert")).toHaveTextContent("预览加载失败");
});

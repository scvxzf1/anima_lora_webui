import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryImageDialog } from "./HistoryImageDialog";

const images = [{ file: "first.png", name: "first", sample: { prompt: "test prompt", seed: 7 } },
  { file: "second.png", name: "second" }];
const props = { image: images[0], images, index: 0, taskId: "task", onIndexChange: vi.fn(), onClose: vi.fn() };

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("zooms, pans, fits, and resets its view when switching the selected sample", async () => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("PointerEvent", MouseEvent);
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(500);
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(400);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 500, height: 400 } as DOMRect);
  Object.defineProperties(HTMLElement.prototype, {
    setPointerCapture: { configurable: true, value: vi.fn() },
    hasPointerCapture: { configurable: true, value: () => false },
  });
  const user = userEvent.setup();
  const { rerender } = render(<HistoryImageDialog {...props} />);
  const load = (name: string) => {
    const image = screen.getByRole("img", { name });
    Object.defineProperties(image, { naturalWidth: { value: 1000 }, naturalHeight: { value: 800 } });
    fireEvent.load(image);
    return image;
  };
  const image = load("first");
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("50%");
  await user.click(screen.getByRole("button", { name: "放大样张" }));
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("63%");
  const host = image.parentElement!;
  fireEvent.wheel(host, { deltaY: -100, clientX: 250, clientY: 200 });
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("73%");
  const before = image.style.transform;
  fireEvent.pointerDown(host, { button: 0, clientX: 200, clientY: 100 });
  fireEvent.pointerMove(host, { clientX: 240, clientY: 120 });
  expect(image.style.transform).not.toBe(before);
  fireEvent.pointerCancel(host);
  expect(host).toHaveAttribute("data-dragging", "false");
  await user.click(screen.getByRole("button", { name: "重置 100%" }));
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("100%");
  await user.click(screen.getByRole("button", { name: "适应" }));
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("50%");
  await user.click(screen.getByRole("button", { name: "放大样张" }));
  rerender(<HistoryImageDialog {...props} image={images[1]} index={1} />);
  load("second");
  expect(screen.getByLabelText("样张缩放比例")).toHaveTextContent("50%");
  expect(screen.getByRole("link", { name: "下载样张" })).toHaveAttribute("download");
  await user.keyboard("{Escape}");
  expect(props.onClose).toHaveBeenCalled();
});

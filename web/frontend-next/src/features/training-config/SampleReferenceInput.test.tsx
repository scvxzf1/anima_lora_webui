import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { useState } from "react";
import { SampleReferenceInput } from "./SampleReferenceInput";
import { importSampleReference } from "./sampleReferenceApi";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const reference = (name: string) => ({ ok: true, reference_image: `/managed/${name}.png`, width: 512, height: 768, url: `/api/config/sample-references/${name}` });
const hash = (digit: string) => digit.repeat(64);
function response(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status }); }
function Editor({ initial = [], maxReferences = 4, onChange = vi.fn(), onBusyChange }: { initial?: string[]; maxReferences?: number; onChange?: (paths: string[]) => void; onBusyChange?: (busy: boolean) => void }) {
  const [value, setValue] = useState(initial);
  return <SampleReferenceInput value={value} maxReferences={maxReferences} disabled={false} onBusyChange={onBusyChange} onChange={(next) => { setValue(next); onChange(next); }} />;
}

it("uploads multiple files in selection order through the single-image API", async () => {
  const fetcher = vi.fn().mockResolvedValueOnce(response(reference(hash("a")))).mockResolvedValueOnce(response(reference(hash("b"))));
  vi.stubGlobal("fetch", fetcher);
  const onChange = vi.fn();
  render(<Editor onChange={onChange} />);
  expect(screen.getByText("缩放后参考图总像素上限 4 Mi。")).toBeInTheDocument();
  await userEvent.upload(screen.getByLabelText("选择参考图文件"), [new File(["a"], "a.png", { type: "image/png" }), new File(["b"], "b.png", { type: "image/png" })]);
  await waitFor(() => expect(onChange).toHaveBeenLastCalledWith([reference(hash("a")).reference_image, reference(hash("b")).reference_image]));
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect((fetcher.mock.calls[0][1].body as FormData).get("file")).toBeInstanceOf(File);
  expect(screen.getAllByRole("img")).toHaveLength(2);
});

it("appends a validated server path and reports API errors", async () => {
  const fetcher = vi.fn().mockResolvedValue(response({ ok: false, error: "图片不存在" }, 400)); vi.stubGlobal("fetch", fetcher);
  await expect(importSampleReference("relative.png")).rejects.toThrow("绝对路径");
  const onBusyChange = vi.fn();
  render(<Editor initial={["/existing.png"]} onBusyChange={onBusyChange} />);
  await userEvent.type(screen.getByLabelText("参考图服务器绝对路径"), "/图片 数据/参考 图.png");
  await userEvent.click(screen.getByRole("button", { name: "载入服务器参考图" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("图片不存在");
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ path: "/图片 数据/参考 图.png" });
  expect(screen.getByText("/existing.png")).toBeInTheDocument();
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
});

it("supports ordered removal and rejects over-limit drops without importing", async () => {
  const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher);
  const onChange = vi.fn();
  const { container } = render(<Editor initial={["/a.png", "/b.png"]} maxReferences={2} onChange={onChange} />);
  fireEvent.drop(container.firstChild!, { dataTransfer: { files: [new File(["c"], "c.png", { type: "image/png" })] } });
  expect(screen.getByRole("alert")).toHaveTextContent("最多 2 张");
  expect(fetcher).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "上移参考图 2" }));
  expect(onChange).toHaveBeenLastCalledWith(["/b.png", "/a.png"]);
  await userEvent.click(screen.getByRole("button", { name: "移除参考图 2" }));
  expect(onChange).toHaveBeenLastCalledWith(["/b.png"]);
});

it("ignores an in-flight upload after removing a reference", async () => {
  let resolve!: (value: Response) => void;
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((done) => { resolve = done; })));
  const onChange = vi.fn();
  const { container } = render(<Editor initial={["/old.png"]} onChange={onChange} />);
  fireEvent.drop(container.firstChild!, { dataTransfer: { files: [new File(["a"], "a.png", { type: "image/png" })] } });
  await userEvent.click(screen.getByRole("button", { name: "移除参考图 1" }));
  await act(async () => resolve(response(reference(hash("a")))));
  expect(onChange).toHaveBeenLastCalledWith([]);
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
});

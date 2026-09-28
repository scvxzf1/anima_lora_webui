import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PreviewAssets } from "./PreviewAssets";
import { PreviewSettings } from "./PreviewSettings";

vi.mock("../../components/ResilientImage", () => ({ ResilientImage: (props: { alt: string }) => <img alt={props.alt} /> }));
afterEach(cleanup);

describe("PreviewAssets", () => {
  it("keeps aggregate group images read-only", () => {
    render(<PreviewAssets images={[{ file: "a.png", name: "A" }]} weights={[]} readOnlyGroup onDelete={vi.fn()} onHotstart={vi.fn()} />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: /删除所选/ })).toBeDisabled();
  });

  it("requires explicit image selection before enabling delete", () => {
    const onDelete = vi.fn();
    render(<PreviewAssets images={[{ file: "a.png", name: "A" }]} weights={[]} readOnlyGroup={false} onDelete={onDelete} onHotstart={vi.fn()} />);
    const button = screen.getByRole("button", { name: /删除所选/ });
    expect(button).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox"));
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(onDelete).toHaveBeenCalledWith(["a.png"]);
  });

  it("cannot delete a selection after that image leaves the current result", () => {
    const onDelete = vi.fn();
    const view = render(<PreviewAssets images={[{ file: "a.png", name: "A" }]} weights={[]} readOnlyGroup={false} onDelete={onDelete} onHotstart={vi.fn()} />);
    fireEvent.click(screen.getByRole("checkbox"));
    view.rerender(<PreviewAssets images={[{ file: "b.png", name: "B" }]} weights={[]} readOnlyGroup={false} onDelete={onDelete} onHotstart={vi.fn()} />);
    const button = screen.getByRole("button", { name: /删除所选/ });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onDelete).not.toHaveBeenCalled();
  });

  it("locks path settings until the existing settings have loaded", () => {
    render(<PreviewSettings settings={{ training_dir: "sample" }} dirty saving={false} locked onChange={vi.fn()} onSave={vi.fn()} onDefaults={vi.fn()} />);
    expect(screen.getByLabelText("训练样张目录")).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存路径设置" })).toBeDisabled();
  });
});

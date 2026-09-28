import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PreviewAssets } from "./PreviewAssets";

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
});

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { TRAINING_LIBRARY_LAYOUT_KEY, TRAINING_LIBRARY_SIZE_KEYS, TrainingWorkspaceLayout, useTrainingLibraryLayout } from "./TrainingWorkspaceLayout";

function LayoutSwitch() {
  const layout = useTrainingLibraryLayout()!;
  return <button onClick={() => layout.setMode(layout.mode === "top" ? "auto" : "top")}>{layout.mode}</button>;
}

function renderLayout() {
  return render(
    <TrainingWorkspaceLayout>
      <aside className="training-config-library">配置库</aside>
      <div className="training-editor-column">编辑器</div>
      <LayoutSwitch />
    </TrainingWorkspaceLayout>,
  );
}

describe("TrainingWorkspaceLayout", () => {
  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it("keeps side and top percentages independent across remounts and resets", () => {
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.side, "35");
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.top, "45");
    const view = renderLayout();
    const layout = document.querySelector<HTMLElement>(".training-workspace-layout")!;
    const side = screen.getByRole("separator", { name: "调整配置库宽度" });
    const top = screen.getByRole("separator", { name: "调整配置库高度" });
    expect(layout.style.getPropertyValue("--training-library-side-size")).toBe("35%");
    expect(layout.style.getPropertyValue("--training-library-top-size")).toBe("45%");

    fireEvent.keyDown(side, { key: "End" });
    fireEvent.keyDown(top, { key: "Home" });
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.side)).toBe("42");
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.top)).toBe("10");
    expect(side).toHaveAttribute("aria-valuenow", "42");
    expect(top).toHaveAttribute("aria-valuenow", "10");

    view.unmount();
    renderLayout();
    const restored = document.querySelector<HTMLElement>(".training-workspace-layout")!;
    expect(restored.style.getPropertyValue("--training-library-side-size")).toBe("42%");
    expect(restored.style.getPropertyValue("--training-library-top-size")).toBe("10%");
    fireEvent.doubleClick(screen.getByRole("separator", { name: "调整配置库宽度" }));
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.side)).toBeNull();
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.top)).toBe("10");
    expect(restored.style.getPropertyValue("--training-library-side-size")).toBe("");
  });

  it("clamps side and top preferences to their usable ranges", () => {
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.side, "999");
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.top, "invalid");
    renderLayout();
    const layout = document.querySelector<HTMLElement>(".training-workspace-layout")!;
    const side = screen.getByRole("separator", { name: "调整配置库宽度" });
    expect(layout.style.getPropertyValue("--training-library-side-size")).toBe("42%");
    expect(layout.style.getPropertyValue("--training-library-top-size")).toBe("");
    fireEvent.keyDown(side, { key: "ArrowLeft", shiftKey: true });
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.side)).toBe("37");
    fireEvent.keyDown(side, { key: "Home" });
    fireEvent.keyDown(side, { key: "ArrowLeft" });
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.side)).toBe("10");
  });

  it("persists explicit placement without changing either saved size", () => {
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.side, "32");
    localStorage.setItem(TRAINING_LIBRARY_SIZE_KEYS.top, "24");
    const view = renderLayout();
    const layout = document.querySelector<HTMLElement>(".training-workspace-layout")!;
    expect(layout.dataset.orientation).toBe("top");
    fireEvent.click(screen.getByRole("button", { name: "auto" }));
    expect(layout.dataset.orientation).toBe("top");
    expect(localStorage.getItem(TRAINING_LIBRARY_LAYOUT_KEY)).toBe("top");
    view.unmount();
    renderLayout();
    expect(document.querySelector<HTMLElement>(".training-workspace-layout")!.dataset.orientation).toBe("top");
    fireEvent.click(screen.getByRole("button", { name: "top" }));
    expect(localStorage.getItem(TRAINING_LIBRARY_LAYOUT_KEY)).toBeNull();
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.side)).toBe("32");
    expect(localStorage.getItem(TRAINING_LIBRARY_SIZE_KEYS.top)).toBe("24");
  });
});

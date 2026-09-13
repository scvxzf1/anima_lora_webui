import { describe, expect, it } from "vitest";
import { historyKeyboardCoordinates } from "./historyDrag";

const config = { kind: "config", collection: "Studio", region: "nav", label: "Portrait" };
const row = (left: number, top: number) => ({ left, top, width: 112, height: 36 });

function coordinates(code: string, target = row(362, 735)) {
  const args = {
    currentCoordinates: { x: 244, y: 735 },
    context: {
      active: { id: "a", data: { current: config } },
      over: { id: "a" },
      collisionRect: { ...row(244, 735), height: 41.5 },
      droppableRects: new Map([["a", row(244, 735)], ["b", target]]),
      droppableContainers: {
        getEnabled: () => ["a", "b"].map((id) => ({ id, data: { current: config } })),
      },
    },
  } as unknown as Parameters<typeof historyKeyboardCoordinates>[1];
  return historyKeyboardCoordinates(new KeyboardEvent("keydown", { code }), args);
}

describe("history keyboard drag geometry", () => {
  it("moves down to the next same-row item even when the overlay is taller", () => {
    expect(coordinates("ArrowDown")).toEqual({ x: 362, y: 732.25 });
    expect(coordinates("ArrowRight")).toEqual({ x: 362, y: 732.25 });
  });

  it("keeps vertical navigation and rejects the opposite direction", () => {
    expect(coordinates("ArrowDown", row(244, 780))).toEqual({ x: 244, y: 777.25 });
    expect(coordinates("ArrowUp", row(244, 780))).toBeUndefined();
    expect(coordinates("ArrowLeft")).toBeUndefined();
  });
});

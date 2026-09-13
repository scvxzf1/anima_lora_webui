import { describe, expect, it } from "vitest";
import {
  orderedHistoryTasks,
  historyStacks,
  historyConfigKey,
  historyConfigOrder,
  historyDragSelection,
  moveHistoryOrder,
} from "./historyOrder";
import { acceptsHistoryDrop } from "./historyDrag";

describe("history collection order", () => {
  it("does not combine unrelated records without config metadata", () => {
    expect(historyStacks([{ id: "one" }, { id: "two" }])).toHaveLength(2);
    expect(historyStacks([{ name: "one" }, { name: "two" }])).toHaveLength(2);
  });
  it("stacks matching configs including preprocessing, keeping collections separate", () => {
    const tasks = [
      { id: "1", group: "A", history_group_key: "config", job: "training" },
      { id: "2", group: "A", history_group_key: "other" },
      { id: "3", group: "A", history_group_key: "config", job: "preprocess" },
      { id: "4", group: "B", history_group_key: "config" },
    ];
    expect(
      historyStacks(tasks).map((stack) => stack.tasks.map((task) => task.id)),
    ).toEqual([["1", "3"], ["2"], ["4"]]);
    expect(historyConfigKey({ history_source_config_file: "file.toml" })).toBe(
      "file.toml",
    );
    expect(
      historyConfigKey({
        methods_subdir: "methods",
        variant: "lora",
        preset: "default",
      }),
    ).toBe("methods:lora:default");
  });

  it("preserves saved unseen order entries while reordering loaded configs", () => {
    const settings = {
      collection_order: ["A"],
      config_group_order: { A: ["unloaded", "b", "a"] },
    };
    const order = historyConfigOrder(
      [{ group: "A", history_group_key: "c" }],
      "A",
      settings,
    );
    expect(moveHistoryOrder(order, "a", "b")).toEqual([
      "unloaded",
      "a",
      "b",
      "c",
    ]);
    expect(order).toEqual(["unloaded", "b", "a", "c"]);
    expect(moveHistoryOrder(order, "missing", "b")).toBe(order);
  });

  it("moves selected records only when the drag source intersects selection", () => {
    expect(historyDragSelection(["a", "b"], ["b", "c"])).toEqual(["b", "c"]);
    expect(historyDragSelection(["a"], ["b", "c"])).toEqual(["a"]);
    expect(historyDragSelection(["a", "a"], [])).toEqual(["a"]);
  });

  it("restricts sorting to its own collection and region but allows moving to unclassified", () => {
    const config = {
      kind: "config" as const,
      collection: "A",
      region: "stack",
      label: "A",
    };
    expect(
      acceptsHistoryDrop(config, {
        kind: "collection",
        collection: "",
        label: "",
      }),
    ).toBe(true);
    expect(acceptsHistoryDrop(config, { ...config, collection: "B" })).toBe(
      false,
    );
    expect(acceptsHistoryDrop(config, { ...config, region: "nav" })).toBe(
      false,
    );
    expect(acceptsHistoryDrop(config, { ...config, key: "other" })).toBe(true);
    expect(
      acceptsHistoryDrop(
        { kind: "collection", collection: "A", label: "A" },
        { kind: "collection", collection: "", label: "" },
      ),
    ).toBe(false);
  });
  it("orders collections and config groups without mutating the server list", () => {
    const tasks = [
      { id: "a", group: "A", history_group_key: "old" },
      { id: "b", group: "B" },
      { id: "c", group: "A", history_group_key: "new" },
    ];
    expect(
      orderedHistoryTasks(tasks, {
        collection_order: ["B", "A"],
        config_group_order: { A: ["new", "old"] },
      }).map((task) => task.id),
    ).toEqual(["b", "c", "a"]);
    expect(tasks.map((task) => task.id)).toEqual(["a", "b", "c"]);
  });
});

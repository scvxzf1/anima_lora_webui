import { expect, it } from "vitest";
import { modelGroups } from "./modelLibrary";

it("preserves model group ordering and the configured item order", () => {
  const groups = [
    { id: "second", label: "第二组", item_ids: ["c"] },
    { id: "first", label: "第一组", item_ids: ["b", "a"] },
  ];
  expect(
    modelGroups({
      revision: "r1",
      default_id: "a",
      groups,
      items: [
        { id: "a", name: "A", model_family: "anima", pretrained_model_name_or_path: "a", qwen3: "qa", vae: "va" },
        { id: "b", name: "B", model_family: "anima", pretrained_model_name_or_path: "b", qwen3: "qb", vae: "vb" },
        { id: "c", name: "C", model_family: "anima", pretrained_model_name_or_path: "c", qwen3: "qc", vae: "vc" },
      ],
    }),
  ).toEqual(groups);
});

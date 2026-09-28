import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { fieldsForConfig } from "./fieldCatalog";
import { draftFromMerged } from "./trainingForm";
import { normalizeBooleanConfigValue } from "./domain/config-field-types.js";

afterEach(cleanup);

it("renders boolean fields as switches and reports boolean values on change", async () => {
  const sparseFields = fieldsForConfig({});
  for (const key of [
    "use_ortho",
    "use_timestep_mask",
    "route_per_layer",
    "dim_from_weights",
    "use_custom_down_autograd",
    "reuse_vae_latents",
  ]) {
    expect(sparseFields.find((field) => field.key === key)?.kind, key).toBe("boolean");
  }
  expect(normalizeBooleanConfigValue("use_ortho", undefined)).toBe(false);
  const config = {
    model_family: "anima",
    use_ortho: false,
    route_per_layer: false,
    use_moe_style: "false",
    use_cmmd: false,
  };
  const fields = fieldsForConfig(config).filter((field) =>
    ["use_ortho", "route_per_layer", "use_moe_style", "use_cmmd"].includes(field.key),
  );
  const draft = draftFromMerged(config, fields);
  const onChange = vi.fn();
  render(
    <TrainingFieldEditor
      fields={fields}
      draft={draft}
      ownKeys={new Set(Object.keys(config))}
      disabled={false}
      onChange={onChange}
      method="ortholora"
    />,
  );

  const ortho = screen.getByRole("checkbox", { name: "启用 OrthoLoRA" });
  expect(fields.find((field) => field.key === "use_cmmd")?.kind).toBe("boolean");
  expect(fields.find((field) => field.key === "use_moe_style")?.kind).toBe("select");
  expect(ortho).not.toBeChecked();
  await userEvent.click(ortho);
  expect(onChange).toHaveBeenCalledWith("use_ortho", true);
  expect(screen.getByLabelText("MoE 结构").tagName).toBe("SELECT");
  expect(normalizeBooleanConfigValue("use_ortho", "yes")).toBe(true);
  expect(normalizeBooleanConfigValue("use_ortho", "no", true)).toBe(false);
});

import { expect, it } from "vitest";
import { overviewProgress } from "./overviewProgress";

it("does not mistake max_train_steps for the actual target of epoch-based runs", () => {
  expect(overviewProgress({ config_toml: "max_train_steps = 100\nmax_train_epochs = 3" })).toEqual({ epochTarget: 3, stepTarget: undefined });
  expect(overviewProgress({ config_toml: "max_train_steps = 100" })).toEqual({ epochTarget: undefined, stepTarget: 100 });
  for (const config_toml of ["", "broken = [", "max_train_steps = -1", "max_train_steps = 1.5"]) {
    expect(overviewProgress({ config_toml }).stepTarget).toBeUndefined();
  }
});

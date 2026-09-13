import { describe, expect, it } from "vitest";
import { pageScale, validScale } from "./uiPreferences";

describe("UI scale preferences", () => {
  it("validates legacy bounds and empty overrides", () => {
    expect(validScale("", 125)).toBe(125);
    expect(validScale("bad")).toBe(100);
    expect(validScale(0)).toBe(25);
    expect(validScale(500)).toBe(400);
  });
  it("uses absolute page overrides rather than multiplying scales", () => {
    const settings = { ui_scale: 150, ui_scale_config: 125 };
    expect(pageScale(settings, "/training")).toBe(125);
    expect(pageScale(settings, "/queue")).toBe(150);
  });
});

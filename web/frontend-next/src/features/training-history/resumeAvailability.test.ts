import { describe, expect, it } from "vitest";
import { resumeAvailability } from "./resumeAvailability";
const checkpoint = { path: "checkpoint", name: "checkpoint", step: 100, target_total_steps: 100, state_integrity: { ok: true }, resume_available: false };

describe("resume availability", () => {
  it("only permits target overrides for complete, target-reached checkpoints", () => {
    expect(resumeAvailability(checkpoint).available).toBe(false);
    expect(resumeAvailability(checkpoint, 200).available).toBe(true);
    expect(resumeAvailability(checkpoint, 200).appendSteps).toBe(100);
    expect(resumeAvailability({ ...checkpoint, step: undefined }, 200).available).toBe(false);
    expect(resumeAvailability({ ...checkpoint, state_integrity: { ok: false } }, 200).available).toBe(false);
    expect(resumeAvailability({ ...checkpoint, state_integrity: undefined }, 200).available).toBe(false);
    expect(resumeAvailability({ ...checkpoint, state_complete: false }, 200).available).toBe(false);
    expect(resumeAvailability({ ...checkpoint, state_integrity: { ok: true, scheduler: false, scheduler_required: false } }, 200).available).toBe(true);
    expect(resumeAvailability({ ...checkpoint, target_total_steps: 1000, unavailable_reason: "文件不可读" }, 200)).toMatchObject({ available: false, reason: "文件不可读" });
  });
  it("rejects invalid targets and does not treat unknown state as available", () => {
    for (const target of [NaN, Infinity, 100, -1, 100.5]) expect(resumeAvailability(checkpoint, target).available).toBe(false);
    expect(resumeAvailability(undefined).available).toBe(false);
    expect(resumeAvailability({ ...checkpoint, resume_available: undefined }, 200).available).toBe(true);
    expect(resumeAvailability({ ...checkpoint, resume_available: true, target_total_steps: 200 }).available).toBe(true);
  });
});

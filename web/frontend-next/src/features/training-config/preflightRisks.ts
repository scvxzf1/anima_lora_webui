import type { PreflightCheck, TrainingPreflightResponse } from "./api";

export function isOverridableRisk(check: PreflightCheck) {
  return (
    check.level === "error" &&
    check.overridable === true &&
    check.risk_code === "mixed_nf4_compile"
  );
}

export function canOverridePreflight(result?: TrainingPreflightResponse) {
  if (
    !result ||
    result.ok ||
    result.can_override !== true ||
    !result.errors.length
  )
    return false;
  const errorChecks = result.checks.filter((check) => check.level === "error");
  return (
    result.summary.errors === result.errors.length &&
    errorChecks.length === result.errors.length &&
    result.errors.every(isOverridableRisk) &&
    errorChecks.every(isOverridableRisk)
  );
}

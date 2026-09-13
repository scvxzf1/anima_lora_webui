import { finiteNumber } from "../../components/trainingNumbers";
import type { ResumeCheckpoint } from "./api";

export function resumeAvailability(checkpoint: ResumeCheckpoint | undefined, target?: number) {
  const step = finiteNumber(checkpoint?.step);
  const originalTarget = finiteNumber(checkpoint?.target_total_steps);
  const targetValid = target === undefined || (step !== undefined && Number.isSafeInteger(step) && step >= 0 && Number.isSafeInteger(target) && target > step);
  const appendSteps = target !== undefined && targetValid && step !== undefined ? target - step : undefined;
  const reachedTarget = step != null && originalTarget != null && originalTarget > 0 && step >= originalTarget;
  const incomplete = checkpoint?.state_integrity?.ok === false || checkpoint?.state_complete === false;
  const complete = checkpoint?.state_integrity?.ok === true && !incomplete;
  const override = complete && reachedTarget && target !== undefined && targetValid;
  const available = Boolean(checkpoint && !incomplete && targetValid && (override || (!reachedTarget && checkpoint.resume_available === true)));
  const reason = !checkpoint ? "暂无可恢复检查点"
    : incomplete ? checkpoint.unavailable_reason || "检查点状态不完整，不能恢复优化器状态"
      : !targetValid ? "新目标总步数必须大于检查点步数"
        : reachedTarget && !override ? "历史目标已达到；提高总步数后可继续"
          : !available ? checkpoint.unavailable_reason || "尚未确认检查点可恢复"
            : "";
  return { available, targetValid, reason, appendSteps };
}

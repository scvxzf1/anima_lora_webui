import { useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { TrainingDevices } from "./TrainingDevices";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import "./TrainingLaunchAvailability.css";

export function TrainingLaunchAvailability({ state }: {
  state: ReturnType<typeof useTrainingWorkspace>;
}) {
  const [open, setOpen] = useState(false);
  const reason = state.deviceState.issue || (state.pipelineUnavailable
    ? "当前配置启用了尚未接入主训练入口的流水线并行。请选择单卡或数据并行。"
    : "");
  return (
    <>
      {reason && (
        <div className="training-launch-availability" role="status">
          <span>暂不能启动或入队：{reason}</span>
          <button type="button" onClick={() => setOpen(true)}>
            检查训练设备
          </button>
        </div>
      )}
      {open && (
        <CommandDialog title="检查训练设备" onClose={() => setOpen(false)}>
          <TrainingDevices state={state} />
          <p>设备选择仅保存在当前浏览器，不会启动训练或加入队列。</p>
        </CommandDialog>
      )}
    </>
  );
}

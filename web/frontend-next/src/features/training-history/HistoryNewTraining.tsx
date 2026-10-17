import { lazy, Suspense, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { trainingContextKeys } from "../../api/trainingContext";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import type { HistoryTaskDetail } from "./api";

const TrainingRawEditor = lazy(() => import("../training-config/TrainingRawEditor").then((module) => ({ default: module.TrainingRawEditor })));

export function HistoryNewTraining({ detail, disabled }: { detail: HistoryTaskDetail; disabled: boolean }) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const content = detail.config_toml || "";
  return <>
    <button type="button" className="history-new-training" disabled={disabled || !content.trim()}
      title={content.trim() ? "从历史快照创建独立配置，不会自动开始训练" : "此任务未保存配置快照"}
      onClick={() => setOpen(true)}><Plus size={16} />以此配置新建训练</button>
    {open && <Suspense fallback={<p role="status">正在加载配置编辑器</p>}>
      <TrainingRawEditor content={content} onClose={() => setOpen(false)} onReload={async () => {}}
        description="从本次训练的实际配置快照创建独立副本，不覆盖原配置，也不会自动启动训练。快照可能包含旧输出、缓存路径或续训参数，请在创建前检查 TOML，并在训练页核对硬件预设。"
        onSaved={(file) => {
          void queryClient.invalidateQueries({ queryKey: trainingContextKeys.files() });
          useTrainingContextStore.getState().selectConfigFile(file);
          setOpen(false);
          navigate("/training");
        }} />
    </Suspense>}
  </>;
}

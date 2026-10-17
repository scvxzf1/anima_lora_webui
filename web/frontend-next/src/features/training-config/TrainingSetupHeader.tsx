import { TrainingContextBar } from "../../app/TrainingContextBar";
import { TrainingModelPicker } from "./TrainingExtras";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";

export function TrainingSetupHeader({ state }: {
  state: ReturnType<typeof useTrainingWorkspace>;
}) {
  const { context, guardedContext, contextSelectionBusy, locked, dirty, busy } = state;
  return (
    <div className="training-setup-header">
      <header className="training-config-header">
        <div>
          <p className="eyebrow">TRAINING BLUEPRINT</p>
          <h1>训练配置</h1>
        </div>
        <span className="training-readonly-badge" data-editable={!locked}>
          {locked ? "系统只读 · 可另存" : dirty ? "有未保存修改" : "已同步"}
        </span>
      </header>
      <TrainingContextBar context={guardedContext} selectionDisabled={contextSelectionBusy}>
        {!context.error && !state.rawQuery.error && (
          <TrainingModelPicker
            disabled={busy || context.isPending}
            onSelect={(item) => {
              state.setDraft((current) => ({
                ...current,
                model_family: item.model_family,
                pretrained_model_name_or_path: item.pretrained_model_name_or_path,
                qwen3: item.qwen3,
                vae: item.vae,
              }));
              state.preview.reset();
              state.preflight.reset();
            }}
          />
        )}
      </TrainingContextBar>
    </div>
  );
}

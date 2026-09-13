import { useEffect, useState } from "react";
import { Calculator, ClipboardCheck } from "lucide-react";
import { CommandDialog } from "../../components/CommandDialog";
import { TrainingEstimate } from "./TrainingEstimate";
import { TrainingPatchPreview } from "./TrainingPatchPreview";
import { TrainingPreflightPanel } from "./TrainingPreflightPanel";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";

export function TrainingTools({
  state,
}: {
  state: ReturnType<typeof useTrainingWorkspace>;
}) {
  const [open, setOpen] = useState<"estimate" | "validation" | null>(null);
  const { preview, preflight } = state;
  useEffect(() => {
    if (preview.submittedAt) setOpen("validation");
  }, [preview.submittedAt]);
  useEffect(() => {
    if (preflight.submittedAt) setOpen("validation");
  }, [preflight.submittedAt]);
  return (
    <>
      <div className="training-tools" aria-label="训练辅助工具">
        <button type="button" onClick={() => setOpen("estimate")}>
          <Calculator size={16} />
          训练量估算
        </button>
        <button type="button" onClick={() => setOpen("validation")}>
          <ClipboardCheck size={16} />
          预览与预检
        </button>
      </div>
      {open && (
        <CommandDialog
          title={open === "estimate" ? "训练量估算" : "预览与预检"}
          onClose={() => setOpen(null)}
        >
          {open === "estimate" ? (
            <TrainingEstimate
              file={state.selectedFile}
              preset={state.context.selectedPreset}
              dirty={state.dirty}
            />
          ) : (
            <>
              <div className="toolbar">
                <button
                  type="button"
                  disabled={
                    !state.dirty || state.busy || Boolean(state.patch.error)
                  }
                  onClick={() => preview.mutate()}
                >
                  预览变更
                </button>
                <button
                  type="button"
                  disabled={state.commandBlocked || Boolean(state.deviceState.issue)}
                  onClick={() => state.beforeAction("preflight")}
                >
                  {state.dirty ? "保存并预检" : "运行预检测"}
                </button>
              </div>
              <TrainingPatchPreview
                preview={preview.data}
                pending={preview.isPending}
                error={preview.error?.message}
              />
              <TrainingPreflightPanel
                result={preflight.data}
                pending={preflight.isPending}
                error={preflight.error?.message}
                onLocate={(key) => {
                  const target = state.fields.find((field) => field.key === key);
                  if (target) state.setActiveStage(target.group);
                  setOpen(null);
                  state.setFieldView("all");
                  state.setFieldSearch(key);
                  requestAnimationFrame(() => {
                    const field = document.getElementById(
                      `training-field-${key}`,
                    );
                    field?.scrollIntoView({ block: "center" });
                    field
                      ?.querySelector<HTMLElement>("input, select, textarea")
                      ?.focus({ preventScroll: true });
                  });
                }}
              />
            </>
          )}
        </CommandDialog>
      )}
    </>
  );
}

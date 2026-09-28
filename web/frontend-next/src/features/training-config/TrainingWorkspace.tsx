import { useState } from "react";
import { TrainingEditor } from "./TrainingEditor";
import { TrainingCommands } from "./TrainingCommands";
import { trainingContextKeys } from "../../api/trainingContext";
import { TrainingContextBar } from "../../app/TrainingContextBar";
import { TrainingTools } from "./TrainingTools";
import { TrainingSaveAsDialog } from "./TrainingSaveAsDialog";
import { TrainingConfigLibrary } from "./TrainingConfigLibrary";
import { TrainingWorkspaceLayout } from "./TrainingWorkspaceLayout";
import { TrainingLaunchDialog } from "./TrainingLaunchDialog";
import { TrainingRawEditor } from "./TrainingRawEditor";
import { TrainingSamplePrompts } from "./TrainingExtras";
import {
  invalidateTrainingQueries,
  useTrainingWorkspace,
} from "./useTrainingWorkspace";
import "./TrainingWorkspace.css";
import "./TrainingWorkspaceLayout.css";

export function TrainingWorkspace() {
  const state = useTrainingWorkspace();
  const [libraryExpanded, setLibraryExpanded] = useState(() => {
    try {
      const saved = localStorage.getItem("dragon-next.training-library-expanded");
      return saved === null
        ? !window.matchMedia("(max-width: 800px)").matches
        : saved !== "false";
    } catch {
      return true;
    }
  });
  const toggleLibrary = () => {
    setLibraryExpanded(!libraryExpanded);
    try {
      localStorage.setItem("dragon-next.training-library-expanded", String(!libraryExpanded));
    } catch {
      // Keep the toggle usable when storage is unavailable.
    }
  };
  const {
    queryClient,
    context,
    selectedFile,
    rawQuery,
    setHydratedKey,
    saveAsOpen,
    setSaveAsOpen,
    launchMode,
    setLaunchMode,
    rawMode,
    setRawMode,
    promptsOpen,
    setPromptsOpen,
    mergedConfig,
    changes,
    dirty,
    ownKeys,
    saveAs,
    confirmDiscard,
    guardedContext,
    locked,
    busy,
  } = state;
  return (
    <div className="training-config-shell">
      <main className="training-config-page">
        <header className="training-config-header">
          <div>
            <p className="eyebrow">TRAINING BLUEPRINT</p>
            <h1>训练配置</h1>
          </div>
          <span className="training-readonly-badge" data-editable={!locked}>
            {locked ? "系统只读 · 可另存" : dirty ? "有未保存修改" : "已同步"}
          </span>
        </header>
        <TrainingContextBar context={guardedContext} />

        {context.error || rawQuery.error ? (
          <section className="training-config-error" role="alert">
            <h2>无法读取训练配置</h2>
            <p>{(context.error || rawQuery.error)?.message}</p>
            <button
              type="button"
              disabled={queryClient.isFetching({ queryKey: trainingContextKeys.all }) > 0 || rawQuery.isFetching}
              onClick={() => {
                if (context.error) void queryClient.refetchQueries({ queryKey: trainingContextKeys.all });
                if (rawQuery.error) void rawQuery.refetch();
              }}
            >
              重试读取
            </button>
          </section>
        ) : (
          <>
            <section
              className="training-config-source"
              aria-busy={context.isPending || rawQuery.isPending}
            >
              <div>
                <span>方法文件</span>
                <strong>{selectedFile?.path || "—"}</strong>
              </div>
              <div>
                <span>硬件预设</span>
                <strong>{context.selectedPreset}</strong>
              </div>
              <div>
                <span>文件自有字段</span>
                <strong>{ownKeys.size}</strong>
              </div>
              <div>
                <span>待保存</span>
                <strong>{Object.keys(changes).length}</strong>
              </div>
            </section>

            <TrainingCommands state={state} libraryExpanded={libraryExpanded} onToggleLibrary={toggleLibrary}>
              <TrainingTools state={state} />
            </TrainingCommands>
            <TrainingWorkspaceLayout>
              <TrainingConfigLibrary
                expanded={libraryExpanded}
                files={context.files}
                libraryGroups={context.groups}
                dirty={dirty}
                selectedPath={selectedFile?.path}
                disabled={context.isPending || rawQuery.isPending || busy}
                onSelect={guardedContext.selectConfigFile}
                onCreate={() => {
                  if (confirmDiscard("新建配置")) setRawMode("new");
                }}
              />
              <TrainingEditor state={state} />
            </TrainingWorkspaceLayout>
          </>
        )}
      </main>
      {promptsOpen && selectedFile && (
        <TrainingSamplePrompts
          file={selectedFile}
          promptFile={String(
            mergedConfig.sample_prompts || "",
          )}
          configRevision={rawQuery.data?.revision}
          modelFamily={String(state.draft.model_family ?? mergedConfig.model_family ?? "anima")}
          trainingTask={String(state.draft.qwen_image_2_1_task ?? mergedConfig.qwen_image_2_1_task ?? "t2i")}
          onClose={() => setPromptsOpen(false)}
          onSaved={async () => {
            await invalidateTrainingQueries(
              queryClient,
              selectedFile.path,
              context.selectedPreset,
            );
            setHydratedKey("");
          }}
        />
      )}
      {rawMode && (
        <TrainingRawEditor
          file={rawMode === "edit" ? selectedFile?.path : undefined}
          content={rawMode === "edit" ? rawQuery.data?.content || "" : ""}
          revision={rawMode === "edit" ? rawQuery.data?.revision : undefined}
          locked={rawMode === "edit" && locked}
          onClose={() => setRawMode(null)}
          onReload={async () => {
            await invalidateTrainingQueries(queryClient, selectedFile!.path, context.selectedPreset);
            setHydratedKey("");
            setRawMode(null);
          }}
          onSaved={async (file) => {
            await queryClient.invalidateQueries({
              queryKey: trainingContextKeys.files(),
            });
            await invalidateTrainingQueries(
              queryClient,
              file,
              context.selectedPreset,
            );
            context.selectConfigFile(file);
            setHydratedKey("");
            setRawMode(null);
          }}
        />
      )}
      {launchMode && selectedFile && (
        <TrainingLaunchDialog
          file={selectedFile}
          preset={context.selectedPreset}
          mode={launchMode}
          gpuIds={state.deviceState.gpuIds}
          deviceSummary={state.deviceState.summary}
          deviceIssue={state.deviceState.issue}
          onClose={() => setLaunchMode(null)}
        />
      )}
      {saveAsOpen && selectedFile ? (
        <TrainingSaveAsDialog
          initialName={`${selectedFile.filename?.replace(/\.toml$/i, "") || selectedFile.method || "training"}_copy`}
          busy={saveAs.isPending}
          error={saveAs.error?.message}
          onCancel={() => {
            if (!saveAs.isPending) setSaveAsOpen(false);
          }}
          onConfirm={(name) => saveAs.mutate(name)}
        />
      ) : null}
    </div>
  );
}

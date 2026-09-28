import type { ReactNode } from "react";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import {
  Save,
  Files,
  FileDiff,
  FileCode,
  Image,
  ShieldCheck,
  Play,
  ListPlus,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCcw,
} from "lucide-react";

export function TrainingCommands({
  state,
  libraryExpanded,
  onToggleLibrary,
  children,
}: {
  state: ReturnType<typeof useTrainingWorkspace>;
  libraryExpanded: boolean;
  onToggleLibrary: () => void;
  children?: ReactNode;
}) {
  const {
    capabilities,
    selectedFile,
    rawQuery,
    draft,
    setSaveAsOpen,
    notice,
    setRawMode,
    patch,
    dirty,
    preview,
    save,
    saveConflict,
    reloadConflictedConfig,
    saveAs,
    preflight,
    beforeAction,
    locked,
    busy,
    commandBlocked,
    restorePageDefaults,
    restoreDefaultsBlocked,
  } = state;
  return (
    <>
      <div className="training-command-bar" aria-label="训练配置操作">
        <button
          type="button"
          className="icon-button"
          title={libraryExpanded ? "收起配置库" : "展开配置库"}
          aria-label={libraryExpanded ? "收起配置库" : "展开配置库"}
          aria-expanded={libraryExpanded}
          aria-controls="training-config-library"
          onClick={onToggleLibrary}
        >
          {libraryExpanded ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
        </button>
        <button
          type="button"
          className="compact-tool"
          title="恢复页面默认值"
          aria-label="恢复页面默认值"
          onClick={restorePageDefaults}
          disabled={restoreDefaultsBlocked}
        >
          <RotateCcw size={16} />
          <span>恢复页面默认值</span>
        </button>
        <button
          type="button"
          className="compact-tool"
          title="预览变更"
          aria-label="预览变更"
          onClick={() => preview.mutate()}
          disabled={!dirty || busy || Boolean(patch.error)}
        >
          <FileDiff size={16} />
          <span>预览变更</span>
        </button>
        <button
          type="button"
          className="primary-command compact-tool"
          title="保存配置"
          aria-label="保存配置"
          onClick={() => save.mutate({ ...draft })}
          disabled={!dirty || locked || busy || saveConflict || Boolean(patch.error)}
        >
          <Save size={16} />
          <span>保存配置</span>
        </button>
        <button
          type="button"
          className="compact-tool"
          title="另存配置"
          aria-label="另存配置"
          onClick={() => {
            saveAs.reset();
            setSaveAsOpen(true);
          }}
          disabled={!selectedFile || busy}
        >
          <Files size={16} />
          <span>另存配置</span>
        </button>
        <button
          type="button"
          className="compact-tool"
          title={dirty ? "保存并预检" : "运行预检测"}
          aria-label={dirty ? "保存并预检" : "运行预检测"}
          onClick={() => beforeAction("preflight")}
          disabled={commandBlocked || Boolean(state.deviceState.issue)}
        >
          <ShieldCheck size={16} />
          <span>{dirty ? "保存并预检" : "运行预检测"}</span>
        </button>
        <button
          type="button"
          className="primary-command"
          onClick={() => beforeAction("start")}
          disabled={commandBlocked || Boolean(state.deviceState.issue) || state.pipelineUnavailable}
        >
          <Play size={16} />
          {dirty ? "保存并启动" : "立即启动"}
        </button>
        <button
          type="button"
          onClick={() => beforeAction("queue")}
          disabled={commandBlocked || Boolean(state.deviceState.issue) || state.pipelineUnavailable}
        >
          <ListPlus size={16} />
          {dirty ? "保存并入队" : "加入队列"}
        </button>
        <button
          type="button"
          className="compact-tool"
          title="采样样张"
          aria-label="采样样张"
          onClick={() => beforeAction("prompts")}
          disabled={commandBlocked || locked}
        >
          <Image size={16} />
          <span>采样样张</span>
        </button>
        <button
          type="button"
          className="compact-tool"
          title="TOML"
          aria-label="TOML"
          disabled={dirty || !rawQuery.data || busy}
          onClick={() => setRawMode("edit")}
        >
          <FileCode size={16} />
          <span>TOML</span>
        </button>
        {children}
      </div>
      {notice ? (
        <p className="training-notice" role="status">
          {notice}
        </p>
      ) : null}
      {save.error ? (
        <p className="training-command-error" role="alert">
          {save.error.message}
          {saveConflict && <button type="button" onClick={() => void reloadConflictedConfig()}>重新加载配置</button>}
        </p>
      ) : null}
      {patch.error && (
        <p className="form-error" role="alert">
          结构化字段 JSON 无效：{patch.error}
        </p>
      )}
      {capabilities.error && (
        <p className="form-error" role="alert">
          模型能力目录不可用：{capabilities.error.message}
          <button type="button" onClick={() => capabilities.refetch()}>
            重试
          </button>
        </p>
      )}
    </>
  );
}

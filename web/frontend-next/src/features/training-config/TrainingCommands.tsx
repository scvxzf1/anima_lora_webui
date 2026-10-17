import { TrainingDisplayMenu } from "./TrainingDisplayMenu";
import { TrainingLaunchAvailability } from "./TrainingLaunchAvailability";
import type { useTrainingEditorView } from "./useTrainingEditorView";
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import { downloadTextFile } from "../dataset-editor/downloadTextFile";
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
  ChevronDown,
  MoreHorizontal,
} from "lucide-react";

function CommandPopover({
  label,
  icon,
  children,
}: {
  label: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const close = (restoreFocus = true) => {
    setOpen(false);
    if (restoreFocus) {
      requestAnimationFrame(() => triggerRef.current?.focus());
    }
  };
  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) {
        close(false);
      }
    };
    const dismissOnBlur = (event: FocusEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) {
        close(false);
      }
    };
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("focusin", dismissOnBlur);
    panelRef.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("focusin", dismissOnBlur);
    };
  }, [open]);
  return (
    <div
      className="training-command-popover"
      ref={rootRef}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.preventDefault();
          close();
        }
      }}
    >
      <button
        type="button"
        className="compact-tool training-command-menu-trigger"
        ref={triggerRef}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if ((event.key === "ArrowDown" || event.key === "ArrowUp") && !open) {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        {icon}
        <span>{label}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open && (
        <div
          className="training-command-menu-panel"
          ref={panelRef}
          role="menu"
          aria-label={label}
          onClick={(event) => {
            if ((event.target as HTMLElement).closest("button:not(:disabled)")) {
              const preserveFocus = (event.target as HTMLElement).closest("[data-preserve-focus]");
              close(!preserveFocus);
            }
          }}
          onKeyDown={(event) => {
            const buttons = Array.from(
              panelRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [],
            );
            const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
            const next = event.key === "ArrowDown"
              ? (current + 1) % buttons.length
              : event.key === "ArrowUp"
                ? (current - 1 + buttons.length) % buttons.length
                : event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? buttons.length - 1
                    : null;
            if (next !== null && buttons.length) {
              event.preventDefault();
              buttons[next]?.focus();
            }
          }}
        >
          <div data-preserve-focus="true">{children}</div>
        </div>
      )}
    </div>
  );
}

function CommandMenuItem({
  onClick,
  disabled,
  preserveFocus,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  preserveFocus?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      data-preserve-focus={preserveFocus || undefined}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

export function TrainingCommands({
  state,
  libraryExpanded,
  onToggleLibrary,
  children,
  editorView,
}: {
  state: ReturnType<typeof useTrainingWorkspace>;
  libraryExpanded: boolean;
  onToggleLibrary: () => void;
  children?: ReactNode;
  editorView?: ReturnType<typeof useTrainingEditorView>;
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
      <div className="training-command-bar" role="group" aria-label="训练配置操作">
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
        <div className="training-save-group">
          <button
            type="button"
            className="training-save-command compact-tool"
            title="保存配置"
            aria-label="保存配置"
            onClick={() => save.mutate({ ...draft })}
            disabled={!dirty || locked || busy || saveConflict || Boolean(patch.error)}
          >
            <Save size={16} />
            <span>保存</span>
          </button>
          <CommandPopover label="保存菜单" icon={<ChevronDown size={14} aria-hidden="true" />}>
            <CommandMenuItem
              preserveFocus
              onClick={() => {
                saveAs.reset();
                setSaveAsOpen(true);
              }}
              disabled={!selectedFile || busy}
            >
              <Files size={15} /> 另存配置
            </CommandMenuItem>
            <CommandMenuItem
              onClick={() => {
                if (!rawQuery.data || !selectedFile) return;
                const filename = selectedFile.path.split("/").pop() || "training.toml";
                downloadTextFile(filename, rawQuery.data.content);
              }}
              disabled={!selectedFile || !rawQuery.data || busy}
            >
              <FileCode size={15} /> 导出 TOML
            </CommandMenuItem>
            <CommandMenuItem onClick={restorePageDefaults} disabled={restoreDefaultsBlocked}>
              <RotateCcw size={15} /> 恢复默认值
            </CommandMenuItem>
          </CommandPopover>
        </div>
        <button
          type="button"
          className="compact-tool training-preview-command"
          title="预览变更"
          aria-label="预览变更"
          data-dirty={dirty || undefined}
          onClick={() => preview.mutate()}
          disabled={!dirty || busy || Boolean(patch.error)}
        >
          <span className="training-preview-icon">
            <FileDiff size={16} />
            {dirty && <span className="training-dirty-dot" aria-hidden="true" />}
          </span>
          <span>预览变更</span>
        </button>
        <button
          type="button"
          className="compact-tool training-sample-command"
          title="采样"
          aria-label="采样"
          onClick={() => beforeAction("prompts")}
          disabled={commandBlocked || locked}
        >
          <Image size={16} />
          <span>采样</span>
        </button>
        <button
          type="button"
          className="compact-tool training-toml-command"
          title="TOML"
          aria-label="TOML"
          disabled={dirty || !rawQuery.data || busy}
          onClick={() => setRawMode("edit")}
        >
          <FileCode size={16} />
          <span>TOML</span>
        </button>
        {children}
        {editorView && (
          <TrainingDisplayMenu
            view={editorView.view}
            onViewChange={editorView.changeView}
          />
        )}
        {!editorView && <TrainingDisplayMenu />}
        <CommandPopover label="更多" icon={<MoreHorizontal size={16} aria-hidden="true" />}>
          <CommandMenuItem preserveFocus onClick={() => save.mutate({ ...draft })} disabled={!dirty || locked || busy || saveConflict || Boolean(patch.error)}>
            <Save size={15} /> 保存配置
          </CommandMenuItem>
          <CommandMenuItem preserveFocus onClick={() => setRawMode("edit")} disabled={dirty || !rawQuery.data || busy}>
            <FileCode size={15} /> 编辑 TOML
          </CommandMenuItem>
          <CommandMenuItem preserveFocus onClick={() => preview.mutate()} disabled={!dirty || busy || Boolean(patch.error)}>
            <FileDiff size={15} /> 预览变更
          </CommandMenuItem>
        </CommandPopover>
        <div className="training-launch-actions" role="group" aria-label="训练启动">
          <button
            type="button"
            className="training-preflight-command"
            aria-label={dirty ? "保存并预检" : "预检"}
            onClick={() => beforeAction("preflight")}
            disabled={commandBlocked || Boolean(state.deviceState.issue)}
            title={state.deviceState.issue || undefined}
          >
            <ShieldCheck size={16} />
            <span>预检</span>
          </button>
          <button
            type="button"
            className="training-queue-command"
            aria-label={dirty ? "保存并入队" : "加入队列"}
            onClick={() => beforeAction("queue")}
            disabled={commandBlocked || Boolean(state.deviceState.issue) || state.pipelineUnavailable}
            title={state.deviceState.issue || (state.pipelineUnavailable ? "当前配置的流水线并行尚未接入主训练入口" : undefined)}
          >
            <ListPlus size={16} />
            <span>{dirty ? "保存并入队" : "加入队列"}</span>
          </button>
          <button
            type="button"
            className="primary-command training-start-command"
            aria-label={dirty ? "保存并启动" : "立即启动"}
            onClick={() => beforeAction("start")}
            disabled={commandBlocked || Boolean(state.deviceState.issue) || state.pipelineUnavailable}
            title={state.deviceState.issue || (state.pipelineUnavailable ? "当前配置的流水线并行尚未接入主训练入口" : undefined)}
          >
            <Play size={16} />
            <span>{dirty ? "保存并启动" : "立即启动"}</span>
          </button>
        </div>
      </div>
      <TrainingLaunchAvailability state={state} />
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

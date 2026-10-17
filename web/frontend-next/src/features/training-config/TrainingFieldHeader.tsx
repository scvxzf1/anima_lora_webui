import type { ReactNode } from "react";
import { HelpCircle, RotateCcw, Undo2 } from "lucide-react";

type Props = {
  label: string; fieldKey: string; own: boolean; dirty: boolean;
  overridden?: boolean; disabled: boolean; onHelp: () => void;
  onUndo: () => void; onReset?: () => void; children?: ReactNode;
};

export function TrainingFieldHeader({ label, fieldKey, own, dirty, overridden, disabled, onHelp, onUndo, onReset, children }: Props) {
  const source = dirty ? "未保存草稿" : own ? "当前文件" : overridden ? "当前生效值" : "继承值(所选预设及方法默认)";
  return <div className="training-field-header">
    <div className="training-field-heading">
      <span className="training-field-name">{label}</span>
      <div className="training-field-metadata">
        <code className="training-field-tag" title={`${fieldKey} · 来源：${source}`}>{fieldKey}</code>
      </div>
    </div>
    <div className="training-field-actions">
      {(own || dirty || overridden) && <div className="training-field-state">
        {own && <span className="training-field-own" title="此参数显式保存在当前 TOML 文件中">当前文件</span>}
        {overridden && <span className="training-field-override" role={dirty ? "status" : undefined}>已覆写({dirty ? "未保存" : own ? "当前文件" : "当前值"})</span>}
        {dirty && !overridden && <span className="training-field-dirty" role="status">已修改</span>}
      </div>}
      {children}
      <button type="button" className="training-field-help-trigger" title={`查看${label}帮助`} aria-label={`查看${label}帮助`} onClick={onHelp}><HelpCircle size={16} aria-hidden="true" /></button>
      {dirty && <button type="button" className="training-field-undo" aria-label={`撤销${label}修改`} title="撤销未保存修改，恢复已保存值" disabled={disabled} onClick={onUndo}><Undo2 size={15} aria-hidden="true" /></button>}
      {overridden && onReset && <button type="button" className="training-field-undo" aria-label={`恢复${label}继承值`} title="恢复继承值(所选预设及方法默认)" disabled={disabled} onClick={onReset}><RotateCcw size={14} aria-hidden="true" /></button>}
    </div>
  </div>;
}

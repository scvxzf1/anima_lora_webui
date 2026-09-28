import type { FormEvent } from "react";
import type { PreviewSettings as Settings } from "./api";

export function PreviewSettings({ settings, dirty, saving, locked = false, onChange, onSave, onDefaults }: {
  settings: Settings; dirty: boolean; saving: boolean; locked?: boolean; onChange: (settings: Settings) => void; onSave: () => void; onDefaults: () => void;
}) {
  const fields = [["training_dir", "训练样张目录"], ["inference_dir", "推理输出目录"], ["custom_dir", "自定义目录"]] as const;
  const submit = (event: FormEvent) => { event.preventDefault(); onSave(); };
  return <details className="preview-settings" onToggle={(event) => { if ((event.currentTarget as HTMLDetailsElement).open) onChange(settings); }}>
    <summary>预览路径设置{dirty ? " · 有未保存修改" : ""}</summary>
    <form onSubmit={submit}>
      {fields.map(([key, label]) => <label key={key}>{label}<input disabled={locked || saving} value={settings[key] || ""} onChange={(event) => onChange({ ...settings, [key]: event.target.value })} /></label>)}
      <p>相对路径以项目根目录为基准；保存将写入全局预览设置。</p>
      <div className="toolbar"><button type="button" disabled={locked} onClick={onDefaults}>恢复默认</button><button type="submit" disabled={locked || !dirty || saving}>保存路径设置</button></div>
    </form>
  </details>;
}

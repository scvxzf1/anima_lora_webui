import { useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import type { CaptionItem } from "./api";
import { batchPreview, deltasAreCurrent, tagCounts, type BatchOperation, type CaptionDelta, type Drafts } from "./captionBatch";
import "./CaptionBatchToolbar.css";
export function CaptionBatchToolbar({ items, drafts, checked, filtered, filter, onFilter, allowed, disabled, onApply, onSave, undo, redo, canUndo, canRedo }: {
  items: CaptionItem[]; drafts: Drafts; checked: string[]; filtered: CaptionItem[]; filter: string;
  onFilter: (tag: string) => void; allowed: boolean; disabled: boolean; onApply: (deltas: CaptionDelta[]) => void;
  onSave: () => void; undo: () => void; redo: () => void; canUndo: boolean; canRedo: boolean;
}) {
  const [enabled, setEnabled] = useState(allowed), [search, setSearch] = useState("");
  const [scope, setScope] = useState("checked"), [operation, setOperation] = useState<BatchOperation>("append");
  const [source, setSource] = useState(""), [target, setTarget] = useState("");
  const [preview, setPreview] = useState<CaptionDelta[] | null>(null);
  const [previewBases, setPreviewBases] = useState<Drafts>({}), [previewError, setPreviewError] = useState("");
  const ids = (scope === "all" ? items.map((i) => i.id) : scope === "filtered" ? filtered.map((i) => i.id) : checked);
  const valid = operation === "dedupe" || (!!source.trim() && (operation !== "replace" || !!target.trim()));
  const deltas = enabled && valid ? batchPreview(items, drafts, ids, operation, source, target) : [];
  const counts = enabled ? tagCounts(items, drafts).filter(([tag]) => tag.toLowerCase().includes(search.toLowerCase())) : [];
  return <div className="caption-batch">
    <div className="toolbar"><strong>本任务标签管理 · {items.length} 个候选（含跨页）</strong>
      {!enabled && <button type="button" disabled={disabled} onClick={() => { if (window.confirm("标签管理会按逗号或换行拆分候选；自然语言可能含逗号。确认将本任务按标签处理？")) setEnabled(true); }}>切换到标签管理</button>}
      <button type="button" disabled={disabled || !canUndo} onClick={undo}>撤销</button>
      <button type="button" disabled={disabled || !canRedo} onClick={redo}>重做</button>
      <button type="button" disabled={disabled || !Object.keys(drafts).length || Object.values(drafts).some((text) => !text.trim())} onClick={onSave}>保存全部草稿 ({Object.keys(drafts).length})</button>
    </div>
    {enabled && <>
      <label>搜索标签<input value={search} onChange={(e) => setSearch(e.target.value)} /></label>
      <div className="caption-batch-tags"><button type="button" aria-pressed={!filter} onClick={() => onFilter("")}>全部候选</button>{counts.map(([tag, count]) => <button type="button" key={tag} aria-pressed={filter === tag} onClick={() => onFilter(tag)}>{tag} ({count})</button>)}</div>
      <div className="toolbar">
        <label>操作范围<select value={scope} onChange={(e) => setScope(e.target.value)}><option value="checked">勾选项 ({checked.length})</option><option value="filtered">当前筛选 ({filtered.length})</option><option value="all">全任务 ({items.length})</option></select></label>
        <label>批量操作<select value={operation} onChange={(e) => setOperation(e.target.value as BatchOperation)}><option value="append">追加标签</option><option value="delete">删除标签</option><option value="replace">替换标签</option><option value="front">触发词置顶</option><option value="dedupe">去重</option></select></label>
        {operation !== "dedupe" && <label>标签（逗号分隔）<input value={source} onChange={(e) => setSource(e.target.value)} /></label>}
        {operation === "replace" && <label>替换为<input value={target} onChange={(e) => setTarget(e.target.value)} /></label>}
        <button type="button" disabled={disabled || !deltas.length} onClick={() => { setPreview(deltas); setPreviewBases(Object.fromEntries(items.map((item) => [item.id, item.proposed_caption]))); setPreviewError(""); }}>预览差异 ({deltas.length})</button>
      </div><p>筛选 {filtered.length} 项；空勾选不会扩大到全任务。标签按原文精确匹配，空格、下划线与大小写不互换。仅变更本地草稿，不写 TXT；清空候选的操作不应用。</p>
    </>}
    {preview && <CommandDialog title="批量标签差异预览" onClose={() => setPreview(null)}>
      <p>范围：{scope === "all" ? "全任务" : scope === "filtered" ? "当前筛选" : "勾选项"}；受影响 {preview.length} 项。应用仅更新本地草稿。</p>
      <div className="caption-batch-diff">{preview.map((delta) => <article key={delta.id}><strong>{items.find((i) => i.id === delta.id)?.name}</strong><p>原：{delta.before}</p><p>新：{delta.after}</p></article>)}</div>
      {previewError && <p role="alert">{previewError}</p>}
      <button type="button" disabled={disabled} onClick={() => {
        if (!deltasAreCurrent(items, drafts, preview) || preview.some((delta) => items.find((item) => item.id === delta.id)?.proposed_caption !== previewBases[delta.id])) {
          setPreviewError("候选或草稿已改变，请关闭此弹窗并重新预览。" ); return;
        }
        onApply(preview); setPreview(null);
      }}>应用到草稿</button>
    </CommandDialog>}
  </div>;
}

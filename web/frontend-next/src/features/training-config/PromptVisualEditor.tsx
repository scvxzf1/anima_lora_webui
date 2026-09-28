import { useState } from "react";
import { Plus, Copy, Trash2, ArrowUp, ArrowDown } from "lucide-react";
import { parseSamplePromptLine } from "../../../../static/js/features/sample-prompts/model.js";
import { isQwenSampleFamily, movePromptLine, promptLines, promptRowError, supportsEditSamples, updatePromptLine } from "./promptDocument";
import { SampleReferenceInput } from "./SampleReferenceInput";
import "./PromptVisualEditor.css";

const FIELDS = [["prompt", "正向提示词"], ["negative_prompt", "负向提示词"], ["width", "宽度"], ["height", "高度"], ["steps", "步数"], ["cfg", "CFG"], ["seed", "种子"], ["flow_shift", "Flow shift"], ["sample_sampler", "采样器"], ["extra", "其他参数"]] as const;

export function PromptVisualEditor({ content, disabled, onChange, onEditing, modelFamily, supportedPreviewTasks = ["t2i"], defaultTask = "t2i", maxPreviewReferences = 4 }: {
  content: string; disabled: boolean; onChange: (value: string) => void; onEditing: (value: boolean, dirty: boolean) => void;
  modelFamily?: string; supportedPreviewTasks?: readonly string[]; defaultTask?: string; maxPreviewReferences?: number;
}) {
  const [mode, setMode] = useState("visual");
  const [referenceBusy, setReferenceBusy] = useState(false);
  const [editing, updateEditing] = useState<{ index: number; row: ReturnType<typeof parseSamplePromptLine> } | null>(null);
  const setEditing = (value: typeof editing) => {
    updateEditing(value);
    const original = parseSamplePromptLine(value && value.index >= 0 ? content.split(/(?<=\n)/)[value.index] : "");
    onEditing(Boolean(value), Boolean(value && JSON.stringify(original) !== JSON.stringify(value.row)));
  };
  const canEdit = supportsEditSamples(modelFamily, supportedPreviewTasks);
  const qwen = isQwenSampleFamily(modelFamily);
  const error = editing ? promptRowError(editing.row, modelFamily, supportedPreviewTasks, maxPreviewReferences) : "";
  const rows = promptLines(content);
  const append = (line: string) => onChange(content + (content && !content.endsWith("\n") ? "\n" : "") + line);
  return <div className="prompt-visual-editor">
    <div className="toolbar" role="group" aria-label="提示词编辑模式">
      <button type="button" aria-pressed={mode === "visual"} disabled={!!editing} onClick={() => setMode("visual")}>图形化</button>
      <button type="button" aria-pressed={mode === "raw"} disabled={!!editing} onClick={() => setMode("raw")}>原文</button>
    </div>
    {mode === "raw" ? <textarea aria-label="样张提示词内容" rows={16} value={content} disabled={disabled} onChange={(e) => onChange(e.target.value)} /> : <>
      {!editing && rows.map(({ index, row }, position) => <div className="prompt-row" key={index}>
        <button type="button" className="prompt-row-title" disabled={disabled || !!editing} onClick={() => setEditing({ index, row })}>样张 {position + 1}{row.sample_task === "edit" ? ` · 编辑 · ${row.reference_images.length} 图` : ""}: {row.prompt}</button>
        <div className="toolbar">
          {[[Copy, "复制", () => append(rows[position].raw.trimEnd()), false], [ArrowUp, "上移", () => onChange(movePromptLine(content, index, -1)), position === 0], [ArrowDown, "下移", () => onChange(movePromptLine(content, index, 1)), position === rows.length - 1], [Trash2, "删除", () => onChange(content.split(/(?<=\n)/).filter((_, i) => i !== index).join("")), false]].map(([Icon, title, action, blocked]) => {
            const Tool = Icon as typeof Copy;
            return <button key={String(title)} type="button" title={String(title)} aria-label={`${title}样张 ${position + 1}`} disabled={disabled || !!editing || Boolean(blocked)} onClick={action as () => void}><Tool size={16} /></button>;
          })}
        </div>
      </div>)}
      <button type="button" disabled={disabled || !!editing} onClick={() => setEditing({ index: -1, row: {
        ...parseSamplePromptLine(""), sample_task: defaultTask === "edit" && canEdit ? "edit" : "t2i",
        ...(qwen ? { cfg: "1", sample_sampler: "euler" } : {}),
      } })}><Plus size={16} /> 新增样张</button>
      {editing && <fieldset disabled={disabled} className="prompt-fields">
        <legend>{editing.index < 0 ? "新增样张" : "编辑样张"}</legend>
        <label className="prompt-task">样张任务<select aria-label="样张任务" value={editing.row.sample_task} onChange={(event) => setEditing({ ...editing, row: {
          ...editing.row, sample_task: event.target.value, ...(event.target.value === "t2i" ? {
            reference_image: "", reference_images: [],
            ...(editing.row.json ? { json: Object.fromEntries(Object.entries(editing.row.json).filter(([key]) => key !== "reference_image" && key !== "reference_images")) } : {}),
          } : {}),
        } })}>
          <option value="t2i">文生图</option><option value="edit" disabled={!canEdit}>图像编辑</option>
          {!["t2i", "edit"].includes(editing.row.sample_task) && <option value={editing.row.sample_task}>{editing.row.sample_task}</option>}
        </select></label>
        {editing.row.sample_task === "edit" && <SampleReferenceInput value={editing.row.reference_images} maxReferences={maxPreviewReferences} disabled={disabled || !canEdit} onBusyChange={setReferenceBusy} onChange={(reference_images) => setEditing({ ...editing, row: {
          ...editing.row, reference_image: "", reference_images,
          ...(editing.row.json ? { json: Object.fromEntries(Object.entries(editing.row.json).filter(([key]) => key !== "reference_image" && key !== "reference_images")) } : {}),
        } })} />}
        {FIELDS.filter(([key]) => !(qwen && key === "flow_shift") && !((editing.row.json || editing.row.sample_task === "edit") && key === "extra" && !editing.row.extra)).map(([key, label]) => {
          const title = key === "prompt" && editing.row.sample_task === "edit" ? "编辑指令" : label;
          return <label key={key} className={["prompt", "negative_prompt", "extra"].includes(key) ? "prompt-wide-field" : undefined}>{title}
          {key === "sample_sampler" ? <select aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })}>
            <option value="">继承训练配置</option>
            {[...new Set([...(qwen ? ["euler"] : ["euler", "er_sde", "lcm"]), editing.row[key]])].filter(Boolean).map((value) => <option key={value} disabled={qwen && value !== "euler"}>{value}</option>)}
          </select> : key === "prompt" || key === "negative_prompt" ? <textarea rows={3} aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })} /> :
          <input type={key === "extra" ? "text" : "number"} min={0} step={["cfg", "flow_shift"].includes(key) ? "any" : "1"} aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })} />}
        </label>; })}
        {error && <p role="alert">{error}</p>}
        <button type="button" onClick={() => setEditing(null)}>取消编辑</button>
        <button type="button" disabled={!!error || referenceBusy} onClick={() => {
          if (editing.index < 0) append(updatePromptLine("", 0, editing.row));
          else onChange(updatePromptLine(content, editing.index, editing.row));
          setEditing(null);
        }}>应用样张</button>
      </fieldset>}
    </>}
  </div>;
}

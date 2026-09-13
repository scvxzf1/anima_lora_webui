import { useState } from "react";
import { Plus, Copy, Trash2, ArrowUp, ArrowDown } from "lucide-react";
import { parseSamplePromptLine } from "../../../../static/js/features/sample-prompts/model.js";
import { movePromptLine, promptLines, promptRowError, updatePromptLine } from "./promptDocument";
import "./PromptVisualEditor.css";

const FIELDS = [["prompt", "正向提示词"], ["negative_prompt", "负向提示词"], ["width", "宽度"], ["height", "高度"], ["steps", "步数"], ["cfg", "CFG"], ["seed", "种子"], ["flow_shift", "Flow shift"], ["sample_sampler", "采样器"], ["extra", "其他参数"]] as const;

export function PromptVisualEditor({ content, disabled, onChange, onEditing }: { content: string; disabled: boolean; onChange: (value: string) => void; onEditing: (value: boolean, dirty: boolean) => void }) {
  const [mode, setMode] = useState("visual");
  const [editing, updateEditing] = useState<{ index: number; row: ReturnType<typeof parseSamplePromptLine> } | null>(null);
  const setEditing = (value: typeof editing) => {
    updateEditing(value);
    const original = parseSamplePromptLine(value && value.index >= 0 ? content.split(/(?<=\n)/)[value.index] : "");
    onEditing(Boolean(value), Boolean(value && Object.keys(original).some((key) =>
      original[key as keyof typeof original] !== value.row[key as keyof typeof original])));
  };
  const rows = promptLines(content);
  const append = (line: string) => onChange(content + (content && !content.endsWith("\n") ? "\n" : "") + line);
  return <div className="prompt-visual-editor">
    <div className="toolbar" role="group" aria-label="提示词编辑模式">
      <button type="button" aria-pressed={mode === "visual"} disabled={!!editing} onClick={() => setMode("visual")}>图形化</button>
      <button type="button" aria-pressed={mode === "raw"} disabled={!!editing} onClick={() => setMode("raw")}>原文</button>
    </div>
    {mode === "raw" ? <textarea aria-label="样张提示词内容" rows={16} value={content} disabled={disabled} onChange={(e) => onChange(e.target.value)} /> : <>
      {!editing && rows.map(({ index, row }, position) => <div className="prompt-row" key={index}>
        <button type="button" className="prompt-row-title" disabled={disabled || !!editing} onClick={() => setEditing({ index, row })}>样张 {position + 1}: {row.prompt}</button>
        <div className="toolbar">
          {[[Copy, "复制", () => append(rows[position].raw.trimEnd()), false], [ArrowUp, "上移", () => onChange(movePromptLine(content, index, -1)), position === 0], [ArrowDown, "下移", () => onChange(movePromptLine(content, index, 1)), position === rows.length - 1], [Trash2, "删除", () => onChange(content.split(/(?<=\n)/).filter((_, i) => i !== index).join("")), false]].map(([Icon, title, action, blocked]) => {
            const Tool = Icon as typeof Copy;
            return <button key={String(title)} type="button" title={String(title)} aria-label={`${title}样张 ${position + 1}`} disabled={disabled || !!editing || Boolean(blocked)} onClick={action as () => void}><Tool size={16} /></button>;
          })}
        </div>
      </div>)}
      <button type="button" disabled={disabled || !!editing} onClick={() => setEditing({ index: -1, row: parseSamplePromptLine("") })}><Plus size={16} /> 新增样张</button>
      {editing && <fieldset disabled={disabled} className="prompt-fields">
        <legend>{editing.index < 0 ? "新增样张" : "编辑样张"}</legend>
        {FIELDS.map(([key, title]) => <label key={key}>{title}
          {key === "sample_sampler" ? <select aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })}>
            <option value="">继承训练配置</option>
            {[...new Set(["euler", "er_sde", "lcm", editing.row[key]])].filter(Boolean).map((value) => <option key={value}>{value}</option>)}
          </select> : key === "prompt" || key === "negative_prompt" ? <textarea rows={3} aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })} /> :
          <input type={key === "extra" ? "text" : "number"} min={0} step={["cfg", "flow_shift"].includes(key) ? "any" : "1"} aria-label={title} value={editing.row[key]} onChange={(e) => setEditing({ ...editing, row: { ...editing.row, [key]: e.target.value } })} />}
        </label>)}
        {promptRowError(editing.row) && <p role="alert">{promptRowError(editing.row)}</p>}
        <button type="button" onClick={() => setEditing(null)}>取消编辑</button>
        <button type="button" disabled={!!promptRowError(editing.row)} onClick={() => {
          if (editing.index < 0) append(updatePromptLine("", 0, editing.row));
          else onChange(updatePromptLine(content, editing.index, editing.row));
          setEditing(null);
        }}>应用样张</button>
      </fieldset>}
    </>}
  </div>;
}

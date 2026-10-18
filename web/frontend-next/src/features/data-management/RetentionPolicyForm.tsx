import { ArrowDownUp, Filter, ShieldCheck, Trash2 } from "lucide-react";
import type { RetentionPolicy, RetentionSortBy } from "./retentionTypes";

const DIRECTION_LABELS: Record<RetentionSortBy, { desc: string; asc: string }> = {
  progress: { desc: "从新到旧（保留最新进度）", asc: "从旧到新（保留最早进度）" },
  mtime: { desc: "最新修改优先", asc: "最早修改优先" },
  size: { desc: "从大到小（优先清理大文件）", asc: "从小到大" },
};

const SORT_LABELS: Record<RetentionSortBy, string> = {
  progress: "按训练步数 / Epoch（自然序）",
  mtime: "按修改时间",
  size: "按文件体积",
};

function CountInput({
  value,
  onChange,
  disabled,
  label,
}: {
  value: number;
  onChange: (value: number) => void;
  disabled: boolean;
  label: string;
}) {
  return <input
    type="number"
    min={0}
    max={200}
    value={value}
    disabled={disabled}
    aria-label={label}
    onChange={(event) => {
      const next = Number.parseInt(event.target.value, 10);
      onChange(Number.isFinite(next) ? Math.min(200, Math.max(0, next)) : 0);
    }}
  />;
}

export function RetentionPolicyForm({
  policy,
  patternText,
  onPatternTextChange,
  onChange,
  disabled,
}: {
  policy: RetentionPolicy;
  patternText: string;
  onPatternTextChange: (value: string) => void;
  onChange: (patch: Partial<RetentionPolicy>) => void;
  disabled: boolean;
}) {
  return <div className="retention-form">
    <fieldset disabled={disabled}>
      <legend><Filter size={14} aria-hidden="true" /> 1. 检查点排序</legend>
      <label className="retention-field">排序基准
        <select
          value={policy.sort_by}
          aria-label="检查点排序基准"
          onChange={(event) => onChange({ sort_by: event.target.value as RetentionSortBy })}
        >
          {(Object.keys(SORT_LABELS) as RetentionSortBy[]).map((value) => (
            <option key={value} value={value}>{SORT_LABELS[value]}</option>
          ))}
        </select>
      </label>
      <div className="retention-radios" role="group" aria-label="排序方向">
        {(["desc", "asc"] as const).map((order) => <label key={order} className="retention-radio">
          <input
            type="radio"
            name="retention-sort-order"
            checked={policy.sort_order === order}
            onChange={() => onChange({ sort_order: order })}
            disabled={policy.mode === "delete_all_checkpoints"}
          />
          <span>{DIRECTION_LABELS[policy.sort_by][order]}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset disabled={disabled}>
      <legend><ArrowDownUp size={14} aria-hidden="true" /> 2. 保留 / 清理数量（每个实验目录独立计算）</legend>
      <label className="retention-radio">
        <input type="radio" name="retention-mode" checked={policy.mode === "keep_top_n"}
          onChange={() => onChange({ mode: "keep_top_n" })} />
        <span>保留排名前 <CountInput label="保留检查点数量" value={policy.keep_count} disabled={disabled}
          onChange={(value) => onChange({ keep_count: value })} /> 个检查点，清理其余</span>
      </label>
      <label className="retention-radio">
        <input type="radio" name="retention-mode" checked={policy.mode === "delete_top_m"}
          onChange={() => onChange({ mode: "delete_top_m" })} />
        <span>清理排名前 <CountInput label="清理检查点数量" value={policy.delete_count} disabled={disabled}
          onChange={(value) => onChange({ delete_count: value })} /> 个检查点，保留其余</span>
      </label>
      <label className="retention-radio">
        <input type="radio" name="retention-mode" checked={policy.mode === "keep_first_last"}
          onChange={() => onChange({ mode: "keep_first_last" })} />
        <span>首尾保留：最早 <CountInput label="保留最早检查点数量" value={policy.keep_first} disabled={disabled}
          onChange={(value) => onChange({ keep_first: value })} /> 个 + 最新 <CountInput label="保留最新检查点数量"
          value={policy.keep_last} disabled={disabled} onChange={(value) => onChange({ keep_last: value })} /> 个，清理中间</span>
      </label>
      <label className="retention-radio">
        <input type="radio" name="retention-mode" checked={policy.mode === "delete_all_checkpoints"}
          onChange={() => onChange({ mode: "delete_all_checkpoints" })} />
        <span>清理全部中间检查点（仅保留最终权重与下方规则命中的文件）</span>
      </label>
    </fieldset>

    <fieldset disabled={disabled}>
      <legend><ShieldCheck size={14} aria-hidden="true" /> 3. 权重与附属产物规则</legend>
      <label className="retention-check">
        <input type="checkbox" checked={policy.keep_final_weights}
          onChange={(event) => onChange({ keep_final_weights: event.target.checked })} />
        <span>始终保留最终 / 最佳权重（无编号主权重，或名称含 final / best / last / ema）</span>
      </label>
      <label className="retention-check">
        <input type="checkbox" checked={policy.ensure_at_least_one_weight}
          onChange={(event) => onChange({ ensure_at_least_one_weight: event.target.checked })} />
        <span>保底安全锁：若目录下没有最终权重，至少保留排序最靠前的 1 个权重</span>
      </label>
      <label className="retention-field">自定义保留权重匹配词（逗号分隔，支持 *best* 通配）
        <input
          type="text"
          value={patternText}
          placeholder="*best*, *final*, *ema*"
          aria-label="自定义保留权重匹配词"
          onChange={(event) => {
            const next = event.target.value;
            onPatternTextChange(next);
            onChange({
              keep_weight_patterns: next.split(",").map((item) => item.trim()).filter(Boolean),
            });
          }}
        />
      </label>
      <label className="retention-check">
        <input type="checkbox" checked={policy.clean_optimizer_states}
          onChange={(event) => onChange({ clean_optimizer_states: event.target.checked })} />
        <span>顺便清理优化器与训练恢复状态（optimizer / scheduler / random_states / *-state）</span>
      </label>
      <label className="retention-check">
        <input type="checkbox" checked={policy.clean_data_cache}
          onChange={(event) => onChange({ clean_data_cache: event.target.checked })} />
        <span>清理实验内预处理缓存（cache/、latents 目录与中间 .npz）</span>
      </label>
      <label className="retention-check">
        <input type="checkbox" checked={policy.clean_analysis_tensors}
          onChange={(event) => onChange({ clean_analysis_tensors: event.target.checked })} />
        <span>清理分析与调试中间张量（dump/、activation/ 目录与 *.npy；汇总 .json / .tsv 始终保留）</span>
      </label>
      <p className="retention-note"><Trash2 size={13} aria-hidden="true" /> 配置、日志与指标（.json / .tsv / .log / .png 等）始终保留，不受上方规则影响。</p>
      <p className="retention-note"><ShieldCheck size={13} aria-hidden="true" /> 符号链接在扫描阶段静默跳过：不跟随、不删除，也不作为错误阻断。</p>
      {policy.clean_optimizer_states ? <p className="retention-warn">清理恢复状态后，即使保留的检查点也无法直接继续训练。</p> : null}
    </fieldset>
  </div>;
}

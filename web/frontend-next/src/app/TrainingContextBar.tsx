import './TrainingContextBar.css';

import type { TrainingContextController } from './useTrainingContext';

type Props = {
  context: TrainingContextController;
};

export function TrainingContextBar({ context }: Props) {
  return (
    <section className="training-context-bar" aria-label="当前训练上下文">
      <div className="training-context-copy">
        <p className="eyebrow">TRAINING CONTEXT</p>
        <strong>当前训练配置</strong>
        <span>数据集应用和后续启动操作都会作用于这里。</span>
      </div>
      <label>
        <span>训练配置</span>
        <select
          aria-label="当前训练配置"
          value={context.selectedFile?.path || ''}
          disabled={context.isPending || !context.files.length}
          onChange={(event) => context.selectConfigFile(event.target.value)}
        >
          {context.files.map((file) => (
            <option key={file.path} value={file.path}>
              {file.label || file.filename || file.path.split('/').pop() || file.path}
              {file.locked ? '（只读）' : ''}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>硬件预设</span>
        <select
          aria-label="当前硬件预设"
          value={context.selectedPreset}
          disabled={context.isPending || !context.presets.length}
          onChange={(event) => context.selectPreset(event.target.value)}
        >
          {context.presets.map((preset) => <option key={preset} value={preset}>{preset}</option>)}
        </select>
      </label>
      <div className="training-context-state" data-tone={context.error ? 'danger' : 'neutral'}>
        {context.error
          ? `上下文读取失败：${context.error.message}`
          : context.isPending
            ? '正在同步训练上下文'
            : `${context.maxTrainSteps || '—'} steps`}
      </div>
    </section>
  );
}

import { ArrowDown, ArrowUp, Plus, Settings2, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { UseFormReturn } from 'react-hook-form';

import type { DatasetFormValues } from './datasetForm';
import {
  addStageScheduleStage,
  applyStageTemplate,
  defaultStageSchedule,
  deleteStageScheduleStage,
  moveStageScheduleBinding,
  normalizeStageSchedule,
  pctLabel,
  type StageScheduleStage,
  updateStageScheduleStage,
  validateStageSchedule,
} from './stageSchedule';
import { trapDialogFocus } from './trapDialogFocus';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  disabled: boolean;
};

export function StageScheduleEditor({ form, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const enabled = form.watch('stage_schedule_enabled');
  const stages = form.watch('stage_schedule');
  const datasets = form.watch('datasets');
  const issues = enabled ? validateStageSchedule(stages, datasets.length) : [];

  function setEnabled(nextEnabled: boolean) {
    form.setValue('stage_schedule_enabled', nextEnabled, { shouldDirty: true, shouldValidate: true });
    if (nextEnabled && !stages.length) {
      form.setValue('stage_schedule', defaultStageSchedule(datasets.length), {
        shouldDirty: true,
        shouldValidate: true,
      });
    }
  }

  return (
    <section className="stage-schedule-panel" aria-labelledby="stage-schedule-heading">
      <header>
        <div>
          <h3 id="stage-schedule-heading">分阶段调度</h3>
          <p>按训练进度切换数据子集，阶段必须连续覆盖 0% 到 100%。</p>
        </div>
        <label className="stage-schedule-toggle">
          <input
            type="checkbox"
            checked={enabled}
            disabled={disabled}
            onChange={(event) => setEnabled(event.target.checked)}
          />
          <span>{enabled ? '已启用' : '已关闭'}</span>
        </label>
      </header>

      {enabled ? (
        <div className="stage-schedule-summary">
          <div className="stage-schedule-timeline" aria-label="阶段时间线">
            {stages.map((stage, index) => (
              <span
                key={`${stage.name}-${index}`}
                style={{ width: `${Math.max(0, stage.end_pct - stage.start_pct) * 100}%` }}
                title={`${stage.name}: ${pctLabel(stage.start_pct)} - ${pctLabel(stage.end_pct)}`}
              >
                {index + 1}
              </span>
            ))}
          </div>
          <p>{stages.length} 个阶段，关联 {new Set(stages.map((stage) => stage.subset_index)).size} 个子集</p>
          {issues.length ? <p className="stage-schedule-error" role="alert">{issues[0].message}</p> : null}
        </div>
      ) : (
        <p className="stage-schedule-disabled">训练期间使用数据集预设中的全部子集。</p>
      )}

      <button
        ref={triggerRef}
        type="button"
        className="stage-schedule-command"
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        <Settings2 size={15} aria-hidden="true" />
        配置阶段
      </button>

      {open ? (
        <StageScheduleDialog
          initialStages={stages}
          datasets={datasets}
          returnFocus={triggerRef.current}
          onCancel={() => setOpen(false)}
          onApply={(nextStages) => {
            form.setValue('stage_schedule_enabled', true, { shouldDirty: true, shouldValidate: true });
            form.setValue('stage_schedule', nextStages, { shouldDirty: true, shouldValidate: true });
            setOpen(false);
          }}
        />
      ) : null}
    </section>
  );
}

type DialogProps = {
  initialStages: StageScheduleStage[];
  datasets: DatasetFormValues['datasets'];
  returnFocus: HTMLElement | null;
  onCancel: () => void;
  onApply: (stages: StageScheduleStage[]) => void;
};

function StageScheduleDialog({ initialStages, datasets, returnFocus, onCancel, onApply }: DialogProps) {
  const dialogRef = useRef<HTMLElement>(null);
  const [stages, setStages] = useState(() => {
    const normalized = normalizeStageSchedule(initialStages);
    return normalized.length ? normalized : defaultStageSchedule(datasets.length);
  });
  const [operationError, setOperationError] = useState('');
  const issues = useMemo(() => validateStageSchedule(stages, datasets.length), [datasets.length, stages]);

  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.querySelector<HTMLElement>('button, input, select')?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancel();
        return;
      }
      trapDialogFocus(event, dialogRef.current);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      returnFocus?.focus();
    };
  }, [onCancel, returnFocus]);

  function replace(nextStages: StageScheduleStage[]) {
    setOperationError('');
    setStages(nextStages);
  }

  function addStage() {
    try {
      replace(addStageScheduleStage(stages, datasets.length));
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : '无法新增阶段');
    }
  }

  return createPortal(
    <div
      className="dataset-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <section
        ref={dialogRef}
        className="dataset-dialog stage-schedule-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="stage-schedule-dialog-title"
      >
        <header>
          <div>
            <h3 id="stage-schedule-dialog-title">配置分阶段调度</h3>
            <p>移动阶段只交换名称和子集绑定，不改变时间区间。</p>
          </div>
          <button type="button" className="icon-command" onClick={onCancel} aria-label="关闭阶段调度">
            <X size={17} aria-hidden="true" />
          </button>
        </header>

        <div className="stage-schedule-toolbar" aria-label="阶段模板和操作">
          <button type="button" onClick={() => replace(applyStageTemplate(2, datasets.length))}>两段模板</button>
          <button type="button" onClick={() => replace(applyStageTemplate(3, datasets.length))}>三段模板</button>
          <button type="button" onClick={() => replace(applyStageTemplate(stages.length, datasets.length))}>
            均分当前阶段
          </button>
          <button type="button" onClick={addStage} disabled={stages.length >= 12}>
            <Plus size={14} aria-hidden="true" />
            新增阶段
          </button>
        </div>

        <div className="stage-schedule-timeline stage-schedule-dialog-timeline" aria-label="编辑中的阶段时间线">
          {stages.map((stage, index) => (
            <span
              key={`${stage.name}-${index}`}
              style={{ width: `${Math.max(0, stage.end_pct - stage.start_pct) * 100}%` }}
              title={`${stage.name}: ${pctLabel(stage.start_pct)} - ${pctLabel(stage.end_pct)}`}
            >
              {stage.name}
            </span>
          ))}
        </div>

        <div className="stage-schedule-table-wrap">
          <table>
            <thead>
              <tr>
                <th>阶段</th>
                <th>数据子集</th>
                <th>开始</th>
                <th>结束</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {stages.map((stage, index) => (
                <tr key={index}>
                  <td>
                    <input
                      aria-label={`阶段 ${index + 1} 名称`}
                      value={stage.name}
                      onChange={(event) => replace(updateStageScheduleStage(stages, index, { name: event.target.value }))}
                    />
                  </td>
                  <td>
                    <select
                      aria-label={`阶段 ${index + 1} 数据子集`}
                      value={stage.subset_index}
                      onChange={(event) => replace(updateStageScheduleStage(stages, index, {
                        subset_index: Number(event.target.value),
                      }))}
                    >
                      {datasets.map((dataset, datasetIndex) => (
                        <option key={datasetIndex} value={datasetIndex}>
                          {datasetLabel(dataset.source_dir, datasetIndex)}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="0.1"
                      aria-label={`阶段 ${index + 1} 开始百分比`}
                      value={Math.round(stage.start_pct * 1000) / 10}
                      disabled={index === 0}
                      onChange={(event) => replace(updateStageScheduleStage(stages, index, {
                        start_pct: Number(event.target.value) / 100,
                      }))}
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="0.1"
                      aria-label={`阶段 ${index + 1} 结束百分比`}
                      value={Math.round(stage.end_pct * 1000) / 10}
                      disabled={index === stages.length - 1}
                      onChange={(event) => replace(updateStageScheduleStage(stages, index, {
                        end_pct: Number(event.target.value) / 100,
                      }))}
                    />
                  </td>
                  <td>
                    <div className="stage-schedule-row-actions">
                      <button
                        type="button"
                        aria-label={`上移阶段 ${index + 1} 绑定`}
                        disabled={index === 0}
                        onClick={() => replace(moveStageScheduleBinding(stages, index, -1))}
                      >
                        <ArrowUp size={14} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        aria-label={`下移阶段 ${index + 1} 绑定`}
                        disabled={index === stages.length - 1}
                        onClick={() => replace(moveStageScheduleBinding(stages, index, 1))}
                      >
                        <ArrowDown size={14} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        aria-label={`删除阶段 ${index + 1}`}
                        disabled={stages.length <= 1}
                        onClick={() => replace(deleteStageScheduleStage(stages, index))}
                      >
                        <Trash2 size={14} aria-hidden="true" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {operationError ? <p className="stage-schedule-error" role="alert">{operationError}</p> : null}
        {issues.length ? (
          <ul className="stage-schedule-issues" role="alert">
            {issues.map((issue, index) => <li key={`${issue.message}-${index}`}>{issue.message}</li>)}
          </ul>
        ) : null}

        <footer>
          <button type="button" onClick={onCancel}>取消</button>
          <button
            type="button"
            className="primary-command"
            disabled={Boolean(issues.length)}
            onClick={() => onApply(stages)}
          >
            应用到当前预设
          </button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function datasetLabel(sourceDir: string, index: number) {
  const normalized = String(sourceDir || '').replace(/\\/g, '/').replace(/\/$/, '');
  const name = normalized.split('/').pop();
  return `子集 ${index + 1}${name ? ` - ${name}` : ''}`;
}

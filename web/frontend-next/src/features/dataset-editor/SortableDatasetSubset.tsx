import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { ArrowDown, ArrowUp, GripVertical, Images } from 'lucide-react';
import { useEffect, useState, type KeyboardEvent } from 'react';
import type { FieldPath, UseFormReturn } from 'react-hook-form';

import type { DatasetFormValues } from './datasetForm';
import { DatasetSettingsFields } from './DatasetSettingsFields';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  fieldId: string;
  fieldIds: string[];
  index: number;
  rowCount: number;
  selected: boolean;
  disabled: boolean;
  previewDisabled: boolean;
  onSelect: () => void;
  onPreview: (trigger: HTMLElement) => void;
  onMove: (index: number) => void;
  onRemove: () => void;
  onCopyExperimental: (sourceId: string, targetIds: string[]) => void;
};

export function SortableDatasetSubset({
  form,
  fieldId,
  fieldIds,
  index,
  rowCount,
  selected,
  disabled,
  previewDisabled,
  onSelect,
  onPreview,
  onMove,
  onRemove,
  onCopyExperimental,
}: Props) {
  const sortable = useSortable({ id: fieldId, disabled: disabled || rowCount <= 1 });
  const style = {
    transform: CSS.Transform.toString(sortable.transform),
    transition: sortable.transition,
  };
  const path = (key: string) => `datasets.${index}.${key}` as FieldPath<DatasetFormValues>;
  const row = form.watch(`datasets.${index}`);

  function handleSortKey(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.altKey && event.key === 'ArrowUp' && index > 0) {
      event.preventDefault();
      onMove(index - 1);
      return;
    }
    if (event.altKey && event.key === 'ArrowDown' && index < rowCount - 1) {
      event.preventDefault();
      onMove(index + 1);
      return;
    }
    sortable.listeners?.onKeyDown?.(event);
  }

  return (
    <fieldset
      ref={sortable.setNodeRef}
      style={style}
      className="dataset-row-editor"
      aria-label={`子集 ${index + 1}`}
      data-selected={selected}
      data-dragging={sortable.isDragging}
      disabled={disabled}
      onFocusCapture={onSelect}
    >
      <legend className="dataset-row-legend">
        <span>子集 {index + 1}</span>
        <button
          type="button"
          aria-label={`预览子集 ${index + 1} 图片和标注`}
          title={previewDisabled ? '请先保存当前预设，并确保没有未保存修改' : '预览图片和标注'}
          disabled={previewDisabled}
          onClick={(event) => onPreview(event.currentTarget)}
        >
          <Images aria-hidden="true" size={15} />
          预览
        </button>
      </legend>
      <div className="dataset-row-heading">
        <div className="dataset-row-order-controls">
          <button
            ref={sortable.setActivatorNodeRef}
            type="button"
            className="dataset-sort-button dataset-drag-handle"
            aria-label={`拖动排序子集 ${index + 1}`}
            title={rowCount <= 1 ? '至少两个子集时可排序' : '拖动、键盘拖放或 Alt+方向键排序'}
            disabled={disabled || rowCount <= 1}
            {...sortable.attributes}
            {...sortable.listeners}
            onKeyDown={handleSortKey}
          >
            <GripVertical size={15} aria-hidden="true" />
          </button>
          <button type="button" className="dataset-sort-button" aria-label={`上移子集 ${index + 1}`} disabled={disabled || index <= 0} onClick={() => onMove(index - 1)}>
            <ArrowUp size={15} aria-hidden="true" />
          </button>
          <button type="button" className="dataset-sort-button" aria-label={`下移子集 ${index + 1}`} disabled={disabled || index >= rowCount - 1} onClick={() => onMove(index + 1)}>
            <ArrowDown size={15} aria-hidden="true" />
          </button>
        </div>
        <label className="dataset-checkbox-field">
          <input type="checkbox" {...form.register(path('is_reg'))} />
          <span>正则数据</span>
        </label>
        <button type="button" className="danger-command" onClick={onRemove} disabled={disabled || rowCount <= 1}>
          删除子集
        </button>
      </div>

      <label className="dataset-wide-field">
        <span>原始图片目录</span>
        <input {...form.register(path('source_dir'))} />
        <FieldError form={form} path={path('source_dir')} />
      </label>
      <label>
        <span>处理图片目录</span>
        <input {...form.register(path('image_dir'))} />
      </label>
      <label>
        <span>缓存目录</span>
        <input {...form.register(path('cache_dir'))} />
      </label>
      <label>
        <span>重复次数</span>
        <input type="number" min="1" {...form.register(path('num_repeats'), { valueAsNumber: true })} />
        <FieldError form={form} path={path('num_repeats')} />
      </label>
      <label className="dataset-checkbox-field">
        <input type="checkbox" {...form.register(path('recursive'))} />
        <span>递归扫描子目录</span>
      </label>
      <label>
        <span>路径筛选</span>
        <input {...form.register(path('path_pattern'))} placeholder="*" />
        <FieldError form={form} path={path('path_pattern')} />
      </label>

      <details className="dataset-row-advanced dataset-wide-field">
        <summary>高级规则</summary>
        <DatasetSettingsFields form={form} prefix={`datasets.${index}.settings`} />
        <section className="dataset-experimental-settings">
          <h4>实验规则</h4>
          <div>
            <label className="dataset-checkbox-field">
              <input type="checkbox" {...form.register(path('nl_tag_mix.enabled'))} />
              <span>NL/Tag 标签混合</span>
            </label>
            <label>
              <span>Tag 占比</span>
              <input type="number" min="0" max="1" step="0.05" {...form.register(path('nl_tag_mix.tag_ratio'), { valueAsNumber: true })} />
              <FieldError form={form} path={path('nl_tag_mix.tag_ratio')} />
            </label>
            <label className="dataset-checkbox-field">
              <input type="checkbox" {...form.register(path('trigger_clone.enabled'))} />
              <span>触发词图像复制</span>
            </label>
            <label>
              <span>触发词</span>
              <input {...form.register(path('trigger_clone.prompt'))} placeholder="my_character_token" disabled={!row.trigger_clone.enabled} />
              <FieldError form={form} path={path('trigger_clone.prompt')} />
            </label>
            <label>
              <span>复制循环次数</span>
              <input type="number" min="1" {...form.register(path('trigger_clone.num_repeats'), { valueAsNumber: true })} disabled={!row.trigger_clone.enabled} />
              <FieldError form={form} path={path('trigger_clone.num_repeats')} />
            </label>
          </div>
        </section>
        <ExperimentalScope
          fieldId={fieldId}
          fieldIds={fieldIds}
          onApply={(targets) => onCopyExperimental(fieldId, targets)}
        />
      </details>
    </fieldset>
  );
}

function ExperimentalScope({
  fieldId,
  fieldIds,
  onApply,
}: {
  fieldId: string;
  fieldIds: string[];
  onApply: (targets: string[]) => void;
}) {
  const candidates = fieldIds.filter((id) => id !== fieldId);
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    setSelected((current) => current.filter((id) => candidates.includes(id)));
  }, [fieldIds.join('|')]);

  if (!candidates.length) return null;
  return (
    <section className="dataset-experimental-scope">
      <h4>实验规则生效范围</h4>
      <div>
        {candidates.map((id) => {
          const targetIndex = fieldIds.indexOf(id);
          return (
            <label className="dataset-checkbox-field" key={id}>
              <input
                type="checkbox"
                checked={selected.includes(id)}
                onChange={(event) => setSelected((current) => (
                  event.target.checked ? [...current, id] : current.filter((item) => item !== id)
                ))}
              />
              <span>子集 {targetIndex + 1}</span>
            </label>
          );
        })}
        <button type="button" disabled={!selected.length} onClick={() => onApply(selected)}>
          应用到所选子集
        </button>
      </div>
    </section>
  );
}

function FieldError({ form, path }: { form: UseFormReturn<DatasetFormValues>; path: FieldPath<DatasetFormValues> }) {
  const error = form.getFieldState(path, form.formState).error;
  return typeof error?.message === 'string' ? <small role="alert">{error.message}</small> : null;
}

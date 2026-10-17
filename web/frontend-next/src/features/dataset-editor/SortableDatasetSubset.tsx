import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { ArrowDown, ArrowUp, Images, GripVertical } from 'lucide-react';
import { useEffect, useState, type KeyboardEvent } from 'react';
import type { FieldPath, UseFormReturn } from 'react-hook-form';

import { SettingSwitch } from '../../components/SettingSwitch';
import type { DatasetFormValues } from './datasetForm';
import { DatasetSettingsFields } from './DatasetSettingsFields';
import { DatasetFieldLabel } from './DatasetFieldLabel';
import { DatasetMaskFields } from './DatasetMaskFields';
import { DatasetSourcePathField } from './DatasetSourcePathField';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  fieldId: string;
  fieldIds: string[];
  index: number;
  rowCount: number;
  selected: boolean;
  disabled: boolean;
  workbenchDisabled: boolean;
  onSelect: () => void;
  onImageCount?: (source: string, count: number | null) => void;
  onOpenWorkbench?: () => void;
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
  workbenchDisabled,
  onSelect,
  onImageCount,
  onOpenWorkbench,
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

  function revalidatePairs() {
    if (form.formState.errors.datasets) queueMicrotask(() => void form.trigger('datasets'));
  }

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
      onFocusCapture={onSelect}
    >
      <legend className="dataset-row-legend">
        <span>子集 {index + 1}</span>
        {onOpenWorkbench && <button
          type="button"
          aria-label={`打开子集 ${index + 1} 图片工作台`}
          title={workbenchDisabled ? '请先保存当前预设，并确保没有未保存修改' : '打开此子集的图片工作台'}
          disabled={workbenchDisabled}
          onClick={onOpenWorkbench}
        >
          <Images aria-hidden="true" size={15} />
          图片工作台
        </button>}
      </legend>
      <div className="dataset-row-heading">
        <div className="dataset-row-tools">
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
        <div className="dataset-setting-field">
          <DatasetFieldLabel htmlFor={path('edit_role')} field="edit_role" label="子集角色" />
          <select id={path('edit_role')} disabled={disabled} {...form.register(path('edit_role'), {
            onChange: () => {
              form.setValue(`datasets.${index}.is_reg`, false, { shouldDirty: true, shouldValidate: true });
              revalidatePairs();
            },
          })}>
            <option value="normal">非编辑（普通数据集）</option>
            <option value="before">编辑前</option>
            <option value="after">编辑后</option>
          </select>
        </div>
        {row.edit_role === 'normal' && <div className="dataset-setting-field dataset-checkbox-field">
          <SettingSwitch id={path('is_reg')} aria-label="正则数据" disabled={disabled}
                checked={Boolean(form.watch(path('is_reg')))}
                onCheckedChange={(checked) => form.setValue(path('is_reg'), checked, { shouldDirty: true, shouldValidate: true })} />
          <DatasetFieldLabel htmlFor={path('is_reg')} field="is_reg" label="正则数据" />
        </div>}
        </div>
        <button type="button" className="danger-command" onClick={onRemove} disabled={disabled || rowCount <= 1}>
          删除子集
        </button>
      </div>

      {row.edit_role !== 'normal' && <div className="dataset-setting-field dataset-wide-field">
        <DatasetFieldLabel htmlFor={path('edit_pair_id')} field="edit_pair_id" label="配对编号或名称" />
        <input id={path('edit_pair_id')} disabled={disabled} {...form.register(path('edit_pair_id'), { onChange: revalidatePairs })} placeholder="例如：1" />
        <FieldError form={form} path={path('edit_pair_id')} />
      </div>}

      <DatasetSourcePathField
        form={form}
        path={path('source_dir')}
        helpField="source_dir"
        onImageCount={onImageCount}
        disabled={disabled}
        label={row.edit_role === 'before' ? '编辑前图片目录' : row.edit_role === 'after' ? '目标图原始目录（编辑后）' : '原始图片目录'}
      />
      {row.edit_role !== 'before' && <>
      <div className="dataset-setting-field">
        <DatasetFieldLabel htmlFor={path('image_dir')} field="image_dir" label={row.edit_role === 'after' ? '目标图训练目录（编辑后）' : '处理图片目录'} />
        <input id={path('image_dir')} disabled={disabled} {...form.register(path('image_dir'))} />
      </div>
      <div className="dataset-setting-field">
        <DatasetFieldLabel htmlFor={path('cache_dir')} field="cache_dir" label="缓存目录" />
        <input id={path('cache_dir')} disabled={disabled} {...form.register(path('cache_dir'))} />
      </div>
      <div className="dataset-row-basics dataset-wide-field">
      <div className="dataset-setting-field">
        <DatasetFieldLabel htmlFor={path('num_repeats')} field="num_repeats" label="重复次数" />
        <input id={path('num_repeats')} disabled={disabled} type="number" min="1" {...form.register(path('num_repeats'), { valueAsNumber: true })} />
        <FieldError form={form} path={path('num_repeats')} />
      </div>
      <div className="dataset-setting-field">
        <DatasetFieldLabel htmlFor={path('path_pattern')} field="path_pattern" label="路径筛选" />
        <input id={path('path_pattern')} disabled={disabled} {...form.register(path('path_pattern'))} placeholder="*" />
        <FieldError form={form} path={path('path_pattern')} />
      </div>
      <div className="dataset-setting-field dataset-checkbox-field">
        <SettingSwitch id={path('recursive')} aria-label="递归扫描子目录" disabled={disabled}
                checked={Boolean(form.watch(path('recursive')))}
                onCheckedChange={(checked) => form.setValue(path('recursive'), checked, { shouldDirty: true, shouldValidate: true })} />
        <DatasetFieldLabel htmlFor={path('recursive')} field="recursive" label="递归扫描子目录" />
      </div>
      </div>

      <details className="dataset-row-advanced dataset-wide-field">
        <summary>高级规则</summary>
        <DatasetMaskFields form={form} index={index} disabled={disabled} />
        <DatasetSettingsFields form={form} prefix={`datasets.${index}.settings`} disabled={disabled} />
        <section className="dataset-experimental-settings">
          <h4>实验规则</h4>
          <div>
            <div className="dataset-setting-field dataset-checkbox-field">
              <SettingSwitch id={path('nl_tag_mix.enabled')} aria-label="NL/Tag 标签混合" disabled={disabled}
                checked={Boolean(form.watch(path('nl_tag_mix.enabled')))}
                onCheckedChange={(checked) => form.setValue(path('nl_tag_mix.enabled'), checked, { shouldDirty: true, shouldValidate: true })} />
              <DatasetFieldLabel htmlFor={path('nl_tag_mix.enabled')} field="nl_tag_mix_enabled" label="NL/Tag 标签混合" />
            </div>
            <div className="dataset-setting-field">
              <DatasetFieldLabel htmlFor={path('nl_tag_mix.tag_ratio')} field="tag_ratio" label="Tag 占比" />
              <input id={path('nl_tag_mix.tag_ratio')} disabled={disabled} type="number" min="0" max="1" step="0.05" {...form.register(path('nl_tag_mix.tag_ratio'), { valueAsNumber: true })} />
              <FieldError form={form} path={path('nl_tag_mix.tag_ratio')} />
            </div>
            <div className="dataset-setting-field dataset-checkbox-field">
              <SettingSwitch id={path('trigger_clone.enabled')} aria-label="触发词图像复制" disabled={disabled}
                checked={Boolean(form.watch(path('trigger_clone.enabled')))}
                onCheckedChange={(checked) => form.setValue(path('trigger_clone.enabled'), checked, { shouldDirty: true, shouldValidate: true })} />
              <DatasetFieldLabel htmlFor={path('trigger_clone.enabled')} field="trigger_clone_enabled" label="触发词图像复制" />
            </div>
            <div className="dataset-setting-field">
              <DatasetFieldLabel htmlFor={path('trigger_clone.prompt')} field="trigger_clone_prompt" label="触发词" />
              <input id={path('trigger_clone.prompt')} {...form.register(path('trigger_clone.prompt'))} placeholder="my_character_token" disabled={disabled || !row.trigger_clone.enabled} />
              <FieldError form={form} path={path('trigger_clone.prompt')} />
            </div>
            <div className="dataset-setting-field">
              <DatasetFieldLabel htmlFor={path('trigger_clone.num_repeats')} field="trigger_clone_num_repeats" label="复制循环次数" />
              <input id={path('trigger_clone.num_repeats')} type="number" min="1" {...form.register(path('trigger_clone.num_repeats'), { valueAsNumber: true })} disabled={disabled || !row.trigger_clone.enabled} />
              <FieldError form={form} path={path('trigger_clone.num_repeats')} />
            </div>
          </div>
        </section>
        <ExperimentalScope
          fieldId={fieldId}
          fieldIds={fieldIds}
          disabled={disabled}
          onApply={(targets) => onCopyExperimental(fieldId, targets)}
        />
      </details>
      </>}
    </fieldset>
  );
}

function ExperimentalScope({
  fieldId,
  fieldIds,
  disabled,
  onApply,
}: {
  fieldId: string;
  fieldIds: string[];
  disabled: boolean;
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
              <SettingSwitch
                aria-label={`子集 ${targetIndex + 1}`}
                disabled={disabled}
                checked={selected.includes(id)}
                onCheckedChange={(checked) => setSelected((current) => (
                  checked ? [...current, id] : current.filter((item) => item !== id)
                ))}
              />
              <span>子集 {targetIndex + 1}</span>
            </label>
          );
        })}
        <button type="button" disabled={disabled || !selected.length} onClick={() => onApply(selected)}>
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

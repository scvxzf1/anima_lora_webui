import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useEffect, useState } from 'react';
import { useFieldArray, type UseFormReturn } from 'react-hook-form';

import { emptyDatasetRow, type DatasetFormValues } from './datasetForm';
import { SortableDatasetSubset } from './SortableDatasetSubset';

export function DatasetSubsetList({
  form,
  disabled,
  qwenEditIssue,
  workbenchDisabled,
  onOpenWorkbench,
}: {
  form: UseFormReturn<DatasetFormValues>;
  disabled: boolean;
  qwenEditIssue: string | null;
  workbenchDisabled: boolean;
  onOpenWorkbench?: (index: number) => void;
}) {
  const rows = useFieldArray({ control: form.control, name: 'datasets' });
  const watchedRows = form.watch('datasets');
  const editEnabled = watchedRows.some((row) => row.edit_role !== 'normal');
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );
  const [selectedId, setSelectedId] = useState(rows.fields[0]?.id || '');

  useEffect(() => {
    if (!rows.fields.some((field) => field.id === selectedId)) {
      setSelectedId(rows.fields[0]?.id || '');
    }
  }, [rows.fields, selectedId]);

  function handleDragEnd(event: DragEndEvent) {
    if (!event.over || event.active.id === event.over.id) return;
    const oldIndex = rows.fields.findIndex((field) => field.id === event.active.id);
    const newIndex = rows.fields.findIndex((field) => field.id === event.over?.id);
    if (oldIndex >= 0 && newIndex >= 0) rows.move(oldIndex, newIndex);
  }

  function copyExperimentalRules(sourceId: string, targetIds: string[]) {
    const sourceIndex = rows.fields.findIndex((field) => field.id === sourceId);
    if (sourceIndex < 0) return;
    const source = form.getValues(`datasets.${sourceIndex}`);
    targetIds.forEach((targetId) => {
      const targetIndex = rows.fields.findIndex((field) => field.id === targetId);
      if (targetIndex < 0) return;
      form.setValue(`datasets.${targetIndex}.nl_tag_mix`, { ...source.nl_tag_mix }, {
        shouldDirty: true,
        shouldValidate: true,
      });
      form.setValue(`datasets.${targetIndex}.trigger_clone`, { ...source.trigger_clone }, {
        shouldDirty: true,
        shouldValidate: true,
      });
    });
  }

  function addEditPair() {
    const hadDatasetErrors = Boolean(form.formState.errors.datasets);
    const revalidate = () => {
      if (hadDatasetErrors) queueMicrotask(() => void form.trigger('datasets'));
    };
    const current = form.getValues('datasets');
    const incompleteIndex = current.findIndex((row) => row.edit_role !== 'normal'
      && (!row.edit_pair_id || !current.some((other) => other.edit_pair_id === row.edit_pair_id && other.edit_role !== row.edit_role)));
    const incomplete = current[incompleteIndex];
    if (incomplete) {
      const used = new Set(current.map((row) => row.edit_pair_id));
      let number = 1;
      while (used.has(String(number))) number += 1;
      const pairId = incomplete.edit_pair_id || String(number);
      if (!incomplete.edit_pair_id) form.setValue(`datasets.${incompleteIndex}.edit_pair_id`, pairId, { shouldDirty: true, shouldValidate: true });
      rows.append({ ...emptyDatasetRow(form.getValues('defaults')),
        edit_role: incomplete.edit_role === 'before' ? 'after' : 'before',
        edit_pair_id: pairId });
      revalidate();
      return;
    }
    const used = new Set(current.map((row) => row.edit_pair_id));
    let number = 1;
    while (used.has(String(number))) number += 1;
    const pairId = String(number);
    const defaults = form.getValues('defaults');
    if (rows.fields.length === 1 && form.getValues('datasets.0.edit_role') === 'normal') {
      rows.replace([
        { ...emptyDatasetRow(defaults), edit_role: 'before', edit_pair_id: pairId },
        { ...current[0], is_reg: false, edit_role: 'after', edit_pair_id: pairId },
      ]);
      revalidate();
      return;
    }
    rows.append([
      { ...emptyDatasetRow(defaults), edit_role: 'before', edit_pair_id: pairId },
      { ...emptyDatasetRow(defaults), edit_role: 'after', edit_pair_id: pairId },
    ]);
    revalidate();
  }

  return (
    <section className="dataset-subsets">
      <header>
        <div>
          <h3>数据子集</h3>
          <span>{rows.fields.length} 项</span>
        </div>
        <div className="dataset-subset-commands">
          <button
            type="button"
            onClick={() => rows.append(emptyDatasetRow(form.getValues('defaults')))}
            disabled={disabled || editEnabled}
          >
            添加子集
          </button>
          <button type="button" onClick={addEditPair} disabled={disabled || (watchedRows.length > 1 && watchedRows.some((row) => row.edit_role === 'normal'))}>
            添加编辑配对
          </button>
        </div>
      </header>
      {editEnabled && qwenEditIssue ? (
        <p className="dataset-command-error" role="alert">
          当前训练配置不兼容：{qwenEditIssue}；数据集可继续编辑，应用时需选择兼容配置。
        </p>
      ) : null}
      {editEnabled && rows.fields.some((_field, index) => form.watch(`datasets.${index}.is_reg`)) ? (
        <p className="dataset-command-error" role="alert">编辑 LoRA 首版不支持正则数据，请先移除正则子集。</p>
      ) : null}
      {typeof form.formState.errors.datasets?.message === 'string' ? (
        <p className="dataset-command-error" role="alert">{form.formState.errors.datasets.message}</p>
      ) : null}
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={rows.fields.map((field) => field.id)} strategy={verticalListSortingStrategy}>
          <div className="dataset-row-list">
            {rows.fields.map((field, index) => (
              <SortableDatasetSubset
                key={field.id}
                form={form}
                fieldId={field.id}
                fieldIds={rows.fields.map((item) => item.id)}
                index={index}
                rowCount={rows.fields.length}
                selected={field.id === selectedId}
                disabled={disabled}
                workbenchDisabled={workbenchDisabled}
                onSelect={() => setSelectedId(field.id)}
                onOpenWorkbench={onOpenWorkbench ? () => onOpenWorkbench(index) : undefined}
                onMove={(nextIndex) => rows.move(index, nextIndex)}
                onRemove={() => rows.remove(index)}
                onCopyExperimental={copyExperimentalRules}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>
    </section>
  );
}

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
  previewDisabled,
  onPreview,
}: {
  form: UseFormReturn<DatasetFormValues>;
  disabled: boolean;
  previewDisabled: boolean;
  onPreview: (index: number, trigger: HTMLElement) => void;
}) {
  const rows = useFieldArray({ control: form.control, name: 'datasets' });
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

  return (
    <section className="dataset-subsets">
      <header>
        <div>
          <h3>数据子集</h3>
          <span>{rows.fields.length} 项</span>
        </div>
        <button
          type="button"
          onClick={() => rows.append(emptyDatasetRow(form.getValues('defaults')))}
          disabled={disabled}
        >
          添加子集
        </button>
      </header>
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
                previewDisabled={previewDisabled}
                onSelect={() => setSelectedId(field.id)}
                onPreview={(trigger) => onPreview(index, trigger)}
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

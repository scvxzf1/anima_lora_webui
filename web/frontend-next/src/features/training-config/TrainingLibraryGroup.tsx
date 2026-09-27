import { useEffect, useState } from "react";
import { useTrainingDragFeedback } from "./trainingDragFeedback";
import { TrainingLibraryActions } from "./TrainingLibraryActions";
import { useDroppable } from "@dnd-kit/core";
import { groupDropId } from "./trainingLibraryDrag";
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { GripVertical, ChevronDown, ChevronRight } from "lucide-react";
import {
  type TrainingConfigFile,
  type TrainingConfigGroup,
} from "../../api/trainingContext";

type Props = {
  group: TrainingConfigGroup;
  files: TrainingConfigFile[];
  selectedPath?: string;
  disabled?: boolean;
  dirty?: boolean;
  onSelect: (path: string) => void;
  groups: TrainingConfigGroup[];
  searchActive: boolean;
  detailedManagement: boolean;
};

export function TrainingLibraryGroup({
  group,
  files,
  selectedPath,
  disabled,
  dirty,
  onSelect,
  groups,
  searchActive,
  detailedManagement,
}: Props) {
  const [open, setOpen] = useState(
    () =>
      !group.locked &&
      !group.readonly &&
      group.id !== "gui_methods" &&
      !group.files.every((file) => file.locked || file.readonly),
  );
  const locked = Boolean(
    disabled || dirty || searchActive || group.locked || group.readonly,
  );
  const feedback = useTrainingDragFeedback();
  useEffect(() => {
    if (feedback.pendingGroup === group.id) setOpen(true);
  }, [feedback.pendingGroup, group.id]);
  const { setNodeRef, isOver } = useDroppable({
    id: groupDropId(group.id),
    disabled: locked,
  });
  return (
    <section
      className="training-library-group"
      ref={setNodeRef}
      data-drop-over={isOver && Boolean(feedback.target?.placement)}
      data-drop-end={
        feedback.target?.over === groupDropId(group.id) &&
        Boolean(feedback.target?.placement)
      }
    >
      <header className="training-library-group-heading">
        <button
          type="button"
          className="training-library-group-toggle"
          aria-expanded={open || searchActive}
          onClick={() => setOpen(!open)}
        >
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="training-library-group-title">
            <strong>{group.label}</strong>
            <small>{files.length} 个配置</small>
          </span>
        </button>
        {detailedManagement && !(group.locked || group.readonly || group.system) && (
          <TrainingLibraryActions
            scope="group"
            groups={groups}
            targetGroup={group}
            disabled={Boolean(disabled || dirty || searchActive)}
            onRenamed={onSelect}
          />
        )}
      </header>
      {(open || searchActive) && (
        <SortableContext
          items={files.map((file) => file.path)}
          strategy={verticalListSortingStrategy}
        >
          {files.map((file) => (
            <SortableFile
              key={file.path}
              file={file}
              selected={file.path === selectedPath}
              disabled={Boolean(disabled)}
              dragDisabled={locked || Boolean(file.locked || file.readonly)}
              onSelect={onSelect}
              groups={groups}
              group={group}
              actionsDisabled={Boolean(disabled || dirty || searchActive)}
              detailedManagement={detailedManagement}
            />
          ))}
        </SortableContext>
      )}
    </section>
  );
}

function SortableFile({
  file,
  selected,
  disabled,
  dragDisabled,
  onSelect,
  groups,
  group,
  actionsDisabled,
  detailedManagement,
}: {
  file: TrainingConfigFile;
  selected: boolean;
  disabled: boolean;
  dragDisabled: boolean;
  onSelect: (path: string) => void;
  groups: TrainingConfigGroup[];
  group: TrainingConfigGroup;
  actionsDisabled: boolean;
  detailedManagement: boolean;
}) {
  const { setNodeRef, setActivatorNodeRef, attributes, listeners, isDragging } =
    useSortable({ id: file.path, disabled: dragDisabled });
  const feedback = useTrainingDragFeedback();
  const drop =
    feedback.target?.over === file.path && feedback.target.placement
      ? feedback.target.after
        ? "after"
        : "before"
      : undefined;
  const label = file.label || file.filename || file.path.split("/").pop();
  return (
    <div
      ref={setNodeRef}
      className="training-library-row"
      data-dragging={isDragging}
      data-drop-position={drop}
    >
      {detailedManagement && !dragDisabled && (
        <button
          type="button"
          className="training-library-drag"
          ref={setActivatorNodeRef}
          {...attributes}
          {...listeners}
          aria-label={`拖动排序 ${label}`}
          title={`拖动排序 ${label}`}
        >
          <GripVertical size={14} />
        </button>
      )}
      <button
        ref={!detailedManagement && !dragDisabled ? setActivatorNodeRef : undefined}
        type="button"
        className="training-library-item"
        data-selected={selected}
        data-hold-drag={!detailedManagement && !dragDisabled}
        disabled={disabled}
        title={file.path}
        onClick={() => onSelect(file.path)}
        {...(!detailedManagement && !dragDisabled ? { ...attributes, ...listeners } : {})}
      >
        <span className="training-library-file-copy">
          <strong>{label}</strong>
          <small>{file.path}</small>
        </span>
        {file.locked || file.readonly ? <small>只读</small> : null}
      </button>
      {detailedManagement && !(file.locked || file.readonly) && (
        <TrainingLibraryActions
          scope="file"
          groups={groups}
          targetGroup={group}
          file={file}
          disabled={actionsDisabled}
          onRenamed={onSelect}
        />
      )}
    </div>
  );
}

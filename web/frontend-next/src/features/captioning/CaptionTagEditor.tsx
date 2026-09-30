import { useState, type FormEvent } from "react";
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { ArrowDown, ArrowUp, GripVertical, Plus, Trash2 } from "lucide-react";
import { useEffect } from "react";
import {
  appendCaptionTag,
  joinCaptionTags,
  moveCaptionTag,
  splitCaptionTags,
  updateCaptionTag,
} from "./captionTags";
import "./CaptionTagEditor.css";

type Props = {
  itemId: string;
  value: string;
  tagModeAllowed: boolean;
  disabled?: boolean;
  onChange: (value: string) => void;
};
let nextTagId = 0;
const createTag = (value: string) => ({
  id: `caption-tag-${++nextTagId}`,
  value,
});

export function CaptionTagEditor({
  itemId,
  value,
  tagModeAllowed,
  disabled = false,
  onChange,
}: Props) {
  const [modeState, setModeState] = useState<{
    itemId: string;
    mode: "tags" | "raw";
  }>({ itemId, mode: "raw" });
  const mode = modeState.itemId === itemId ? modeState.mode : "raw";
  const [tags, setTags] = useState(() =>
    splitCaptionTags(value).map(createTag),
  );
  const [newTagState, setNewTagState] = useState({ itemId, text: "" });
  const newTag = newTagState.itemId === itemId ? newTagState.text : "";
  useEffect(() => {
    setTags(splitCaptionTags(value).map(createTag));
  }, [itemId, value]);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  function publish(next: typeof tags) {
    setTags(next);
    onChange(joinCaptionTags(next.map((tag) => tag.value)));
  }
  function dragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    publish(
      arrayMove(
        tags,
        tags.findIndex((tag) => tag.id === active.id),
        tags.findIndex((tag) => tag.id === over.id),
      ),
    );
  }
  function addTag(event: FormEvent) {
    event.preventDefault();
    const next = appendCaptionTag(
      tags.map((tag) => tag.value),
      newTag,
    );
    if (next.length !== tags.length)
      publish([...tags, createTag(next.at(-1)!)]);
    setNewTagState({ itemId, text: "" });
  }
  return (
    <div className="caption-tag-editor">
      {tagModeAllowed && (
        <div
          className="caption-tag-mode"
          role="group"
          aria-label="候选标注编辑模式"
        >
          <button
            type="button"
            aria-pressed={mode === "tags"}
            onClick={() => setModeState({ itemId, mode: "tags" })}
          >
            标签
          </button>
          <button
            type="button"
            aria-pressed={mode === "raw"}
            onClick={() => setModeState({ itemId, mode: "raw" })}
          >
            原始文本
          </button>
        </div>
      )}
      {!tagModeAllowed || mode === "raw" ? (
        <label className="caption-tag-raw">
          <span>候选标注</span>
          <textarea
            aria-label="候选标注"
            rows={7}
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
          />
        </label>
      ) : (
        <>
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={dragEnd}
          >
            <SortableContext
              items={tags.map((tag) => tag.id)}
              strategy={verticalListSortingStrategy}
            >
              <ol className="caption-tag-list" aria-label="标签顺序">
                {tags.map((tag, index) => (
                  <SortableTag
                    key={tag.id}
                    id={tag.id}
                    tag={tag.value}
                    index={index}
                    count={tags.length}
                    disabled={disabled}
                    onEdit={(text) => {
                      const values = updateCaptionTag(
                        tags.map((entry) => entry.value),
                        index,
                        text,
                      );
                      publish(
                        tags.map((entry, i) =>
                          i === index ? { ...entry, value: values[i] } : entry,
                        ),
                      );
                    }}
                    onDelete={() => publish(tags.filter((_, i) => i !== index))}
                    onMove={(to) => {
                      const reordered = moveCaptionTag(
                        tags.map((entry) => entry.id),
                        index,
                        to,
                      );
                      publish(
                        reordered.map((id) =>
                          tags.find((entry) => entry.id === id)!,
                        ),
                      );
                    }}
                  />
                ))}
                {!tags.length && (
                  <li className="caption-tag-empty">暂无标签</li>
                )}
              </ol>
            </SortableContext>
          </DndContext>
          <form className="caption-tag-add" onSubmit={addTag}>
            <input
              aria-label="添加标签"
              value={newTag}
              disabled={disabled}
              onChange={(event) =>
                setNewTagState({ itemId, text: event.target.value })
              }
              placeholder="添加标签"
            />
            <button
              type="submit"
              title="添加标签"
              aria-label="添加标签"
              disabled={disabled || !newTag.trim()}
            >
              <Plus size={16} />
            </button>
          </form>
        </>
      )}
    </div>
  );
}

function SortableTag({
  id,
  tag,
  index,
  count,
  disabled,
  onEdit,
  onDelete,
  onMove,
}: {
  id: string;
  tag: string;
  index: number;
  count: number;
  disabled: boolean;
  onEdit: (value: string) => void;
  onDelete: () => void;
  onMove: (to: number) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id, disabled });
  const [draft, setDraft] = useState(tag);
  useEffect(() => setDraft(tag), [tag]);
  return (
    <li
      ref={setNodeRef}
      className="caption-tag-row"
      style={{ transform: CSS.Transform.toString(transform), transition }}
      data-dragging={isDragging}
    >
      <button
        type="button"
        className="caption-tag-grip"
        title="拖动重排"
        aria-label={`拖动重排 ${tag}`}
        disabled={disabled}
        {...attributes}
        {...listeners}
      >
        <GripVertical size={16} />
      </button>
      <input
        aria-label={`编辑标签 ${tag}`}
        value={draft}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          const cleanDraft = draft.trim();
          if (!cleanDraft || cleanDraft === tag) setDraft(tag);
          else onEdit(draft);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
      />
      <button
        type="button"
        title="上移"
        aria-label={`上移 ${tag}`}
        disabled={disabled || index === 0}
        onClick={() => onMove(index - 1)}
      >
        <ArrowUp size={15} />
      </button>
      <button
        type="button"
        title="下移"
        aria-label={`下移 ${tag}`}
        disabled={disabled || index === count - 1}
        onClick={() => onMove(index + 1)}
      >
        <ArrowDown size={15} />
      </button>
      <button
        type="button"
        title="删除标签"
        aria-label={`删除 ${tag}`}
        disabled={disabled}
        onClick={onDelete}
      >
        <Trash2 size={15} />
      </button>
    </li>
  );
}

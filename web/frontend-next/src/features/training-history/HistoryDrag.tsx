import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  KeyboardSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  useIsMutating,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { GripVertical } from "lucide-react";
import { InlineConfirmDialog } from "../../components/InlineConfirmDialog";
import {
  batchUpdateHistoryTasks,
  historyKeys,
  saveHistoryCollections,
  type HistoryCollections,
  type HistoryTaskSummary,
} from "./api";
import {
  historyCollectionNames,
  historyConfigOrder,
  historyDragSelection,
  moveHistoryOrder,
} from "./historyOrder";
import {
  historyCollision,
  acceptsHistoryDrop,
  historyKeyboardCoordinates,
  type HistoryDragData,
} from "./historyDrag";

const DragBusy = createContext(false);

export function HistoryDrag({
  children,
  tasks,
  settings,
  selected,
  disabled,
  onMoved,
}: {
  children: ReactNode;
  tasks: HistoryTaskSummary[];
  settings?: HistoryCollections;
  selected: string[];
  disabled: boolean;
  onMoved: () => void;
}) {
  const qc = useQueryClient();
  const lock = useRef(false);
  const pending = useIsMutating({ mutationKey: ["training-history"] }) > 0;
  const [active, setActive] = useState<HistoryDragData>();
  const [confirmation, setConfirmation] = useState<{
    message: string;
    command: () => Promise<unknown>;
    danger?: boolean;
  }>();
  useEffect(() => {
    if (disabled) {
      setActive(undefined);
      setConfirmation(undefined);
    }
  }, [disabled]);
  const mutation = useMutation({
    mutationKey: historyKeys.collections,
    mutationFn: (command: () => Promise<unknown>) => command(),
    retry: false,
    onSettled: async () => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: historyKeys.collections }),
        qc.invalidateQueries({ queryKey: historyKeys.list }),
      ]);
      lock.current = false;
    },
  });
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: historyKeyboardCoordinates }),
  );
  function drop({ active, over }: DragEndEvent) {
    setActive(undefined);
    if (!over || active.id === over.id || disabled || pending || lock.current || !settings)
      return;
    const source = active.data.current as HistoryDragData;
    const target = over.data.current as HistoryDragData;
    if (!acceptsHistoryDrop(source, target)) return;
    let command: (() => Promise<unknown>) | undefined;
    if (source.kind === "collection" && target.kind === "collection") {
      command = () =>
        saveHistoryCollections({
          ...settings,
          collection_order: moveHistoryOrder(
            historyCollectionNames(tasks, settings),
            source.collection,
            target.collection,
          ),
        });
    } else if (source.kind === "config" && target.kind === "config") {
      command = () =>
        saveHistoryCollections({
          ...settings,
          config_group_order: {
            ...settings.config_group_order,
            [source.collection]: moveHistoryOrder(
              historyConfigOrder(tasks, source.collection, settings),
              source.key!,
              target.key!,
            ),
          },
        });
    } else if (target.kind === "collection") {
      const ids = historyDragSelection(source.taskIds || [], selected);
      if (
        !ids.length ||
        tasks
          .filter((task) => ids.includes(task.id || ""))
          .every((task) => (task.group || "") === target.collection)
      )
        return;
      command = async () => {
        await batchUpdateHistoryTasks({
          action: "set_group",
          task_ids: ids,
          group: target.collection,
        });
        onMoved();
      };
    }
    if (command) {
      if (target.kind === "collection" && source.kind !== "collection") {
        setConfirmation({
          message: `将 ${source.taskIds?.length || 0} 条已加载记录移到“${target.collection || "未分类"}”？同配置关联的全部历史记录会一起调整，训练文件保持不变。`,
          command,
        });
        return;
      }
      lock.current = true;
      mutation.mutate(command);
    }
  }
  return (
    <DragBusy.Provider value={disabled || pending || !settings}>
      {mutation.error && (
        <p className="history-error" role="alert">
          {mutation.error.message}
        </p>
      )}
      <DndContext
        sensors={sensors}
        collisionDetection={historyCollision}
        onDragStart={({ active }) =>
          setActive(active.data.current as HistoryDragData)
        }
        onDragCancel={() => setActive(undefined)}
        onDragEnd={drop}
      >
        {children}
        {createPortal(
          <DragOverlay dropAnimation={null}>
            {active && (
              <div className="history-drag-overlay">{active.label}</div>
            )}
          </DragOverlay>,
          document.body,
        )}
      </DndContext>
      {confirmation && (
        <InlineConfirmDialog
          title="移动历史记录"
          message={confirmation.message}
          onCancel={() => setConfirmation(undefined)}
          onConfirm={() => {
            if (disabled) { setConfirmation(undefined); return; }
            const command = confirmation.command;
            setConfirmation(undefined);
            lock.current = true;
            mutation.mutate(command);
          }}
        />
      )}
    </DragBusy.Provider>
  );
}

export function HistoryDragItem({
  id,
  data,
  children,
  disabled = false,
  handle = true,
}: {
  id: string;
  data: HistoryDragData;
  children: ReactNode;
  disabled?: boolean;
  handle?: boolean;
}) {
  const busy = useContext(DragBusy) || disabled;
  const drag = useDraggable({ id, data, disabled: busy || !handle });
  const drop = useDroppable({
    id,
    data,
    disabled: busy || data.kind === "task",
  });
  const setNodeRef = useCallback(
    (node: HTMLDivElement | null) => {
      drag.setNodeRef(node);
      drop.setNodeRef(node);
    },
    [drag.setNodeRef, drop.setNodeRef],
  );
  return (
    <div
      ref={setNodeRef}
      className="history-drag-row"
      data-history-drag={id}
      data-over={drop.isOver}
      data-dragging={drag.isDragging}
    >
      {handle && (
        <button
          type="button"
          className="history-drag-handle"
          disabled={busy}
          ref={drag.setActivatorNodeRef}
          {...drag.attributes}
          {...drag.listeners}
          aria-label={`拖动${data.kind === "collection" ? "集合" : data.kind === "config" ? "配置组" : "任务"} ${data.label}`}
          title={`拖动${data.kind === "collection" ? "集合" : data.kind === "config" ? "配置组" : "任务"} ${data.label}`}
        >
          <GripVertical size={16} />
        </button>
      )}
      {children}
    </div>
  );
}

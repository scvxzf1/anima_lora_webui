import { useRef, useState, type ReactNode } from "react";
import { TrainingDragFeedback, type DragTarget } from "./trainingDragFeedback";
import { createPortal } from "react-dom";
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
  pointerWithin,
  closestCenter,
  type CollisionDetection,
} from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "../../api/client";
import {
  trainingContextKeys,
  type TrainingConfigGroup,
} from "../../api/trainingContext";
import { trainingDropAfter, trainingFilePlacement } from "./trainingLibraryDrag";

export function TrainingLibraryDrag({
  groups,
  disabled,
  children,
}: {
  groups: TrainingConfigGroup[];
  disabled: boolean;
  children: (busy: boolean) => ReactNode;
}) {
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [target, setTarget] = useState<DragTarget | null>(null);
  const targetRef = useRef<DragTarget | null>(null);
  const collision: CollisionDetection = (args) => {
    const hits = args.pointerCoordinates
      ? pointerWithin(args)
      : closestCenter(args);
    const files = hits.filter(
      (hit) => !String(hit.id).startsWith("training-group:"),
    );
    const candidates = files.length ? files : hits;
    const hit = candidates[0];
    const rect = hit && args.droppableRects.get(hit.id);
    const after = trainingDropAfter(
      groups, String(args.active.id), String(hit?.id ?? ""),
      args.pointerCoordinates?.y, rect,
    );
    targetRef.current = hit
      ? {
          over: String(hit.id),
          after,
          placement: disabled
            ? null
            : trainingFilePlacement(
                groups,
                String(args.active.id),
                String(hit.id),
                after,
              ),
        }
      : null;
    return candidates;
  };
  const syncTarget = () =>
    setTarget((previous) => {
      const next = targetRef.current;
      return previous?.over === next?.over && previous?.after === next?.after
        ? previous
        : next;
    });
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: (body: NonNullable<ReturnType<typeof trainingFilePlacement>>) =>
      apiRequest("/api/config/file-groups/place", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    retry: false,
    onMutate: async (body) => {
      const key = trainingContextKeys.files();
      await queryClient.cancelQueries({ queryKey: key });
      const previous = queryClient.getQueryData<TrainingConfigGroup[]>(key);
      const file = previous
        ?.flatMap((group) => group.files)
        .find((file) => file.path === body.file);
      if (file)
        queryClient.setQueryData(
          key,
          previous!.map((group) => {
            const files = group.files.filter((item) => item.path !== body.file);
            if (group.id === body.group) files.splice(body.index, 0, file);
            return { ...group, files };
          }),
        );
      return previous;
    },
    onError: (_error, _body, previous) => {
      if (previous)
        queryClient.setQueryData(trainingContextKeys.files(), previous);
    },
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: trainingContextKeys.files() }),
  });
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const destination = groups.find(
    (group) => group.id === target?.placement?.group,
  );
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={collision}
      onDragStart={({ active }) => {
        mutation.reset();
        targetRef.current = null;
        setTarget(null);
        setActiveFile(String(active.id));
      }}
      onDragMove={syncTarget}
      onDragOver={syncTarget}
      onDragCancel={() => {
        setActiveFile(null);
        setTarget(null);
        targetRef.current = null;
      }}
      onDragEnd={() => {
        setActiveFile(null);
        const placement = targetRef.current?.placement;
        setTarget(null);
        targetRef.current = null;
        if (!disabled && !mutation.isPending && placement)
          mutation.mutate(placement);
      }}
    >
      <TrainingDragFeedback.Provider
        value={{
          active: Boolean(activeFile),
          target,
          pendingGroup: mutation.isPending
            ? mutation.variables?.group
            : undefined,
        }}
      >
        {children(mutation.isPending)}
      </TrainingDragFeedback.Provider>
      {mutation.isPending && <p role="status">正在保存排序</p>}
      {mutation.error && (
        <p className="form-error" role="alert">
          {mutation.error.message}
        </p>
      )}
      {createPortal(
        <DragOverlay dropAnimation={null}>
          {activeFile && (
            <div
              className="training-library-drag-overlay"
              data-valid={Boolean(target?.placement)}
            >
              <strong>{activeFile.split("/").pop()}</strong>
              <small>
                {destination && target?.placement
                  ? `${destination.label} · 第 ${target.placement.index + 1} 位`
                  : target?.over === activeFile
                    ? "原位置"
                    : "不可放置"}
              </small>
            </div>
          )}
        </DragOverlay>,
        document.body,
      )}
    </DndContext>
  );
}

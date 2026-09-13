import {
  closestCenter,
  pointerWithin,
  type CollisionDetection,
  type KeyboardCoordinateGetter,
} from "@dnd-kit/core";

export type HistoryDragData = {
  kind: "collection" | "config" | "task";
  collection: string;
  key?: string;
  region?: string;
  label: string;
  taskIds?: string[];
};

export function acceptsHistoryDrop(
  source?: HistoryDragData,
  target?: HistoryDragData,
) {
  if (!source || !target) return false;
  if (source.kind === "collection")
    return target.kind === "collection" && Boolean(target.collection);
  if (target.kind === "collection") return true;
  return (
    source.kind === "config" &&
    target.kind === "config" &&
    source.collection === target.collection &&
    source.region === target.region
  );
}

export const historyCollision: CollisionDetection = (args) => {
  const droppableContainers = args.droppableContainers.filter((item) =>
    acceptsHistoryDrop(
      args.active.data.current as HistoryDragData,
      item.data.current as HistoryDragData,
    ),
  );
  const filtered = { ...args, droppableContainers };
  return args.pointerCoordinates
    ? pointerWithin(filtered)
    : closestCenter(filtered);
};

// Keyboard moves between valid targets, rather than stepping through empty pixels.
export const historyKeyboardCoordinates: KeyboardCoordinateGetter = (
  event,
  { context, currentCoordinates },
) => {
  const { active, over, collisionRect, droppableContainers, droppableRects } =
    context;
  if (!active || !collisionRect || !event.code.startsWith("Arrow")) return;
  event.preventDefault();
  const x = collisionRect.left + collisionRect.width / 2;
  const y = collisionRect.top + collisionRect.height / 2;
  // Overlay dimensions can differ from the row; use row geometry for direction.
  const origin = droppableRects.get(over?.id ?? active.id) ?? collisionRect;
  const originX = origin.left + origin.width / 2;
  const originY = origin.top + origin.height / 2;
  const candidates = droppableContainers
    .getEnabled()
    .flatMap((item) => {
      const rect = droppableRects.get(item.id);
      if (
        item.id === (over?.id ?? active.id) ||
        !rect ||
        !acceptsHistoryDrop(
          active.data.current as HistoryDragData,
          item.data.current as HistoryDragData,
        )
      )
        return [];
      const dx = rect.left + rect.width / 2 - x;
      const dy = rect.top + rect.height / 2 - y;
      const directionX = rect.left + rect.width / 2 - originX;
      const directionY = rect.top + rect.height / 2 - originY;
      const sameRow = Math.abs(directionY) <= 1;
      const valid =
        event.code === "ArrowDown"
          ? directionY > 1 || (sameRow && directionX > 1)
          : event.code === "ArrowUp"
            ? directionY < -1 || (sameRow && directionX < -1)
            : event.code === "ArrowRight"
              ? directionX > 1
              : directionX < -1;
      return valid ? [{ id: item.id, dx, dy, distance: Math.hypot(dx, dy) }] : [];
    })
    .sort((a, b) => a.distance - b.distance);
  const next = candidates[0];
  if (next) {
    return { x: currentCoordinates.x + next.dx, y: currentCoordinates.y + next.dy };
  }
};

import {
  closestCenter,
  pointerWithin,
  type CollisionDetection,
} from "@dnd-kit/core";
import { insertPath, isDatasetMoveTarget } from "./datasetOrdering";
import type { DatasetLibraryGroup } from "./types";

export type DatasetDrop = {
  groupId: string;
  file?: string;
  position: "before" | "after";
};

export function datasetCollision(
  onTarget: (target: DatasetDrop | null) => void,
): CollisionDetection {
  return (args) => {
    const groupDrag = args.active.data.current?.type === "group";
    const droppableContainers = args.droppableContainers.filter((container) => {
      const type = container.data.current?.type;
      return groupDrag
        ? type === "group"
        : type === "preset" || type === "group-drop";
    });
    const filtered = { ...args, droppableContainers };
    const collisions = args.pointerCoordinates
      ? pointerWithin(filtered)
      : closestCenter(filtered);
    const container = droppableContainers.find(
      (entry) => entry.id === collisions[0]?.id,
    );
    const data = container?.data.current;
    const rect = container && args.droppableRects.get(container.id);
    if (!groupDrag && data?.groupId && rect) {
      const y =
        args.pointerCoordinates?.y ??
        args.collisionRect.top + args.collisionRect.height / 2;
      onTarget({
        groupId: data.groupId,
        file: data.file,
        position: y < rect.top + rect.height / 2 ? "before" : "after",
      });
    } else onTarget(null);
    return collisions;
  };
}

export function datasetDropOrder(
  groups: DatasetLibraryGroup[],
  file: string,
  target: DatasetDrop,
) {
  const group = groups.find((entry) => entry.id === target.groupId);
  if (!group || !isDatasetMoveTarget(group) || target.file === file)
    return null;
  const paths = group.files
    .map((entry) => entry.path)
    .filter((path) => path !== file);
  const anchor = target.file ? paths.indexOf(target.file) : paths.length;
  if (anchor < 0) return null;
  return insertPath(
    paths,
    file,
    anchor + (target.file && target.position === "after" ? 1 : 0),
  );
}

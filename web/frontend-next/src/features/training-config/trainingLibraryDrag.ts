import type { TrainingConfigGroup } from "../../api/trainingContext";

export const groupDropId = (id: string) => `training-group:${id}`;

export function trainingDropAfter(
  groups: TrainingConfigGroup[],
  file: string,
  over: string,
  pointerY: number | undefined,
  rect: { top: number; height: number } | undefined,
) {
  if (pointerY !== undefined)
    return Boolean(rect && pointerY > rect.top + rect.height / 2);
  // Keyboard hits are centered on the target; use list direction, not its midpoint.
  const order = groups.flatMap((group) => group.files.map((item) => item.path));
  const source = order.indexOf(file);
  const target = order.indexOf(over);
  return source >= 0 && target > source;
}

export function trainingFilePlacement(
  groups: TrainingConfigGroup[],
  file: string,
  over: string,
  after = false,
) {
  const source = groups.find((group) =>
    group.files.some((item) => item.path === file),
  );
  const item = source?.files.find((item) => item.path === file);
  const target = groups.find(
    (group) =>
      groupDropId(group.id) === over ||
      group.files.some((item) => item.path === over),
  );
  if (
    !source ||
    !item ||
    !target ||
    source.locked ||
    source.readonly ||
    item.locked ||
    item.readonly ||
    target.locked ||
    target.readonly ||
    file === over
  )
    return null;
  const order = target.files
    .filter((item) => item.path !== file)
    .map((item) => item.path);
  const anchor = order.indexOf(over);
  const index = anchor < 0 ? order.length : anchor + Number(after);
  order.splice(index, 0, file);
  if (
    source.id === target.id &&
    order.every((path, i) => path === target.files[i]?.path)
  )
    return null;
  return { target: "file", file, group: target.id, index };
}

import { createContext, useContext } from "react";
import type { trainingFilePlacement } from "./trainingLibraryDrag";

export type DragTarget = {
  over: string;
  after: boolean;
  placement: ReturnType<typeof trainingFilePlacement>;
};
export const TrainingDragFeedback = createContext<{
  active: boolean;
  target: DragTarget | null;
  pendingGroup?: string;
}>({ active: false, target: null });
export const useTrainingDragFeedback = () => useContext(TrainingDragFeedback);

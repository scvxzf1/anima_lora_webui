import { useEffect, useLayoutEffect, useRef } from "react";
import {
  trainingScrollRoot,
  type TrainingEditorView,
} from "./useTrainingEditorView";

function stageAtScrollPosition(root: HTMLElement) {
  const maxScroll = root.scrollHeight - root.clientHeight;
  // When every section fits, navigation can select any section without scrolling.
  if (maxScroll <= 1) return;
  const sections = Array.from(
    document.querySelectorAll<HTMLElement>("[data-training-stage]"),
  );
  if (root.scrollTop >= maxScroll - 2)
    return sections.at(-1)?.dataset.trainingStage;
  const nav = document.querySelector(".training-stage-nav");
  const top =
    Math.max(
      root.getBoundingClientRect().top,
      nav?.getBoundingClientRect().bottom ?? 0,
    ) + 12;
  return sections.find(
    (section) => section.getBoundingClientRect().bottom > top,
  )?.dataset.trainingStage;
}

export function useTrainingStageNavigation(
  view: TrainingEditorView,
  activeStage: string,
  setActiveStage: (stage: string) => void,
) {
  const previous = useRef({ view, activeStage });
  const currentStage = useRef(activeStage);
  currentStage.current = activeStage;
  useLayoutEffect(() => {
    if (
      view === "groups" &&
      previous.current.view === view &&
      previous.current.activeStage !== activeStage
    ) {
      const root = trainingScrollRoot();
      if (root?.classList.contains("training-edit-groups")) root.scrollTop = 0;
      else
        document
          .getElementById(`training-stage-${activeStage}`)
          ?.scrollIntoView({ block: "start", behavior: "instant" });
    }
    previous.current = { view, activeStage };
  }, [view, activeStage]);
  useEffect(() => {
    if (view !== "stream") return;
    let frame = 0;
    let root = trainingScrollRoot();
    const update = () => {
      frame = 0;
      if (!root) return;
      const stage = stageAtScrollPosition(root);
      if (stage && stage !== currentStage.current) setActiveStage(stage);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    const resize = () => {
      root?.removeEventListener("scroll", schedule);
      root = trainingScrollRoot();
      root?.addEventListener("scroll", schedule, { passive: true });
      schedule();
    };
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(schedule);
    document
      .querySelectorAll("[data-training-stage]")
      .forEach((section) => observer?.observe(section));
    root?.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", resize);
    return () => {
      observer?.disconnect();
      root?.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(frame);
    };
  }, [view, setActiveStage]);
  return (stage: string) => {
    (document.activeElement as HTMLElement | null)?.blur();
    setActiveStage(stage);
    if (view === "stream")
      document
        .getElementById(`training-stage-${stage}`)
        ?.scrollIntoView({ block: "start", behavior: "instant" });
  };
}

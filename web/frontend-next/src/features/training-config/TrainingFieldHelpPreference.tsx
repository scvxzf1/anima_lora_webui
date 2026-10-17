import { createContext, useContext, useState, type ReactNode } from "react";

export type TrainingFieldHelpDisplay = "hidden" | "single" | "full";
const STORAGE_KEY = "dragon-next.training-field-help-display.v1";
const TrainingFieldHelpContext = createContext<{
  display: TrainingFieldHelpDisplay;
  setDisplay: (display: TrainingFieldHelpDisplay) => void;
}>({ display: "single", setDisplay: () => {} });

function readDisplay(): TrainingFieldHelpDisplay {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "hidden" || saved === "full" ? saved : "single";
  } catch {
    return "single";
  }
}

export function TrainingFieldHelpProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [display, updateDisplay] = useState(readDisplay);
  const setDisplay = (next: TrainingFieldHelpDisplay) => {
    updateDisplay(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Keep the current page preference usable when storage is unavailable.
    }
  };
  return (
    <TrainingFieldHelpContext.Provider value={{ display, setDisplay }}>
      {children}
    </TrainingFieldHelpContext.Provider>
  );
}

export function useTrainingFieldHelpDisplay() {
  return useContext(TrainingFieldHelpContext);
}

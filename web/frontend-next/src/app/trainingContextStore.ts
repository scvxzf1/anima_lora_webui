import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type TrainingContextState = {
  configFile: string;
  preset: string;
  selectConfigFile: (configFile: string) => void;
  selectPreset: (preset: string) => void;
};

export const useTrainingContextStore = create<TrainingContextState>()(persist(
  (set) => ({
    configFile: '',
    preset: 'default',
    selectConfigFile: (configFile) => set({ configFile }),
    selectPreset: (preset) => set({ preset }),
  }),
  { name: 'dragon-next-training-context' },
));

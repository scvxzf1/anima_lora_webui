import { describe, expect, it } from 'vitest';

import { draftFromMerged, importedTrainingPath, rawConfigOwnKeys, trainingPatchValues } from './trainingForm';

describe('training config form domain', () => {
  it('keeps absent numeric values blank and only emits actual field differences', () => {
    const baseline = draftFromMerged({ output_name: 'run', gradient_checkpointing: true });
    expect(baseline.max_train_steps).toBe('');
    expect(baseline.network_dim).toBe('');

    expect(trainingPatchValues({ ...baseline, output_name: 'edited' }, baseline)).toEqual({
      output_name: 'edited',
    });
  });

  it('tracks only top-level TOML ownership and normalizes imported save-as paths', () => {
    expect([...rawConfigOwnKeys([
      'output_name = "run"',
      'max_train_steps = 100',
      '[variant]',
      'family = "lora"',
    ].join('\n'))]).toEqual(['output_name', 'max_train_steps']);
    expect(importedTrainingPath('dragon copy.toml')).toBe('configs/imported/dragon_copy.toml');
    expect(importedTrainingPath('../escape')).toBe('configs/imported/escape.toml');
  });
});

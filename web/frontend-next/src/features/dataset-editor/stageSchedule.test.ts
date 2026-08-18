import { describe, expect, it } from 'vitest';

import {
  addStageScheduleStage,
  applyStageTemplate,
  deleteStageScheduleStage,
  moveStageScheduleBinding,
  normalizeStageSchedule,
  updateStageScheduleStage,
  validateStageSchedule,
} from './stageSchedule';

describe('stage schedule domain model', () => {
  it('normalizes aliases and percentage values', () => {
    expect(normalizeStageSchedule([
      { name: 'warmup', subsetIndex: 1, startPct: 0, endPct: 40 },
      { name: 'finish', dataset_index: 0, start_pct: 40, end_pct: 100 },
    ])).toEqual([
      { name: 'warmup', subset_index: 1, start_pct: 0, end_pct: 0.4 },
      { name: 'finish', subset_index: 0, start_pct: 0.4, end_pct: 1 },
    ]);
  });

  it('creates contiguous N-stage templates', () => {
    const stages = applyStageTemplate(5, 3);
    expect(stages).toHaveLength(5);
    expect(stages[0].start_pct).toBe(0);
    expect(stages[4].end_pct).toBe(1);
    expect(validateStageSchedule(stages, 3)).toEqual([]);
  });

  it('keeps adjacent boundaries attached when editing a percentage', () => {
    const stages = updateStageScheduleStage(applyStageTemplate(3, 3), 0, { end_pct: 0.4 });
    expect(stages[0].end_pct).toBe(0.4);
    expect(stages[1].start_pct).toBe(0.4);
    expect(validateStageSchedule(stages, 3)).toEqual([]);
  });

  it('adds by splitting the final stage and deletes without gaps', () => {
    const added = addStageScheduleStage(applyStageTemplate(2, 3), 3);
    expect(added).toHaveLength(3);
    expect(validateStageSchedule(added, 3)).toEqual([]);
    const deleted = deleteStageScheduleStage(added, 1);
    expect(deleted).toHaveLength(2);
    expect(validateStageSchedule(deleted, 3)).toEqual([]);
  });

  it('moves only stage bindings and preserves intervals', () => {
    const stages = applyStageTemplate(2, 2);
    const moved = moveStageScheduleBinding(stages, 1, -1);
    expect(moved.map((stage) => stage.subset_index)).toEqual([1, 0]);
    expect(moved.map((stage) => [stage.start_pct, stage.end_pct])).toEqual([[0, 0.5], [0.5, 1]]);
  });

  it('reports gaps and subset references outside the current dataset', () => {
    const issues = validateStageSchedule([
      { name: 'one', subset_index: 0, start_pct: 0, end_pct: 0.4 },
      { name: 'two', subset_index: 2, start_pct: 0.5, end_pct: 1 },
    ], 2);
    expect(issues.map((issue) => issue.message)).toEqual(expect.arrayContaining([
      '阶段1 与阶段2 未贴齐',
      '阶段2 的子集索引超出范围',
    ]));
  });
});

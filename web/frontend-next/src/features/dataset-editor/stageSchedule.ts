import { z } from 'zod';

export const MAX_STAGE_COUNT = 12;
export const MIN_STAGE_SPAN = 0.001;

export const stageScheduleStageSchema = z.object({
  name: z.string(),
  subset_index: z.number().int().min(0),
  start_pct: z.number().min(0).max(1),
  end_pct: z.number().min(0).max(1),
});

export type StageScheduleStage = z.infer<typeof stageScheduleStageSchema>;

export type StageScheduleIssue = {
  message: string;
  stageIndex?: number;
  field?: keyof StageScheduleStage;
};

export class StageScheduleOperationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StageScheduleOperationError';
  }
}

export function normalizeStageSchedule(raw: unknown): StageScheduleStage[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item, index) => {
    if (!item || typeof item !== 'object') return [];
    const value = item as Record<string, unknown>;
    return [{
      name: String(value.name || `阶段${index + 1}`).trim() || `阶段${index + 1}`,
      subset_index: Math.max(0, integer(value.subset_index ?? value.subsetIndex ?? value.dataset_index, index)),
      start_pct: fraction(value.start_pct ?? value.startPct, 0),
      end_pct: fraction(value.end_pct ?? value.endPct, 1),
    }];
  });
}

export function defaultStageSchedule(subsetCount: number) {
  return [
    { name: '阶段1', subset_index: 0, start_pct: 0, end_pct: 0.5 },
    {
      name: '阶段2',
      subset_index: Math.min(1, Math.max(0, subsetCount - 1)),
      start_pct: 0.5,
      end_pct: 1,
    },
  ] satisfies StageScheduleStage[];
}

export function applyStageTemplate(count: number, subsetCount: number) {
  const stageCount = Math.max(1, Math.min(MAX_STAGE_COUNT, integer(count, 2)));
  return Array.from({ length: stageCount }, (_, index) => ({
    name: `阶段${index + 1}`,
    subset_index: Math.min(index, Math.max(0, subsetCount - 1)),
    start_pct: index / stageCount,
    end_pct: (index + 1) / stageCount,
  }));
}

export function updateStageScheduleStage(
  input: StageScheduleStage[],
  index: number,
  patch: Partial<StageScheduleStage>,
) {
  const stages = normalizeStageSchedule(input).map((stage) => ({ ...stage }));
  if (!stages[index]) return stages;
  const next = { ...stages[index], ...patch };
  if ('name' in patch) next.name = String(patch.name || '').trim() || `阶段${index + 1}`;
  if ('subset_index' in patch) next.subset_index = Math.max(0, integer(patch.subset_index, 0));

  if ('end_pct' in patch) {
    const minEnd = stages[index].start_pct + MIN_STAGE_SPAN;
    const maxEnd = index < stages.length - 1
      ? 1 - MIN_STAGE_SPAN * (stages.length - 1 - index)
      : 1;
    next.end_pct = clamp(Math.max(minEnd, Math.min(maxEnd, Number(patch.end_pct))));
    if (stages[index + 1]) {
      stages[index + 1].start_pct = next.end_pct;
      if (stages[index + 1].end_pct <= stages[index + 1].start_pct) {
        stages[index + 1].end_pct = Math.min(1, stages[index + 1].start_pct + MIN_STAGE_SPAN);
      }
    }
  }
  if ('start_pct' in patch) {
    const minStart = index === 0 ? 0 : MIN_STAGE_SPAN * index;
    const maxStart = stages[index].end_pct - MIN_STAGE_SPAN;
    next.start_pct = clamp(Math.max(minStart, Math.min(maxStart, Number(patch.start_pct))));
    if (stages[index - 1]) {
      stages[index - 1].end_pct = next.start_pct;
      if (stages[index - 1].end_pct <= stages[index - 1].start_pct) {
        stages[index - 1].start_pct = Math.max(0, stages[index - 1].end_pct - MIN_STAGE_SPAN);
      }
    }
  }
  stages[index] = next;
  stages[0].start_pct = 0;
  stages[stages.length - 1].end_pct = 1;
  return stages;
}

export function addStageScheduleStage(input: StageScheduleStage[], subsetCount: number) {
  const stages = normalizeStageSchedule(input).map((stage) => ({ ...stage }));
  if (!stages.length) return defaultStageSchedule(subsetCount);
  if (stages.length >= MAX_STAGE_COUNT) throw new StageScheduleOperationError('最多支持 12 个阶段');
  const last = stages[stages.length - 1];
  const span = last.end_pct - last.start_pct;
  if (span <= MIN_STAGE_SPAN * 2) {
    throw new StageScheduleOperationError('最后一段太短，无法继续拆分');
  }
  const middle = last.start_pct + span / 2;
  last.end_pct = middle;
  stages.push({
    name: `阶段${stages.length + 1}`,
    subset_index: Math.min(stages.length, Math.max(0, subsetCount - 1)),
    start_pct: middle,
    end_pct: 1,
  });
  return stages;
}

export function deleteStageScheduleStage(input: StageScheduleStage[], index: number) {
  const stages = normalizeStageSchedule(input).map((stage) => ({ ...stage }));
  if (stages.length <= 1 || !stages[index]) return stages;
  const [removed] = stages.splice(index, 1);
  if (index > 0) stages[index - 1].end_pct = Math.max(stages[index - 1].end_pct, removed.end_pct);
  else stages[0].start_pct = 0;
  stages[0].start_pct = 0;
  stages[stages.length - 1].end_pct = 1;
  for (let cursor = 1; cursor < stages.length; cursor += 1) {
    stages[cursor].start_pct = stages[cursor - 1].end_pct;
    if (stages[cursor].end_pct <= stages[cursor].start_pct) {
      const remaining = stages.length - cursor;
      stages[cursor].end_pct = Math.min(
        1,
        stages[cursor].start_pct + Math.max(MIN_STAGE_SPAN, (1 - stages[cursor].start_pct) / remaining),
      );
    }
  }
  return stages;
}

export function moveStageScheduleBinding(input: StageScheduleStage[], index: number, direction: -1 | 1) {
  const stages = normalizeStageSchedule(input).map((stage) => ({ ...stage }));
  const target = index + direction;
  if (!stages[index] || !stages[target]) return stages;
  const binding = { name: stages[index].name, subset_index: stages[index].subset_index };
  stages[index].name = stages[target].name;
  stages[index].subset_index = stages[target].subset_index;
  stages[target].name = binding.name;
  stages[target].subset_index = binding.subset_index;
  return stages;
}

export function validateStageSchedule(stages: StageScheduleStage[], subsetCount: number) {
  const issues: StageScheduleIssue[] = [];
  if (!stages.length) return [{ message: '阶段表为空' }];
  if (Math.abs(stages[0].start_pct) > 1e-6) {
    issues.push({ message: '第一阶段必须从 0% 开始', stageIndex: 0, field: 'start_pct' });
  }
  if (Math.abs(stages[stages.length - 1].end_pct - 1) > 1e-6) {
    issues.push({ message: '最后一阶段必须到 100%', stageIndex: stages.length - 1, field: 'end_pct' });
  }
  stages.forEach((stage, index) => {
    if (stage.end_pct <= stage.start_pct + 1e-9) {
      issues.push({ message: `阶段${index + 1} 区间为空`, stageIndex: index, field: 'end_pct' });
    }
    if (subsetCount <= 0 || stage.subset_index < 0 || stage.subset_index >= subsetCount) {
      issues.push({
        message: `阶段${index + 1} 的子集索引超出范围`,
        stageIndex: index,
        field: 'subset_index',
      });
    }
    if (index > 0) {
      const previous = stages[index - 1];
      if (Math.abs(previous.end_pct - stage.start_pct) > 1e-6) {
        issues.push({ message: `阶段${index} 与阶段${index + 1} 未贴齐`, stageIndex: index, field: 'start_pct' });
      }
      if (stage.start_pct < previous.end_pct - 1e-6) {
        issues.push({ message: `阶段${index} 与阶段${index + 1} 区间重叠`, stageIndex: index, field: 'start_pct' });
      }
    }
  });
  return issues;
}

export function stageStepRange(stage: StageScheduleStage, totalSteps: number) {
  if (totalSteps <= 0) return '未设置总步数';
  return `${Math.floor(totalSteps * stage.start_pct)}-${Math.floor(totalSteps * stage.end_pct)}`;
}

export function pctLabel(value: number) {
  return `${Math.round(clamp(value) * 1000) / 10}%`;
}

function fraction(value: unknown, fallback: number) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return clamp(number > 1 ? number / 100 : number);
}

function integer(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number) : fallback;
}

function clamp(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

import { z } from 'zod';

import type {
  DatasetPresetMutationResponse,
  DatasetPresetResponse,
  DatasetPresetWritePayload,
} from './types';
import {
  normalizeStageSchedule,
  stageScheduleStageSchema,
  validateStageSchedule,
} from './stageSchedule';

export const captionSourceModes = ['auto', 'txt', 'json', 'captions_json'] as const;

const settingsShape = {
  resolution: z.number().int().min(64, '分辨率至少为 64'),
  batch_size: z.number().int().min(1, '批大小至少为 1'),
  prior_loss_weight: z.number().min(0, '正则损失权重不能小于 0'),
  enable_bucket: z.boolean(),
  min_bucket_reso: z.number().int().min(64, '最小桶尺寸至少为 64'),
  max_bucket_reso: z.number().int().min(64, '最大桶尺寸至少为 64'),
  bucket_reso_steps: z.number().int().min(1, '桶步长至少为 1'),
  bucket_no_upscale: z.boolean(),
  validation_split: z.number().min(0).max(1, '验证集比例必须在 0 到 1 之间'),
  validation_split_num: z.number().int().min(0, '验证集数量不能小于 0'),
  validation_seed: z.number().int().min(0, '验证随机种子不能小于 0'),
  caption_extension: z.string().trim().min(1, '请填写标注扩展名'),
  prefer_json_caption: z.boolean(),
  caption_source_mode: z.enum(captionSourceModes),
};

const datasetSettingsSchema = z.object(settingsShape).catchall(z.unknown());
const datasetDefaultsSchema = z.object({
  ...settingsShape,
  keep_tokens: z.number().int().min(0, '保留 Token 数不能小于 0'),
}).catchall(z.unknown());

const nlTagMixSchema = z.object({
  enabled: z.boolean(),
  tag_ratio: z.number().min(0).max(1, '标签比例必须在 0 到 1 之间'),
}).catchall(z.unknown());

const triggerCloneSchema = z.object({
  enabled: z.boolean(),
  prompt: z.string().trim(),
  num_repeats: z.number().int().min(1, '复制次数至少为 1'),
}).catchall(z.unknown());

const datasetRowSchema = z.object({
  source_dir: z.string().trim().min(1, '请输入原始图片目录'),
  image_dir: z.string().trim(),
  cache_dir: z.string().trim(),
  num_repeats: z.number().int().min(1, '重复次数至少为 1'),
  is_reg: z.boolean(),
  recursive: z.boolean(),
  path_pattern: z.string().trim().min(1, '路径筛选不能为空'),
  settings: datasetSettingsSchema,
  nl_tag_mix: nlTagMixSchema,
  trigger_clone: triggerCloneSchema,
}).catchall(z.unknown());

export const datasetFormSchema = z.object({
  defaults: datasetDefaultsSchema,
  datasets: z.array(datasetRowSchema).min(1, '至少保留一个数据子集'),
  stage_schedule_enabled: z.boolean(),
  stage_schedule: z.array(stageScheduleStageSchema),
}).superRefine((value, context) => {
  validateBucketSettings(value.defaults, ['defaults'], context);
  value.datasets.forEach((row, index) => {
    validateBucketSettings(row.settings, ['datasets', index, 'settings'], context);
    if (row.trigger_clone.enabled && !row.trigger_clone.prompt) {
      context.addIssue({
        code: 'custom',
        path: ['datasets', index, 'trigger_clone', 'prompt'],
        message: '启用触发词复制后必须填写触发词',
      });
    }
  });
  if (value.datasets.every((row) => row.is_reg)) {
    context.addIssue({
      code: 'custom',
      path: ['datasets'],
      message: '至少保留一个普通训练数据子集',
    });
  }
  if (value.stage_schedule_enabled) {
    validateStageSchedule(value.stage_schedule, value.datasets.length).forEach((issue) => {
      context.addIssue({
        code: 'custom',
        path: issue.stageIndex === undefined
          ? ['stage_schedule']
          : ['stage_schedule', issue.stageIndex, issue.field || 'name'],
        message: issue.message,
      });
    });
  }
});

export type DatasetFormValues = z.infer<typeof datasetFormSchema>;
export type DatasetSettingsValues = DatasetFormValues['datasets'][number]['settings'];

export function emptyDatasetRow(defaults: Partial<DatasetSettingsValues> = {}): DatasetFormValues['datasets'][number] {
  return {
    source_dir: '',
    image_dir: '',
    cache_dir: '',
    num_repeats: 1,
    is_reg: false,
    recursive: true,
    path_pattern: '*',
    settings: normalizeSettings(defaults),
    nl_tag_mix: { enabled: false, tag_ratio: 0.7 },
    trigger_clone: { enabled: false, prompt: '', num_repeats: 1 },
  };
}

export function emptyDatasetForm(): DatasetFormValues {
  const defaults = normalizeDefaults({});
  return {
    defaults,
    datasets: [emptyDatasetRow(defaults)],
    stage_schedule_enabled: false,
    stage_schedule: [],
  };
}

export function datasetFormFromPreset(
  preset: DatasetPresetResponse | DatasetPresetMutationResponse,
): DatasetFormValues {
  const defaults = normalizeDefaults(preset.defaults || {});
  return {
    defaults,
    datasets: preset.datasets.length
      ? preset.datasets.map((row) => ({
          ...row,
          source_dir: String(row.source_dir || ''),
          image_dir: String(row.image_dir || ''),
          cache_dir: String(row.cache_dir || ''),
          num_repeats: positiveInt(row.num_repeats, 1),
          is_reg: Boolean(row.is_reg),
          recursive: row.recursive !== false,
          path_pattern: String(row.path_pattern || '*').trim() || '*',
          settings: normalizeSettings({ ...defaults, ...(row.settings || {}) }),
          nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
          trigger_clone: normalizeTriggerClone(row.trigger_clone),
        }))
      : [emptyDatasetRow(defaults)],
    stage_schedule_enabled: Boolean(preset.stage_schedule_enabled),
    stage_schedule: normalizeStageSchedule(preset.stage_schedule),
  };
}

export function datasetWritePayload(values: DatasetFormValues): DatasetPresetWritePayload {
  return {
    datasets: values.datasets,
    defaults: values.defaults,
    stage_schedule_enabled: values.stage_schedule_enabled,
    stage_schedule: values.stage_schedule,
  };
}

export function datasetPresetPathFromName(name: string) {
  const stem = String(name || '')
    .replace(/\.toml$/i, '')
    .replace(/\\/g, '/')
    .split('/')
    .pop()
    ?.replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'dataset';
  return `configs/datasets/${stem}.toml`;
}

export function datasetPresetStem(file: string) {
  return file.split('/').pop()?.replace(/\.toml$/i, '') || 'dataset';
}

function normalizeDefaults(raw: Record<string, unknown>): DatasetFormValues['defaults'] {
  return {
    ...raw,
    ...normalizeSettings(raw),
    keep_tokens: nonnegativeInt(raw.keep_tokens, 3),
  };
}

function normalizeSettings(raw: Record<string, unknown>): DatasetSettingsValues {
  const resolution = positiveInt(raw.resolution, 1024);
  const minBucket = positiveInt(raw.min_bucket_reso, 256);
  return {
    ...raw,
    resolution,
    batch_size: positiveInt(raw.batch_size, 1),
    prior_loss_weight: nonnegativeNumber(raw.prior_loss_weight, 1),
    enable_bucket: raw.enable_bucket !== false,
    min_bucket_reso: minBucket,
    max_bucket_reso: Math.max(resolution, positiveInt(raw.max_bucket_reso, 1024), minBucket),
    bucket_reso_steps: positiveInt(raw.bucket_reso_steps, 64),
    bucket_no_upscale: Boolean(raw.bucket_no_upscale),
    validation_split: boundedNumber(raw.validation_split, 0, 0, 1),
    validation_split_num: nonnegativeInt(raw.validation_split_num, 0),
    validation_seed: nonnegativeInt(raw.validation_seed, 42),
    caption_extension: String(raw.caption_extension || '.txt').trim() || '.txt',
    prefer_json_caption: Boolean(raw.prefer_json_caption),
    caption_source_mode: captionModeOr(raw.caption_source_mode),
  };
}

function normalizeNlTagMix(raw: unknown): DatasetFormValues['datasets'][number]['nl_tag_mix'] {
  const value = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  const ratio = Number(value.tag_ratio ?? 0.7);
  return {
    ...value,
    enabled: Boolean(value.enabled),
    tag_ratio: Math.max(0, Math.min(1, Number.isFinite(ratio) ? (ratio > 1 ? ratio / 100 : ratio) : 0.7)),
  };
}

function normalizeTriggerClone(raw: unknown): DatasetFormValues['datasets'][number]['trigger_clone'] {
  const value = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    ...value,
    enabled: Boolean(value.enabled),
    prompt: String(value.prompt || '').trim(),
    num_repeats: positiveInt(value.num_repeats, 1),
  };
}

function validateBucketSettings(
  settings: DatasetSettingsValues,
  path: Array<string | number>,
  context: z.RefinementCtx,
) {
  if (settings.min_bucket_reso > settings.max_bucket_reso) {
    context.addIssue({
      code: 'custom',
      path: [...path, 'min_bucket_reso'],
      message: '最小桶尺寸不能大于最大桶尺寸',
    });
  }
  if (settings.max_bucket_reso < settings.resolution) {
    context.addIssue({
      code: 'custom',
      path: [...path, 'max_bucket_reso'],
      message: '最大桶尺寸不能小于训练分辨率',
    });
  }
}

function captionModeOr(value: unknown): typeof captionSourceModes[number] {
  return captionSourceModes.includes(value as typeof captionSourceModes[number])
    ? value as typeof captionSourceModes[number]
    : 'auto';
}

function positiveInt(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 1 ? Math.trunc(number) : fallback;
}

function nonnegativeInt(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.trunc(number) : fallback;
}

function nonnegativeNumber(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(min, Math.min(max, number)) : fallback;
}

import { describe, expect, it } from 'vitest';

import {
  datasetFormFromPreset,
  datasetFormSchema,
  datasetWritePayload,
  emptyDatasetForm,
  emptyDatasetRow,
} from './datasetForm';

describe('dataset form domain', () => {
  it('hydrates every persisted defaults and subset field', () => {
    const form = datasetFormFromPreset({
      ok: true,
      file: 'configs/datasets/advanced.toml',
      name: 'advanced',
      content: '',
      readonly: false,
      summary: {},
      defaults: {
        resolution: 768,
        batch_size: 2,
        min_bucket_reso: 320,
        max_bucket_reso: 960,
        validation_split: 0.1,
        keep_tokens: 5,
        caption_source_mode: 'json',
      },
      datasets: [{
        source_dir: 'image_dataset/advanced',
        image_dir: 'post_image_dataset/advanced',
        cache_dir: 'post_image_dataset/advanced_cache',
        num_repeats: 4,
        is_reg: false,
        recursive: false,
        path_pattern: 'character_*',
        settings: { validation_seed: 99, bucket_no_upscale: true },
        nl_tag_mix: { enabled: true, tag_ratio: 70 },
        trigger_clone: { enabled: true, prompt: 'token', num_repeats: 3 },
      }],
    });

    expect(form.defaults).toMatchObject({
      resolution: 768,
      batch_size: 2,
      min_bucket_reso: 320,
      max_bucket_reso: 960,
      validation_split: 0.1,
      keep_tokens: 5,
      caption_source_mode: 'json',
    });
    expect(form.datasets[0]).toMatchObject({
      recursive: false,
      path_pattern: 'character_*',
      settings: { resolution: 768, validation_seed: 99, bucket_no_upscale: true },
      nl_tag_mix: { enabled: true, tag_ratio: 0.7 },
      trigger_clone: { enabled: true, prompt: 'token', num_repeats: 3 },
    });
  });

  it('rejects invalid bucket ranges and enabled trigger clones without prompts', () => {
    const form = emptyDatasetForm();
    form.datasets[0].source_dir = 'image_dataset/source';
    form.datasets[0].settings.min_bucket_reso = 1024;
    form.datasets[0].settings.max_bucket_reso = 512;
    form.datasets[0].trigger_clone.enabled = true;

    const result = datasetFormSchema.safeParse(form);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map((issue) => issue.path.join('.'))).toEqual(expect.arrayContaining([
        'datasets.0.settings.min_bucket_reso',
        'datasets.0.settings.max_bucket_reso',
        'datasets.0.trigger_clone.prompt',
      ]));
    }
  });

  it('rejects an empty dataset list and a form whose only dataset is regularization data', () => {
    const empty = emptyDatasetForm();
    empty.datasets = [];
    const emptyResult = datasetFormSchema.safeParse(empty);
    expect(emptyResult.success).toBe(false);
    if (!emptyResult.success) {
      expect(emptyResult.error.issues.map((issue) => issue.path.join('.'))).toContain('datasets');
    }

    const regularizationOnly = emptyDatasetForm();
    regularizationOnly.datasets[0].source_dir = 'image_dataset/reg';
    regularizationOnly.datasets[0].is_reg = true;
    const regularizationResult = datasetFormSchema.safeParse(regularizationOnly);
    expect(regularizationResult.success).toBe(false);
    if (!regularizationResult.success) {
      expect(regularizationResult.error.issues).toContainEqual(expect.objectContaining({
        path: ['datasets'],
        message: '至少保留一个普通训练数据子集',
      }));
    }
  });

  it('inherits defaults when creating a new subset', () => {
    const row = emptyDatasetRow({ resolution: 640, validation_seed: 7, caption_source_mode: 'txt' });

    expect(row.settings).toMatchObject({ resolution: 640, validation_seed: 7, caption_source_mode: 'txt' });
    expect(row).toMatchObject({ recursive: true, path_pattern: '*' });
  });

  it('rejects regularization datasets with stage scheduling', () => {
    const form = emptyDatasetForm();
    form.datasets[0].source_dir = 'image_dataset/train';
    form.datasets.push({
      ...emptyDatasetRow(form.defaults),
      source_dir: 'image_dataset/reg',
      is_reg: true,
    });
    form.stage_schedule_enabled = true;
    form.stage_schedule = [{ name: 'train', subset_index: 0, start_pct: 0, end_pct: 1 }];

    const result = datasetFormSchema.safeParse(form);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map((issue) => issue.path.join('.'))).toContain('stage_schedule_enabled');
    }
  });

  it('requires one before and one after row for every named edit pair', () => {
    const form = emptyDatasetForm();
    form.datasets = [
      { ...emptyDatasetRow(), source_dir: 'refs/1', edit_role: 'before', edit_pair_id: '1' },
      { ...emptyDatasetRow(), source_dir: 'targets/1', edit_role: 'after', edit_pair_id: '1' },
      { ...emptyDatasetRow(), source_dir: 'refs/2', edit_role: 'before', edit_pair_id: '2' },
      { ...emptyDatasetRow(), source_dir: 'targets/2', edit_role: 'after', edit_pair_id: '2' },
    ];
    expect(datasetFormSchema.safeParse(form).success).toBe(true);
    expect(datasetWritePayload(form).defaults.qwen_edit_enabled).toBe(true);
    form.datasets[3].edit_pair_id = '1';
    expect(datasetFormSchema.safeParse(form).success).toBe(false);
  });

  it('migrates legacy reference directories into editable before/after rows', () => {
    const form = datasetFormFromPreset({
      ok: true, file: 'configs/datasets/legacy.toml', name: 'legacy', content: '', readonly: false,
      summary: {}, defaults: { qwen_edit_enabled: true },
      datasets: [{ source_dir: 'original', image_dir: 'target', reference_image_dir: 'reference' }],
    });
    expect(form.datasets.map((row) => [row.edit_role, row.edit_pair_id, row.source_dir])).toEqual([
      ['before', '1', 'reference'], ['after', '1', 'original'],
    ]);
    expect(datasetWritePayload(form).datasets.every((row) => row.reference_image_dir === '')).toBe(true);
  });
});

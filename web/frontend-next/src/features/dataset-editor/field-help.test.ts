import { describe, expect, it } from 'vitest';

import { captionSourceModes } from './datasetForm';
import { datasetMaskModes } from './datasetMask';
import { DATASET_FIELD_HELP } from './field-help';

describe('dataset field help', () => {
  it('covers the dataset editor fields that need user guidance', () => {
    const required = [
      'resolution', 'batch_size', 'prior_loss_weight', 'enable_bucket', 'min_bucket_reso',
      'max_bucket_reso', 'bucket_reso_steps', 'bucket_no_upscale', 'validation_split',
      'validation_split_num', 'validation_seed', 'caption_extension', 'keep_tokens', 'prefer_json_caption',
      'caption_source_mode', 'source_dir', 'image_dir', 'cache_dir', 'num_repeats',
      'recursive', 'path_pattern', 'is_reg', 'edit_role', 'edit_pair_id', 'mask_mode', 'mask_dir',
      'nl_tag_mix_enabled', 'tag_ratio', 'trigger_clone_enabled', 'trigger_clone_prompt',
      'trigger_clone_num_repeats', 'stage_schedule_enabled', 'stage_name', 'stage_subset_index', 'stage_start_pct', 'stage_end_pct',
    ] as const;

    required.forEach((key) => {
      expect(DATASET_FIELD_HELP[key].summary).toBeTruthy();
      expect(DATASET_FIELD_HELP[key].recommendation).toBeTruthy();
    });
    expect(Object.keys(DATASET_FIELD_HELP).sort()).toEqual([...required].sort());
  });

  it('documents the same enum values accepted by the dataset form', () => {
    expect(DATASET_FIELD_HELP.caption_source_mode.options.map((option) => option.value)).toEqual(captionSourceModes);
    expect(DATASET_FIELD_HELP.mask_mode.options.map((option) => option.value)).toEqual(datasetMaskModes);
    expect(DATASET_FIELD_HELP.edit_role.options.map((option) => option.value)).toEqual(['normal', 'before', 'after']);
  });
});

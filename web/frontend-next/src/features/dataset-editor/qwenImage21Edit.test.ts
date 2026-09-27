import { describe, expect, it } from 'vitest';

import { emptyDatasetRow } from './datasetForm';
import { isQwenImage21Config, qwenImage21DatasetApplyIssue, qwenImage21EditApplyIssue, qwenImage21PlainLoraIssue } from './qwenImage21Edit';

const plainConfig = {
  model_family: 'qwen_image_2_1',
  network_module: 'networks.lora_anima',
  use_moe_style: false,
  route_per_layer: false,
  router_source: 'none',
  cache_latents: true,
  cache_text_encoder_outputs: true,
  mixed_precision: 'bf16',
  base_compute: 'bf16',
};

describe('Qwen Image 2.1 edit training gates', () => {
  it('recognizes the Qwen family and accepts a plain LoRA configuration', () => {
    expect(isQwenImage21Config(plainConfig)).toBe(true);
    expect(qwenImage21PlainLoraIssue(plainConfig)).toBeNull();
    expect(qwenImage21EditApplyIssue(plainConfig)).toBeNull();
  });

  it.each([
    [{ ...plainConfig, model_family: 'anima' }, '需要 Qwen Image 2.1'],
    [{ ...plainConfig, network_module: 'networks.methods.easycontrol' }, 'plain LoRA'],
    [{ ...plainConfig, use_moe_style: 'shared_A' }, 'plain LoRA'],
    [{ ...plainConfig, use_lokr: true }, 'plain LoRA'],
  ])('rejects incompatible edit adapters and model families', (config, expected) => {
    expect(qwenImage21EditApplyIssue(config)).toContain(expected);
  });

  it('requires both prompt-condition and latent caches plus BF16', () => {
    expect(qwenImage21EditApplyIssue({ ...plainConfig, cache_latents: false })).toContain('latent 缓存');
    expect(qwenImage21EditApplyIssue({ ...plainConfig, cache_text_encoder_outputs: false })).toContain('条件缓存');
    expect(qwenImage21EditApplyIssue({ ...plainConfig, mixed_precision: 'fp16' })).toContain('BF16');
  });

  it('blocks T2I apply while an A reference directory remains', () => {
    const row = { ...emptyDatasetRow(), reference_image_dir: 'references' };
    expect(qwenImage21DatasetApplyIssue(plainConfig, false, [row])).toContain('仍包含参考图目录');
  });

  it('blocks augmented edit datasets before Apply', () => {
    const before = { ...emptyDatasetRow(), edit_role: 'before' as const, edit_pair_id: '1', source_dir: 'references' };
    const after = {
      ...emptyDatasetRow(), edit_role: 'after' as const, edit_pair_id: '1', source_dir: 'targets',
      preserved_subset_fields: { color_aug: true },
    };
    expect(qwenImage21DatasetApplyIssue(plainConfig, true, [before, after])).toContain('增强');
  });
});

import type { DatasetFormValues } from './datasetForm';

type TrainingConfig = Record<string, unknown> | undefined;

const unsupportedAdapterFlags = [
  'use_ip_adapter',
  'use_easycontrol',
  'use_byg',
  'use_lokr',
  'use_loha',
  'use_glora',
  'use_vera',
  'use_ortho',
  'use_chimera_hydra',
  'use_timestep_mask',
  'add_reft',
  'train_llm_adapter',
];

export function isQwenImage21Config(config: TrainingConfig) {
  return String(config?.model_family || '').trim().toLowerCase() === 'qwen_image_2_1';
}

export function qwenImage21PlainLoraIssue(config: TrainingConfig): string | null {
  if (!config) return '训练配置尚未读取完成';
  if (!isQwenImage21Config(config)) return '需要 Qwen Image 2.1 训练配置';

  const networkModule = String(config.network_module || '').trim();
  const moeStyle = String(config.use_moe_style ?? 'false').trim().toLowerCase();
  const routerSource = String(config.router_source ?? 'none').trim().toLowerCase();
  const hasVariant = unsupportedAdapterFlags.some((key) => isEnabled(config[key]));
  const hasAdvancedRouting = !['', '0', 'false', 'none', 'off'].includes(moeStyle)
    || isEnabled(config.route_per_layer)
    || !['', 'none'].includes(routerSource)
    || Number(config.dora_wd || 0) > 0
    || Number(config.step_expert_K || 0) > 1
    || Number(config.functional_loss_weight || 0) > 0;

  if ((networkModule && networkModule !== 'networks.lora_anima') || hasVariant || hasAdvancedRouting) {
    return 'Qwen Image 2.1 Edit 仅支持 plain LoRA';
  }
  return null;
}

export function qwenImage21EditApplyIssue(config: TrainingConfig): string | null {
  const adapterIssue = qwenImage21PlainLoraIssue(config);
  if (adapterIssue) return adapterIssue;
  if (!isEnabled(config?.cache_latents ?? config?.use_vae_cache)) {
    return '编辑 LoRA 需要启用 VAE latent 缓存';
  }
  if (!isEnabled(config?.cache_text_encoder_outputs ?? config?.use_text_cache)) {
    return '编辑 LoRA 需要启用 Qwen3-VL 条件缓存';
  }
  if (String(config?.mixed_precision || 'bf16').toLowerCase() !== 'bf16') {
    return 'Qwen Image 2.1 需要 BF16 混合精度';
  }
  if (String(config?.base_compute || 'bf16').toLowerCase() !== 'bf16') {
    return 'Qwen Image 2.1 需要 BF16 base compute';
  }
  return null;
}

export function qwenImage21DatasetApplyIssue(
  config: TrainingConfig,
  editEnabled: boolean,
  rows: DatasetFormValues['datasets'],
): string | null {
  if (!isQwenImage21Config(config)) return null;
  if (!editEnabled) {
    return rows.some((row) => row.reference_image_dir.trim())
      ? '数据集仍包含参考图目录（编辑前）；请启用编辑 LoRA 或移除该目录'
      : null;
  }
  if (rows.some((row) => row.edit_role === 'normal')) return '编辑 LoRA 不能混用普通数据子集';
  if (rows.some((row) => row.is_reg)) return '编辑 LoRA 不支持正则数据子集';
  const pairs = new Map<string, { before: number; after: number }>();
  for (const row of rows) {
    const id = row.edit_pair_id.trim();
    if (!id) return '编辑子集需要配对名称';
    const pair = pairs.get(id) || { before: 0, after: 0 };
    pair[row.edit_role === 'before' ? 'before' : 'after'] += 1;
    pairs.set(id, pair);
  }
  if ([...pairs.values()].some((pair) => pair.before !== 1 || pair.after !== 1)) {
    return '每组配对必须各有一个编辑前和编辑后子集';
  }
  if (rows.some((row) => row.edit_role === 'before' && !row.source_dir.trim())) {
    return '编辑前图片目录不能为空';
  }
  if (rows.some((row) => row.edit_role === 'after' && row.settings.batch_size !== 1)) {
    return '编辑 LoRA 首版要求所有数据子集 batch_size=1';
  }
  for (const row of rows) {
    if (row.edit_role === 'before') continue;
    const preserved = row.preserved_subset_fields;
    const extra = preserved && typeof preserved === 'object'
      ? preserved as Record<string, unknown>
      : {};
    if (['flip_aug', 'color_aug', 'random_crop'].some((key) => isEnabled(row[key] ?? extra[key]))) {
      return '编辑 LoRA 不支持翻转、颜色或随机裁剪增强';
    }
  }
  return null;
}

function isEnabled(value: unknown) {
  return value === true || ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase());
}

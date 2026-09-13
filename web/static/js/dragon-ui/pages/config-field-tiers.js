export const CONFIG_VISIBILITY_OPTIONS = Object.freeze([
    // Keep the legacy ids so existing localStorage preferences remain valid.
    { id: 'newcomer', label: '精简' },
    { id: 'beginner', label: '默认' },
    { id: 'all', label: '全部适用' },
]);

const VISIBILITY_LEVELS = new Set(CONFIG_VISIBILITY_OPTIONS.map((option) => option.id));
const VISIBILITY_RANK = Object.freeze({ newcomer: 0, beginner: 1, all: 2 });
const FIELD_LEVEL_RANK = Object.freeze({ newcomer: 0, beginner: 1, advanced: 2 });

const COMPACT_FIELD_KEYS = new Set([
    'pretrained_model_name_or_path',
    'qwen3',
    'vae',
    'dataset_config',
    'output_name',
    'max_train_epochs',
    'max_train_steps',
    'learning_rate',
    'save_every_n_epochs',
    'save_last_n_epochs',
    'network_train_unet_only',
    'network_dim',
    'network_alpha',
    'lora_adapter_kind',
    'optimizer_type',
    'lr_scheduler',
    'timestep_sampling',
    'discrete_flow_shift',
    'train_batch_size',
    'gradient_accumulation_steps',
    'sample_ratio',
    'checkpointing_epochs',
    'checkpointing_last_n_epochs',
    'dora_wd',
    'lokr_factor',
    'network_weights',
    'dim_from_weights',
    'log_every_n_steps',
    'logging_dir',
    'log_with',
    'use_shuffled_caption_variants',
    'masked_loss',
    'caption_dropout_rate',
    'sample_prompts',
    'sample_every_n_epochs',
    'sample_every_n_steps',
    'sample_at_first',
    'sample_sampler',
    'seed',
    'blocks_to_swap',
    'auto_block_swap',
    'auto_block_swap_mode',
    'auto_block_swap_vram_reserve_percent',
    'auto_block_swap_preference',
    'block_swap_transfer_dtype',
    'block_swap_restore_mode',
    'selective_checkpoint',
    'base_compute',
    'gradient_checkpointing',
    'precision_preference',
    'lr_warmup_steps',
    'attn_mode',
    'max_data_loader_n_workers',
    'vae_chunk_size',
    'dataloader_pin_memory',
    'persistent_data_loader_workers',
    'use_vae_cache',
    'use_text_cache',
    'force_rebuild_preprocess_cache',
    'save_model_as',
    'save_precision',
    'weight_decay',
    'use_cmmd',
]);

const DEFAULT_EXTRA_FIELD_KEYS = new Set([
    'selective_checkpoint_blocks',
    'preprocess_precision_preference',
    'torch_compile',
    'compile_block_scope',
    'compile_inductor_mode',
    'debug_finite_checks',
    'sigmoid_scale',
    'sigmoid_bias',
    'weighting_scheme',
    'min_snr_gamma',
    'p2_gamma',
    'p2_k',
    'velocity_direction_loss_weight',
    'pipeline_parallel',
    'pipeline_parallel_stages',
    'pipeline_parallel_microbatches',
    'pipeline_parallel_schedule',
    'pipeline_parallel_split',
    'block_swap_profile_jsonl',
    'memory_probe_jsonl',
    'memory_probe_max_steps',
    'peak_probe_jsonl',
    'peak_probe_max_steps',
    'peak_probe_level',
    'gradient_flow_probe_jsonl',
    'gradient_flow_probe_every_n_steps',
    'gradient_flow_probe_dense_steps',
    'preprocess_vae_cache_batch_size',
    'preprocess_text_cache_batch_size',
    'preprocess_memory_profile',
    'reuse_dataset_cache_copy',
    'reuse_vae_latents',
    'reuse_text_encoder_cache',
    'cache_fingerprint_mode',
    'unsloth_offload_checkpointing',
    'disable_block_swap_for_eval',
    'use_custom_down_autograd',
]);

export function normalizeConfigVisibilityLevel(level, fallback = 'beginner') {
    const safeFallback = VISIBILITY_LEVELS.has(fallback) ? fallback : 'beginner';
    return VISIBILITY_LEVELS.has(level) ? level : safeFallback;
}

export function configFieldVisibilityLevel(key) {
    if (COMPACT_FIELD_KEYS.has(key)) return 'newcomer';
    if (DEFAULT_EXTRA_FIELD_KEYS.has(key)) return 'beginner';
    return 'advanced';
}

export function configFieldIsAdvanced(key) {
    return configFieldVisibilityLevel(key) === 'advanced';
}

export function configFieldVisibleAtLevel(fieldLevel, visibilityLevel = 'beginner') {
    const selectedRank = VISIBILITY_RANK[normalizeConfigVisibilityLevel(visibilityLevel)];
    const fieldRank = FIELD_LEVEL_RANK[fieldLevel] ?? FIELD_LEVEL_RANK.advanced;
    return fieldRank <= selectedRank;
}

export function configVisibilityLabel(level) {
    const normalized = normalizeConfigVisibilityLevel(level);
    return CONFIG_VISIBILITY_OPTIONS.find((option) => option.id === normalized)?.label
        || CONFIG_VISIBILITY_OPTIONS.at(-1).label;
}

import { FORM_SECTION_DEFS } from '../../config/catalog/form-layout.js?v=dragon-ui-20260902-lokr-backend-v4';

export const CONFIG_KEY_OVERRIDES = {
    'data-behavior': ['use_shuffled_caption_variants', 'masked_loss', 'caption_dropout_rate', 'path_pattern', 'drop_lowres_images', 'min_pixels'],
    'dataset-filter': ['path_pattern', 'drop_lowres_images', 'min_pixels'],
    optimizer: ['max_train_epochs', 'max_train_steps', 'train_batch_size', 'gradient_accumulation_steps', 'sample_ratio', 'optimizer_type', 'optimizer_args', 'lr_scheduler', 'lr_warmup_steps', 'learning_rate', 'timestep_sampling', 'discrete_flow_shift'],
    timestep: ['timestep_sampling', 'discrete_flow_shift'],
    logging: ['log_every_n_steps', 'logging_dir', 'log_with'],
    'block-swap': ['blocks_to_swap', 'block_swap_transfer_dtype', 'block_swap_restore_mode', 'block_swap_profile_jsonl', 'disable_block_swap_for_eval'],
    'gradient-checkpoint': ['gradient_checkpointing', 'selective_checkpoint', 'selective_checkpoint_blocks', 'unsloth_offload_checkpointing'],
    compile: ['torch_compile', 'compile_block_scope', 'compile_inductor_mode', 'compile_dynamic_seq', 'use_custom_down_autograd', 'debug_finite_checks'],
    'attention-backend': ['attn_mode', 'v100_flash_stability'],
    precision: ['precision_preference', 'base_compute'],
    convrot: ['convrot_group_size', 'convrot_scope', 'convrot_hadamard', 'convrot_min_in_features', 'convrot_largest_in_features_only', 'convrot_large_layer_mode', 'convrot_large_min_in_features'],
    'memory-probe': ['memory_probe_jsonl', 'memory_probe_max_steps', 'peak_probe_jsonl', 'peak_probe_max_steps', 'peak_probe_level'],
    'preprocess-batch': ['preprocess_vae_cache_batch_size', 'preprocess_text_cache_batch_size', 'preprocess_memory_profile', 'preprocess_precision_preference'],
    'cache-reuse': ['use_vae_cache', 'use_text_cache', 'cache_llm_adapter_outputs', 'ip_features_cache_to_disk', 'skip_cache_check', 'reuse_dataset_cache_copy', 'reuse_vae_latents', 'reuse_text_encoder_cache', 'cache_fingerprint_mode', 'force_rebuild_preprocess_cache'],
    'vae-resource': ['vae_chunk_size', 'vae_disable_cache'],
    reft: ['add_reft', 'reft_dim', 'reft_alpha', 'reft_layers'],
    'moe-routing': ['use_moe_style', 'route_per_layer', 'router_source', 'num_experts', 'balance_loss_weight', 'balance_loss_warmup_ratio', 'network_router_lr_scale', 'router_targets', 'sigma_feature_dim', 'per_bucket_balance_weight', 'num_sigma_buckets', 'specialize_experts_by_sigma_buckets', 'sigma_bucket_boundaries', 'router_hidden_dim', 'router_tau'],
    fei: ['fei_feature_dim', 'fei_sigma_low_div', 'fera_fecl_weight', 'fera_num_bands'],
    'chimera-hydra': ['use_chimera_hydra', 'num_experts_content', 'num_experts_freq', 'balance_w_content', 'balance_w_freq', 'network_content_router_lr_scale', 'network_freq_router_lr_scale', 'content_router_source', 'content_router_init_std', 'content_router_layer_norm', 'freq_router_init_std', 'freq_router_layer_norm'],
    'output-format': ['save_model_as', 'save_precision', 'weight_decay', 'use_cmmd', 'ip_diagnostics_epochs'],
    'snr-weighting': ['sigmoid_scale', 'sigmoid_bias', 'weighting_scheme', 'min_snr_gamma', 'p2_gamma', 'p2_k', 'velocity_direction_loss_weight'],
    'no-dataset-reg': ['prior_preservation_weight', 'blank_prompt_preservation', 'diff_output_preservation_trigger', 'diff_output_preservation_class', 'inverted_mask_prior_weight'],
    'soft-tokens': ['n_layers', 'n_t_buckets', 'init_std', 'splice_position', 'contrastive_weight', 'contrastive_k', 'contrastive_every_n', 'contrastive_negative_mode', 'contrastive_objective', 'contrastive_jaccard_alpha', 'contrastive_tau', 'contrastive_warmup_ratio', 'softrank_softness', 'softrank_method', 'dual_bank'],
    spd: ['dit_path', 'data_dir', 'iterations', 'seed', 'channel_scaling_alpha'],
    'data-loading': ['max_data_loader_n_workers', 'dataloader_pin_memory', 'persistent_data_loader_workers'],
    'adapter-basics': ['network_dim', 'network_alpha', 'lora_adapter_kind', 'dora_wd', 'lokr_factor', 'network_weights', 'dim_from_weights', 'vera_projection_prng_key', 'vera_d_initial', 'vera_save_projection', 'lokr_use_einsum', 'lokr_decompose_w2', 'lokr_full_factor', 'lokr_allow_legacy_dim', 'lokr_factor_group_size', 'lokr_project_chunk_bytes', 'lokr_grouped_delta_backend', 'lokr_grouped_delta_backward_backend'],
    lokr: ['lokr_factor', 'lokr_use_einsum', 'lokr_decompose_w2', 'lokr_full_factor', 'lokr_allow_legacy_dim', 'lokr_factor_group_size', 'lokr_project_chunk_bytes', 'lokr_grouped_delta_backend', 'lokr_grouped_delta_backward_backend'],
    'output-save': ['output_name', 'save_every_n_epochs', 'save_last_n_epochs', 'checkpointing_epochs', 'checkpointing_last_n_epochs', 'network_train_unet_only', 'save_model_as', 'save_precision', 'weight_decay', 'use_cmmd', 'ip_diagnostics_epochs', 'log_every_n_steps', 'logging_dir', 'log_with', 'sample_prompts', 'sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first', 'sample_sampler', 'seed'],
    'lora-basics': ['network_module', 'network_args', 'use_ortho', 'use_timestep_mask', 'min_rank', 'alpha_rank_scale', 'channel_scaling_alpha', 'layer_start'],
    'ip-adapter': ['use_ip_adapter', 'ip_image_drop_p', 'validation_baselines', 'ip_pair_mode', 'ip_pair_prob', 'ip_pair_min_level', 'ip_pair_caption_strip_p', 'encoder', 'encoder_dim', 'resampler_layers', 'resampler_heads', 'ip_scale', 'gate_lr', 'pe_lora_enabled', 'pe_lora_rank', 'pe_lora_alpha', 'pe_lora_layer_from'],
    easycontrol: ['use_easycontrol', 'easycontrol_drop_p', 'easycontrol_cond_noise_max', 'b_cond_init', 'cond_scale', 'apply_ffn_lora', 'cond_token_count'],
};

function collectSectionKeys(subItem) {
    return (subItem.sections || []).flatMap((title) =>
        FORM_SECTION_DEFS.find((section) => section.title === title)?.keys || []
    );
}

export function keysForConfigSubItem(subItem) {
    return [...new Set(CONFIG_KEY_OVERRIDES[subItem.id] || collectSectionKeys(subItem))];
}

/** Family-specific values required when changing a training config's base model. */

const MODEL_FAMILY_FORM_DEFAULTS = Object.freeze({
    z_image: Object.freeze({
        mixed_precision: 'bf16',
        base_compute: 'bf16',
        attn_mode: 'torch',
        xformers: false,
        torch_compile: false,
        blocks_to_swap: 0,
        gradient_checkpointing: true,
        selective_checkpoint: 'off',
        cpu_offload_checkpointing: false,
        unsloth_offload_checkpointing: false,
        discrete_flow_shift: 6.0,
        timestep_sampling: 'uniform',
        weighting_scheme: 'none',
        caption_dropout_rate: 0.0,
        weighted_captions: false,
        masked_loss: false,
        cache_llm_adapter_outputs: false,
        use_shuffled_caption_variants: false,
    }),
});

export function modelFamilyFormDefaults(family) {
    const normalized = String(family || '').trim().toLowerCase();
    return Object.entries(MODEL_FAMILY_FORM_DEFAULTS[normalized] || {});
}

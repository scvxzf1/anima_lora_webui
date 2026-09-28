import { describe, expect, it } from "vitest";
import { FIELD_HELP_ZH } from "./field-help.js";
import { FIELD_HELP_ADVANCED_ZH } from "./field-help-advanced.js";
import { FIELD_HELP_SUMMARY_ZH } from "./field-help-summary.js";

const LEGACY_ADVANCED_FIELDS = [
  "path_pattern", "drop_lowres_images", "min_pixels", "use_cmmd",
  "use_chimera_hydra", "num_experts_content", "num_experts_freq",
  "balance_w_content", "balance_w_freq", "network_content_router_lr_scale",
  "network_freq_router_lr_scale", "freq_router_init_std", "freq_router_layer_norm",
  "lokr_grouped_delta_backend", "lokr_grouped_delta_backward_backend", "rank_dropout",
  "cache_fingerprint_mode", "force_rebuild_preprocess_cache", "reuse_dataset_cache_copy",
  "reuse_text_encoder_cache", "reuse_vae_latents", "max_data_loader_n_workers", "seed",
  "weight_decay", "ip_diagnostics_epochs", "contrastive_every_n", "contrastive_k",
  "contrastive_negative_mode", "contrastive_objective", "contrastive_tau", "contrastive_weight",
  "contrastive_warmup_ratio", "contrastive_jaccard_alpha", "softrank_method", "softrank_softness",
  "pe_lora_enabled", "pe_lora_rank", "pe_lora_alpha", "pe_lora_layer_from", "dual_bank",
  "init_std", "n_layers", "n_t_buckets", "splice_position", "timestep_mask_mode",
  "timestep_mask_at_inference", "apply_ffn_lora", "b_cond_init", "channel_scaling_alpha",
  "cond_scale", "cond_token_count", "encoder", "encoder_dim", "gate_lr", "ip_scale",
  "resampler_heads", "resampler_layers", "data_dir", "dit_path", "iterations",
];

describe("Next field help catalog", () => {
  it("keeps all detailed help summaries aligned with the summary catalog", () => {
    expect(Object.keys(FIELD_HELP_ZH).sort()).toEqual(Object.keys(FIELD_HELP_SUMMARY_ZH).sort());
    expect(Object.keys(FIELD_HELP_ZH)).toHaveLength(263);
    for (const [field, help] of Object.entries(FIELD_HELP_ZH)) {
      expect(FIELD_HELP_SUMMARY_ZH[field as keyof typeof FIELD_HELP_SUMMARY_ZH]).toBe(help.summary);
    }
  });

  it("includes the complete legacy advanced-field help list", () => {
    expect(Object.keys(FIELD_HELP_ADVANCED_ZH).sort()).toEqual([...LEGACY_ADVANCED_FIELDS].sort());
  });

  it("preserves the full structured help shape", () => {
    for (const help of Object.values(FIELD_HELP_ZH)) {
      expect(help).toEqual(expect.objectContaining({
        summary: expect.any(String),
        fill: expect.any(String),
        benefit: expect.any(Array),
        cost: expect.any(Array),
        risk: expect.any(Array),
      }));
      expect(typeof help.recommend === "string" || Array.isArray(help.recommend)).toBe(true);
    }
  });
});

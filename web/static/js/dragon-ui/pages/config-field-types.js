/* Shared field-type rules for the Dragon configuration editor.
 *
 * Merged configs are intentionally sparse: method-specific flags are often
 * absent from a plain LoRA file even though the editor still exposes them.
 * Keep those fields typed here so an omitted value never degrades into a
 * free-form text input.
 */
import { FORM_UI_DEFAULTS } from '../../config/catalog/defaults.js?v=dragon-ui-20260830v2';

const BOOLEAN_METHOD_DEFAULTS = Object.freeze({
    // Network flags omitted by plain/imported method files.
    use_ortho: false,
    use_timestep_mask: false,
    add_reft: false,
    use_ip_adapter: false,
    use_easycontrol: false,
    route_per_layer: false,
    specialize_experts_by_sigma_buckets: false,
    use_chimera_hydra: false,

    // Base/network flags that can also be absent in imported files.
    network_train_unet_only: true,
    unsloth_offload_checkpointing: false,
    disable_block_swap_for_eval: false,
    dataloader_pin_memory: true,
    persistent_data_loader_workers: true,
    vae_disable_cache: true,
    torch_compile: true,
    train_adaln: false,
    freq_router_layer_norm: true,
    content_router_layer_norm: true,
    train_llm_adapter: false,
    lora_fp32_compute: false,
    ortho_centered_gate: false,
    chimera_centered_gate: false,
});

const BOOLEAN_DEFAULTS = Object.freeze({
    ...Object.fromEntries(
        Object.entries(FORM_UI_DEFAULTS).filter(([, value]) => typeof value === 'boolean'),
    ),
    ...BOOLEAN_METHOD_DEFAULTS,
});

export const BOOLEAN_CONFIG_DEFAULTS = BOOLEAN_DEFAULTS;
export const BOOLEAN_CONFIG_KEYS = new Set(Object.keys(BOOLEAN_DEFAULTS));

function booleanLiteral(value) {
    if (typeof value === 'boolean') return value;
    if (value === 1 || value === 0) return value === 1;
    const text = String(value ?? '').trim().toLowerCase();
    if (['true', '1', 'yes', 'on'].includes(text)) return true;
    if (['false', '0', 'no', 'off', ''].includes(text)) return false;
    return null;
}

function isBooleanOptionList(options) {
    if (!Array.isArray(options) || options.length !== 2) return false;
    const values = options.map(booleanLiteral);
    return values.every((value) => value !== null) && new Set(values).size === 2;
}

export function isBooleanConfigField(key, value, options = null) {
    if (BOOLEAN_CONFIG_KEYS.has(key)) return true;
    if (isBooleanOptionList(options)) return true;
    return typeof value === 'boolean' && !options;
}

export function booleanDefaultForKey(key, fallback = false) {
    return Object.prototype.hasOwnProperty.call(BOOLEAN_DEFAULTS, key)
        ? BOOLEAN_DEFAULTS[key]
        : fallback;
}

export function normalizeBooleanConfigValue(key, value, fallback = booleanDefaultForKey(key)) {
    const parsed = booleanLiteral(value);
    return parsed === null ? Boolean(fallback) : parsed;
}

/**
 * History config chip helpers (training variant / precisions).
 * Shared by history detail overview; keep in sync with
 * web/services/training/history_config_chips.py for list filters.
 */

export function readConfigString(configText, key) {
    const escapedKey = String(key || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (!escapedKey) return undefined;
    const match = String(configText || '').match(new RegExp(`^\\s*${escapedKey}\\s*=\\s*([^\\n#]+)`, 'm'));
    if (!match) return undefined;
    const raw = String(match[1] || '').trim().replace(/^["']|["']$/g, '').trim();
    return raw || undefined;
}

export function readConfigBool(configText, key) {
    const raw = readConfigString(configText, key);
    if (raw === undefined) return false;
    return ['1', 'true', 'yes', 'on'].includes(String(raw).trim().toLowerCase());
}

export function formatHistoryTrainingPrecision(configText) {
    const preference = readConfigString(configText, 'precision_preference');
    if (preference) return preference.toLowerCase();
    const mixed = readConfigString(configText, 'mixed_precision');
    return mixed ? mixed.toLowerCase() : '-';
}

export function formatHistoryPreprocessPrecision(configText) {
    const value = readConfigString(configText, 'preprocess_precision_preference');
    return value ? value.toLowerCase() : '-';
}

export function formatHistoryBlockSwapPrecision(configText) {
    const value = readConfigString(configText, 'block_swap_transfer_dtype');
    return value ? value.toLowerCase() : '-';
}

export function formatHistoryBaseCompute(configText) {
    const value = readConfigString(configText, 'base_compute');
    return value ? value.toLowerCase() : '-';
}

export function formatHistoryPrecisionPreference(configText) {
    // WebUI virtual field: save path expands precision_preference into
    // mixed_precision (+ clears full_fp16/full_bf16) and drops the key.
    // Rebuild with the same rules as precisionPreferenceFromConfig().
    const explicit = readConfigString(configText, 'precision_preference');
    if (explicit) {
        const normalized = explicit.toLowerCase();
        if (normalized === 'fp16' || normalized === 'fp32' || normalized === 'bf16') {
            return normalized;
        }
    }
    const mixedPrecision = String(readConfigString(configText, 'mixed_precision') || '').trim().toLowerCase();
    if (mixedPrecision === 'no') return 'fp32';
    if (mixedPrecision === 'fp16' || readConfigBool(configText, 'full_fp16')) return 'fp16';
    if (mixedPrecision === 'bf16' || readConfigBool(configText, 'full_bf16')) return 'bf16';
    // No precision signals in the snapshot at all.
    if (!mixedPrecision && explicit === undefined) return '-';
    return 'bf16';
}

export function formatHistoryTrainingVariant(task, configText) {
    const text = String(configText || '');
    const hasSnapshot = Boolean(text.trim())
        && !/^\s*#\s*无配置快照/.test(text)
        && !text.includes('无法生成配置快照');
    const moduleName = String(readConfigString(text, 'network_module') || '').toLowerCase();
    const moeStyle = String(readConfigString(text, 'use_moe_style') || '').trim().toLowerCase();

    if (readConfigBool(text, 'use_chimera_hydra') || moduleName.includes('chimera')) {
        return 'chimera';
    }
    if (readConfigBool(text, 'use_ip_adapter') || moduleName.includes('ip_adapter')) {
        return 'ip_adapter';
    }
    if (readConfigBool(text, 'use_easycontrol') || moduleName.includes('easycontrol')) {
        return 'easycontrol';
    }
    if (moduleName.includes('soft_tokens')) {
        return 'soft_tokens';
    }
    if (readConfigBool(text, 'use_loha')) return 'loha';
    if (readConfigBool(text, 'use_lokr')) return 'lokr';
    if (readConfigBool(text, 'use_vera')) return 'vera';
    if (readConfigBool(text, 'use_glora')) return 'glora';
    if (readConfigBool(text, 'dora_wd') || readConfigBool(text, 'use_dora')) return 'dora';
    if (readConfigBool(text, 'add_reft')) return 'reft';
    if (moeStyle && !['', 'false', 'none', '0', 'off'].includes(moeStyle)) {
        return 'hydralora';
    }
    if (readConfigBool(text, 'use_timestep_mask')) return 'tlora';
    if (readConfigBool(text, 'use_ortho')) return 'ortholora';
    if (moduleName.includes('lora_anima')) return 'lora';
    if (hasSnapshot && !moduleName) return 'lora';

    const known = new Set([
        'lora',
        'lokr',
        'loha',
        'vera',
        'glora',
        'dora',
        'hydralora',
        'reft',
        'tlora',
        'ortholora',
        'chimera',
        'chimera_hydra',
        'soft_tokens',
        'ip_adapter',
        'easycontrol',
    ]);
    const variantKey = String(task?.variant || '').trim().toLowerCase();
    if (known.has(variantKey)) {
        return variantKey === 'chimera_hydra' ? 'chimera' : variantKey;
    }
    const compact = variantKey.replace(/-8gb$/, '');
    if (known.has(compact)) return compact;
    return '-';
}

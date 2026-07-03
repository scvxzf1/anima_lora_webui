export const IMAGE_TEST_DEFAULTS = Object.freeze({
    sampler: 'euler',
    attnMode: 'flash',
    runtimeDtype: 'bf16',
    textEncoderDtype: 'same',
    gpuIndex: '',
    width: 1024,
    height: 1024,
    inferSteps: 28,
    guidanceScale: 4.0,
    flowShift: 1.0,
    loraMultiplier: 1.0,
});

export const IMAGE_TEST_SAMPLER_OPTIONS = Object.freeze([
    { value: 'euler', label: 'Euler' },
    { value: 'er_sde', label: 'ER-SDE' },
    { value: 'lcm', label: 'LCM' },
]);

export const IMAGE_TEST_ATTN_MODE_OPTIONS = Object.freeze([
    { value: 'flash', label: 'flash' },
    { value: 'torch', label: 'torch' },
    { value: 'sageattn', label: 'sageattn' },
    { value: 'flex', label: 'flex' },
    { value: 'xformers', label: 'xformers' },
    { value: 'sdpa', label: 'sdpa' },
]);

export const IMAGE_TEST_RUNTIME_DTYPE_OPTIONS = Object.freeze([
    { value: 'bf16', label: 'bf16' },
    { value: 'fp16', label: 'fp16' },
    { value: 'fp32', label: 'fp32' },
]);

export const IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS = Object.freeze([
    { value: 'same', label: '跟随推理精度' },
    { value: 'bf16', label: 'bf16' },
    { value: 'fp16', label: 'fp16' },
    { value: 'fp32', label: 'fp32' },
]);

export const IMAGE_TEST_HISTORY_RANGE_OPTIONS = Object.freeze([
    { value: '7', label: '近 7 天', days: 7 },
    { value: '14', label: '近 14 天', days: 14 },
    { value: '30', label: '近 30 天', days: 30 },
    { value: 'all', label: '全部', days: null },
]);

export const IMAGE_TEST_SELECTIVE_LORA_MAIN_BLOCKS = Object.freeze(
    Array.from({ length: 28 }, (_, index) => `block_${index}`),
);

export const IMAGE_TEST_SELECTIVE_LORA_ADAPTER_BLOCKS = Object.freeze(
    Array.from({ length: 6 }, (_, index) => `llm_adapter_${index}`),
);

export const IMAGE_TEST_SELECTIVE_LORA_SPECIAL_BLOCKS = Object.freeze([
    'llm_adapter_io',
    'final_layer',
    't_embedder',
    'x_embedder',
    'other_weights',
]);

export const IMAGE_TEST_SELECTIVE_LORA_BLOCKS = Object.freeze([
    ...IMAGE_TEST_SELECTIVE_LORA_MAIN_BLOCKS,
    ...IMAGE_TEST_SELECTIVE_LORA_ADAPTER_BLOCKS,
    ...IMAGE_TEST_SELECTIVE_LORA_SPECIAL_BLOCKS,
]);
export const IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP = 0.05;
export const IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN = 0;
export const IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX = 2;

export const IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS = Object.freeze([
    {
        value: 'default',
        label: 'Default',
        blocks: IMAGE_TEST_SELECTIVE_LORA_BLOCKS,
        strength: 1.0,
    },
    {
        value: 'all_off',
        label: 'All Off',
        blocks: [],
        strength: 0.0,
    },
    {
        value: 'half_strength',
        label: 'Half Strength',
        blocks: IMAGE_TEST_SELECTIVE_LORA_BLOCKS,
        strength: 0.5,
    },
    {
        value: 'main_blocks_only',
        label: 'Main Blocks Only',
        blocks: [
            ...IMAGE_TEST_SELECTIVE_LORA_MAIN_BLOCKS,
            'final_layer',
            't_embedder',
            'x_embedder',
            'other_weights',
        ],
        strength: 1.0,
    },
    {
        value: 'llm_adapter_only',
        label: 'LLM Adapter Only',
        blocks: [
            ...IMAGE_TEST_SELECTIVE_LORA_ADAPTER_BLOCKS,
            'llm_adapter_io',
            'other_weights',
        ],
        strength: 1.0,
    },
    {
        value: 'late_main',
        label: 'Late Main (20-27)',
        blocks: [
            ...Array.from({ length: 8 }, (_, offset) => `block_${offset + 20}`),
            'final_layer',
            't_embedder',
            'x_embedder',
            'other_weights',
        ],
        strength: 1.0,
    },
    {
        value: 'mid_late_main',
        label: 'Mid-Late Main (14-27)',
        blocks: [
            ...Array.from({ length: 14 }, (_, offset) => `block_${offset + 14}`),
            'final_layer',
            't_embedder',
            'x_embedder',
            'other_weights',
        ],
        strength: 1.0,
    },
    {
        value: 'evens_only',
        label: 'Evens Only',
        blocks: [
            ...Array.from({ length: 14 }, (_, offset) => `block_${offset * 2}`),
            'llm_adapter_0',
            'llm_adapter_2',
            'llm_adapter_4',
        ],
        strength: 1.0,
    },
    {
        value: 'odds_only',
        label: 'Odds Only',
        blocks: [
            ...Array.from({ length: 14 }, (_, offset) => `block_${offset * 2 + 1}`),
            'llm_adapter_1',
            'llm_adapter_3',
            'llm_adapter_5',
        ],
        strength: 1.0,
    },
    {
        value: 'custom',
        label: 'Custom',
        blocks: IMAGE_TEST_SELECTIVE_LORA_BLOCKS,
        strength: 1.0,
    },
]);

export const IMAGE_TEST_SELECTIVE_LORA_GROUPS = Object.freeze([
    {
        key: 'main',
        label: 'Main Blocks',
        hint: 'DiT 主干 0-27',
        containerId: 'image-test-layer-blocks-main',
        blocks: IMAGE_TEST_SELECTIVE_LORA_MAIN_BLOCKS,
    },
    {
        key: 'adapter',
        label: 'LLM Adapter',
        hint: '跨模态适配层',
        containerId: 'image-test-layer-blocks-adapter',
        blocks: [...IMAGE_TEST_SELECTIVE_LORA_ADAPTER_BLOCKS, 'llm_adapter_io'],
    },
    {
        key: 'special',
        label: 'Special',
        hint: '输入、输出与剩余权重',
        containerId: 'image-test-layer-blocks-special',
        blocks: ['final_layer', 't_embedder', 'x_embedder', 'other_weights'],
    },
]);

export function normalizeImageTestSelectiveLoraPreset(value, fallback = 'default') {
    const normalized = String(value || '').trim().toLowerCase() || fallback;
    return IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.some((item) => item.value === normalized)
        ? normalized
        : fallback;
}

export function blocksForImageTestSelectiveLoraPreset(value) {
    const preset = IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.find(
        (item) => item.value === normalizeImageTestSelectiveLoraPreset(value),
    );
    return preset ? [...preset.blocks] : [...IMAGE_TEST_SELECTIVE_LORA_BLOCKS];
}

export function strengthForImageTestSelectiveLoraPreset(value) {
    const preset = IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.find(
        (item) => item.value === normalizeImageTestSelectiveLoraPreset(value),
    );
    return preset?.strength ?? 1.0;
}

export function clampImageTestSelectiveLoraStrength(value, fallback = 0) {
    const parsed = Number.parseFloat(String(value ?? '').trim());
    const resolved = Number.isFinite(parsed) ? parsed : fallback;
    const clamped = Math.min(
        IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX,
        Math.max(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN, resolved),
    );
    const stepped = Math.round(clamped / IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP) * IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP;
    return Number(stepped.toFixed(2));
}

export function blockStrengthsForImageTestSelectiveLoraPreset(value) {
    const strength = strengthForImageTestSelectiveLoraPreset(value);
    const enabled = new Set(blocksForImageTestSelectiveLoraPreset(value));
    return Object.fromEntries(
        IMAGE_TEST_SELECTIVE_LORA_BLOCKS.map((blockId) => [
            blockId,
            enabled.has(blockId) ? clampImageTestSelectiveLoraStrength(strength, strength) : 0,
        ]),
    );
}

export function normalizeImageTestSelectiveLoraBlockStrengths(values, preset = 'default') {
    if (!values || typeof values !== 'object' || Array.isArray(values)) {
        return blockStrengthsForImageTestSelectiveLoraPreset(preset);
    }
    return Object.fromEntries(
        IMAGE_TEST_SELECTIVE_LORA_BLOCKS.map((blockId) => [
            blockId,
            clampImageTestSelectiveLoraStrength(values[blockId], 0),
        ]),
    );
}

export function enabledBlocksForImageTestSelectiveLoraStrengths(values) {
    const normalized = normalizeImageTestSelectiveLoraBlockStrengths(values, 'all_off');
    return IMAGE_TEST_SELECTIVE_LORA_BLOCKS.filter((blockId) => normalized[blockId] > 0);
}

export function normalizeImageTestHistoryRange(value, fallback = '7') {
    const normalized = String(value || '').trim().toLowerCase() || fallback;
    return IMAGE_TEST_HISTORY_RANGE_OPTIONS.some((item) => item.value === normalized)
        ? normalized
        : fallback;
}

export function daysForImageTestHistoryRange(value) {
    const option = IMAGE_TEST_HISTORY_RANGE_OPTIONS.find(
        (item) => item.value === normalizeImageTestHistoryRange(value),
    );
    return option?.days ?? 7;
}

export function createImageTestState() {
    return {
        initialized: false,
        syncReady: false,
        hasPersistedDraft: false,
        loadingStatus: false,
        loadingWeights: false,
        loadingGpus: false,
        loadingImages: false,
        starting: false,
        stopping: false,
        imageRequestSeq: 0,
        pollTimer: null,
        configSnapshot: {},
        lastStatus: null,
        lastImagesPayload: null,
        lastWeightsPayload: null,
        lastGpusPayload: null,
        restoredDraftFieldIds: new Set(),
    };
}

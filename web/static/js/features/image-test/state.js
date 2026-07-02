export const IMAGE_TEST_DEFAULTS = Object.freeze({
    sampler: 'euler',
    attnMode: 'flash',
    runtimeDtype: 'bf16',
    textEncoderDtype: 'same',
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

export function createImageTestState() {
    return {
        initialized: false,
        syncReady: false,
        loadingStatus: false,
        loadingWeights: false,
        loadingImages: false,
        starting: false,
        stopping: false,
        imageRequestSeq: 0,
        pollTimer: null,
        configSnapshot: {},
        lastStatus: null,
        lastImagesPayload: null,
        lastWeightsPayload: null,
    };
}

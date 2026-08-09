export const FILE_GROUP_DROP_TARGET_ATTR = 'data-file-group-drop-target';

export const RESOURCE_QUICK_PRESETS = [
    {
        id: 'gpu_full',
        label: '快捷·全 GPU',
        note: '快捷资源：显存充足优先；最快，不做 block swap。',
        values: {
            blocks_to_swap: 0,
            block_swap_transfer_dtype: 'bf16',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'off',
            block_swap_profile_jsonl: 'off',
            memory_probe_jsonl: 'off',
            memory_probe_max_steps: 2,
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
    },
    {
        id: 'balanced_16g',
        label: '快捷·Balanced 16G',
        note: '快捷资源：推荐 16GB；约省 4GB，速度损失较低。不是硬件 preset 本身。',
        values: {
            blocks_to_swap: 12,
            block_swap_transfer_dtype: 'bf16',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'off',
            block_swap_profile_jsonl: 'off',
            memory_probe_jsonl: 'off',
            memory_probe_max_steps: 2,
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
    },
    {
        id: 'fp8_swap_test',
        label: '快捷·FP8 测试',
        note: '快捷资源：实验项；压缩 frozen base block 传输，只用于对照测试。',
        values: {
            blocks_to_swap: 12,
            block_swap_transfer_dtype: 'fp8_e4m3',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'off',
            block_swap_profile_jsonl: 'auto',
            memory_probe_jsonl: 'auto',
            memory_probe_max_steps: 2,
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
    },
    {
        id: 'vram_saver',
        label: '快捷·更省显存',
        note: '快捷资源：交换 16 块；更省显存，训练会更慢。',
        values: {
            blocks_to_swap: 16,
            block_swap_transfer_dtype: 'bf16',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'off',
            block_swap_profile_jsonl: 'off',
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
    },
    {
        id: 'lokr_16g_rescue',
        label: '快捷·LoKr 16G',
        note: '快捷资源：仅 LoKr / 方法变体专用；实测交换 23 块，分组 8 作为当前速度默认。',
        applicableMethods: ['lokr', 'use_lokr'],
        values: {
            blocks_to_swap: 23,
            block_swap_transfer_dtype: 'bf16',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'off',
            block_swap_profile_jsonl: 'off',
            memory_probe_jsonl: 'auto',
            memory_probe_max_steps: 3,
            lokr_factor_group_size: 8,
            lokr_project_chunk_bytes: 4194304,
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
    },
    {
        id: 'oom_fallback',
        label: '快捷·OOM 兜底',
        note: '快捷资源：仍然 OOM 时用；开启 mlp_only 重算。',
        values: {
            blocks_to_swap: 12,
            block_swap_transfer_dtype: 'bf16',
            block_swap_restore_mode: 'slab',
            selective_checkpoint: 'mlp_only',
            block_swap_profile_jsonl: 'off',
            gradient_checkpointing: false,
            unsloth_offload_checkpointing: false,
            torch_compile: true,
        },
        merge: {
            blocks_to_swap: 'max',
            selective_checkpoint: 'checkpoint_strength_max',
        },
    },
];

export const NO_DATASET_REGULARIZATION_FIELD_KEYS = [
    'prior_preservation_weight',
    'blank_prompt_preservation',
    'diff_output_preservation_trigger',
    'diff_output_preservation_class',
    'inverted_mask_prior_weight',
];

export const NO_DATASET_REGULARIZATION_MODE_SPECS = [
    {
        id: 'off',
        label: '关闭',
        note: '普通训练；不额外跑先验保留 forward。',
    },
    {
        id: 'blank',
        label: '空提示先验',
        note: '用 T5("") 做基线，适合先跑通无额外数据集的先验保留。',
    },
    {
        id: 'dop',
        label: 'DOP / class prompt',
        note: '用类提示做 base 预测，适合角色、物体或风格 LoRA。',
    },
    {
        id: 'mask',
        label: '反转遮罩保护',
        note: '只约束遮罩外区域，适合局部编辑或带 alpha mask 的训练。',
    },
];

export const NO_DATASET_REGULARIZATION_DEFAULT_WEIGHT = 0.1;
export const NO_DATASET_REGULARIZATION_CACHE_PATCH = {
    use_text_cache: true,
    cache_llm_adapter_outputs: true,
};
export const NO_DATASET_REGULARIZATION_CONFLICT_MODE = 'conflict';
export const NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY = '显示底层参数';
export const NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY_OPEN = '收起底层参数';
export const NO_DATASET_REGULARIZATION_CONFLICT_MESSAGE = '配置冲突：请只保留空提示、DOP/class prompt、反转遮罩保护中的一种。';
export const NO_DATASET_REGULARIZATION_DOP_CLASS_REQUIRED = 'DOP 类提示要填训练目标的泛化类别，而不是触发词。人物/角色填 woman / man / character；物体填 object / outfit / weapon；风格填 anime style / illustration style。';

export const NO_DATASET_REGULARIZATION_QUICK_PRESETS = [
    {
        id: 'prior_baseline',
        label: '先验基线',
        note: '最适合先跑通；开启文本缓存，只用空提示先验。',
        values: {
            prior_preservation_weight: 0.1,
            blank_prompt_preservation: true,
            diff_output_preservation_trigger: '',
            diff_output_preservation_class: '',
            inverted_mask_prior_weight: 0.0,
            use_text_cache: true,
            cache_llm_adapter_outputs: true,
        },
    },
    {
        id: 'dop_roles',
        label: 'DOP 角色',
        note: 'DOP/class prompt 方案；把类提示补上后再训练。',
        values: {
            prior_preservation_weight: 0.1,
            blank_prompt_preservation: false,
            diff_output_preservation_trigger: 'sks',
            diff_output_preservation_class: '',
            inverted_mask_prior_weight: 0.0,
            use_text_cache: true,
            cache_llm_adapter_outputs: true,
        },
    },
    {
        id: 'mask_guard',
        label: '遮罩保护',
        note: '局部编辑/遮罩训练时更稳；保留非目标区域。',
        values: {
            prior_preservation_weight: 0.0,
            blank_prompt_preservation: false,
            diff_output_preservation_trigger: '',
            diff_output_preservation_class: '',
            inverted_mask_prior_weight: 0.1,
            use_text_cache: true,
            cache_llm_adapter_outputs: true,
        },
    },
    {
        id: 'off',
        label: '关闭',
        note: '快速回到普通训练状态。',
        values: {
            prior_preservation_weight: 0.0,
            blank_prompt_preservation: false,
            diff_output_preservation_trigger: '',
            diff_output_preservation_class: '',
            inverted_mask_prior_weight: 0.0,
        },
    },
];

export const SELECTIVE_CHECKPOINT_STRENGTH = new Map([
    ['off', 0],
    ['peak_blocks_mlp_layer1', 1],
    ['mlp_layer1_only', 2],
    ['peak_blocks_mlp', 3],
    ['peak_blocks_adapter_aware', 4],
    ['mlp_only', 4],
    ['every_other', 5],
    ['adapter_aware', 6],
]);

export const HIDDEN_DATASET_PRESET_FILES = new Set([
    'configs/datasets/easycontrol.toml',
    'configs/datasets/ip_adapter.toml',
]);
export const DATASET_PRESET_REQUEST_TIMEOUT_MS = 15000;
export const DEFAULT_SAMPLE_PROMPTS_PATH = 'configs/sample_prompts.txt';
export const STAGE_RESOLUTION_STEPS_PER_EPOCH = 1000;
export const HISTORY_UNGROUPED_COLLECTION_KEY = 'collection:__ungrouped__';
export const HISTORY_TASK_DRAG_MIME = 'application/x-anima-history-task-ids';
export const HISTORY_COLLECTION_DRAG_MIME = 'application/x-anima-history-collection';
export const HISTORY_CONFIG_GROUP_DRAG_MIME = 'application/x-anima-history-config-group';
export const THEME_STORAGE_KEY = 'anima_lora_theme';
export const LANGUAGE_STORAGE_KEY = 'anima_lora_language';
export const GPU_WHITELIST_STORAGE_KEY = 'anima_lora_gpu_whitelist';
export const LOSS_WEIGHTING_DEPENDENT_FIELDS = new Map([
    ['min_snr_gamma', 'min_snr'],
    ['p2_gamma', 'p2'],
    ['p2_k', 'p2'],
]);

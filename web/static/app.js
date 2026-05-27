/**
 * Anima LoRA Web UI — 主应用逻辑
 */
(function () {
    'use strict';

    // ── 状态 ──
    let fieldHelp = {};
    let currentConfig = {};
    let ws = null;
    let lossChart = null;
    let stepCounter = 0;
    let tomlStatusTimer = null;
    let tomlFiles = [];
    let tomlFileGroups = [];
    let tomlFileMeta = {};
    let currentTomlFile = '';
    let tomlSavedContent = '';
    let tomlDeleteConfirmFile = '';
    let tomlDeleteConfirmTimer = null;
    let tomlSaveConfirmFile = '';
    let tomlSaveConfirmTimer = null;
    let tomlManagerMode = 'project';
    let configSwitchToastTimer = null;
    let sharedDialogBusy = false;
    let tomlGroupActionBusy = false;
    let configLoadSeq = 0;
    let datasetLoadSeq = 0;
    let stepEstimateSeq = 0;
    let samplePromptsLoadSeq = 0;
    let datasetPresetLoadSeq = 0;
    let datasetPreviewLoadSeq = 0;
    let configGroupHintSeq = 0;
    let choiceGuideHintSeq = 0;
    const selectionSnapshot = {
        method: '',
        variant: '',
        preset: '',
    };
    let previewSettings = null;
    let currentPreviewSource = 'training';
    let selectedPreviewTaskId = '';
    let selectedPreviewGroup = null;
    let previewRequestSeq = 0;
    let previewWeightRequestSeq = 0;
    let previewWeightSortDirection = 'asc';
    let currentStepEstimate = null;
    let datasetEditorState = {
        loading: false,
        loaded: false,
        dirty: false,
        dataset_config: '',
        datasets: [],
        defaults: {},
        error: '',
    };
    let datasetPresetState = {
        loading: false,
        dirty: false,
        isNew: false,
        selectedFile: '',
        presets: [],
        datasets: [],
        defaults: {},
        readonly: false,
        error: '',
        status: '',
    };
    const datasetPreviewState = {
        datasetIndex: 0,
        source: 'source',
        payload: null,
    };
    const HIDDEN_DATASET_PRESET_FILES = new Set([
        'configs/datasets/easycontrol.toml',
        'configs/datasets/ip_adapter.toml',
    ]);
    let selectedConfigDatasetFile = '';
    let selectedConfigDatasetSummary = null;
    let outputRunState = {
        loading: false,
        runs: [],
        selectedRun: '',
        selectedKind: 'original',
        search: '',
        content: '',
        file: '',
        outputRoot: '',
        error: '',
        saveAsOpen: false,
    };
    let configDatasetPickerSearch = '';
    let configDatasetPreviewRequestSeq = 0;
    let configDatasetPreviewState = {
        file: '',
        loading: false,
        payload: null,
        error: '',
    };
    let trainingSampleState = null;
    const DEFAULT_SAMPLE_PROMPTS_PATH = 'configs/sample_prompts.txt';
    let samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
    let samplePromptsContent = '';
    let samplePromptsMode = 'editor-inline';
    let viewingHistoryTaskId = '';
    let historyViewMode = 'live';
    let currentHistoryTaskForResume = null;
    let currentHistoryConfigGroup = null;
    let currentHistoryTimelineSelection = [];
    let resumeOptionsState = {
        loading: false,
        taskId: '',
        checkpoints: [],
        defaultCheckpoint: '',
        error: '',
        message: '',
    };
    let continueTrainingSource = null;
    let continueLoraDialogState = {
        loading: false,
        taskId: '',
        weights: [],
        error: '',
        message: '',
    };
    let trainingQueueState = {
        loading: false,
        paused: false,
        items: [],
        error: '',
        currentItemId: '',
    };
    let historyTasks = [];
    let showArchivedHistory = false;
    const THEME_STORAGE_KEY = 'anima_lora_theme';
    const GPU_WHITELIST_STORAGE_KEY = 'anima_lora_gpu_whitelist';
    let availableGpus = [];
    let selectedGpuWhitelist = [];
    let currentTrainingSource = {
        method: 'lora',
        methods_subdir: 'gui-methods',
        file: 'configs/gui-methods/lora.toml',
    };
    const BLANK_PRESET_TEMPLATE_FILE = 'configs/gui-methods/lora.toml';
    const BLANK_PRESET_TEMPLATE_LABEL = 'LoRA 标准训练变体 / lora.toml';
    const FORM_UI_DEFAULTS = {
        train_batch_size: 1,
        gradient_accumulation_steps: 1,
        sample_prompts: '',
        sample_every_n_epochs: '',
        sample_every_n_steps: '',
        gradient_checkpointing: true,
        max_train_epochs: '',
        max_train_steps: 0,
        sample_at_first: false,
        sample_sampler: 'ddim',
        use_lokr: false,
        lokr_factor: 8,
        max_data_loader_n_workers: 0,
        path_pattern: '*',
        drop_lowres_images: true,
        min_pixels: 500000,
        validation_baselines: false,
        ip_pair_mode: 'identity',
        ip_pair_prob: 0.8,
        ip_pair_min_level: 'artist',
        ip_pair_caption_strip_p: 0.0,
        content_router_source: 'crossattn_emb',
        content_router_init_std: 0.001,
        content_router_layer_norm: true,
        use_cmmd: false,
        ip_diagnostics_epochs: 999,
        weight_decay: 0.0,
        dit_path: 'models/diffusion_models/anima-base-v1.0.safetensors',
        data_dir: 'post_image_dataset/lora',
        iterations: 2000,
        seed: 42,
        use_chimera_hydra: false,
        channel_scaling_alpha: 0.5,
        num_experts_content: 4,
        num_experts_freq: 2,
        balance_w_content: 0.000002,
        balance_w_freq: 0.000005,
        network_content_router_lr_scale: 10.0,
        network_freq_router_lr_scale: 2.0,
        freq_router_init_std: 0.02,
        freq_router_layer_norm: true,
        n_layers: 14,
        n_t_buckets: 14,
        init_std: 0.02,
        splice_position: 'front_of_padding',
        contrastive_weight: 0.05,
        contrastive_k: 1,
        contrastive_every_n: 3,
        contrastive_negative_mode: 'hard',
        contrastive_objective: 'agsm',
        agsm_gamma: 0.5,
        agsm_ema_decay: 0.99,
        contrastive_jaccard_alpha: 1.0,
        contrastive_tau: 0.5,
        contrastive_warmup_ratio: 0.1,
        encoder: 'pe',
        encoder_dim: 1024,
        resampler_layers: 2,
        resampler_heads: 8,
        ip_scale: 1.0,
        gate_lr: 0.001,
        pe_lora_enabled: false,
        pe_lora_rank: 16,
        pe_lora_alpha: 16,
        pe_lora_layer_from: 8,
        b_cond_init: -10.0,
        cond_scale: 1.0,
        apply_ffn_lora: true,
        cond_token_count: 4096,
        resolution: 1024,
        batch_size: 1,
        enable_bucket: true,
        min_bucket_reso: 256,
        max_bucket_reso: 1024,
        bucket_reso_steps: 64,
        bucket_no_upscale: false,
        validation_split: 0.025,
        validation_split_num: 0,
        validation_seed: 42,
        caption_extension: '.txt',
        keep_tokens: 3,
    };
    const OPTIONAL_EMPTY_FIELDS = new Set([
        'sample_prompts',
        'sample_every_n_epochs',
        'sample_every_n_steps',
        'max_train_epochs',
    ]);
    const FORM_UI_PERSIST_DEFAULT_FIELDS = new Set([
        'gradient_checkpointing',
    ]);
    const CONFIG_FORM_INTERNAL_KEYS = new Set([
        'dataset_config_picker',
    ]);
    const DEPRECATED_CONFIG_FORM_FIELDS = new Set([
        'compile_mode',
        'static_pad',
        'static_token_count',
    ]);
    const DATASET_EDITOR_COMPAT_FIELDS = new Set([
        'source_image_dir',
        'resized_image_dir',
        'lora_cache_dir',
        'dataset_config',
    ]);
    const DATASET_BLUEPRINT_FIELDS = new Set([
        'dataset_config',
        'source_image_dir',
        'resized_image_dir',
        'lora_cache_dir',
        'resolution',
        'batch_size',
        'enable_bucket',
        'min_bucket_reso',
        'max_bucket_reso',
        'bucket_reso_steps',
        'bucket_no_upscale',
        'validation_split',
        'validation_split_num',
        'validation_seed',
        'caption_extension',
        'keep_tokens',
    ]);
    const DATASET_SETTING_KEYS = new Set([
        'resolution',
        'batch_size',
        'enable_bucket',
        'min_bucket_reso',
        'max_bucket_reso',
        'bucket_reso_steps',
        'bucket_no_upscale',
        'validation_split',
        'validation_split_num',
        'validation_seed',
    ]);
    const NETWORK_ARG_FIELD_SPECS = [
        { family: 'soft_tokens', key: 'n_layers', arg: 'n_layers', default: 14, valueType: 'integer' },
        { family: 'soft_tokens', key: 'n_t_buckets', arg: 'n_t_buckets', default: 14, valueType: 'integer' },
        { family: 'soft_tokens', key: 'init_std', arg: 'init_std', default: 0.02, valueType: 'number' },
        { family: 'soft_tokens', key: 'splice_position', arg: 'splice_position', default: 'front_of_padding', valueType: 'string' },
        { family: 'soft_tokens', key: 'contrastive_weight', arg: 'contrastive_weight', default: 0.05, valueType: 'number' },
        { family: 'soft_tokens', key: 'contrastive_k', arg: 'contrastive_k', default: 1, valueType: 'integer' },
        { family: 'soft_tokens', key: 'contrastive_every_n', arg: 'contrastive_every_n', default: 3, valueType: 'integer' },
        { family: 'soft_tokens', key: 'contrastive_negative_mode', arg: 'contrastive_negative_mode', default: 'hard', valueType: 'string' },
        { family: 'soft_tokens', key: 'contrastive_objective', arg: 'contrastive_objective', default: 'agsm', valueType: 'string' },
        { family: 'soft_tokens', key: 'agsm_gamma', arg: 'agsm_gamma', default: 0.5, valueType: 'number' },
        { family: 'soft_tokens', key: 'agsm_ema_decay', arg: 'agsm_ema_decay', default: 0.99, valueType: 'number' },
        { family: 'soft_tokens', key: 'contrastive_jaccard_alpha', arg: 'contrastive_jaccard_alpha', default: 1.0, valueType: 'number' },
        { family: 'soft_tokens', key: 'contrastive_tau', arg: 'contrastive_tau', default: 0.5, valueType: 'number' },
        { family: 'soft_tokens', key: 'contrastive_warmup_ratio', arg: 'contrastive_warmup_ratio', default: 0.1, valueType: 'number' },
        { family: 'ip_adapter', key: 'encoder', arg: 'encoder', default: 'pe', valueType: 'string' },
        { family: 'ip_adapter', key: 'encoder_dim', arg: 'encoder_dim', default: 1024, valueType: 'integer' },
        { family: 'ip_adapter', key: 'resampler_layers', arg: 'resampler_layers', default: 2, valueType: 'integer' },
        { family: 'ip_adapter', key: 'resampler_heads', arg: 'resampler_heads', default: 8, valueType: 'integer' },
        { family: 'ip_adapter', key: 'ip_scale', arg: 'ip_scale', default: 1.0, valueType: 'number' },
        { family: 'ip_adapter', key: 'gate_lr', arg: 'gate_lr', default: 0.001, valueType: 'number' },
        { family: 'ip_adapter', key: 'pe_lora_enabled', arg: 'pe_lora_enabled', default: false, valueType: 'boolean' },
        { family: 'ip_adapter', key: 'pe_lora_rank', arg: 'pe_lora_rank', default: 16, valueType: 'integer' },
        { family: 'ip_adapter', key: 'pe_lora_alpha', arg: 'pe_lora_alpha', default: 16, valueType: 'number' },
        { family: 'ip_adapter', key: 'pe_lora_layer_from', arg: 'pe_lora_layer_from', default: 8, valueType: 'integer' },
        { family: 'easycontrol', key: 'b_cond_init', arg: 'b_cond_init', default: -10.0, valueType: 'number' },
        { family: 'easycontrol', key: 'cond_scale', arg: 'cond_scale', default: 1.0, valueType: 'number' },
        { family: 'easycontrol', key: 'apply_ffn_lora', arg: 'apply_ffn_lora', default: true, valueType: 'booleanInt' },
        { family: 'easycontrol', key: 'cond_token_count', arg: 'cond_token_count', default: 4096, valueType: 'integer' },
    ];
    const NETWORK_ARG_FIELD_MAP = new Map(NETWORK_ARG_FIELD_SPECS.map((spec) => [spec.key, spec]));
    const NETWORK_ARG_SPEC_BY_ARG = new Map(NETWORK_ARG_FIELD_SPECS.map((spec) => [spec.arg, spec]));
    const SPD_UI_DEFAULT_FIELDS = new Set(['dit_path', 'data_dir', 'iterations', 'seed', 'channel_scaling_alpha']);
    const CHIMERA_UI_DEFAULT_FIELDS = new Set([
        'use_chimera_hydra',
        'channel_scaling_alpha',
        'num_experts_content',
        'num_experts_freq',
        'balance_w_content',
        'balance_w_freq',
        'network_content_router_lr_scale',
        'network_freq_router_lr_scale',
        'freq_router_init_std',
        'freq_router_layer_norm',
    ]);
    const IP_ADAPTER_UI_DEFAULT_FIELDS = new Set(['ip_diagnostics_epochs']);
    const SOFT_TOKENS_UI_DEFAULT_FIELDS = new Set(['weight_decay']);
    const trainingRuntime = {
        state: 'idle',
        lastOutputAt: 0,
        lastUiActivityAt: 0,
        lastGpuUtil: null,
        quietHintShown: false,
        lastLogId: 0,
        logLineCount: 0,
        outputDir: '',
        sampleDir: '',
        sampleConfig: null,
        runDir: '',
        runtimeConfigFile: '',
        originalConfigFile: '',
        datasetConfigFile: '',
        modelCacheDir: '',
        datasetCacheDir: '',
        trainingOutputDir: '',
        logsDir: '',
    };
    const MAX_LOG_LINES = 2000;
    let globalSettings = null;
    const GLOBAL_MODEL_PATH_FIELDS = [
        ['pretrained_model_name_or_path', 'global-pretrained-model-path'],
        ['qwen3', 'global-qwen3-path'],
        ['vae', 'global-vae-path'],
    ];
    const GLOBAL_SETTING_INPUTS = [
        ['output_root', 'global-output-root'],
        ...GLOBAL_MODEL_PATH_FIELDS,
    ];
    const FORM_SECTION_DEFS = [
        {
            title: '基础模型路径',
            description: '训练必须用到的底模、文本编码器和 VAE；路径错误会直接影响预处理和训练启动。',
            open: true,
            className: 'config-group-model',
            keys: [
                'pretrained_model_name_or_path',
                'qwen3',
                'vae',
            ],
        },
        {
            title: '常用训练设置',
            description: '最常改：输出命名、训练时长、适配器容量、学习率、保存频率、优化器和时间步采样。',
            open: true,
            className: 'config-group-primary',
            keys: [
                'output_name',
                'max_train_epochs',
                'learning_rate',
                'save_every_n_epochs',
                'checkpointing_epochs',
                'network_train_unet_only',
                'network_dim',
                'network_alpha',
                'use_lokr',
                'lokr_factor',
                'network_weights',
                'dim_from_weights',
                'optimizer_type',
                'optimizer_args',
                'lr_scheduler',
                'timestep_sampling',
                'discrete_flow_shift',
                'log_every_n_steps',
                'logging_dir',
                'log_with',
            ],
        },
        {
            title: '步数与训练量',
            description: '集中设置批大小、梯度累积和采样比例；下方会实时估算总步数。',
            open: true,
            className: 'config-group-steps',
            keys: [
                'max_train_steps',
                'train_batch_size',
                'gradient_accumulation_steps',
                'sample_ratio',
            ],
        },
        {
            title: '数据集设置',
            description: '选择已保存的数据集预设；路径和分桶蓝图在“数据集”页维护。',
            open: true,
            className: 'config-group-data',
            keys: [
                'use_shuffled_caption_variants',
                'caption_dropout_rate',
                'masked_loss',
            ],
        },
        {
            title: '训练中预览图',
            description: '控制训练过程中是否生成样张；默认关闭，填写提示词文件并设置采样频率后才会出图。',
            open: true,
            className: 'config-group-sampling',
            keys: [
                'sample_prompts',
                'sample_every_n_epochs',
                'sample_every_n_steps',
                'sample_at_first',
                'sample_sampler',
            ],
        },
        {
            title: '显存与速度',
            description: 'OOM、训练速度和编译相关；显存不足时先看这里。',
            open: false,
            className: 'config-group-resource',
            keys: [
                'blocks_to_swap',
                'gradient_checkpointing',
                'unsloth_offload_checkpointing',
                'mixed_precision',
                'attn_mode',
                'torch_compile',
                'compile_inductor_mode',
                'max_data_loader_n_workers',
                'trim_crossattn_kv',
                'vae_chunk_size',
                'vae_disable_cache',
                'use_custom_down_autograd',
                'dataloader_pin_memory',
                'persistent_data_loader_workers',
            ],
        },
        {
            title: '缓存与预处理',
            description: '控制 latent、文本编码器和方法特征缓存；换图片、caption、分桶参数后通常需要重建。',
            open: false,
            keys: [
                'cache_latents',
                'cache_latents_to_disk',
                'cache_text_encoder_outputs',
                'cache_text_encoder_outputs_to_disk',
                'cache_llm_adapter_outputs',
                'skip_cache_check',
                'ip_features_cache_to_disk',
            ],
        },
        {
            title: '更多数据集配置',
            description: '控制训练前预处理对源图的筛选规则；通常和预处理缓存一起重建。',
            open: false,
            keys: [
                'path_pattern',
                'drop_lowres_images',
                'min_pixels',
            ],
        },
        {
            title: 'SPD CLI 实验',
            description: 'SPD 使用 scripts/distill_spd.py 的专用流程；这里用于查看和编辑实验配置，不走 Web 普通训练按钮。',
            open: false,
            className: 'config-group-spd',
            method: 'spd',
            keys: [
                'dit_path',
                'data_dir',
                'iterations',
                'seed',
                'channel_scaling_alpha',
            ],
        },
        {
            title: '输出格式与训练范围',
            description: '模型保存格式、保存精度和训练目标范围。',
            open: false,
            keys: [
                'save_model_as',
                'save_precision',
                'weight_decay',
                'use_cmmd',
                'ip_diagnostics_epochs',
            ],
        },
        {
            title: '方法内部与实验架构',
            description: 'Hydra、FeRA、ReFT、IP-Adapter、EasyControl 等高级方法开关。',
            open: false,
            className: 'config-group-methods',
            keys: [
                'network_module',
                'network_args',
                'use_ortho',
                'use_timestep_mask',
                'min_rank',
                'alpha_rank_scale',
                'channel_scaling_alpha',
                'layer_start',
                'per_channel_scaling',
                'add_reft',
                'reft_dim',
                'reft_alpha',
                'reft_layers',
                'use_repa',
                'repa_weight',
                'repa_layer',
                'repa_lr_scale',
                'use_moe_style',
                'route_per_layer',
                'router_source',
                'num_experts',
                'balance_loss_weight',
                'balance_loss_warmup_ratio',
                'network_router_lr_scale',
                'router_targets',
                'sigma_feature_dim',
                'per_bucket_balance_weight',
                'num_sigma_buckets',
                'specialize_experts_by_sigma_buckets',
                'sigma_bucket_boundaries',
                'use_ip_adapter',
                'ip_image_drop_p',
                'validation_baselines',
                'ip_pair_mode',
                'ip_pair_prob',
                'ip_pair_min_level',
                'ip_pair_caption_strip_p',
                'use_easycontrol',
                'easycontrol_drop_p',
                'easycontrol_cond_noise_max',
                'use_hydra',
                'use_sigma_router',
                'fera_num_bands',
                'fei_feature_dim',
                'fei_sigma_low_div',
                'router_hidden_dim',
                'router_tau',
                'fera_fecl_weight',
                'use_chimera_hydra',
                'num_experts_content',
                'num_experts_freq',
                'balance_w_content',
                'balance_w_freq',
                'network_content_router_lr_scale',
                'network_freq_router_lr_scale',
                'content_router_source',
                'content_router_init_std',
                'content_router_layer_norm',
                'freq_router_init_std',
                'freq_router_layer_norm',
            ],
        },
        {
            title: 'Soft Tokens 参数',
            description: '这些控件会写回 network_args；普通训练无需手改，做 Soft Tokens 对照时再调整。',
            open: false,
            className: 'config-group-soft-tokens',
            keys: [
                'n_layers',
                'n_t_buckets',
                'init_std',
                'splice_position',
                'contrastive_weight',
                'contrastive_k',
                'contrastive_every_n',
                'contrastive_negative_mode',
                'contrastive_objective',
                'agsm_gamma',
                'agsm_ema_decay',
                'contrastive_jaccard_alpha',
                'contrastive_tau',
                'contrastive_warmup_ratio',
            ],
        },
        {
            title: 'IP-Adapter 高级参数',
            description: '这些控件会写回 network_args；identity pair 仍需要先准备 caption-index 与 PE 特征缓存。',
            open: false,
            className: 'config-group-ip-adapter',
            keys: [
                'encoder',
                'encoder_dim',
                'resampler_layers',
                'resampler_heads',
                'ip_scale',
                'gate_lr',
                'pe_lora_enabled',
                'pe_lora_rank',
                'pe_lora_alpha',
                'pe_lora_layer_from',
            ],
        },
        {
            title: 'EasyControl 高级参数',
            description: '这些控件会写回 network_args；用于控制条件流门控、缩放和 FFN LoRA 范围。',
            open: false,
            className: 'config-group-easycontrol',
            keys: [
                'b_cond_init',
                'cond_scale',
                'apply_ffn_lora',
                'cond_token_count',
            ],
        },
    ];
    const CONFIG_COMPACT_FIELD_GROUPS = {
        'config-group-primary': [
            {
                className: 'config-field-grid-2col',
                keys: ['max_train_epochs', 'learning_rate', 'save_every_n_epochs', 'checkpointing_epochs'],
            },
            {
                className: 'config-field-grid-2col',
                keys: ['network_dim', 'network_alpha', 'use_lokr', 'lokr_factor'],
            },
        ],
        'config-group-steps': [
            {
                className: 'config-field-grid-2col',
                keys: ['train_batch_size', 'gradient_accumulation_steps'],
            },
        ],
        'config-group-sampling': [
            {
                className: 'config-field-grid-2col',
                keys: ['sample_every_n_epochs', 'sample_every_n_steps'],
            },
        ],
    };

    const VARIANT_METHOD_FAMILY = {
        lora: 'lora',
        lora_longer: 'lora',
        'lora-8gb': 'lora',
        lora_repa: 'lora',
        lokr: 'lokr',
        ortholora: 'ortholora',
        tlora: 'tlora',
        tlora_ortho: 'tlora',
        'tlora-8gb': 'tlora',
        hydralora: 'hydralora',
        'hydralora-8gb': 'hydralora',
        hydralora_sigma: 'hydralora',
        hydralora_experimental: 'hydralora',
        hydralora_fei: 'hydralora',
        fera: 'hydralora',
        reft: 'reft',
        tlora_ortho_reft: 'reft',
        chimera_hydra: 'chimera',
        ip_adapter: 'ip_adapter',
        easycontrol: 'easycontrol',
        soft_tokens: 'soft_tokens',
        spd: 'spd',
    };

    function help(summary, fill, benefit, cost, risk, recommend) {
        return { summary, fill, benefit, cost, risk, recommend };
    }

    const EXTRA_FIELD_HELP_ZH = {
        max_data_loader_n_workers: help(
            '训练时后台帮忙读取数据的进程数量。',
            '默认 0 表示不用额外进程，最适合 WebUI 和新手排错。数据集非常大、GPU 等数据明显等待时，才考虑调到 1-4。',
            ['流程更稳定，报错位置更容易看懂。'],
            ['少数大数据集可能读取速度偏慢。'],
            ['设太高会占内存和文件句柄，Windows/桌面环境还可能出现卡住或日志混乱。'],
            '新手保持 0；确认瓶颈是读图速度后再小步增加。'
        ),
        drop_lowres_images: help(
            '预处理时是否跳过像素太低的源图。',
            '开启后，低于 min_pixels 的图片不会生成缩放图、VAE 缓存和文本缓存。关闭后不再看 min_pixels，所有图片都会进入预处理和训练。',
            ['减少小图、糊图对训练结果的干扰。'],
            ['关闭后会保留更多素材，但小图也可能被放大、变糊或影响训练质量。'],
            ['开启时如果 min_pixels 设太高，可能突然少掉很多训练图，导致训练不足。'],
            '新手推荐开启；如果你明确想保留全部小图，就关闭它，并知道此时最低像素数不会生效。'
        ),
        min_pixels: help(
            '判断“低分辨率图”的像素数门槛。',
            '只有“过滤低分辨率图 / drop_lowres_images”开启时才生效。宽 x 高低于这个值的图片会被过滤；例如 500000 约等于 0.5MP。',
            ['能用一个数字控制预处理阶段的图片质量下限。'],
            ['关闭低分辨率过滤时，这个字段会被忽略，填多少都不会过滤图片。'],
            ['过滤开启时，门槛太高会让数据集变小；门槛太低则可能保留糊图。'],
            '默认 500000 比较稳；想完全不过滤，优先关闭 drop_lowres_images，或把这里设为 0。'
        ),
        path_pattern: help(
            '从数据集里筛选哪些文件参与训练的路径规则。',
            '默认 * 表示全部图片都参与。只有想临时训练某个子目录或某类文件时才填写更具体的匹配模式。',
            ['不用移动图片文件，就能做小范围数据实验。'],
            ['规则写错会让样本数变少，甚至变成 0。'],
            ['改筛选规则后要重新确认预处理结果、训练步数和样本数量。'],
            '新手保持 *，先让完整数据集跑通。'
        ),
        use_cmmd: help(
            '控制是否启用 CMMD 验证指标。',
            'CMMD 会在验证阶段计算 PE-Core MMD² 类指标，主要服务 IP-Adapter、Chimera 等实验方法。',
            ['比只看 loss 更容易观察条件/风格偏移。'],
            ['验证更慢，并依赖对应特征路径。'],
            ['数据量太少时指标波动较大，不宜单独作为好坏判断。'],
            '日常训练保持 false；做方法对照时再开启。'
        ),
        ip_diagnostics_epochs: help(
            'IP-Adapter 诊断信息输出间隔。',
            '用于控制 gate、IP/text ratio 等诊断日志的频率；数值很大基本等于少输出。',
            ['便于观察 IP 路径是否打开、是否压过文本路径。'],
            ['日志更多，训练查看成本更高。'],
            ['设太小会让历史日志较嘈杂。'],
            '默认 999；调试 IP-Adapter 时可临时调小。'
        ),
        weight_decay: help(
            '优化器权重衰减。',
            'Soft Tokens 等小参数方法可能需要轻微正则；普通 LoRA 多数保持 0。',
            ['能抑制部分小模块过拟合。'],
            ['过高会限制收敛，尤其是低秩参数。'],
            ['不同优化器对 weight_decay 的语义可能略有差异。'],
            '普通 LoRA 保持 0；Soft Tokens 可从 1e-4 起。'
        ),
        dit_path: help(
            'SPD 命令行实验使用的 DiT 底模路径。',
            '这是 scripts/distill_spd.py 专用字段，不是 Web 普通训练使用的基础模型路径。Web 配置页只是允许查看和编辑它。',
            ['SPD 实验可以独立指定底模，不影响普通 LoRA 训练。'],
            ['路径填错会让 SPD CLI 启动失败。'],
            ['把它当成普通训练的基础模型路径来改，会产生误解。'],
            '不跑 SPD CLI 时不用改；需要 SPD 时跟随 configs/methods/spd.toml 默认值。'
        ),
        data_dir: help(
            'SPD 命令行实验读取的数据目录。',
            '通常指向 SPD 脚本要求的数据或缓存。它不是 Web「数据集」页里选择的原始图片目录。',
            ['SPD 实验可以使用独立数据来源。'],
            ['需要自己确认目录内容符合脚本要求。'],
            ['目录不匹配时，distill_spd 可能直接失败，或训练到错误数据。'],
            '不跑 SPD CLI 时保持默认；普通 Web 训练不需要填写它。'
        ),
        iterations: help(
            'SPD CLI 实验的优化迭代次数。',
            '它按 distill_spd 的 optimizer step 计数，不等同于 Web 普通训练的 epoch。',
            ['能精确控制 SPD 实验时长。'],
            ['数值越大训练越久。'],
            ['过小只适合冒烟，过大可能浪费算力。'],
            '默认 4000；先短跑确认流程再放大。'
        ),
        seed: help(
            '随机种子。',
            '用于 SPD 等 CLI 实验复现采样和初始化。',
            ['同配置下更容易复现实验。'],
            ['不同平台/后端仍可能有轻微非确定性。'],
            ['固定种子不能替代多次实验确认稳定性。'],
            '默认 42。'
        ),
        validation_baselines: help(
            'IP-Adapter 验证时输出基线对照。',
            '开启后可对比 self、identity、negative 等参考策略。',
            ['更容易判断 identity pair 是否真的有效。'],
            ['验证更慢，日志/指标更多。'],
            ['小验证集下对照波动会比较明显。'],
            '调试 IP-Adapter identity pair 时开启。'
        ),
        ip_pair_mode: help(
            'IP-Adapter 参考图配对策略。',
            'identity 会找同身份不同图；identity_cross_artist 还要求跨画师；self 使用旧的自配对。',
            ['distinct pair 更能逼迫图像路径学习身份不变量。'],
            ['需要 caption-index 提供 character/copyright/artist 分组。'],
            ['索引缺失时会回退到 self。'],
            '推荐 identity。'
        ),
        ip_pair_prob: help(
            '每一步使用 distinct identity pair 的概率。',
            '剩余比例会混入 self pair，稳定训练。',
            ['平衡身份学习和重建稳定性。'],
            ['概率过高时对索引质量更敏感。'],
            ['概率过低则 identity pair 信号不足。'],
            '推荐 0.8。'
        ),
        ip_pair_min_level: help(
            'identity pair 回退允许的最松层级。',
            'character 最严格，copyright 次之，artist 最宽松。',
            ['控制同身份匹配的保守程度。'],
            ['越严格越容易找不到配对并回退 self。'],
            ['artist 级别可能混入较弱身份关系。'],
            '默认 artist，数据标注好时可收紧。'
        ),
        ip_pair_caption_strip_p: help(
            'distinct-pair 步骤中从 caption 删除身份词的概率。',
            '用于防止身份从文本泄漏而不是从参考图学习；文本缓存开启时该保护基本不生效。',
            ['能更纯粹测试图像条件承载身份。'],
            ['需要关闭/重建对应文本缓存才有意义。'],
            ['过高会让文本条件信息不足。'],
            '默认 0。'
        ),
        content_router_source: help(
            'Chimera 内容路由使用的输入信号。',
            'crossattn_emb 使用文本条件嵌入；其它值用于方法实验。',
            ['让专家按内容条件分工。'],
            ['改变后路由统计和训练行为都会变。'],
            ['错误组合可能导致专家分化不足。'],
            'Chimera 默认 crossattn_emb。'
        ),
        content_router_init_std: help(
            '内容路由器初始化标准差。',
            '非零初始化用于打破均匀路由固定点。',
            ['帮助路由从训练早期开始分化。'],
            ['过大可能导致初期路由偏置太强。'],
            ['和均衡损失权重共同影响稳定性。'],
            '保持 0.001。'
        ),
        content_router_layer_norm: help(
            '内容路由输入是否做无参数 LayerNorm。',
            '用于稳定 pooled 条件特征的尺度。',
            ['路由输入尺度更稳定。'],
            ['关闭后可能更贴近原始特征但更敏感。'],
            ['小数据下尺度噪声可能放大。'],
            '推荐 true。'
        ),
        use_chimera_hydra: help(
            '启用 ChimeraHydra 双路由结构。',
            '打开后内容池与频率池分别路由，并由方法代码固定三轴路由字段。',
            ['能分别观察内容/频率专家分工。'],
            ['字段多、显存和解释成本更高。'],
            ['普通 LoRA 配置里误开会得到实验方法行为。'],
            '只在 chimera_hydra 变体中保持 true。'
        ),
        channel_scaling_alpha: help(
            '按通道缩放的强度系数。',
            '用于 Chimera 与 SPD 等实验路径，影响 LoRA down projection 的通道级输入缩放。',
            ['可缓和部分通道过强更新。'],
            ['调参反馈不直观。'],
            ['过大或过小都可能破坏变体默认平衡。'],
            '保持变体默认值。'
        ),
        num_experts_content: help(
            'Chimera 内容池专家数量。',
            '控制由文本/内容信号路由的专家头数量。',
            ['内容专家越多，分工空间越大。'],
            ['显存、参数和负载均衡难度上升。'],
            ['小数据下专家可能利用不均。'],
            'GUI 默认 4；实验配置可能用 6。'
        ),
        num_experts_freq: help(
            'Chimera 频率池专家数量。',
            '控制由 FEI/sigma 信号路由的频率专家头数量。',
            ['能把不同噪声/频率区域交给不同专家。'],
            ['专家越多越需要观察路由统计。'],
            ['和数据规模不匹配时容易空专家。'],
            '默认 2。'
        ),
        balance_w_content: help(
            'Chimera 内容池负载均衡权重。',
            '只约束内容路由池，不影响频率池的均衡权重。',
            ['能单独压制内容专家坍缩。'],
            ['过高会强迫平均分配，削弱自然分工。'],
            ['需要和 balance_loss_weight 一起理解。'],
            '保持变体默认；观察到内容池坍缩再调。'
        ),
        balance_w_freq: help(
            'Chimera 频率池负载均衡权重。',
            '只约束频率路由池，不影响内容池。',
            ['能单独压制频率专家坍缩。'],
            ['过高会让频率分工变钝。'],
            ['和 freq_router_init_std、sigma_feature_dim 共同影响路由。'],
            '保持变体默认。'
        ),
        network_content_router_lr_scale: help(
            '内容路由器学习率倍率。',
            '相对主学习率放大或缩小内容路由器更新。',
            ['路由器能更快开始分化。'],
            ['过高会震荡或坍缩。'],
            ['和内容池均衡权重耦合明显。'],
            'Chimera 默认 10。'
        ),
        network_freq_router_lr_scale: help(
            '频率路由器学习率倍率。',
            '相对主学习率放大或缩小频率路由器更新。',
            ['频率路由可更快捕捉 FEI/sigma 差异。'],
            ['过高可能让频率池主导训练。'],
            ['需要结合路由统计判断。'],
            'GUI 默认 2，上游实验配置可用 5。'
        ),
        freq_router_init_std: help(
            '频率路由器初始化标准差。',
            '非零初始化帮助频率路由从 step 0 打破均匀固定点。',
            ['更容易早期分化。'],
            ['过大可能带来初始偏置。'],
            ['和均衡损失共同影响稳定性。'],
            '保持变体默认。'
        ),
        freq_router_layer_norm: help(
            '频率路由输入是否做无参数 LayerNorm。',
            '用于稳定 FEI/sigma 特征尺度。',
            ['频率路由输入尺度更稳定。'],
            ['关闭后更依赖原始特征幅度。'],
            ['小数据下尺度噪声可能放大。'],
            '推荐 true。'
        ),
        n_layers: help(
            'Soft Tokens 附加到前多少个 DiT block。',
            '这是 network_args 字段，保存时会写回 network_args 的 n_layers。',
            ['层数越多，软 token 影响范围越大。'],
            ['参数、显存和过拟合风险上升。'],
            ['超过模型层数会在启动时报错。'],
            '默认 14；小显存或快速对照时再下调。'
        ),
        n_t_buckets: help(
            'Soft Tokens 的时间桶数量。',
            '把 sigma/timestep 分桶，每桶学习一组时间偏移。',
            ['更多桶能表达更细的时间步差异。'],
            ['数据不足时许多桶训练很少。'],
            ['桶太多会增加参数并拖慢收敛。'],
            '默认 14；数据量更大时再提高桶数量。'
        ),
        init_std: help(
            'Soft Tokens 基础 token 初始化标准差。',
            '控制软 token 初始幅度。',
            ['较小初始化更接近底模原始行为。'],
            ['太小可能启动慢，太大可能扰动强。'],
            ['和学习率一起影响稳定性。'],
            '默认 0.02。'
        ),
        splice_position: help(
            'Soft Tokens 写入文本条件的位置。',
            'end_of_sequence 覆盖末尾 padding；front_of_padding 从每条 caption 的 padding 前沿插入。',
            ['可控制软 token 与文本 token 的相对位置。'],
            ['不同模式需要配合缓存/推理链路理解。'],
            ['切换会改变训练语义，不能和旧权重简单对照。'],
            '默认 front_of_padding；需要复现旧实验时再改成 end_of_sequence。'
        ),
        contrastive_weight: help(
            'Soft Tokens 对比目标权重。',
            '0 表示关闭额外对比前向；大于 0 会启用负样本目标。',
            ['可能增强提示词区分能力。'],
            ['每次触发会增加额外 DiT forward，训练变慢。'],
            ['权重过高会压过 FM 主损失。'],
            '默认 0.05；想关闭额外目标时设为 0。'
        ),
        contrastive_k: help(
            '每步使用的对比负样本数。',
            '每个负样本都会带来额外前向成本。',
            ['更多负样本使对比信号更强。'],
            ['训练时间近似随 k 增加。'],
            ['过多会明显拖慢并增大调参难度。'],
            '推荐 1-2。'
        ),
        contrastive_every_n: help(
            '每隔多少个 optimizer step 触发一次对比目标。',
            '1 表示每步触发；数值越大平均成本越低。',
            ['可以控制额外前向频率。'],
            ['有效对比强度约随 1/N 下降。'],
            ['如果想维持平均强度，需要同步调整 weight。'],
            '默认 3；需要更强对比信号时再降低。'
        ),
        contrastive_negative_mode: help(
            'Soft Tokens 负样本来源。',
            'shuffled 为随机负样本，jaccard 会按 tag 重叠降权，hard 尝试同画师不同角色等困难负样本。',
            ['hard/jaccard 能提供更有针对性的区分信号。'],
            ['依赖 caption-index 质量。'],
            ['索引缺失或标签差时会退化或引入噪声。'],
            '默认 hard；没有高质量 caption-index 时可退回 shuffled。'
        ),
        contrastive_objective: help(
            'Soft Tokens 对比目标函数。',
            'infonce 使用传统对比分类；agsm 使用有界目标偏移。',
            ['AGSM 可能降低负样本无界发散。'],
            ['仍是实验路径，成本更高。'],
            ['目标函数切换后历史经验不可直接套用。'],
            '默认 agsm；需要复现传统对比分类时再选 infonce。'
        ),
        agsm_gamma: help(
            'AGSM 目标偏移强度。',
            '只在 contrastive_objective=agsm 且 contrastive_weight>0 时生效。',
            ['控制正负目标分离幅度。'],
            ['太高可能让目标偏离主 FM 任务。'],
            ['太低则 AGSM 信号弱。'],
            '默认 0.5。'
        ),
        agsm_ema_decay: help(
            'AGSM 读取自身预测 EMA 的衰减。',
            '越接近 1 越平滑，越小越跟随当前预测。',
            ['稳定目标偏移估计。'],
            ['过高会反应慢。'],
            ['必须在 0 和 1 之间。'],
            '默认 0.99。'
        ),
        contrastive_jaccard_alpha: help(
            'jaccard 负样本模式的 tag 重叠惩罚。',
            '负样本和正样本 tag 越重叠，logit 惩罚越大。',
            ['降低相似 caption 负样本误伤。'],
            ['只对 jaccard 模式生效。'],
            ['过高会让负样本信号过弱。'],
            '默认 1.0。'
        ),
        contrastive_tau: help(
            'InfoNCE 温度参数。',
            '只对 infonce 目标生效，越小 logits 越尖锐。',
            ['可调节对比分类强度。'],
            ['过小容易梯度尖锐，过大信号变弱。'],
            ['AGSM 目标不使用它。'],
            '默认 0.5。'
        ),
        contrastive_warmup_ratio: help(
            '对比目标预热比例。',
            '训练前若干比例 step 内将对比权重保持为 0。',
            ['先让普通 FM 建立基础，再加入对比约束。'],
            ['预热太长会让对比目标影响不足。'],
            ['太短可能早期不稳定。'],
            '默认 0.1。'
        ),
        encoder: help(
            'IP-Adapter 使用的视觉编码器。',
            '当前上游路径主要使用 PE-Core，值为 pe。',
            ['明确图像特征来源。'],
            ['其它值需要对应编码器实现支持。'],
            ['改错会导致启动失败。'],
            '保持 pe。'
        ),
        encoder_dim: help(
            '视觉编码器输出维度。',
            'PE-Core L14-336 默认 1024。',
            ['必须与缓存的 PE 特征维度一致。'],
            ['不匹配会在训练时形状报错。'],
            ['手动改动通常无意义。'],
            '保持 1024。'
        ),
        resampler_layers: help(
            'IP-Adapter Perceiver Resampler 层数。',
            '用于把 PE patch 特征压缩成固定数量 IP token。',
            ['层数越多，图像条件聚合能力越强。'],
            ['显存和计算增加。'],
            ['小数据过深可能过拟合。'],
            '默认 2。'
        ),
        resampler_heads: help(
            'IP-Adapter Resampler 注意力头数。',
            '控制 resampler 内部多头注意力宽度。',
            ['更多头可表达更丰富的图像 token 聚合。'],
            ['计算略增，收益依数据而定。'],
            ['和 encoder_dim 需整除匹配。'],
            '默认 8。'
        ),
        ip_scale: help(
            'IP 条件输出强度倍率。',
            '乘到 IP attention 输出上，影响图像条件对文本 cross-attention 的贡献。',
            ['能快速调强或调弱图像条件。'],
            ['过高会压过文本提示。'],
            ['过低会让参考图影响不足。'],
            '默认 1.0。'
        ),
        gate_lr: help(
            'IP per-block gate 的学习率覆盖。',
            '留空时使用全局学习率；填写后 gate 可更快打开。',
            ['解决 gate 打开太慢的问题。'],
            ['过高可能导致 IP 路径快速压过文本。'],
            ['需要观察诊断指标。'],
            '常用 1e-3；不调试时保持变体默认。'
        ),
        pe_lora_enabled: help(
            '是否训练 PE-Core 视觉编码器 LoRA。',
            '开启后不能使用静态缓存 PE 特征，需要 live 编码。',
            ['能让视觉编码器适配动漫/漫画分布。'],
            ['训练明显更慢，显存更高。'],
            ['和 ip_features_cache_to_disk=true 不兼容。'],
            'identity pair 快速路径保持 false。'
        ),
        pe_lora_rank: help(
            'PE-Core LoRA 秩。',
            '只在 pe_lora_enabled=true 时生效。',
            ['控制视觉编码器 LoRA 容量。'],
            ['rank 越高显存和过拟合风险越高。'],
            ['不开 PE-LoRA 时无效。'],
            '默认 16。'
        ),
        pe_lora_alpha: help(
            'PE-Core LoRA Alpha。',
            '只在 pe_lora_enabled=true 时生效。',
            ['控制 PE-LoRA 更新强度。'],
            ['过高会让视觉特征漂移。'],
            ['不开 PE-LoRA 时无效。'],
            '默认 16。'
        ),
        pe_lora_layer_from: help(
            'PE-LoRA 从哪一层开始训练。',
            '-1 表示所有 PE resblock；正数 N 表示只训练最后 N 层。',
            ['可把容量集中在高层语义。'],
            ['层数越多越慢。'],
            ['选择不当可能适配不足或过拟合。'],
            '默认 8。'
        ),
        b_cond_init: help(
            'EasyControl 条件注意力的初始 bias。',
            '默认 -10 让 step 0 几乎等价底模，随后训练逐步打开条件质量。',
            ['保护初始行为稳定。'],
            ['过低会打开慢，过高会一开始扰动底模。'],
            ['改动会影响 EasyControl 稳定性。'],
            '保持 -10。'
        ),
        cond_scale: help(
            'EasyControl 条件流强度倍率。',
            '控制条件输出对目标流的影响。',
            ['可快速调节条件强弱。'],
            ['过高可能过度依赖条件图。'],
            ['过低控制效果会弱。'],
            '默认 1.0。'
        ),
        apply_ffn_lora: help(
            'EasyControl 是否在 FFN 层也应用条件 LoRA。',
            '关闭后只在注意力相关路径加条件 LoRA。',
            ['开启表达力更强。'],
            ['参数和计算更多。'],
            ['小数据时可能过拟合。'],
            '默认开启。'
        ),
        cond_token_count: help(
            'EasyControl 条件 latent 静态 padding 的 token 数。',
            '默认 4096，与常见 Anima 桶 token 数对齐。',
            ['保证条件流形状稳定。'],
            ['数值越大显存越高。'],
            ['太小会拒绝较大条件 latent。'],
            '默认 4096；低显存实验可谨慎降低。'
        ),
    };

    const FIELD_LABEL_ZH = {
        add_reft: '启用 ReFT',
        agsm_ema_decay: 'AGSM EMA 衰减',
        agsm_gamma: 'AGSM 目标偏移强度',
        alpha_rank_scale: '按秩缩放 Alpha',
        apply_ffn_lora: 'EasyControl FFN LoRA',
        attn_mode: '注意力后端',
        balance_loss_warmup_ratio: '均衡损失预热比例',
        balance_loss_weight: '均衡损失权重',
        balance_w_content: '内容池均衡权重',
        balance_w_freq: '频率池均衡权重',
        b_cond_init: '条件注意力初始门控',
        blocks_to_swap: 'CPU/GPU 交换块数',
        cache_latents: '缓存潜变量',
        cache_latents_to_disk: '潜变量写入磁盘',
        cache_llm_adapter_outputs: '缓存 LLM 适配器输出',
        cache_text_encoder_outputs: '缓存文本编码器输出',
        cache_text_encoder_outputs_to_disk: '文本编码器缓存写入磁盘',
        batch_size: '数据集批大小',
        bucket_no_upscale: '禁止放大图片',
        bucket_reso_steps: '桶尺寸步长',
        caption_extension: 'Caption 扩展名',
        caption_dropout_rate: '标题丢弃率',
        channel_scaling_alpha: '通道缩放 Alpha',
        checkpointing_epochs: '训练状态保存间隔',
        compile_inductor_mode: 'Inductor 编译模式',
        cond_scale: 'EasyControl 条件强度',
        cond_token_count: 'EasyControl 条件 Token 数',
        content_router_init_std: '内容路由初始化标准差',
        content_router_layer_norm: '内容路由 LayerNorm',
        content_router_source: '内容路由信号来源',
        contrastive_every_n: '对比损失触发间隔',
        contrastive_jaccard_alpha: 'Jaccard 负样本惩罚',
        contrastive_k: '对比负样本数',
        contrastive_negative_mode: '对比负样本模式',
        contrastive_objective: '对比目标函数',
        contrastive_tau: 'InfoNCE 温度',
        contrastive_warmup_ratio: '对比损失预热比例',
        contrastive_weight: '对比损失权重',
        dataloader_pin_memory: '固定内存加载',
        data_dir: 'SPD 数据目录',
        dataset_config: '数据集配置',
        discrete_flow_shift: 'Flow 偏移',
        dit_path: 'SPD DiT 模型路径',
        drop_lowres_images: '过滤低分辨率图',
        easycontrol_cond_noise_max: 'EasyControl 条件噪声上限',
        easycontrol_drop_p: 'EasyControl 条件丢弃率',
        enable_bucket: '启用长宽比分桶',
        encoder: 'IP 视觉编码器',
        encoder_dim: 'IP 编码器维度',
        fei_feature_dim: 'FEI 特征维度',
        fei_sigma_low_div: 'FEI 低 Sigma 除数',
        fera_fecl_weight: 'FeRA FECL 权重',
        fera_num_bands: 'FeRA 分带数',
        freq_router_init_std: '频率路由初始化标准差',
        freq_router_layer_norm: '频率路由 LayerNorm',
        gate_lr: 'IP Gate 学习率',
        train_batch_size: '批大小',
        gradient_accumulation_steps: '梯度累积步数',
        gradient_checkpointing: '梯度检查点',
        init_std: 'Soft Tokens 初始化标准差',
        ip_features_cache_to_disk: 'IP 特征写入磁盘',
        ip_diagnostics_epochs: 'IP 诊断间隔',
        ip_image_drop_p: 'IP 图像条件丢弃率',
        ip_pair_caption_strip_p: '身份对 Caption 去标概率',
        ip_pair_min_level: '身份对最低层级',
        ip_pair_mode: '身份对采样模式',
        ip_pair_prob: '身份对采样概率',
        ip_scale: 'IP 条件强度',
        iterations: 'SPD 迭代步数',
        keep_tokens: '保留 Token 数',
        learning_rate: '学习率',
        log_every_n_steps: '日志记录间隔',
        log_with: '日志后端',
        logging_dir: '日志目录',
        lora_cache_dir: 'LoRA 缓存目录',
        lr_scheduler: '学习率调度',
        masked_loss: '遮罩损失',
        max_train_epochs: '最大训练轮数',
        max_train_steps: '最大训练步数',
        max_data_loader_n_workers: 'DataLoader 进程数',
        max_bucket_reso: '最大桶边长',
        min_rank: '最小秩',
        min_bucket_reso: '最小桶边长',
        min_pixels: '最低像素数',
        mixed_precision: '混合精度',
        network_alpha: 'LoRA Alpha',
        network_args: '网络额外参数',
        network_content_router_lr_scale: '内容路由学习率倍率',
        network_dim: 'LoRA 秩',
        network_freq_router_lr_scale: '频率路由学习率倍率',
        network_module: '网络模块',
        network_router_lr_scale: '路由器学习率倍率',
        network_train_unet_only: '仅训练 DiT',
        num_experts: '专家数量',
        num_sigma_buckets: 'Sigma 桶数量',
        lokr_factor: 'LoKr Factor',
        use_lokr: '启用 LoKr',
        optimizer_args: '优化器参数',
        optimizer_type: '优化器',
        output_dir: '输出目录',
        output_name: '输出名称',
        path_pattern: '数据路径匹配',
        pe_lora_alpha: 'PE-LoRA Alpha',
        pe_lora_enabled: '启用 PE-LoRA',
        pe_lora_layer_from: 'PE-LoRA 起始层',
        pe_lora_rank: 'PE-LoRA 秩',
        per_bucket_balance_weight: '分桶均衡权重',
        persistent_data_loader_workers: '常驻数据加载进程',
        pretrained_model_name_or_path: '基础模型路径',
        preflight: '训练前预检测',
        preprocess_environment: '预处理启动环境',
        qwen3: 'Qwen3 文本编码器路径',
        reft_alpha: 'ReFT Alpha',
        reft_dim: 'ReFT 秩',
        reft_layers: 'ReFT 层范围',
        repa_layer: 'REPA 层',
        repa_lr_scale: 'REPA 学习率倍率',
        repa_weight: 'REPA 权重',
        resized_image_dir: '缩放图像目录',
        resolution: '数据集分辨率',
        resampler_heads: 'IP Resampler 头数',
        resampler_layers: 'IP Resampler 层数',
        route_per_layer: '逐层路由',
        router_hidden_dim: '路由器隐藏维度',
        router_source: '路由信号来源',
        router_targets: '路由目标层',
        router_tau: '路由温度',
        sample_ratio: '数据采样比例',
        sample_at_first: '开始前生成样张',
        sample_every_n_epochs: '按轮生成样张',
        sample_every_n_steps: '按步生成样张',
        sample_prompts: '样张提示词文件',
        sample_sampler: '样张采样器',
        save_every_n_epochs: '模型保存间隔',
        save_model_as: '模型保存格式',
        save_precision: '保存精度',
        seed: '随机种子',
        sigma_bucket_boundaries: 'Sigma 桶边界',
        sigma_feature_dim: 'Sigma 特征维度',
        skip_cache_check: '跳过缓存检查',
        source_image_dir: '源图像目录',
        specialize_experts_by_sigma_buckets: '按 Sigma 桶专门化专家',
        splice_position: 'Soft Tokens 拼接位置',
        timestep_sampling: '时间步采样',
        torch_compile: '启用 torch.compile',
        trim_crossattn_kv: '裁剪交叉注意力 KV',
        unsloth_offload_checkpointing: 'Unsloth 检查点卸载',
        use_custom_down_autograd: '自定义 Down 反向',
        use_chimera_hydra: '启用 ChimeraHydra',
        use_cmmd: '启用 CMMD 验证',
        use_easycontrol: '启用 EasyControl',
        use_ip_adapter: '启用 IP-Adapter',
        use_moe_style: 'MoE 结构',
        use_ortho: '启用 OrthoLoRA',
        use_repa: '启用 REPA',
        use_shuffled_caption_variants: '使用打乱标题变体',
        use_timestep_mask: '启用 T-LoRA',
        validation_seed: '验证随机种子',
        validation_baselines: '验证基线对照',
        validation_split: '验证集比例',
        validation_split_num: '固定验证数量',
        vae: 'VAE 路径',
        vae_chunk_size: 'VAE 分块大小',
        vae_disable_cache: '禁用 VAE 缓存',
        weight_decay: '权重衰减',
        n_layers: 'Soft Tokens 层数',
        n_t_buckets: '时间桶数量',
        num_experts_content: '内容专家数',
        num_experts_freq: '频率专家数',
    };

    const FIELD_OPTIONS = {
        attn_mode: ['flash', 'flex'],
        compile_inductor_mode: ['default', 'reduce-overhead', 'max-autotune'],
        apply_ffn_lora: [true, false],
        contrastive_negative_mode: ['shuffled', 'jaccard', 'hard'],
        contrastive_objective: ['infonce', 'agsm'],
        content_router_layer_norm: [true, false],
        content_router_source: ['crossattn_emb', 'pooled_text', 'fei', 'sigma'],
        dataset_config: [
            'configs/datasets/ip_adapter.toml',
            'configs/datasets/easycontrol.toml',
            'configs/datasets/rokkotsu_goddess.toml',
        ],
        drop_lowres_images: [true, false],
        encoder: ['pe'],
        freq_router_layer_norm: [true, false],
        pe_lora_enabled: [false, true],
        splice_position: ['end_of_sequence', 'front_of_padding'],
        validation_baselines: [false, true],
        log_with: ['tensorboard'],
        use_chimera_hydra: [false, true],
        use_cmmd: [false, true],
        use_lokr: [false, true],
        lokr_factor: [2, 4, 8, 16],
        ip_pair_mode: ['identity', 'identity_cross_artist', 'self'],
        ip_pair_min_level: ['artist', 'copyright', 'character'],
        lr_scheduler: ['constant', 'cosine', 'cosine_with_restarts', 'polynomial'],
        max_data_loader_n_workers: [0, 2, 4, 8],
        min_pixels: [0, 262144, 500000, 786432, 1048576],
        min_rank: [1, 2, 4, 8],
        mixed_precision: ['bf16', 'fp16', 'no'],
        network_alpha: [4, 8, 16, 32, 64],
        network_dim: [4, 8, 16, 32, 64, 128],
        network_module: [
            'networks.lora_anima',
            'networks.methods.ip_adapter',
            'networks.methods.easycontrol',
            'networks.methods.soft_tokens',
        ],
        num_experts: [2, 4, 6, 8],
        num_sigma_buckets: [2, 3, 4],
        optimizer_type: ['AdamW', 'CAME', 'AdamW8bit', 'Lion', 'Prodigy', 'ProdigyPlusScheduleFree'],
        reft_alpha: [16, 32, 64, 128],
        reft_dim: [16, 32, 64, 128],
        reft_layers: ['last_8', 'first_4', 'stride_2', 'all'],
        route_per_layer: [true, false],
        router_source: ['fei', 'sigma', 'input', 'none'],
        sample_ratio: [1.0, 0.5, 0.25, 0.1, 0.01],
        sample_sampler: [
            'ddim',
            'euler',
            'euler_a',
            'dpmsolver++',
            'dpmsolver',
            'heun',
            'lms',
            'pndm',
            'k_euler',
            'k_euler_a',
            'k_lms',
        ],
        save_model_as: ['safetensors'],
        save_precision: ['bf16', 'fp16', 'float'],
        timestep_sampling: ['sigmoid', 'uniform', 'shift'],
        use_moe_style: ['false', 'shared_A', 'independent_A'],
        vae_chunk_size: [16, 32, 64, 128],
    };

    const METHOD_GUIDE_ZH = {
        lora: choiceHelp(
            'LoRA 家族',
            '最基础、兼容性最好的低秩微调方法，适合大多数角色、画风和概念训练。',
            '好处是稳定、可合并、推理链路最简单；代价是表达力主要靠 rank 和训练轮数。',
            '新手优先选它。'
        ),
        ortholora: choiceHelp(
            'OrthoLoRA',
            '在 LoRA 更新里加入正交约束，目标是减少无关概念互相污染。',
            '更适合希望风格/概念更干净的训练；代价是方法更复杂，收益依数据而定。',
            '想比普通 LoRA 更稳一点时选。'
        ),
        tlora: choiceHelp(
            'T-LoRA',
            '让 LoRA 有效秩随去噪时间步变化，把容量偏向结构更关键的阶段。',
            '保存结果仍接近普通 LoRA 工作流；代价是多了时间步相关超参。',
            '泛用进阶推荐 tlora_ortho。'
        ),
        hydralora: choiceHelp(
            'HydraLoRA / FeRA',
            'MoE 专家路由类方法，让不同专家处理不同时间步或特征区域。',
            '容量和分工更强；代价是显存、速度、推理兼容性和调参复杂度都更高。',
            '只在普通 LoRA 不够表达时再试。'
        ),
        lokr: choiceHelp(
            'LoKr',
            '使用 Kronecker 积分解代替标准低秩分解，参数效率更高。',
            '适合复杂画风或多角色；代价是推理需要 LyCORIS/LoKr 兼容加载器。',
            '需要 LoKr 训练时选，简单单角色仍可用普通 LoRA。'
        ),
        reft: choiceHelp(
            'ReFT',
            '在 DiT 块残差流上做可训练干预，可和 LoRA/T-LoRA 组合。',
            '表达力强；代价是合并/推理兼容性不如普通 LoRA。',
            '需要更强语义干预时选。'
        ),
        chimera: choiceHelp(
            'Chimera',
            '双路由 Hydra 实验方法，把内容路由和频率路由拆开观察。',
            '适合方法对照；代价是字段更多、显存和解释成本都高。',
            '只在需要 ChimeraHydra 实验时选。'
        ),
        soft_tokens: choiceHelp(
            'Soft Tokens',
            '训练可学习软文本 token，让条件侧获得少量可训练容量。',
            '参数量小、适合文本条件实验；代价是推理链路和普通 LoRA 不同。',
            '只建议做方法实验或已有配套加载器时使用。'
        ),
        ip_adapter: choiceHelp(
            'IP-Adapter',
            '图像条件适配器，用参考图像特征参与训练。',
            '能学习图像条件控制；新版支持 identity pair，通常先跑 caption-index 和 preprocess-pe 准备索引/PE 特征。',
            '需要参考图/图像条件时选。'
        ),
        easycontrol: choiceHelp(
            'EasyControl',
            '图像控制条件方法，使用专用数据集和缓存目录。',
            '控制信号更直接；代价是数据准备和训练路径更专门。',
            '需要图像控制训练时选。'
        ),
        spd: choiceHelp(
            'SPD CLI 实验',
            '逐级分辨率轨迹适配器实验配置，由 scripts/distill_spd.py 读取，不走普通 train.py。',
            '能在 WebUI 查看和编辑配置；启动训练请使用 tasks.py exp-spd 或对应 CLI 命令。',
            '只把这里当配置入口，不要点击 Web 普通训练按钮。'
        ),
    };

    const VARIANT_GUIDE_ZH = {
        lora: choiceHelp(
            '普通 LoRA',
            '默认基础变体，rank 32、学习率 2e-5、4 轮训练。',
            '最稳、最容易和其他工具链配合；表达力不如 MoE/实验方法激进。',
            '新手和大多数正式训练从这里开始。'
        ),
        lora_longer: choiceHelp(
            '更长 LoRA',
            '架构接近普通 LoRA，但偏向更长或更充分的训练配置。',
            '适合默认轮数还欠拟合的数据；代价是训练更久、过拟合风险更高。',
            '样张还不够像时再切。'
        ),
        'lora-8gb': choiceHelp(
            '低显存 LoRA',
            '面向 8GB/低显存环境，开启梯度检查点和卸载相关设置。',
            '更不容易 OOM；代价是训练明显变慢。',
            '默认配置爆显存时选。'
        ),
        lora_repa: choiceHelp(
            'LoRA + REPA',
            '普通 LoRA 叠加 REPA 表征对齐辅助损失。',
            '可能改善中间表征对齐；风险是辅助损失过强会压过主目标。',
            '想做 REPA 实验时选。'
        ),
        ortholora: choiceHelp(
            'OrthoLoRA',
            '普通 LoRA 加正交约束，保存时仍偏普通 LoRA 使用方式。',
            '更重视结构化更新；代价是训练机制更复杂。',
            '概念容易互相污染时试。'
        ),
        tlora: choiceHelp(
            'T-LoRA',
            '启用时间步 rank mask，不加正交约束。',
            '比普通 LoRA 更关注去噪阶段差异；代价是多一个 min_rank 维度。',
            '想单独测试 T-LoRA 时选。'
        ),
        tlora_ortho: choiceHelp(
            'T-LoRA + OrthoLoRA',
            '时间步 rank mask 和正交约束叠加。',
            '泛用进阶配置，兼顾结构阶段和更新约束；训练理解成本比普通 LoRA 高。',
            '有经验后可作为默认进阶选择。'
        ),
        reft: choiceHelp(
            'ReFT',
            '只启用 ReFT 残差流干预，rank 和学习率与普通 LoRA 不同。',
            '干预强、学习快；兼容性和合并能力更弱。',
            '做 ReFT 专项实验时选。'
        ),
        tlora_ortho_reft: choiceHelp(
            'T-LoRA + Ortho + ReFT',
            '把 T-LoRA、OrthoLoRA 和 ReFT 叠加。',
            '表达力强；代价是变量多，出现问题更难定位。',
            '只建议对照实验使用。'
        ),
        hydralora_sigma: choiceHelp(
            'Hydra Sigma',
            '共享 down 矩阵，多专家 up，按 sigma/时间步路由。',
            '专家能按去噪阶段分工；代价是训练和推理更复杂。',
            '想研究时间步专家分工时选。'
        ),
        hydralora_experimental: choiceHelp(
            'Hydra 实验版',
            '更激进的 Hydra Sigma 配置，包含更多专家或硬分桶设置。',
            '探索空间更大；风险是专家利用不均、调参成本高。',
            '只建议实验用。'
        ),
        hydralora_fei: choiceHelp(
            'Hydra FEI',
            'Hydra 结构使用 FEI 特征作为路由信号。',
            '比纯 sigma 路由多一个内容/特征维度；代价是依赖 FEI 特征缓存和路由稳定性。',
            '需要 FEI 路由时选。'
        ),
        fera: choiceHelp(
            'FeRA',
            '独立 A 矩阵的 FEI 路由专家结构。',
            '容量更高、专家更独立；代价是参数、显存和训练复杂度更高。',
            '普通 Hydra 不够时再试。'
        ),
        lokr: choiceHelp(
            'LoKr',
            '输出 LyCORIS 兼容的 lokr_w1/lokr_w2 权重，默认 factor=8。',
            '收敛快、参数效率高；过拟合风险更高，推理侧需要 LoKr 支持。',
            '多角色/复杂画风可试；注意控制训练轮数。'
        ),
        chimera_hydra: choiceHelp(
            'ChimeraHydra',
            '双路由 Hydra 组合变体，内容路由和频率路由都启用。',
            '更适合方法实验；代价是字段多、解释成本高。',
            '做 Chimera 对照实验时选。'
        ),
        ip_adapter: choiceHelp(
            'IP-Adapter',
            '训练图像条件 cross-attention 适配器。',
            '新版支持 identity pair；通常需要先跑 caption-index 与 preprocess-pe，参考图像特征和索引都要准备好。',
            '需要参考图控制时选。'
        ),
        easycontrol: choiceHelp(
            'EasyControl',
            '训练图像条件 self-attention/FFN 控制适配。',
            '控制路径更强；需要 easycontrol-dataset 和专用缓存。',
            '需要 EasyControl 图像条件时选。'
        ),
        soft_tokens: choiceHelp(
            'Soft Tokens',
            '训练软文本 token，当前偏实验/训练侧路径。',
            '参数量小；推理链路和普通 LoRA 不同。',
            '只建议方法实验。'
        ),
        spd: choiceHelp(
            'SPD 实验配置',
            'configs/methods/spd.toml 是专用 distill_spd 脚本配置，包含数据目录、迭代数和 SPD schedule。',
            'Web 表单可补全常用字段；普通 Web 训练/预处理会明确拦截，避免误走 train.py。',
            '运行时使用 CLI：tasks.py exp-spd，测试使用 exp-test-spd。'
        ),
    };

    const PRESET_GUIDE_ZH = {
        default: choiceHelp(
            '默认预设',
            '不额外改硬件/采样覆盖，使用方法变体自己的训练配置。',
            '行为最可预测；如果显存不足，需要切低显存方案。',
            '新手默认选这个。'
        ),
        low_vram: choiceHelp(
            '低显存预设',
            '开启梯度检查点和 CPU 卸载，降低显存峰值。',
            '更不容易 OOM；代价是训练速度下降。',
            '显存不够时选。'
        ),
        graft: choiceHelp(
            '交换块预设',
            '提高 blocks_to_swap，把更多 DiT 块放到 CPU/GPU 间交换。',
            '进一步省显存；代价是训练会更慢。',
            '低显存仍 OOM 时再试。'
        ),
        half: choiceHelp(
            '半量采样',
            '每轮约使用 50% 数据。',
            '快速试跑；结果不能代表完整训练。',
            '验证流程或粗调参数时用。'
        ),
        quarter: choiceHelp(
            '四分之一采样',
            '每轮约使用 25% 数据。',
            '更快试跑；训练信号更不完整。',
            '只用于快速排错。'
        ),
        tenth: choiceHelp(
            '十分之一采样',
            '每轮约使用 10% 数据。',
            '启动和验证最快；几乎不能判断最终质量。',
            '只用于流程冒烟。'
        ),
        debug: choiceHelp(
            '调试采样',
            '每轮约使用 1% 数据。',
            '最快发现配置/代码错误；完全不适合看训练效果。',
            '开发排错用，不建议正式训练。'
        ),
    };

    function choiceHelp(title, summary, tradeoff, recommend) {
        return { title, summary, tradeoff, recommend };
    }

    // 中文字段说明。每项按固定栏目渲染，避免用户只看到一句模糊解释。
    const FIELD_HELP_ZH = {
        network_dim: help(
            "LoRA 的容量大小，也叫 rank 或秩。",
            "它决定 LoRA 能学习多少新特征。常用 16/32/64：数据少、显存小用 16-32；画风复杂、角色细节很多时再考虑 64。",
            ["数值越高，越能记住细节、画风和角色差异。"],
            ["显存、训练时间和最终权重文件都会变大。"],
            ["小数据集设太高容易过拟合，把训练图里的噪声、构图和瑕疵也学进去。"],
            "新手从 32 开始；8GB 显存或少量图片优先用 16-32。"
        ),
        network_alpha: help(
            "LoRA 的缩放强度，影响训练结果最终作用有多猛。",
            "最简单的填法是和 network_dim 填一样，例如 dim=32 时 alpha=32。想让 LoRA 更保守时，可以填 dim 的一半。",
            ["可以控制 LoRA 对底模的影响幅度。"],
            ["太低会学得慢，可能需要更多轮数才看得出效果。"],
            ["太高容易风格过冲、颜色变脏，或让提示词服从性变差。"],
            "新手推荐 alpha = dim；只有效果明显太重、太像训练图时再降低。"
        ),
        network_module: help(
            "训练时加载的网络实现模块。",
            "普通 LoRA 家族保持 networks.lora_anima；IP-Adapter、EasyControl、Soft Tokens 等变体由对应 TOML 自动填写。",
            ["允许不同训练方法共用同一个 Web 表单。"],
            ["改错模块会导致启动失败，或加载到不匹配的参数。"],
            ["不要手动改成未注册/不存在的 Python 模块。"],
            "新手不要手动改；选择方法/变体后保持默认。"
        ),
        network_args: help(
            "传给 network_module 的额外参数列表。",
            "按 TOML 字符串数组填写，例如 [\"mode=cond\", \"cond_hidden_dim=256\"]。",
            ["让实验方法可以暴露内部开关，不必新增表单控件。"],
            ["格式比普通字段更容易写错。"],
            ["key 拼错通常不会得到预期行为；值类型错误可能在训练启动后才暴露。"],
            "优先选择已有变体自动填充；手改前先看对应方法 TOML。"
        ),
        network_train_unet_only: help(
            "只训练 DiT 主模型侧的适配器，冻结文本编码器。",
            "普通 LoRA 训练保持 true。",
            ["显存更稳，训练结果更容易复用。"],
            ["无法直接微调文本编码器本身的语言理解。"],
            ["关闭后训练面更大，资源和失控风险都明显上升。"],
            "新手保持 true；本项目的常规 LoRA 训练不需要改文本编码器。"
        ),
        network_weights: help(
            "从已有适配器检查点热启动训练。",
            "普通新训练留空；续训或二次微调用已有 .safetensors 路径。",
            ["能在已有效果上继续细化，节省从零训练时间。"],
            ["继承旧模型的偏差和过拟合。"],
            ["检查点方法或 rank 不匹配会加载失败或效果异常。"],
            "新手留空；热启动时配合 dim_from_weights 使用。"
        ),
        dim_from_weights: help(
            "从热启动检查点读取 rank，而不是使用表单里的 network_dim。",
            "只有填写 network_weights 时才开启。",
            ["避免 rank 不一致导致加载失败。"],
            ["会忽略当前表单的 network_dim。"],
            ["误开时可能让你以为改了 dim，但实际沿用了旧检查点。"],
            "热启动推荐 true；从零训练保持 false。"
        ),
        use_ortho: help(
            "启用 OrthoLoRA，用正交参数化约束 LoRA 更新。",
            "想减少概念互相污染时开启；普通 LoRA 可以关闭。",
            ["更新更结构化，通常更不容易干扰无关概念。"],
            ["训练计算略重，实验解释成本更高。"],
            ["并不保证一定更好，小数据或很简单的风格可能收益不明显。"],
            "泛用训练可选 tlora_ortho 或 ortholora 变体。"
        ),
        use_timestep_mask: help(
            "启用 T-LoRA，让有效 rank 随去噪时间步变化。",
            "想把容量集中在高噪声结构阶段时开启，并设置 min_rank。",
            ["结构学习更集中，保存结果仍是可合并的普通 LoRA。"],
            ["多一个时间步相关超参，调试复杂度上升。"],
            ["min_rank 太低可能削弱低噪声阶段的细节修正。"],
            "推荐直接使用 tlora 或 tlora_ortho 变体。"
        ),
        use_lokr: help(
            "启用 LoKr（Low-Rank Kronecker Product）。",
            "用 Kronecker 积分解 ΔW = kron(W1, W2)，保存为 lokr_w1/lokr_w2。",
            ["参数效率高，常适合复杂画风或多角色训练。"],
            ["不是普通 LoRA 的低秩矩阵形式，推理侧需要 LyCORIS/LoKr 兼容加载器。"],
            ["与 OrthoLoRA、Hydra/FeRA、DoRA 互斥；小数据更要注意过拟合。"],
            "推荐使用 lokr 变体默认值：learning_rate=1e-4，factor=8。"
        ),
        lokr_factor: help(
            "LoKr 的 Kronecker 分解因子。",
            "W1 为 factor×factor，W2 为 (out/factor)×(in/factor)。",
            ["factor 越大，W2 越小，结构约束越强。"],
            ["factor 必须能整除输入/输出维度，否则运行时会自动降级。"],
            ["过大可能限制表达，过小则更接近大矩阵更新、参数量上升。"],
            "Anima DiT 默认用 8；不确定时保持默认。"
        ),
        min_rank: help(
            "T-LoRA 在低噪声时间步保留的最小活跃 rank。",
            "常用 1/2/4。rank 总量较低时不要设太低。",
            ["能减少低噪声阶段的无效更新。"],
            ["过低会牺牲细节和局部修正能力。"],
            ["与 network_dim 差距过大时，训练行为会更激进。"],
            "默认 1 适合现有 T-LoRA 变体；不稳定时升到 2 或 4。"
        ),
        alpha_rank_scale: help(
            "T-LoRA 降低活跃 rank 时是否同步缩放 alpha。",
            "通常填 1.0，保持不同时间步的有效学习率更接近。",
            ["减少 rank 变化带来的更新强度波动。"],
            ["关闭后行为更原始，但更难预测。"],
            ["错误缩放可能让某些时间步训练过强或过弱。"],
            "推荐保持 1.0。"
        ),
        use_moe_style: help(
            "选择 MoE 专家结构，false 表示不用专家路由。",
            "shared_A 是 HydraLoRA；independent_A 是 FeRA 风格。",
            ["让不同专家学习不同条件或时间步下的更新。"],
            ["参数、显存、训练时间和解释成本都高于普通 LoRA。"],
            ["专家可能坍缩到少数分支，或推理兼容性变差。"],
            "新手先用普通 LoRA；需要 Hydra/FeRA 时选择对应变体后保持默认。"
        ),
        route_per_layer: help(
            "控制路由器是每层独立，还是全模型共享。",
            "true 更细粒度；false 更稳定、更接近全局路由。",
            ["每层路由能给不同层分配不同专家偏好。"],
            ["路由器更多，训练更慢，也更容易需要均衡约束。"],
            ["小数据下 per-layer 可能过拟合或专家利用不均。"],
            "Hydra 实验变体可用 true；FeRA 通常用 false。"
        ),
        router_source: help(
            "专家路由使用的信号来源。",
            "sigma 按去噪时间步路由；fei 按 FEI 特征路由；input 按输入特征；none 关闭路由信号。",
            ["让专家分工跟时间步或内容特征绑定。"],
            ["不同来源需要不同缓存或特征，调参成本较高。"],
            ["选错来源会削弱专家分化，甚至让 MoE 只增加成本不增益。"],
            "Hydra Sigma 选 sigma；Hydra FEI/FeRA 选 fei。"
        ),
        num_experts: help(
            "MoE/Hydra/FeRA 的专家数量。",
            "常用 4；简单任务 2-4，复杂实验可到 6-8。",
            ["专家越多，容量和分工空间越大。"],
            ["显存、参数量、训练速度成本随专家数上升。"],
            ["专家过多容易数据不够分，出现空专家或不稳定路由。"],
            "默认 4；只有明确需要更多分工时再升。"
        ),
        balance_loss_weight: help(
            "专家负载均衡损失权重。",
            "MoE 变体中保持 TOML 默认；普通 LoRA 无需设置。",
            ["降低路由器只用单一专家的概率。"],
            ["过高会强迫平均分配，影响专家自然分化。"],
            ["过低可能专家坍缩，MoE 退化为单专家。"],
            "Hydra/FeRA 用默认值；除非观察到专家坍缩再调。"
        ),
        balance_loss_warmup_ratio: help(
            "训练前多少比例的步数暂不启用均衡损失。",
            "填 0.3-0.5 表示先让专家自由分化，再开始约束。",
            ["减少一开始就被强制平均导致的分工不足。"],
            ["训练早期专家可能短暂不均衡。"],
            ["太晚开启会来不及纠正专家坍缩。"],
            "MoE 变体推荐 0.4 左右；不懂就保持默认。"
        ),
        network_router_lr_scale: help(
            "路由器学习率相对主学习率的倍率。",
            "MoE/FeRA 变体按默认值填写，普通训练不要改。",
            ["让路由器更快学会分配专家。"],
            ["倍率越高越可能震荡。"],
            ["过高会让路由变化压过专家权重学习。"],
            "FeRA 默认 10；只有路由明显不动时再调整。"
        ),
        router_targets: help(
            "限制哪些线性层参与路由适配的正则表达式。",
            "常见值是 .*(mlp\\.layer[12])$，把 MoE 限在 FFN 子层。",
            ["控制 MoE 影响范围，减少显存和干扰。"],
            ["正则写错会匹配不到层，或匹配过多层。"],
            ["范围过宽可能训练慢且不稳定。"],
            "除非你清楚网络层名，否则保持变体默认。"
        ),
        sigma_feature_dim: help(
            "sigma 路由器的时间步特征维度。",
            "Hydra Sigma 默认通常 16 或 128，按变体保留。",
            ["维度更高能表达更复杂的时间步偏置。"],
            ["维度越高，路由器计算和过拟合风险略增。"],
            ["小数据集用太高维度可能没有实际收益。"],
            "使用当前 Hydra Sigma 变体默认值。"
        ),
        per_bucket_balance_weight: help(
            "每个 sigma 桶内部的额外负载均衡权重。",
            "常用 0.3，配合 num_sigma_buckets 使用。",
            ["鼓励不同时间步桶内也保持专家多样性。"],
            ["增加一项路由约束，过高会削弱专家专门化。"],
            ["与 balance_loss_weight 叠加后可能约束过强。"],
            "默认 0.3；不观察路由统计时不要频繁改。"
        ),
        num_sigma_buckets: help(
            "把时间步划分成多少个 sigma 桶。",
            "常用 3，分别近似低/中/高噪声。",
            ["让路由均衡和专家分工更贴近扩散阶段。"],
            ["桶越多，每个桶的数据越少。"],
            ["桶太多会让统计噪声变大，专家更难稳定分化。"],
            "推荐 3。"
        ),
        specialize_experts_by_sigma_buckets: help(
            "是否把专家硬分配给不同 sigma 桶。",
            "Hydra Sigma 实验中可开启；普通 MoE 不需要。",
            ["强制专家按去噪阶段分工，效果更可解释。"],
            ["减少路由自由度，可能牺牲整体最优。"],
            ["专家数和桶数不匹配时，部分专家利用会不均。"],
            "只在 Hydra Sigma 实验变体中保持默认。"
        ),
        sigma_bucket_boundaries: help(
            "自定义 sigma 桶边界。",
            "填写递增数组，长度为 num_sigma_buckets + 1，例如 [0.0, 0.5, 0.8, 1.0]。",
            ["能把专家分工压到你关心的噪声区间。"],
            ["需要理解 sigma 分布，调参成本高。"],
            ["边界不递增或长度不对会导致配置错误。"],
            "不确定就使用变体默认边界或留给代码默认。"
        ),
        add_reft: help(
            "启用 ReFT，在 DiT 块残差流上添加可训练干预。",
            "想做更强的局部/语义干预时开启，通常直接选 reft 变体。",
            ["表达力强，可和 LoRA 叠加。"],
            ["不一定能合并进普通 LoRA 推理路径，兼容性成本更高。"],
            ["过强时可能破坏底模已有能力。"],
            "需要 ReFT 时选 reft 或 tlora_ortho_reft；普通训练关闭。"
        ),
        reft_dim: help(
            "ReFT 干预秩。",
            "常用 32-64；越大越强。",
            ["提高 ReFT 干预容量。"],
            ["参数和过拟合风险增加。"],
            ["小数据用太大可能学到偶然噪声。"],
            "默认 32 或 64，跟随变体。"
        ),
        reft_alpha: help(
            "ReFT 缩放因子。",
            "通常填得和 reft_dim 一样。",
            ["让干预强度有清晰比例。"],
            ["过低收敛慢，过高干预过强。"],
            ["和 reft_dim 不匹配时更难判断实际强度。"],
            "推荐 reft_alpha = reft_dim。"
        ),
        reft_layers: help(
            "哪些 DiT 块启用 ReFT。",
            "可填 last_8、first_4、stride_2、all，或逗号分隔层号。",
            ["能控制干预位置，减少不必要的参数。"],
            ["层选得越多，成本和风险越高。"],
            ["选错层可能效果弱，或影响整体构图。"],
            "默认 last_8；想更强再扩大范围。"
        ),
        use_repa: help(
            "启用 REPA 对齐损失。",
            "只在 lora_repa 等实验变体中开启。",
            ["帮助中间表征向参考特征对齐。"],
            ["额外损失会改变训练目标。"],
            ["权重过高可能压过原本的 flow matching 损失。"],
            "普通 LoRA 关闭；实验时从 lora_repa 默认开始。"
        ),
        repa_weight: help(
            "REPA 损失权重。",
            "默认 0.5；如果压过主损失，可降到 0.1-0.25。",
            ["提高表征对齐强度。"],
            ["越高越可能牺牲生成多样性。"],
            ["过高会让模型追逐辅助目标，画面质量可能下降。"],
            "先用默认；出现主损失被压制再降低。"
        ),
        repa_layer: help(
            "应用 REPA 的 DiT 块索引。",
            "按变体默认使用；只有做方法实验才改。",
            ["可指定中间层对齐位置。"],
            ["需要理解模型层级，试错成本高。"],
            ["层选错可能几乎无收益。"],
            "默认 8。"
        ),
        repa_lr_scale: help(
            "REPA 相关头部学习率倍率。",
            "通常 1.0，表示与主学习率一致。",
            ["可单独加快或放慢辅助头学习。"],
            ["倍率越高越容易震荡。"],
            ["不当倍率会影响 REPA 稳定性。"],
            "保持 1.0。"
        ),
        layer_start: help(
            "从第几层开始应用 LoRA。",
            "0 表示从开头应用；较大值会跳过前面层。",
            ["可减少参数和低层干扰。"],
            ["跳过层越多，表达力越低。"],
            ["跳过关键早期层可能导致风格学不进去。"],
            "普通训练保持默认。"
        ),
        per_channel_scaling: help(
            "启用每通道输入预缩放。",
            "只在对应实验变体要求时开启。",
            ["可能改善部分通道尺度不均的训练。"],
            ["增加方法复杂度。"],
            ["没有配套实验时收益不确定。"],
            "不懂就保持变体默认。"
        ),
        use_ip_adapter: help(
            "启用 IP-Adapter 图像条件训练。",
            "只有 ip_adapter 变体中开启，并准备参考图像/图像条件数据。",
            ["让模型学习图像条件，而不只依赖文本。"],
            ["需要额外视觉特征缓存，训练路径不同。"],
            ["数据准备不匹配时，训练会失败或条件无效。"],
            "新手普通训练保持关闭；需要图像参考训练时选择 ip_adapter 变体后保持默认。"
        ),
        ip_image_drop_p: help(
            "训练时丢弃图像条件的概率。",
            "常用 0.1；想让模型更能无条件工作可略升。",
            ["提升模型在缺失图像条件时的鲁棒性。"],
            ["过高会削弱图像条件的绑定强度。"],
            ["图像条件本来就弱时，过高会让适配器学不到。"],
            "默认 0.1。"
        ),
        ip_features_cache_to_disk: help(
            "是否把 IP-Adapter 图像特征缓存到磁盘。",
            "数据集较大或内存有限时开启。",
            ["降低内存占用，复用预处理结果。"],
            ["增加磁盘占用和 I/O。"],
            ["缓存过期时需要重新预处理，否则会使用旧特征。"],
            "推荐 true。"
        ),
        use_easycontrol: help(
            "启用 EasyControl 图像条件方法。",
            "只有 easycontrol 变体中开启。",
            ["提供更直接的图像控制信号。"],
            ["需要专用数据集和缓存目录。"],
            ["普通 LoRA 数据目录无法直接替代 EasyControl 数据。"],
            "新手普通训练保持关闭；需要 EasyControl 时选择 easycontrol 变体后保持默认。"
        ),
        easycontrol_drop_p: help(
            "训练时丢弃 EasyControl 条件的概率。",
            "常用 0.1。",
            ["让模型不完全依赖条件图。"],
            ["过高会削弱条件控制能力。"],
            ["条件本来稀疏时会进一步降低有效训练信号。"],
            "默认 0.1。"
        ),
        easycontrol_cond_noise_max: help(
            "给 EasyControl 条件图加入噪声的最大强度。",
            "0.0 表示不加噪；想让条件变成更粗的提示时才升高。",
            ["可提升对低质量/有噪条件图的鲁棒性。"],
            ["条件越模糊，控制越弱。"],
            ["过高会让条件退化成不可靠提示。"],
            "默认 0.0；除非明确需要鲁棒性实验。"
        ),
        use_hydra: help(
            "旧式 HydraLoRA 开关。",
            "新配置优先使用 use_moe_style/router_source 三轴字段；如果旧变体出现该项，按原 TOML 保持。",
            ["兼容旧配置理解。"],
            ["和新三轴字段混用会增加心智负担。"],
            ["手动混搭可能得到未预期路由结构。"],
            "新配置不要手动添加；直接用 Hydra 变体。"
        ),
        use_sigma_router: help(
            "旧式 sigma 路由开关。",
            "新配置优先用 router_source = \"sigma\"。",
            ["表达专家随时间步变化的意图。"],
            ["旧字段与新字段共存时容易混淆。"],
            ["配置迁移不完整可能导致行为和预期不一致。"],
            "使用现有 Hydra Sigma 变体，不手动新增。"
        ),
        fera_num_bands: help(
            "FeRA 将 sigma/FEI 空间划分的带数。",
            "通常用 3。",
            ["帮助专家按阶段或特征带分工。"],
            ["带数越多，每个带的数据越少。"],
            ["过多会让路由统计不稳定。"],
            "FeRA 默认 3。"
        ),
        fei_feature_dim: help(
            "FEI 路由特征维度。",
            "按 FeRA/Hydra FEI 变体默认填写。",
            ["控制 FEI 信号输入路由器的大小。"],
            ["维度越高越复杂，收益不一定增加。"],
            ["改错可能让路由器输入不匹配。"],
            "推荐 2。"
        ),
        fei_sigma_low_div: help(
            "FEI 中低 sigma 区域的缩放除数。",
            "保持默认 4.0。",
            ["帮助平衡不同 sigma 区间的 FEI 信号。"],
            ["属于方法内部调参，直觉不强。"],
            ["随意改会改变路由分布。"],
            "推荐 4.0。"
        ),
        router_hidden_dim: help(
            "路由器隐藏层宽度。",
            "MoE/FeRA 默认 64，复杂任务可实验性上调。",
            ["提升路由器表达能力。"],
            ["参数和过拟合风险略增。"],
            ["路由器过强可能学习到数据偏差。"],
            "默认 64。"
        ),
        router_tau: help(
            "路由温度，控制专家分配尖锐程度。",
            "较低更尖锐，较高更平均。保持默认最稳。",
            ["可调节专家选择的确定性。"],
            ["偏离默认后需要观察专家利用率。"],
            ["过低会早早坍缩，过高会分工不清。"],
            "默认 0.7。"
        ),
        fera_fecl_weight: help(
            "FeRA 的 FECL 辅助损失权重。",
            "默认 0.0 表示关闭。",
            ["可用于进一步约束特征/专家关系。"],
            ["增加额外训练目标。"],
            ["非零值未经验证时可能影响主目标。"],
            "保持 0.0，除非正在复现实验。"
        ),
        learning_rate: help(
            "学习率，决定每一步参数改动有多大。",
            "普通 LoRA 常用 2e-5。ReFT、IP-Adapter、Soft Tokens 等实验方法可能用不同值，选择变体后先不要手动改。",
            ["学习率合适时，loss 和样张会比较平稳地变好。"],
            ["太低会学得很慢，训练很多轮也变化不明显。"],
            ["太高会让 loss 抖动、画面变脏，严重时训练直接跑坏。"],
            "新手普通 LoRA 从 2e-5 开始；只有样张明显欠拟合或过拟合时，再小幅调整。"
        ),
        max_train_epochs: help(
            "最大训练轮数，也就是数据集会被完整看多少遍。",
            "一轮约等于把当前数据集完整训练一遍。小数据集可以先从 4-12 轮观察；大数据集通常不需要太多轮。",
            ["轮数越多，模型越有机会学会你的角色、风格或概念。"],
            ["训练时间会变长，也会生成更多保存点和样张。"],
            ["轮数太高会过拟合，生成图越来越像训练图，泛化变差。"],
            "新手先用变体默认；样张还学不会再增加，已经像训练图照搬就降低。"
        ),
        max_train_steps: help(
            "固定训练总步数，用 step 而不是轮数来控制训练多久。",
            "默认 0 表示不启用。只有 max_train_epochs 为空时，正数 max_train_steps 才会作为训练总时长。",
            ["适合做精确实验，比如只想跑 1000 step。"],
            ["比“训练几轮”更难直观理解，因为它会受图片数量、批大小和重复次数影响。"],
            ["max_train_epochs 为空且这里也是 0 时，训练时长没有配置，启动训练会要求补一个。"],
            "新手保持 0，用最大训练轮数控制训练量。"
        ),
        train_batch_size: help(
            "每个训练 step 同时送进 GPU 的图片数量。",
            "1024 分辨率或显存紧张时保持 1。显存很充足时可尝试 2 或 4。",
            ["批大小更大时，每次更新看到的数据更多，loss 可能更平稳。"],
            ["显存占用会上升很明显，最容易触发 OOM。"],
            ["调大后每轮 step 数会变少，学习率和训练总量也要重新理解。"],
            "新手保持 1；想让有效批更大时，优先用梯度累积。"
        ),
        save_every_n_epochs: help(
            "每隔多少轮保存一次普通模型权重。",
            "它会保存可用于推理/挑版本的 .safetensors，例如第 1、2、4 轮的效果对比。它不是完整续训状态。",
            ["方便回看不同轮数效果，训练过头时还能拿较早的权重。"],
            ["保存越频繁，磁盘占用越多。"],
            ["只有普通权重时，不能完整恢复 optimizer、scheduler、随机状态等训练现场。"],
            "新手建议先设 1；训练稳定后可调到 2-5 省磁盘。"
        ),
        checkpointing_epochs: help(
            "每隔多少轮保存一次可恢复训练状态。",
            "它会成对写入 <output_name>-checkpoint.safetensors 和 <output_name>-checkpoint-state/；重新开始同一配置时会自动从这里续训。",
            ["中断后可恢复 adapter 权重、当前 step/epoch、optimizer、scheduler、随机状态等。"],
            ["只保留最近一份续训点并覆盖更新，但 checkpoint-state/ 体积可能比普通权重大。"],
            ["如果设得太大，中断时只能回到上一次续训点；如果只剩普通 .safetensors 而没有 checkpoint-state/，不能完整续训。"],
            "新手建议设 1。想减少中断损失时，让它小于或等于 save_every_n_epochs。"
        ),
        gradient_accumulation_steps: help(
            "累积多少个小批次后，再真正更新一次参数。",
            "它可以在 batch_size=1 的低显存情况下，模拟更大的有效批大小。例如 batch_size=1、累积 4，就相当于每次更新看 4 张图。",
            ["显存不变或少量增加，但训练更新更稳定。"],
            ["参数更新频率变低，训练同样轮数会更慢。"],
            ["设太高会让反馈变慢，也可能需要重新调学习率。"],
            "新手用默认；显存小但 loss 很抖时可试 2-4。"
        ),
        use_shuffled_caption_variants: help(
            "训练时使用预处理生成的 caption 打乱变体。",
            "如果预处理时生成了多 caption 变体，开启；没有则会退回单 caption。",
            ["提升对标签顺序的鲁棒性，减少死记固定 caption。"],
            ["需要先在预处理阶段生成对应缓存。"],
            ["caption 质量差时，打乱会放大噪声。"],
            "推荐 true，前提是你的 caption 本身干净。"
        ),
        caption_dropout_rate: help(
            "每个样本丢弃 caption 的概率。",
            "角色/概念 LoRA 用 0.0-0.05；画风训练用 0.1-0.25。",
            ["让风格更像无条件偏置，提示词变化时也能保持。"],
            ["会削弱 caption 对姿势、构图、细节的约束。"],
            ["太高会降低提示词服从性和多样性。"],
            "当前默认 0.1 偏风格训练；角色 LoRA 可降到 0.0-0.05。"
        ),
        optimizer_type: help(
            "优化器算法。",
            "默认 AdamW；可选 CAME、AdamW8bit、Lion、Prodigy、ProdigyPlusScheduleFree 等。",
            ["不同优化器适合不同内存和收敛偏好。"],
            ["CAME 是内存友好的自适应优化器，但通常需要重新确认学习率。"],
            ["ProdigyPlusScheduleFree 属于实验优化器；推荐 learning_rate=1.0、lr_scheduler=constant、max_grad_norm=0。"],
            ["随意切换会让历史经验不再适用。"],
            "先用 AdamW；想实验 ProdigyPlusScheduleFree 时，先用上游推荐的 constant scheduler。"
        ),
        optimizer_args: help(
            "传给优化器的额外参数。",
            "按字符串数组填写，例如 [\"fused=True\"]。",
            ["能开启 fused 等性能优化。"],
            ["依赖 PyTorch/平台支持。"],
            ["不支持的参数会导致启动失败。"],
            "保持 base 默认，除非你知道当前优化器支持该参数。"
        ),
        lr_scheduler: help(
            "学习率调度策略。",
            "constant 表示固定学习率；也可用 cosine 等调度。",
            ["调度可以让训练后期更平滑。"],
            ["多一个超参维度，需要搭配总步数理解。"],
            ["不合适的调度可能过早降低学习率。"],
            "默认 constant，先保持。"
        ),
        timestep_sampling: help(
            "训练时如何采样去噪时间步。",
            "flow matching 训练推荐 sigmoid。",
            ["让训练更关注有效时间步区间。"],
            ["改变后会影响模型学到的噪声阶段分布。"],
            ["不匹配方法假设时可能降低质量。"],
            "推荐 sigmoid。"
        ),
        discrete_flow_shift: help(
            "flow matching 噪声调度偏移参数。",
            "默认 1.0。",
            ["控制时间步/噪声分布形状。"],
            ["属于底层采样超参，调参反馈不直观。"],
            ["随意改可能让训练分布偏离推理预期。"],
            "保持 1.0。"
        ),
        sample_ratio: help(
            "每轮使用的数据比例。",
            "0.5 表示只采样一半数据；用于快速试跑。",
            ["能更快验证配置和流程。"],
            ["有效数据减少，结果不能代表完整训练。"],
            ["长期训练使用过低比例会欠拟合或偏向子集。"],
            "正式训练用 1.0 或不设置；试跑可用 half/quarter/tiny 预设。"
        ),
        sample_prompts: help(
            "训练过程中用来生成预览图的提示词。",
            "一行写一条提示词。建议放 1-4 条最能检查效果的提示词，例如角色正面、半身、不同姿势或目标画风。",
            ["不用等训练结束，就能边训练边看模型是否学对方向。"],
            ["每条提示词都会额外出图，提示词越多训练暂停采样的时间越长。"],
            ["提示词太复杂或和数据集无关，会让你误判训练效果。"],
            "新手至少写 1 条，再把按轮生成样张设为 1 或 2。"
        ),
        sample_every_n_epochs: help(
            "每隔多少轮生成一次预览图。",
            "填 1 表示每轮结束都出图；填 2 表示每 2 轮出一次；留空表示不按轮采样。",
            ["最容易理解，适合和每轮保存的权重一起对比。"],
            ["采样会打断训练一小段时间，提示词越多越慢。"],
            ["数据集很小、轮数很多时，填 1 可能生成大量样张。"],
            "新手建议填 1 或 2；长训练且只想偶尔看效果可调大。"
        ),
        sample_every_n_steps: help(
            "每隔多少训练步生成一次预览图。",
            "例如 500 表示每 500 step 出一次图；留空表示不按步采样。它适合单轮特别长的大数据集。",
            ["不用等一整轮结束，也能提前看到训练趋势。"],
            ["数值太小会频繁打断训练，整体速度明显变慢。"],
            ["step 不如 epoch 直观，初学者容易设得过密或过稀。"],
            "多数情况用按轮采样即可；只有一轮很久时再填 500、1000 这类值。"
        ),
        sample_at_first: help(
            "训练开始前先生成一组初始样张。",
            "开启后可对比训练前后变化；仍需要 sample_prompts 文件。",
            ["能确认提示词和采样链路是否正常。"],
            ["启动训练时会多等一轮采样。"],
            ["显存紧张时，首次采样也可能触发 OOM。"],
            "排查预览图是否能生成时开启；稳定训练可关闭。"
        ),
        sample_sampler: help(
            "训练中样张使用的采样器。",
            "常用 ddim、euler、euler_a、dpmsolver++。",
            ["会影响样张风格和速度。"],
            ["和最终推理采样器不同，样张观感会有差异。"],
            ["频繁切换会让训练过程对比不直观。"],
            "默认 ddim；想贴近常用推理体验可试 euler_a 或 dpmsolver++。"
        ),
        attn_mode: help(
            "注意力计算使用的后端实现。",
            "flash 通常更快、更省显存，但依赖显卡、CUDA 和 PyTorch 支持；flex 更偏兼容。",
            ["选对后端能明显影响训练速度和显存占用。"],
            ["高性能后端首次启动或编译可能更慢。"],
            ["不兼容时可能启动失败、报 CUDA 错，或速度异常变慢。"],
            "新手先用配置默认；flash 报错时再切到 flex。"
        ),
        gradient_checkpointing: help(
            "用更多计算换更低显存的训练开关。",
            "开启后，反向传播时会重新计算一部分中间结果，而不是全部存在显存里。",
            ["能明显降低显存占用，是低显存训练最常用的救命开关。"],
            ["训练会变慢，因为部分计算会做第二遍。"],
            ["和 full compile、block swap 等性能组合可能有兼容限制。"],
            "8GB/低显存推荐 true；显存充足且追求速度时可测试 false。"
        ),
        unsloth_offload_checkpointing: help(
            "把梯度检查点卸载到 CPU 内存。",
            "需要 gradient_checkpointing=true；极低显存时开启。",
            ["进一步节省 GPU 显存。"],
            ["CPU 内存和 PCIe 传输压力上升，速度下降明显。"],
            ["CPU 内存不足也会导致训练不稳定或被系统杀掉。"],
            "只有 OOM 时开启。"
        ),
        blocks_to_swap: help(
            "把多少个 DiT 模块临时放到 CPU，以减少 GPU 显存占用。",
            "0 表示尽量都放在 GPU。显存不足时可以增加，但每增加一些都会让训练更慢。",
            ["能降低 GPU 显存峰值，让低显存机器也可能跑起来。"],
            ["CPU/GPU 来回搬运会明显拖慢训练。"],
            ["设太高会慢到不实用，也可能受 CPU 内存和硬盘交换影响。"],
            "显存够用保持 0；OOM 时先用 low_vram 或 lora-8gb 预设。"
        ),
        torch_compile: help(
            "是否让 PyTorch 先编译模型计算图再训练。",
            "开启后会使用上游新的 native flatten + compile_blocks 路径。第一次启动会花时间编译；编译完成后通常更快。遇到 torch.compile/inductor 报错时可以关闭。",
            ["长时间训练时可能提高速度。"],
            ["首次启动更慢，还会在缓存目录写入编译缓存。"],
            ["block swap、梯度检查点和不同显卡驱动组合仍可能触发编译问题。"],
            "新手保持默认；如果报 torch.compile/inductor/triton 相关错误，再关闭排查。"
        ),
        compile_inductor_mode: help(
            "Inductor 编译器优化模式。",
            "default 最稳；reduce-overhead 更偏减少运行开销。",
            ["可影响 compile 后性能。"],
            ["不同环境收益不稳定。"],
            ["模式不兼容时会导致编译失败。"],
            "保持变体默认。"
        ),
        trim_crossattn_kv: help(
            "移除交叉注意力 KV 中的零填充以提升效率。",
            "默认 false；只有确认当前注意力后端支持时开启。",
            ["可能减少无效注意力计算。"],
            ["依赖后端实现细节。"],
            ["处理 padding 不当可能影响图像质量。"],
            "保持默认 false。"
        ),
        cache_llm_adapter_outputs: help(
            "把 LLM adapter 输出缓存到磁盘。",
            "Hydra/FeRA 等路由方法通常需要开启。",
            ["避免每轮重复计算文本投影，支持部分路由特征。"],
            ["占用磁盘并依赖缓存有效性。"],
            ["配置或 tokenizer 变化后旧缓存可能不匹配。"],
            "LoRA 变体通常保持 true；改文本处理后重建缓存。"
        ),
        masked_loss: help(
            "只在非遮罩区域计算损失。",
            "有 masks/merged、masks/sam 或 masks/mit 时开启。",
            ["可减少文字气泡等区域污染训练。"],
            ["需要额外生成并维护 mask。"],
            ["mask 错误会忽略本该学习的区域。"],
            "漫画/带字数据推荐 true；无 mask 或普通图集可关闭。"
        ),
        mixed_precision: help(
            "训练使用的数值精度。",
            "现代 NVIDIA GPU 优先 bf16；旧显卡不支持 bf16 时才考虑 fp16。",
            ["能降低显存占用，并提升训练吞吐。"],
            ["依赖显卡和 PyTorch 支持。"],
            ["fp16 更容易数值不稳定；bf16 在旧卡上可能不可用。"],
            "新手优先用 bf16；启动时报不支持再换 fp16。"
        ),
        vae_chunk_size: help(
            "VAE 解码/编码时的分块大小。",
            "常用 64；显存不足时降低。",
            ["越大通常越快。"],
            ["越大显存峰值越高。"],
            ["太大可能在预处理或采样时 OOM。"],
            "默认 64；OOM 时逐步降低。"
        ),
        vae_disable_cache: help(
            "禁用 VAE 内部缓存。",
            "显存紧张时保持 true。",
            ["降低 VAE 阶段显存占用。"],
            ["可能牺牲少量速度。"],
            ["关闭后预处理/采样阶段可能占更多显存。"],
            "推荐 true。"
        ),
        cache_latents: help(
            "缓存 VAE 编码后的 latent。",
            "训练前预处理/缓存流程中保持开启。",
            ["避免每轮重复编码图像。"],
            ["需要内存或磁盘保存缓存。"],
            ["图像或预处理参数变化后必须重建缓存。"],
            "推荐 true。"
        ),
        cache_latents_to_disk: help(
            "把 latent 缓存写到磁盘。",
            "数据集较大时开启。",
            ["降低 RAM 占用，训练可复用缓存。"],
            ["占磁盘，读取依赖 I/O。"],
            ["旧缓存不更新会训练到过期数据。"],
            "推荐 true。"
        ),
        cache_text_encoder_outputs: help(
            "缓存文本编码器输出。",
            "本项目延迟加载流程需要开启。",
            ["编码后可释放文本编码器，给 DiT 腾显存。"],
            ["caption 改动后需要重新缓存。"],
            ["缓存和 caption 不一致会导致训练内容不对。"],
            "推荐 true。"
        ),
        cache_text_encoder_outputs_to_disk: help(
            "把文本编码器输出写到磁盘。",
            "保持开启，配合释放文本编码器显存。",
            ["支持大数据集和低显存训练。"],
            ["占磁盘，首次预处理更久。"],
            ["tokenizer/padding 改动后旧缓存必须重建。"],
            "推荐 true。"
        ),
        skip_cache_check: help(
            "启动时跳过缓存完整性检查。",
            "确认缓存有效时可开启。",
            ["启动更快。"],
            ["不会提前发现缺失或过期缓存。"],
            ["缓存坏了可能训练中途才报错。"],
            "稳定复训可 true；刚改数据/配置时建议 false 或重建缓存。"
        ),
        use_custom_down_autograd: help(
            "使用自定义 LoRA down 矩阵反向实现。",
            "保持 base 默认。",
            ["可能降低显存或改善性能。"],
            ["属于底层优化，不方便调试。"],
            ["若遇到 autograd 异常，需要作为排错开关。"],
            "默认 true；出错时再尝试关闭。"
        ),
        log_every_n_steps: help(
            "每多少训练步记录一次日志。",
            "数值越小日志越密。",
            ["便于观察 loss 和速度变化。"],
            ["日志过密会略增 I/O 和界面刷新压力。"],
            ["太大则难以及时发现异常。"],
            "默认 2；长训可适当调大。"
        ),
        dataloader_pin_memory: help(
            "DataLoader 是否使用 pinned memory。",
            "GPU 训练通常开启。",
            ["加快 CPU 到 GPU 的数据传输。"],
            ["占用更多主机内存。"],
            ["低内存机器上可能增加系统压力。"],
            "默认 true。"
        ),
        persistent_data_loader_workers: help(
            "DataLoader worker 是否跨 epoch 常驻。",
            "多轮训练保持开启。",
            ["减少每轮重启 worker 的开销。"],
            ["会持续占用进程和内存。"],
            ["数据加载逻辑变化时，常驻 worker 不利于调试。"],
            "默认 true；调试数据加载时可关闭。"
        ),
        pretrained_model_name_or_path: help(
            "基础 DiT 模型权重路径，也就是 LoRA 要挂在哪个底模上训练。",
            "填写本机已有的 .safetensors 文件路径，通常在 models/diffusion_models 下。新建配置时可以用“填写全局路径配置”从全局设置自动带入。",
            ["决定训练结果依附的底模，路径正确是启动训练的前提。"],
            ["模型文件很大，首次下载和读取都需要时间。"],
            ["路径错会启动失败；换底模后，旧 LoRA 可能不能直接通用。"],
            "新手先在“全局设置”填好基础 DiT 路径，再回配置页自动填入。"
        ),
        qwen3: help(
            "Qwen3 文本编码器路径，用来把 caption 和提示词变成模型能理解的条件。",
            "保持下载脚本或全局设置里填写的默认路径。普通 LoRA 不需要更换文本编码器。",
            ["caption 能否被正确编码，直接影响训练内容是否学对。"],
            ["模型较大，会占用磁盘和加载时间。"],
            ["路径错误会让预处理或训练启动失败；换编码器后旧文本缓存需要重建。"],
            "新手在“全局设置”填好 Qwen3 路径后，用按钮带入当前配置。"
        ),
        vae: help(
            "VAE 模型路径，负责把图片和训练用 latent 互相转换。",
            "保持 models/vae 下的默认权重路径。普通训练不需要频繁换 VAE。",
            ["VAE 正确时，预处理缓存和训练样张才能正常生成。"],
            ["更换 VAE 后，需要重新生成 latent 缓存。"],
            ["路径错会导致预处理、训练或采样失败；旧缓存也可能不兼容。"],
            "新手使用默认 qwen_image_vae，并通过全局设置自动填入。"
        ),
        output_dir: help(
            "旧配置里的训练输出目录字段。",
            "在 WebUI 启动训练时，真实输出目录会被全局设置里的“输出文件夹”自动接管，并写入本次运行目录的 training_output。直接编辑 TOML 时仍能看到这个旧字段。",
            ["保留它可以兼容命令行、旧配置和历史 TOML。"],
            ["Web 训练里改它通常不会改变最终产物位置。"],
            ["如果以为 Web 会使用这里的路径，可能会找错权重和样张目录。"],
            "WebUI 用户去“全局设置”改输出文件夹；这里不用改。"
        ),
        output_name: help(
            "保存权重文件时使用的文件名前缀。",
            "用简短英文、数字或下划线命名，避免空格和特殊符号。例如 roleA_lora、style_test。",
            ["以后在预览图、下载权重和历史任务里更容易认出是哪次训练。"],
            ["名字太长会让文件列表难读。"],
            ["同一个运行目录里如果前缀混乱，后面挑权重会很痛苦。"],
            "新手建议写“角色或数据集简称 + 方法”，例如 rokkotsu_lora。"
        ),
        save_model_as: help(
            "模型保存格式。",
            "保持 safetensors。",
            ["加载快，格式更安全。"],
            ["与只支持其他格式的旧工具可能不兼容。"],
            ["改成不支持格式会保存失败。"],
            "推荐 safetensors。"
        ),
        save_precision: help(
            "保存权重时使用的精度。",
            "通常 bf16。",
            ["减小文件体积，匹配训练精度。"],
            ["低精度会丢失少量数值细节。"],
            ["不支持的推理环境可能需要转换。"],
            "推荐 bf16。"
        ),
        source_image_dir: help(
            "旧配置里的原始数据集路径字段。",
            "WebUI 现在主要通过“数据集”页和 dataset_config 管理数据集。这个字段会保留给旧脚本兼容，并通常镜像第 1 组数据集原始路径。",
            ["旧 TOML 和命令行流程仍能读到原始路径。"],
            ["它只能代表第 1 组，不能完整表达多数据集。"],
            ["直接改这里可能和“数据集”页保存的真实数据集配置不一致。"],
            "新手去“数据集”页填写原始数据集路径；这里保持自动同步即可。"
        ),
        resized_image_dir: help(
            "旧配置里的缩放图目录字段。",
            "WebUI 每次运行会在全局输出文件夹下创建独立 dataset_cache/dataset-xx/resized，并在运行时配置里覆盖这里。普通用户不需要手动填写。",
            ["保留它可以兼容旧配置和命令行流程。"],
            ["Web 运行时会使用本次运行目录，占用额外磁盘。"],
            ["如果手动改旧路径，可能和 Web 本次运行目录不一致，排错会更绕。"],
            "新手不要改；需要重新生成缩放图时，在 WebUI 里重新预处理。"
        ),
        lora_cache_dir: help(
            "旧配置里的 LoRA 缓存目录字段。",
            "WebUI 每次运行会在全局输出文件夹下创建独立 dataset_cache/dataset-xx/lora，并在运行时配置里覆盖这里。",
            ["缓存 latent 和文本编码后，训练阶段不用重复编码。"],
            ["缓存会占用磁盘，尤其是大数据集。"],
            ["旧缓存和新图片、新 caption 不匹配时，会训练到过期内容。"],
            "新手不要手动改；改数据或 caption 后重新预处理。"
        ),
        dataset_config: help(
            "当前训练使用的数据集配置 TOML。",
            "它记录每组数据集的原始路径、重复次数、分辨率和缓存设置。WebUI 启动训练时会生成本次运行专用的 dataset.runtime.toml。",
            ["支持多数据集，并让预处理和训练使用同一份数据布局。"],
            ["需要先在“数据集”页保存好数据集预设，再回配置页引用。"],
            ["路径错、数据集预设丢失或字段不匹配时，预处理/训练会失败。"],
            "新手先去“数据集”页建好预设，再在配置页选择它；不要手写这个路径。"
        ),
        resolution: help(
            "数据集预处理和训练使用的基础分辨率。",
            "常用 1024。显存不足时可以降低；想训练更高细节时才能谨慎升高，并准备更大的显存。",
            ["分辨率越高，模型能看到的细节越多。"],
            ["显存、训练时间、缓存体积都会明显增加。"],
            ["改分辨率后必须重新预处理，否则旧缩放图和缓存不匹配。"],
            "新手保持 1024；OOM 时先降分辨率或使用低显存预设。"
        ),
        batch_size: help(
            "数据集蓝图里的批大小。",
            "这里通常保持 1。真正影响训练显存和有效批大小的，主要是训练设置里的“批大小”和“梯度累积步数”。",
            ["保留在数据集配置里，方便兼容底层数据加载器。"],
            ["调大可能增加显存占用。"],
            ["和训练批大小一起改时，初学者容易算错总步数。"],
            "新手保持 1。"
        ),
        enable_bucket: help(
            "启用长宽比分桶。",
            "开启后，预处理会按图片长宽比选择合适的训练尺寸，把结果写入本次运行目录的数据集缓存；训练会读取这些预处理结果。",
            ["更好保留原图构图比例，避免所有图片被硬塞进同一个正方形尺寸。"],
            ["换分桶参数后，缩放图和 LoRA 缓存都需要重新生成，旧缓存不能混用。"],
            ["不会额外生成“桶文件夹”；如果以为训练还会直接读原图，可能会误判数据是否已经更新。"],
            "新手推荐开启。"
        ),
        min_bucket_reso: help(
            "允许生成的最小桶边长。",
            "桶是预处理/训练使用的一组宽高尺寸，例如 512x1024、768x768。这个值限制桶里较短边不能小于多少像素。",
            ["避免生成太小的训练尺寸，减少细节损失。"],
            ["设得太高会减少可选桶，窄图或小图可能被裁切/缩放得更明显。"],
            ["必须不大于训练分辨率，并且通常应能被桶尺寸步长整除。"],
            "默认 256；大多数训练不需要改。"
        ),
        max_bucket_reso: help(
            "允许生成的最大桶边长。",
            "这个值限制桶里较长边最多到多少像素；1024 分辨率训练通常填 1024 或跟随项目默认。",
            ["控制显存上限，避免极端长图生成过大的桶。"],
            ["设得太低会压缩长边细节。"],
            ["不能小于训练分辨率，否则训练端会报错。"],
            "1024 训练保持 1024；更高分辨率训练再同步调高。"
        ),
        bucket_reso_steps: help(
            "桶尺寸之间的间隔。",
            "填 64 表示桶尺寸按 64 像素递增，例如 512、576、640。数值越小，可选桶越多，越贴近原图比例。",
            ["桶更多时图片变形和 padding 更少。"],
            ["桶更多会让预处理/缓存分组更碎，首次预处理和训练加载更复杂。"],
            ["训练代码要求这个值能被 16 整除。"],
            "默认 64；想更精细可试 32，但不建议新手改。"
        ),
        bucket_no_upscale: help(
            "不要把小图放大到桶尺寸。",
            "开启后，小图尽量不被强行放大；关闭时会按桶规则缩放到目标尺寸。当前预处理脚本生成缩放图后，训练会直接使用这些结果。",
            ["避免小图被放大后变糊。"],
            ["不同尺寸会更分散，训练批次可能更碎。"],
            ["数据分辨率参差不齐时，效果和速度更难预估。"],
            "默认关闭；小图很多且不想放大时再开启。"
        ),
        validation_split: help(
            "从数据集中拿出一小部分图片不参与训练，用来做训练效果检查。",
            "填 0-1 的小数。例: 0.025 表示 1000 张里约留 25 张做验证；0 表示不按比例预留。",
            ["可以用同一批没训练过的图观察是否过拟合，而不是只看训练图。"],
            ["被留作验证的图片不会参与训练，比例越大训练可用图片越少。"],
            ["小数据集用比例可能忽多忽少，比如 30 张 x 0.025 约等于 1 张。"],
            "新手保持 0.025；小数据集更建议用下面的固定验证数量。"
        ),
        validation_split_num: help(
            "直接指定要留出多少张图片做验证。",
            "填 0 表示不用固定数量，改用上面的验证集比例；填 4 就固定留 4 张，填 16 就固定留 16 张。",
            ["小数据集时比比例更直观，不会出现比例算出来只有 0-1 张的情况。"],
            ["这些图片会从训练集中扣掉，不参与训练。"],
            ["数量填太大时，真正用于训练的图片会变少，模型可能学不充分。"],
            "少量角色图可填 2-6；几十到几百张可填 8-16；不确定就保持 0。"
        ),
        validation_seed: help(
            "决定哪几张图片会被选进验证集的随机编号。",
            "同样的数据集、验证比例/数量和 seed，会每次选中同一批验证图片；换 seed 就会换一批。",
            ["方便公平比较不同参数，因为验证用的是同一批图片。"],
            ["想重新抽一批验证图时才需要改它。"],
            ["对比实验时如果 seed 变了，效果差异可能只是验证图片换了。"],
            "默认 42；做参数对比时保持不变。"
        ),
        caption_extension: help(
            "caption 标注文件扩展名。",
            "常用 .txt；如果标注文件是 .caption 等格式就改成对应后缀。",
            ["让数据加载器找到图片同名标注。"],
            ["支持不同标注文件命名习惯。"],
            ["扩展名填错会导致读不到 caption。"],
            "默认 .txt。"
        ),
        keep_tokens: help(
            "Caption 前部固定保留的 token 数。",
            "用于 caption shuffle/dropout 时保留触发词或关键主体词。",
            ["减少关键触发词被打乱或丢弃。"],
            ["保留太多会降低 shuffle 的随机化效果。"],
            ["caption 开头不是关键 tag 时，保留意义会变弱。"],
            "默认 3。"
        ),
        logging_dir: help(
            "训练日志目录，主要给 TensorBoard 和历史曲线使用。",
            "WebUI 启动训练时会把日志写到本次运行目录的 model_cache/logs。旧 TOML 里的值保留用于命令行兼容。",
            ["便于查看 loss、速度和历史训练曲线。"],
            ["长时间训练会积累日志文件，占用少量磁盘。"],
            ["手动把多个实验指到同一日志目录，会让曲线混在一起难分辨。"],
            "WebUI 用户不用改；要换根目录请去“全局设置”改输出文件夹。"
        ),
        log_with: help(
            "训练日志后端。",
            "通常填 tensorboard。",
            ["能接入可视化曲线。"],
            ["需要对应依赖和日志目录。"],
            ["填错后可能没有可视化日志。"],
            "推荐 tensorboard。"
        ),
    };

    // ── 初始化 ──
    document.addEventListener('DOMContentLoaded', async () => {
        initThemeToggle();
        setupTabs();
        lossChart = new MetricsChart(document.getElementById('loss-chart'));
        lossChart.setTheme(chartTheme());
        setupEventListeners();
        initGpuPickerEvents();
        await loadInitialData();
        if (location.protocol !== 'file:') {
            connectWebSocket();
            pollStatus();
            setInterval(pollStatus, 10000);
            setInterval(refreshTrainingHealth, 1000);
        }
    });

    function currentTheme() {
        return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
    }

    function storedTheme() {
        try {
            return localStorage.getItem(THEME_STORAGE_KEY);
        } catch (_) {
            return null;
        }
    }

    function saveTheme(theme) {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, theme);
        } catch (_) {
            // 忽略浏览器禁用本地存储的情况，当前页面仍然可以完成切换。
        }
    }

    function applyTheme(theme) {
        const safeTheme = theme === 'light' ? 'light' : 'dark';
        document.documentElement.dataset.theme = safeTheme;
        const toggle = document.getElementById('theme-toggle');
        const label = document.getElementById('theme-toggle-text');
        if (toggle) {
            const isLight = safeTheme === 'light';
            toggle.setAttribute('aria-pressed', String(isLight));
            toggle.title = isLight ? '切换到深色主题' : '切换到浅色主题';
        }
        if (label) label.textContent = safeTheme === 'light' ? '深色主题' : '浅色主题';
        lossChart?.setTheme?.(chartTheme());
    }

    function initThemeToggle() {
        applyTheme(storedTheme() || currentTheme());
        const toggle = document.getElementById('theme-toggle');
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            const next = currentTheme() === 'light' ? 'dark' : 'light';
            applyTheme(next);
            saveTheme(next);
        });
    }

    function loadStoredGpuWhitelist() {
        try {
            const parsed = JSON.parse(localStorage.getItem(GPU_WHITELIST_STORAGE_KEY) || '[]');
            if (!Array.isArray(parsed)) return [];
            return parsed
                .map((item) => Number(item))
                .filter((item, index, list) => Number.isInteger(item) && item >= 0 && list.indexOf(item) === index);
        } catch (_) {
            return [];
        }
    }

    function saveGpuWhitelist() {
        try {
            localStorage.setItem(GPU_WHITELIST_STORAGE_KEY, JSON.stringify(selectedGpuWhitelist));
        } catch (_) {
            // 浏览器禁用 localStorage 时，本次页面内选择仍然有效。
        }
    }

    async function loadGpuOptions() {
        selectedGpuWhitelist = loadStoredGpuWhitelist();
        renderGpuPicker();
        if (location.protocol === 'file:') {
            updateGpuPickerNote('静态打开无法读取本机 GPU；选择会在服务模式下生效。');
            return;
        }
        try {
            const payload = await api('/api/training/gpus');
            availableGpus = Array.isArray(payload.gpus) ? payload.gpus : [];
            selectedGpuWhitelist = sanitizeGpuWhitelist(selectedGpuWhitelist);
            saveGpuWhitelist();
            renderGpuPicker();
        } catch (e) {
            availableGpus = [];
            renderGpuPicker();
            updateGpuPickerNote('读取 GPU 列表失败，训练会使用默认可见 GPU。');
        }
    }

    function sanitizeGpuWhitelist(list) {
        const selected = Array.isArray(list) ? list : [];
        if (!availableGpus.length) return selected.filter((item) => Number.isInteger(item) && item >= 0);
        const known = new Set(availableGpus.map((gpu) => Number(gpu.index)));
        return selected.filter((item) => known.has(item));
    }

    function renderGpuPicker() {
        const toggle = document.getElementById('gpu-picker-toggle');
        const list = document.getElementById('gpu-option-list');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!toggle || !list || !allCheckbox) return;

        const selected = new Set(selectedGpuWhitelist);
        const allSelected = selectedGpuWhitelist.length === 0;
        allCheckbox.checked = allSelected;
        allCheckbox.indeterminate = false;
        allCheckbox.disabled = allSelected;
        toggle.textContent = gpuPickerSummary();
        toggle.title = gpuPickerTitle();
        list.innerHTML = '';

        if (!availableGpus.length) {
            const empty = document.createElement('div');
            empty.className = 'gpu-picker-note';
            empty.textContent = '未读取到 NVIDIA GPU；保持“全部 GPU”时会沿用系统默认可见设备。';
            list.appendChild(empty);
            updateGpuPickerNote('选择为空表示不限制 GPU，训练使用系统默认可见设备。');
            return;
        }

        for (const gpu of availableGpus) {
            const index = Number(gpu.index);
            const option = document.createElement('label');
            option.className = 'gpu-option';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = String(index);
            checkbox.checked = selected.has(index);
            checkbox.addEventListener('change', () => toggleGpuSelection(index, checkbox.checked));

            const body = document.createElement('span');
            const name = document.createElement('span');
            name.className = 'gpu-option-name';
            name.textContent = gpu.label || `GPU ${index} · ${gpu.name || '未命名显卡'}`;
            body.appendChild(name);
            const meta = document.createElement('span');
            meta.className = 'gpu-option-meta';
            meta.textContent = gpu.memory_total_gb
                ? `显存 ${gpu.memory_total_gb} GB · 训练时写入 CUDA_VISIBLE_DEVICES=${index}`
                : `训练时写入 CUDA_VISIBLE_DEVICES=${index}`;
            body.appendChild(meta);
            option.append(checkbox, body);
            list.appendChild(option);
        }

        updateGpuPickerNote(allSelected
            ? '当前不限制 GPU，训练会使用系统默认可见设备。'
            : `当前训练白名单: ${selectedGpuWhitelist.join(', ')}`);
    }

    function gpuPickerSummary() {
        if (!selectedGpuWhitelist.length) return '全部 GPU';
        const names = selectedGpuWhitelist.map((index) => {
            const gpu = availableGpus.find((item) => Number(item.index) === Number(index));
            return gpu?.name ? `GPU ${index} · ${gpu.name}` : `GPU ${index}`;
        });
        if (names.length <= 2) return names.join(' / ');
        return `${names.slice(0, 2).join(' / ')} 等 ${names.length} 张`;
    }

    function gpuPickerTitle() {
        return [
            '选择训练时允许使用的 GPU 白名单。',
            '留空/全部 GPU 表示不覆盖系统默认可见设备。',
            '选择会保存在本机浏览器，并在开始训练或自动预处理后训练时生效。',
        ].join('\n');
    }

    function updateGpuPickerNote(text) {
        const note = document.getElementById('gpu-picker-note');
        if (note) note.textContent = text;
    }

    function setGpuWhitelist(next) {
        selectedGpuWhitelist = sanitizeGpuWhitelist(next);
        saveGpuWhitelist();
        renderGpuPicker();
    }

    function toggleGpuSelection(index, checked) {
        const selected = new Set(selectedGpuWhitelist);
        if (checked) selected.add(index);
        else selected.delete(index);
        setGpuWhitelist([...selected].sort((a, b) => a - b));
    }

    function selectedGpuPayload() {
        return selectedGpuWhitelist.slice().sort((a, b) => a - b);
    }

    function closeGpuPickerPanel() {
        const panel = document.getElementById('gpu-picker-panel');
        const toggle = document.getElementById('gpu-picker-toggle');
        if (!panel || !toggle) return;
        panel.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
    }

    function initGpuPickerEvents() {
        const picker = document.getElementById('gpu-picker');
        const toggle = document.getElementById('gpu-picker-toggle');
        const panel = document.getElementById('gpu-picker-panel');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!picker || !toggle || !panel || !allCheckbox) return;
        toggle.addEventListener('click', () => {
            const nextOpen = panel.hidden;
            panel.hidden = !nextOpen;
            toggle.setAttribute('aria-expanded', String(nextOpen));
        });
        allCheckbox.addEventListener('change', () => setGpuWhitelist([]));
        document.addEventListener('click', (event) => {
            if (!picker.contains(event.target)) closeGpuPickerPanel();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeGpuPickerPanel();
        });
    }

    function chartTheme() {
        const styles = getComputedStyle(document.documentElement);
        return {
            color: styles.getPropertyValue('--accent').trim() || '#4fc3f7',
            grid: styles.getPropertyValue('--chart-grid').trim() || '#2a3a5e',
            text: styles.getPropertyValue('--text-dim').trim() || '#8892a4',
            tooltipBg: styles.getPropertyValue('--bg-card').trim() || '#16213e',
            tooltipBorder: styles.getPropertyValue('--border').trim() || '#2a3a5e',
            tooltipText: styles.getPropertyValue('--text').trim() || '#e0e0e0',
            highlight: styles.getPropertyValue('--warning').trim() || '#f0c36a',
            crosshair: styles.getPropertyValue('--accent').trim() || '#4fc3f7',
        };
    }

    function isHistoryReviewMode() {
        return historyViewMode !== 'live' || Boolean(viewingHistoryTaskId);
    }

    function openTutorialDialog() {
        const dialog = document.getElementById('tutorial-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }

    // ── Tab 切换 ──
    function setupTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
                if (btn.dataset.tab === 'datasets') {
                    loadDatasetPresets();
                }
                if (btn.dataset.tab === 'training' && lossChart?.resize) {
                    lossChart.resize();
                }
                if (btn.dataset.tab === 'preview') {
                    loadPreviewImages();
                }
                if (btn.dataset.tab === 'settings') {
                    loadGlobalSettings();
                }
            });
        });
    }

    // ── 加载初始数据 ──
    async function loadInitialData() {
        if (location.protocol === 'file:') {
            await loadGpuOptions();
            showStandaloneWarning();
            return;
        }
        try {
            const [methods, presets, help] = await Promise.all([
                api('/api/methods'),
                api('/api/presets'),
                api('/api/config/field-help'),
            ]);
            fieldHelp = help;
            populateSelect('method-select', methods, 'lora');
            populateSelect('preset-select', presets, 'default');
            await loadGpuOptions();
            await loadVariants();
            await loadDatasetPresets({ selectCurrent: false });
            await loadConfig();
            await loadTomlFileList();
            rememberSelectionSnapshot();
            await loadTrainingQueue();
            await loadTrainingHistoryList();
            await loadPreviewSettings();
            await loadGlobalSettings();
            returnToLiveTraining({ refresh: false });
        } catch (e) {
            console.error('初始化失败:', e);
        }
    }

    function showStandaloneWarning() {
        const form = document.getElementById('config-form');
        form.innerHTML = '';
        const panel = document.createElement('div');
        panel.className = 'standalone-warning';
        panel.innerHTML = [
            '<strong>当前是 file:// 静态打开模式，无法读取或保存项目配置。</strong>',
            '<p>请在项目根目录启动 Web 服务后访问 <code>http://127.0.0.1:20102/</code>：</p>',
            '<pre>.venv/bin/python -m web --host 127.0.0.1 --port 20102</pre>',
        ].join('');
        form.appendChild(panel);
        setTomlStatus('error', '静态打开没有后端 API，保存/另存为/读取配置不可用', { persist: true });
        setPreviewEmpty('静态打开没有后端 API，无法读取项目预览图。');
    }

    async function loadVariants({ reset = false } = {}) {
        const method = val('method-select');
        const variants = await api(`/api/methods/${method}/variants`);
        populateSelect('variant-select', variants, reset ? (variants[0] || method) : method);
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
    }

    async function loadConfig() {
        const requestSeq = ++configLoadSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        if (!variant) return;
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const data = await api(`/api/config/merged?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}`);
        if (requestSeq !== configLoadSeq) return;
        if (data?.ok === false) {
            setTomlStatus('error', data.error || '读取配置失败');
            return;
        }
        currentConfig = data;
        selectedConfigDatasetFile = currentConfig.dataset_config || '';
        selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
        renderConfigForm(currentConfig);
        renderContinueTrainingSource();
        if (continueTrainingSource?.abs_path) {
            await refreshContinueTrainingSourceCompatibility();
        }
        if (samplePromptsMode === 'editor-file') {
            loadSamplePrompts(samplePromptsPath, requestSeq);
        } else {
            samplePromptsLoadSeq += 1;
        }
        loadStepEstimate(requestSeq);
        updateChoiceGuide();
        // 同步加载对应的 TOML 文件到右侧编辑器
        const tomlFile = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlFiles.includes(tomlFile) && currentTomlFile !== tomlFile) {
            loadTomlFile(tomlFile, { force: true });
        }
    }

    async function reloadCurrentConfig() {
        if (!(await confirmDiscardTomlChanges('当前配置有未保存修改，刷新会重新读取表单和数据集设置并丢弃这些修改。是否继续？'))) {
            return;
        }
        await loadConfig();
        rememberSelectionSnapshot();
    }

    // ── 配置表单渲染 ──
    function renderConfigForm(config) {
        const container = document.getElementById('config-form');
        container.innerHTML = '';

        const fieldsByKey = {};
        for (const [key, value] of Object.entries(config)) {
            if (key === 'output_dir') continue;
            if (key === 'general' || key === 'datasets') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) continue;
            fieldsByKey[key] = value;
        }
        for (const [key, value] of Object.entries(FORM_UI_DEFAULTS)) {
            if (key === 'output_dir') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (!shouldExposeUiDefaultField(key, config, fieldsByKey)) continue;
            if (!(key in fieldsByKey)) fieldsByKey[key] = value;
        }
        applyNetworkArgFields(fieldsByKey, config);
        fieldsByKey.sample_prompts = currentSamplePromptText(config);

        const consumed = new Set();
        for (const section of FORM_SECTION_DEFS) {
            if (!shouldRenderConfigSection(section, config)) continue;
            const fields = collectSectionFields(fieldsByKey, section.keys, consumed);
            if (fields.length > 0) {
                container.appendChild(createGroup(
                    section.title,
                    fields,
                    section.open,
                    section.className || '',
                    section.description || ''
                ));
            }
        }

        const otherFields = Object.entries(fieldsByKey).filter(([key]) => !consumed.has(key));
        if (otherFields.length > 0) {
            container.appendChild(createGroup(
                '其他高级选项',
                otherFields,
                false,
                '',
                '未归类的新字段或低频字段；保留给高级调试使用。'
            ));
        }
    }

    function shouldRenderConfigSection(section, config = currentConfig) {
        if (!section?.method) return true;
        return activeMethodKey(config) === section.method;
    }

    function shouldExposeUiDefaultField(key, config, fieldsByKey = {}) {
        if (key in fieldsByKey) return true;
        if (NETWORK_ARG_FIELD_MAP.has(key)) return false;
        const family = activeMethodKey(config);
        if (SPD_UI_DEFAULT_FIELDS.has(key)) return family === 'spd';
        if (CHIMERA_UI_DEFAULT_FIELDS.has(key)) return family === 'chimera';
        if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key)) return family === 'ip_adapter';
        if (SOFT_TOKENS_UI_DEFAULT_FIELDS.has(key)) return family === 'soft_tokens';
        return true;
    }

    function applyNetworkArgFields(fieldsByKey, config) {
        const specs = activeNetworkArgSpecs(config);
        if (!specs.length) return;
        const argMap = parseNetworkArgMap(config?.network_args);
        for (const spec of specs) {
            const rawValue = argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default;
            fieldsByKey[spec.key] = coerceNetworkArgValue(rawValue, spec);
        }
    }

    function isActiveNetworkArgFieldKey(key, config = currentConfig) {
        return activeNetworkArgSpecs(config).some((spec) => spec.key === key);
    }

    function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    function activeNetworkArgSpecs(config = currentConfig) {
        const families = activeNetworkArgFamilies(config);
        const argMap = parseNetworkArgMap(config?.network_args);
        return NETWORK_ARG_FIELD_SPECS.filter((spec) =>
            families.has(spec.family) || argMap.has(spec.arg)
        );
    }

    function activeNetworkArgFamilies(config = currentConfig) {
        const families = new Set();
        const moduleName = String(config?.network_module || '');
        const method = activeMethodKey(config);
        if (method === 'soft_tokens' || moduleName.includes('soft_tokens')) families.add('soft_tokens');
        if (method === 'ip_adapter' || isTruthy(config?.use_ip_adapter) || moduleName.includes('ip_adapter')) {
            families.add('ip_adapter');
        }
        if (method === 'easycontrol' || isTruthy(config?.use_easycontrol) || moduleName.includes('easycontrol')) {
            families.add('easycontrol');
        }
        return families;
    }

    function parseNetworkArgMap(networkArgs) {
        const map = new Map();
        for (const raw of normalizeNetworkArgArray(networkArgs)) {
            const parsed = parseNetworkArgEntry(raw);
            if (parsed) map.set(parsed.arg, parsed.value);
        }
        return map;
    }

    function normalizeNetworkArgArray(networkArgs) {
        if (Array.isArray(networkArgs)) return networkArgs.map((item) => String(item));
        if (typeof networkArgs === 'string' && networkArgs.trim()) return parseArrayValue(networkArgs).map((item) => String(item));
        return [];
    }

    function parseNetworkArgEntry(raw) {
        const text = String(raw || '').trim();
        const splitAt = text.indexOf('=');
        if (splitAt <= 0) return null;
        const arg = text.slice(0, splitAt).trim();
        if (!arg) return null;
        return {
            arg,
            value: stripNetworkArgQuotes(text.slice(splitAt + 1).trim()),
            raw: text,
        };
    }

    function stripNetworkArgQuotes(value) {
        const text = String(value || '').trim();
        if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
            return text.slice(1, -1);
        }
        return text;
    }

    function coerceNetworkArgValue(value, spec) {
        if (spec.valueType === 'boolean' || spec.valueType === 'booleanInt') {
            return parseBooleanNetworkArg(value, spec.default);
        }
        if (spec.valueType === 'integer') {
            const n = Number(value);
            return Number.isFinite(n) ? Math.trunc(n) : spec.default;
        }
        if (spec.valueType === 'number') {
            const n = Number(value);
            return Number.isFinite(n) ? n : spec.default;
        }
        return String(value ?? spec.default ?? '');
    }

    function parseBooleanNetworkArg(value, fallback = false) {
        if (typeof value === 'boolean') return value;
        if (value === 1 || value === 0) return Boolean(value);
        const text = String(value ?? '').trim().toLowerCase();
        if (['1', 'true', 'yes', 'on'].includes(text)) return true;
        if (['0', 'false', 'no', 'off'].includes(text)) return false;
        return Boolean(fallback);
    }

    async function loadStepEstimate(parentSeq = configLoadSeq) {
        const requestSeq = ++stepEstimateSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            currentStepEstimate = null;
            updateStepEstimatePanel();
            return;
        }
        try {
            const datasetParam = selectedConfigDatasetFile ? `&dataset_config=${encodeURIComponent(selectedConfigDatasetFile)}` : '';
            const data = await api(`/api/config/steps?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}${datasetParam}`);
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            currentStepEstimate = data?.ok === false ? null : data;
        } catch {
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            currentStepEstimate = null;
        }
        updateStepEstimatePanel();
    }

    async function loadDatasetEditor(parentSeq = configLoadSeq) {
        const requestSeq = ++datasetLoadSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                error: 'SPD 是 CLI 实验配置，不使用 Web 数据集编辑器。',
            };
            renderDatasetEditor();
            return;
        }
        datasetEditorState.loading = true;
        datasetEditorState.error = '';
        renderDatasetEditor();
        try {
            const data = await api(`/api/config/datasets?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}`);
            if (parentSeq !== configLoadSeq || requestSeq !== datasetLoadSeq) return;
            if (!data.ok) {
                throw new Error(data.error || '读取数据集配置失败');
            }
            datasetEditorState = {
                loading: false,
                loaded: true,
                dirty: false,
                dataset_config: data.dataset_config || '',
                datasets: normalizeDatasetEditorRows(data.datasets || []),
                defaults: normalizeDatasetDefaults(data.defaults || {}),
                error: '',
            };
        } catch (e) {
            if (parentSeq !== configLoadSeq || requestSeq !== datasetLoadSeq) return;
            datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                error: e.message || '读取数据集配置失败',
            };
        }
        renderDatasetEditor();
    }

    async function loadDatasetPresets(options = {}) {
        if (location.protocol === 'file:') return;
        const requestSeq = ++datasetPresetLoadSeq;
        datasetPresetState.loading = true;
        renderDatasetPresetList();
        try {
            const data = await api('/api/config/dataset-presets');
            if (requestSeq !== datasetPresetLoadSeq) return;
            if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
            const presets = (Array.isArray(data.presets) ? data.presets : [])
                .filter((preset) => !HIDDEN_DATASET_PRESET_FILES.has(preset.path));
            datasetPresetState.presets = presets;
            datasetPresetState.loading = false;
            datasetPresetState.error = '';
            const preserveDirtySelection = datasetPresetState.dirty;
            const selectedDatasetVisible = presets.some((preset) => preset.path === datasetPresetState.selectedFile);
            if (!selectedDatasetVisible && !preserveDirtySelection) {
                datasetPresetState.selectedFile = '';
            }
            if (!preserveDirtySelection && options.selectCurrent !== false && selectedConfigDatasetFile && !datasetPresetState.selectedFile && presets.some((preset) => preset.path === selectedConfigDatasetFile)) {
                datasetPresetState.selectedFile = selectedConfigDatasetFile;
            }
            if (!preserveDirtySelection && !datasetPresetState.selectedFile && presets.length) {
                datasetPresetState.selectedFile = presets[0].path;
            }
            selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
            renderConfigDatasetPicker();
            renderDatasetPresetList();
            renderDatasetPresetHeader();
            if (datasetPresetState.selectedFile && !datasetPresetState.dirty) {
                await loadDatasetPreset(datasetPresetState.selectedFile);
            } else {
                renderDatasetEditor();
            }
        } catch (e) {
            if (requestSeq !== datasetPresetLoadSeq) return;
            datasetPresetState.loading = false;
            datasetPresetState.error = e.message || '读取数据集预设失败';
            renderDatasetPresetList();
            renderDatasetPresetHeader();
        }
    }

    async function loadDatasetPreset(file) {
        if (!file) return;
        if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，切换会丢弃这些修改。是否继续？'))) {
            renderDatasetPresetList();
            return;
        }
        datasetPresetState.selectedFile = file;
        datasetPresetState.loading = true;
        datasetPresetState.error = '';
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
        try {
            const data = await api(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
            if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
            datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                dirty: false,
                isNew: false,
                selectedFile: data.file || file,
                datasets: normalizeDatasetEditorRows(data.datasets || []),
                defaults: normalizeDatasetDefaults(data.defaults || {}),
                readonly: Boolean(data.readonly || data.meta?.locked),
                error: '',
                status: '',
            };
        } catch (e) {
            datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                error: e.message || '读取数据集预设失败',
            };
        }
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    function createStepEstimatePanel() {
        const panel = document.createElement('div');
        panel.id = 'step-estimate-panel';
        panel.className = 'step-estimate-panel';
        panel.innerHTML = [
            '<div class="step-estimate-title">预计训练步数</div>',
            '<div class="step-estimate-grid">',
            '<div><span>数据集</span><strong id="step-dataset-count">-</strong></div>',
            '<div><span>训练图片</span><strong id="step-train-images">-</strong></div>',
            '<div><span>重复后样本</span><strong id="step-repeated-images">-</strong></div>',
            '<div><span>有效批大小</span><strong id="step-effective-batch">-</strong></div>',
            '<div><span>每轮步数</span><strong id="step-per-epoch">-</strong></div>',
            '<div><span>总步数</span><strong id="step-total">-</strong></div>',
            '</div>',
            '<div id="step-dataset-breakdown" class="step-dataset-breakdown"></div>',
            '<p id="step-estimate-note" class="step-estimate-note"></p>',
        ].join('');
        return panel;
    }

    function updateStepEstimatePanel() {
        const panel = document.getElementById('step-estimate-panel');
        if (!panel || !currentStepEstimate) return;

        const epochs = readOptionalLiveNumber('max_train_epochs');
        const batchSize = readLiveNumber('train_batch_size', currentStepEstimate.train_batch_size || 1);
        const gradAccum = readLiveNumber('gradient_accumulation_steps', currentStepEstimate.gradient_accumulation_steps || 1);
        const sampleRatio = readLiveNumber('sample_ratio', currentStepEstimate.sample_ratio || 1);
        const maxTrainSteps = readNonnegativeLiveNumber('max_train_steps', currentStepEstimate.max_train_steps ?? 0);
        const datasets = liveDatasetRowsForEstimate();
        const trainImages = datasets.reduce((sum, row) => sum + Number(row.train_image_count || 0), 0);
        const weightedImages = datasets.reduce((sum, row) => sum + (Number(row.train_image_count || 0) * Number(row.num_repeats || 1)), 0);
        const effectiveBatch = Math.max(1, batchSize * gradAccum);
        const repeatedImages = Math.max(0, Math.floor(weightedImages * sampleRatio));
        const stepsPerEpoch = repeatedImages ? Math.ceil(repeatedImages / effectiveBatch) : 0;
        const durationMode = epochs ? 'epochs' : (maxTrainSteps > 0 ? 'steps' : 'unset');
        const totalSteps = durationMode === 'epochs' ? stepsPerEpoch * epochs : maxTrainSteps;

        setText('step-dataset-count', String(datasets.length || 0));
        setText('step-train-images', String(trainImages));
        setText('step-repeated-images', `${repeatedImages} = ${weightedImages} x ${sampleRatio}`);
        setText('step-effective-batch', `${effectiveBatch} = ${batchSize} x ${gradAccum}`);
        setText('step-per-epoch', String(stepsPerEpoch));
        setText('step-total', durationMode === 'unset' ? '未配置' : String(totalSteps));
        renderStepDatasetBreakdown(datasets);
        const note = durationMode === 'epochs'
            ? `公式: Σ(每组训练图片 x 重复次数) x sample_ratio / (train_batch_size x gradient_accumulation_steps) x max_train_epochs。max_train_epochs 已设置，max_train_steps 此时不生效。`
            : (durationMode === 'steps'
                ? `当前未设置 max_train_epochs，训练将直接按 max_train_steps=${maxTrainSteps} 作为固定总步数运行。若填写 epoch，则会按每轮步数重新推导总步数。`
                : `当前未设置 max_train_epochs，且 max_train_steps=0 表示不启用固定步数。启动训练前需要设置最大训练轮数，或把最大训练步数填成正数。`);
        setText('step-estimate-note', note);
    }

    function liveDatasetRowsForEstimate() {
        const baseRows = Array.isArray(currentStepEstimate?.datasets) ? currentStepEstimate.datasets : [];
        return baseRows.length ? baseRows : [{
            index: 1,
            source_dir: currentStepEstimate?.source_dir || '',
            image_dir: currentStepEstimate?.resized_dir || '',
            cache_dir: currentStepEstimate?.lora_cache_dir || '',
            source_image_count: currentStepEstimate?.source_image_count || 0,
            resized_image_count: currentStepEstimate?.resized_image_count || 0,
            train_image_count: currentStepEstimate?.train_image_count || 0,
            num_repeats: currentStepEstimate?.dataset_num_repeats || 1,
            weighted_image_count: currentStepEstimate?.weighted_image_count || 0,
            uses_preprocessed_images: currentStepEstimate?.uses_preprocessed_images || false,
        }];
    }

    function renderStepDatasetBreakdown(datasets) {
        const container = document.getElementById('step-dataset-breakdown');
        if (!container) return;
        container.innerHTML = '';
        if (!datasets.length) {
            const empty = document.createElement('div');
            empty.className = 'step-dataset-row muted';
            empty.textContent = '还没有可估算的数据集。';
            container.appendChild(empty);
            return;
        }
        for (const row of datasets) {
            const item = document.createElement('div');
            item.className = 'step-dataset-row';
            const trainCount = Number(row.train_image_count || 0);
            const repeats = Number(row.num_repeats || 1);
            const weighted = trainCount * repeats;
            const source = row.uses_preprocessed_images ? '缩放图' : '原始图';
            item.innerHTML = [
                `<strong>第 ${row.index || 1} 组</strong>`,
                `<span>${source} ${trainCount} 张 x 重复 ${repeats} = ${weighted} 样本</span>`,
                `<code>${escapeHtml(row.source_dir || row.image_dir || '-')}</code>`,
            ].join('');
            container.appendChild(item);
        }
    }

    function readLiveNumber(key, fallback) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return Number(fallback) || 0;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const n = Number(raw);
        return Number.isFinite(n) && n > 0 ? n : (Number(fallback) || 0);
    }

    function readNonnegativeLiveNumber(key, fallback = 0) {
        const fallbackNumber = Math.max(0, Number(fallback) || 0);
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return fallbackNumber;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return fallbackNumber;
        const n = Number(trimmed);
        return Number.isFinite(n) && n >= 0 ? n : fallbackNumber;
    }

    function readOptionalLiveNumber(key) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return null;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return null;
        const n = Number(trimmed);
        return Number.isFinite(n) && n > 0 ? n : null;
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function createGroup(name, fields, open, extraClass = '', description = '') {
        const section = document.createElement('section');
        section.className = ['config-group', extraClass].filter(Boolean).join(' ');

        const header = document.createElement('div');
        header.className = 'config-group-title';
        const title = document.createElement('span');
        title.textContent = `${name} (${fields.length} 项)`;
        header.appendChild(title);

        const content = document.createElement('div');
        content.className = 'config-group-body';
        let hint = null;
        if (description) {
            const hintId = `config-group-hint-${++configGroupHintSeq}`;
            const btn = document.createElement('button');
            btn.className = 'info-toggle config-group-info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '展开分组说明';
            btn.setAttribute('aria-label', `${name} 说明`);
            btn.setAttribute('aria-controls', hintId);
            btn.setAttribute('aria-expanded', 'false');
            header.appendChild(btn);

            hint = document.createElement('p');
            hint.className = 'config-group-hint';
            hint.id = hintId;
            hint.hidden = true;
            hint.textContent = description;
            btn.addEventListener('click', () => {
                const nextVisible = hint.hidden;
                hint.hidden = !nextVisible;
                btn.classList.toggle('active', nextVisible);
                btn.setAttribute('aria-expanded', String(nextVisible));
                btn.title = nextVisible ? '收起分组说明' : '展开分组说明';
            });
            content.appendChild(hint);
        }
        if (extraClass === 'config-group-model') {
            header.appendChild(createFillGlobalModelPathsButton());
        }
        section.appendChild(header);
        if (extraClass === 'config-group-data') {
            content.appendChild(createConfigDatasetPicker());
        }
        appendFieldRows(content, fields, extraClass);
        if (extraClass === 'config-group-steps') {
            content.appendChild(createStepEstimatePanel());
            updateStepEstimatePanel();
        }
        section.appendChild(content);
        return section;
    }

    function createFillGlobalModelPathsButton() {
        const btn = document.createElement('button');
        btn.id = 'btn-fill-global-model-paths';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action';
        btn.textContent = '填写全局路径配置';
        btn.title = '用全局设置里的三项基础模型路径覆盖当前配置表单';
        btn.addEventListener('click', () => {
            fillGlobalModelPathsIntoConfigForm().catch((e) => {
                setTomlStatus('error', '填写全局路径配置失败: ' + e.message);
            });
        });
        return btn;
    }

    async function fillGlobalModelPathsIntoConfigForm() {
        if (!globalSettings && location.protocol !== 'file:') {
            await loadGlobalSettings();
        }
        const overrides = getGlobalModelPathOverrides();
        const entries = GLOBAL_MODEL_PATH_FIELDS
            .map(([key]) => [key, overrides[key]])
            .filter(([, value]) => String(value || '').trim());
        if (!entries.length) {
            setTomlStatus('error', '全局设置里还没有可填写的基础模型路径');
            return;
        }

        const confirmed = await showAppConfirmDialog({
            title: '是否确认覆盖',
            description: '填写全局路径配置',
            message: '将用全局设置里的基础模型路径覆盖当前配置表单中的同名字段。',
            confirmText: '是',
            cancelText: '否',
        });
        if (!confirmed) return;

        let applied = 0;
        for (const [key, value] of entries) {
            const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
            if (!input) continue;
            input.value = value;
            applied += 1;
        }
        handleFormFieldChange();
        setTomlStatus(
            applied ? '已填写全局路径配置，请保存当前配置后再训练' : '当前表单没有可覆盖的基础模型路径字段',
            applied ? 'ok' : 'error'
        );
    }

    function appendFieldRows(content, fields, groupClass) {
        const compactGroups = CONFIG_COMPACT_FIELD_GROUPS[groupClass] || [];
        const usedLayouts = new Set();
        let index = 0;

        while (index < fields.length) {
            const [key] = fields[index];
            const compactLayout = compactGroups.find((layout) => {
                if (usedLayouts.has(layout)) return false;
                return layout.keys.includes(key);
            });

            if (!compactLayout) {
                content.appendChild(createFieldRow(fields[index][0], fields[index][1]));
                index += 1;
                continue;
            }

            usedLayouts.add(compactLayout);
            const compactKeys = new Set(compactLayout.keys);
            const grid = document.createElement('div');
            grid.className = ['config-field-grid', compactLayout.className].filter(Boolean).join(' ');

            while (index < fields.length && compactKeys.has(fields[index][0])) {
                const [compactKey, compactValue] = fields[index];
                const row = createFieldRow(compactKey, compactValue);
                row.classList.add('field-row-compact');
                grid.appendChild(row);
                index += 1;
            }

            content.appendChild(grid);
        }
    }

    function createConfigDatasetPicker() {
        const panel = document.createElement('div');
        panel.id = 'config-dataset-picker';
        panel.className = 'config-dataset-picker';
        renderConfigDatasetPicker(panel);
        return panel;
    }

    function renderConfigDatasetPicker(existingPanel = null) {
        const panel = existingPanel || document.getElementById('config-dataset-picker');
        if (!panel) return;
        panel.innerHTML = '';

        const header = document.createElement('div');
        header.className = 'config-dataset-picker-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>数据集预设</strong><span>当前配置只保留选择摘要；搜索、选择和预览在弹窗中完成。</span>';
        const actions = document.createElement('div');
        actions.className = 'config-dataset-picker-actions';
        const openBtn = document.createElement('button');
        openBtn.type = 'button';
        openBtn.className = 'btn btn-small';
        openBtn.textContent = selectedConfigDatasetFile ? '更换预设' : '选择预设';
        openBtn.title = '打开数据集预设弹窗，可以搜索并查看第一张原始图预览。';
        openBtn.addEventListener('click', openConfigDatasetPickerDialog);
        const manageBtn = document.createElement('button');
        manageBtn.type = 'button';
        manageBtn.className = 'btn btn-small';
        manageBtn.textContent = '管理数据集';
        manageBtn.addEventListener('click', () => document.querySelector('[data-tab="datasets"]')?.click());
        const unnamedBtn = document.createElement('button');
        unnamedBtn.id = 'btn-open-unnamed-dataset-dialog';
        unnamedBtn.type = 'button';
        unnamedBtn.className = 'btn btn-small';
        unnamedBtn.textContent = '待命名';
        unnamedBtn.addEventListener('click', openUnnamedDatasetDialog);
        const refreshBtn = document.createElement('button');
        refreshBtn.type = 'button';
        refreshBtn.className = 'btn btn-small';
        refreshBtn.textContent = '刷新预设';
        refreshBtn.addEventListener('click', () => loadDatasetPresets({ selectCurrent: false }));
        actions.append(openBtn, manageBtn, unnamedBtn, refreshBtn);
        header.append(title, actions);
        panel.appendChild(header);

        const body = document.createElement('div');
        body.className = 'config-dataset-picker-body';
        body.appendChild(createConfigDatasetCurrentSummary());
        panel.appendChild(body);
        if (isConfigDatasetPickerDialogOpen()) {
            renderConfigDatasetPickerDialog();
        }
        ensureConfigDatasetPreview();
    }

    function createConfigDatasetCurrentSummary() {
        const preset = datasetPresetByFile(selectedConfigDatasetFile);
        const summary = selectedConfigDatasetSummary || preset?.summary || {};
        const wrap = document.createElement('div');
        wrap.className = 'config-dataset-current';

        const info = document.createElement('div');
        info.className = 'config-dataset-current-info';
        const label = document.createElement('span');
        label.className = 'config-dataset-current-label';
        label.textContent = selectedConfigDatasetFile ? '当前选中' : '当前状态';
        const title = document.createElement('strong');
        title.textContent = selectedConfigDatasetFile
            ? (preset?.label || preset?.filename || selectedConfigDatasetFile)
            : '不使用独立数据集预设';
        const path = document.createElement('code');
        path.textContent = selectedConfigDatasetFile || '沿用当前训练配置文件中的数据集字段';
        info.append(label, title, path);

        const meta = document.createElement('div');
        meta.className = 'config-dataset-current-meta';
        const state = document.createElement('span');
        const isDirtySelection = selectedConfigDatasetFile !== (currentConfig.dataset_config || '');
        state.className = [
            'config-dataset-current-state',
            isDirtySelection ? 'dirty' : 'synced',
        ].join(' ');
        state.textContent = isDirtySelection
            ? '未保存'
            : '已同步';
        const count = document.createElement('span');
        count.textContent = selectedConfigDatasetFile
            ? `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}`
            : '当前配置';
        const source = document.createElement('span');
        source.textContent = selectedConfigDatasetFile && summary.source_dir
            ? `原始路径: ${summary.source_dir}`
            : '保存当前配置后才会写入训练 TOML';
        meta.append(state, count, source);

        wrap.append(info, meta);
        return wrap;
    }

    function isConfigDatasetPickerDialogOpen() {
        return Boolean(document.getElementById('config-dataset-picker-dialog')?.open);
    }

    function openConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        if (!dialog) return;
        renderConfigDatasetPickerDialog();
        ensureConfigDatasetPreview();
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        const search = dialog.querySelector('.config-dataset-search');
        if (search) {
            search.focus({ preventScroll: true });
            search.setSelectionRange(search.value.length, search.value.length);
        }
    }

    function closeConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        if (dialog?.open) dialog.close();
    }

    function openUnnamedDatasetDialog() {
        const dialog = document.getElementById('unnamed-dataset-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }

    function renderContinueTrainingSource() {
        const summary = document.getElementById('continue-training-source-summary');
        const chooseBtn = document.getElementById('btn-open-continue-lora-dialog');
        const clearBtn = document.getElementById('btn-clear-continue-lora-source');
        if (!summary || !chooseBtn || !clearBtn) return;
        summary.innerHTML = '';
        if (!continueTrainingSource) {
            const title = document.createElement('strong');
            title.textContent = '从零开始';
            const detail = document.createElement('span');
            detail.textContent = '不加载已有权重';
            summary.append(title, detail);
            summary.className = 'continue-training-source-summary';
            chooseBtn.textContent = '选择 LoRA/LoKr';
            clearBtn.hidden = true;
            return;
        }
        const title = document.createElement('strong');
        title.textContent = `继续训练 ${continueTrainingSource.kind || 'LoRA'} · ${continueTrainingSource.name || '未命名权重'}`;
        const path = document.createElement('code');
        path.textContent = continueTrainingSource.abs_path || '';
        const state = document.createElement('span');
        state.className = continueTrainingSource.compatible === false ? 'warning' : 'ok';
        state.textContent = continueTrainingSource.compatible === false
            ? (continueTrainingSource.message || '当前配置不兼容')
            : '兼容 · 启动时会使用 --network_weights 与 --dim_from_weights';
        summary.append(title, path, state);
        summary.className = [
            'continue-training-source-summary',
            continueTrainingSource.compatible === false ? 'incompatible' : 'selected',
        ].join(' ');
        chooseBtn.textContent = '更换';
        clearBtn.hidden = false;
    }

    function continueTrainingRequestPayload() {
        if (!continueTrainingSource) return {};
        return {
            continue_from_weight_abs_path: continueTrainingSource.abs_path || '',
            continue_from_weight_name: continueTrainingSource.name || '',
            continue_from_weight_kind: continueTrainingSource.kind || '',
        };
    }

    function clearContinueTrainingSource() {
        continueTrainingSource = null;
        renderContinueTrainingSource();
        setTomlStatus('ok', '已恢复为从零开始训练');
    }

    async function openContinueLoraDialog() {
        const dialog = document.getElementById('continue-lora-dialog');
        if (!dialog) return;
        if (!historyTasks.length) {
            await loadTrainingHistoryList();
        }
        renderContinueLoraHistoryTasks();
        const input = document.getElementById('continue-lora-path-input');
        if (input && continueTrainingSource?.abs_path) {
            input.value = continueTrainingSource.abs_path;
        }
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        await loadContinueLoraWeights();
        document.getElementById('continue-lora-path-input')?.focus({ preventScroll: true });
    }

    function renderContinueLoraHistoryTasks() {
        const select = document.getElementById('continue-lora-history-task');
        if (!select) return;
        const previous = continueLoraDialogState.taskId;
        const tasks = historyTasks.filter((task) => task.job === 'training');
        select.innerHTML = '';
        const latest = document.createElement('option');
        latest.value = '';
        latest.textContent = '最近一次训练输出';
        select.appendChild(latest);
        for (const task of tasks) {
            const option = document.createElement('option');
            option.value = task.id || '';
            option.textContent = historyTaskDisplayName(task) || task.id || '训练任务';
            select.appendChild(option);
        }
        if (previous && tasks.some((task) => task.id === previous)) {
            select.value = previous;
        } else {
            continueLoraDialogState.taskId = '';
            select.value = '';
        }
    }

    async function loadContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        continueLoraDialogState.loading = true;
        continueLoraDialogState.error = '';
        renderContinueLoraWeights();
        try {
            const params = new URLSearchParams();
            if (continueLoraDialogState.taskId) {
                params.set('task_id', continueLoraDialogState.taskId);
            }
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/preview/weights${suffix}`);
            continueLoraDialogState = {
                ...continueLoraDialogState,
                loading: false,
                weights: payload.weights || [],
                error: payload.ok === false ? (payload.error || '读取权重失败') : '',
                message: payload.message || '',
            };
        } catch (e) {
            continueLoraDialogState = {
                ...continueLoraDialogState,
                loading: false,
                weights: [],
                error: e.message || '读取权重失败',
            };
        }
        renderContinueLoraWeights();
    }

    function renderContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        list.innerHTML = '';
        if (continueLoraDialogState.loading) {
            list.textContent = '正在读取历史权重...';
            return;
        }
        if (continueLoraDialogState.error) {
            list.textContent = continueLoraDialogState.error;
            return;
        }
        if (!continueLoraDialogState.weights.length) {
            list.textContent = continueLoraDialogState.message || '没有可选择的 .safetensors 权重。';
            return;
        }
        for (const item of continueLoraDialogState.weights) {
            const row = document.createElement('div');
            row.className = 'continue-lora-weight-item';
            const info = document.createElement('div');
            const name = document.createElement('strong');
            name.textContent = item.name || '未命名权重';
            const path = document.createElement('code');
            path.textContent = item.abs_path || item.file || '';
            info.append(name, path);
            const useBtn = document.createElement('button');
            useBtn.type = 'button';
            useBtn.className = 'btn btn-small btn-primary';
            useBtn.textContent = '继续训练';
            useBtn.addEventListener('click', () => selectContinueLoraWeight(item.abs_path || item.file || ''));
            row.append(info, useBtn);
            list.appendChild(row);
        }
    }

    function setContinueLoraStatus(message, state = '') {
        const status = document.getElementById('continue-lora-inspect-status');
        if (!status) return;
        status.className = ['continue-lora-status', state].filter(Boolean).join(' ');
        status.textContent = message || '';
    }

    async function requestContinueLoraInspection(path) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        return api('/api/training/continue-lora/inspect', {
            method: 'POST',
            body: JSON.stringify({
                path,
                variant,
                preset,
                methods_subdir: methodsSubdir,
                config_file: currentTrainingConfigFile(),
            }),
        });
    }

    async function selectContinueLoraWeight(path, options = {}) {
        const rawPath = String(path || '').trim();
        if (!rawPath) {
            setContinueLoraStatus('请填写 .safetensors 权重绝对路径。', 'error');
            return false;
        }
        setContinueLoraStatus('正在检查权重结构与当前变体兼容性...', 'pending');
        try {
            const payload = await requestContinueLoraInspection(rawPath);
            if (!payload.ok) {
                setContinueLoraStatus(payload.error || '权重检测失败。', 'error');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.error || '权重检测失败。');
                }
                return false;
            }
            if (!payload.compatible) {
                setContinueLoraStatus(payload.message || '当前配置与这个权重不兼容。', 'warning');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.message || '当前配置与这个权重不兼容。');
                }
                return false;
            }
            continueTrainingSource = payload;
            renderContinueTrainingSource();
            setContinueLoraStatus(payload.message || '已选择继续训练权重。', 'ok');
            setTomlStatus('ok', `训练来源已设置为继续训练 ${payload.kind} · ${payload.name}`);
            if (options.switchToConfig !== false) {
                document.querySelector('[data-tab="config"]')?.click();
            }
            const dialog = document.getElementById('continue-lora-dialog');
            if (dialog?.open && options.keepDialogOpen !== true) dialog.close();
            return true;
        } catch (e) {
            setContinueLoraStatus('权重检测请求失败: ' + e.message, 'error');
            if (!document.getElementById('continue-lora-dialog')?.open) {
                alert('权重检测请求失败: ' + e.message);
            }
            return false;
        }
    }

    async function refreshContinueTrainingSourceCompatibility() {
        if (!continueTrainingSource?.abs_path) {
            renderContinueTrainingSource();
            return true;
        }
        let payload;
        try {
            payload = await requestContinueLoraInspection(continueTrainingSource.abs_path);
        } catch (e) {
            continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: '无法重新检查继续训练权重: ' + e.message,
            };
            renderContinueTrainingSource();
            return false;
        }
        if (!payload.ok) {
            continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: payload.error || '无法重新检查继续训练权重。',
            };
            renderContinueTrainingSource();
            return false;
        }
        continueTrainingSource = payload;
        renderContinueTrainingSource();
        return Boolean(payload.compatible);
    }

    function renderConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        const body = document.getElementById('config-dataset-picker-dialog-body');
        if (!dialog || !body) return;
        body.innerHTML = '';

        const toolbar = document.createElement('div');
        toolbar.className = 'config-dataset-dialog-toolbar';
        const search = document.createElement('input');
        search.type = 'search';
        search.className = 'field-input config-dataset-search';
        search.placeholder = '搜索数据集预设、路径或原始目录';
        search.value = configDatasetPickerSearch;
        search.addEventListener('input', () => {
            const cursor = search.selectionStart ?? search.value.length;
            configDatasetPickerSearch = search.value;
            renderConfigDatasetPickerDialog();
            const nextSearch = document.querySelector('#config-dataset-picker-dialog .config-dataset-search');
            if (nextSearch) {
                nextSearch.focus();
                nextSearch.setSelectionRange(cursor, cursor);
            }
        });
        toolbar.appendChild(search);
        body.appendChild(toolbar);

        const workspace = document.createElement('div');
        workspace.className = 'config-dataset-workspace config-dataset-dialog-workspace';
        workspace.appendChild(createConfigDatasetPresetList());
        workspace.appendChild(createConfigDatasetPresetPreview());
        body.appendChild(workspace);
    }

    function datasetPresetOptionLabel(preset) {
        const summary = preset?.summary || {};
        const name = preset?.label || preset?.filename || preset?.path || '未命名预设';
        const count = Number(summary.dataset_count || 0);
        const repeats = Number(summary.repeat_total || 0);
        const lock = preset?.readonly ? '只读 · ' : '';
        return `${lock}${name} · ${count || 0} 组 · 重复 ${repeats || 0}`;
    }

    function createConfigDatasetPresetList() {
        const list = document.createElement('div');
        list.className = 'config-dataset-preset-list';
        const noneBtn = createConfigDatasetPresetButton(null);
        list.appendChild(noneBtn);

        const presets = filteredConfigDatasetPresets();
        if (!presets.length && configDatasetPickerSearch.trim()) {
            const empty = document.createElement('p');
            empty.className = 'config-dataset-picker-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return list;
        }

        for (const preset of presets) {
            list.appendChild(createConfigDatasetPresetButton(preset));
        }
        return list;
    }

    function createConfigDatasetPresetButton(preset) {
        const isNone = !preset;
        const file = isNone ? '' : preset.path;
        const summary = preset?.summary || {};
        const active = file === selectedConfigDatasetFile;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'config-dataset-preset-option',
            active ? 'active' : '',
            preset?.readonly ? 'readonly' : '',
        ].filter(Boolean).join(' ');
        btn.dataset.file = file;
        const title = document.createElement('strong');
        title.textContent = isNone
            ? '不使用独立数据集预设'
            : (preset.label || preset.filename || preset.path || '未命名预设');
        const path = document.createElement('span');
        path.textContent = isNone ? '沿用当前训练配置文件中的数据集字段' : preset.path;
        const meta = document.createElement('small');
        meta.textContent = isNone
            ? '当前配置'
            : `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}`;
        btn.append(title, path, meta);
        btn.addEventListener('click', () => selectConfigDatasetPreset(file));
        return btn;
    }

    function filteredConfigDatasetPresets() {
        const keyword = configDatasetPickerSearch.trim().toLowerCase();
        const presets = datasetPresetState.presets || [];
        if (!keyword) return presets;
        return presets.filter((preset) => {
            const summary = preset.summary || {};
            return [
                preset.label,
                preset.filename,
                preset.path,
                summary.source_dir,
                            ].some((value) => String(value || '').toLowerCase().includes(keyword));
        });
    }

    function createConfigDatasetPresetPreview() {
        const preview = document.createElement('div');
        preview.className = 'config-dataset-preview';
        const summary = document.createElement('div');
        summary.className = 'config-dataset-summary';
        summary.appendChild(createConfigDatasetSummary());
        preview.appendChild(summary);
        preview.appendChild(createConfigDatasetPreviewImage());
        return preview;
    }

    function createConfigDatasetPreviewImage() {
        const box = document.createElement('div');
        box.className = 'config-dataset-preview-image';
        const state = configDatasetPreviewState;
        if (!selectedConfigDatasetFile) {
            box.classList.add('empty');
            box.textContent = '选择一个数据集预设后，这里会显示第一张原始图。';
            return box;
        }
        if (state.file !== selectedConfigDatasetFile || state.loading) {
            box.classList.add('empty');
            box.textContent = '正在读取第一张原始图...';
            return box;
        }
        if (state.error) {
            box.classList.add('empty');
            box.textContent = state.error;
            return box;
        }
        const image = Array.isArray(state.payload?.images) ? state.payload.images[0] : null;
        if (!image) {
            box.classList.add('empty');
            box.textContent = state.payload?.message || '没有找到可预览的原始图。';
            return box;
        }
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '数据集预览图';
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            box.classList.add('empty');
            box.textContent = '预览图加载失败。';
        });
        const caption = document.createElement('div');
        caption.className = 'config-dataset-preview-caption';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const path = document.createElement('span');
        path.textContent = state.payload?.directory || image.file || '';
        caption.append(name, path);
        box.append(img, caption);
        return box;
    }

    function createConfigDatasetSummary() {
        const wrap = document.createElement('div');
        const preset = datasetPresetByFile(selectedConfigDatasetFile);
        const summary = selectedConfigDatasetSummary || preset?.summary || {};
        if (!selectedConfigDatasetFile) {
            wrap.className = 'config-dataset-summary-empty';
            wrap.textContent = '未选择独立数据集预设；训练会沿用当前配置文件里的数据集字段。';
            return wrap;
        }
        const items = [
            ['预设文件', selectedConfigDatasetFile],
            ['数据组数', String(summary.dataset_count || 0)],
            ['重复合计', String(summary.repeat_total || 0)],
            ['第 1 组原始路径', summary.source_dir || '-'],
        ];
        if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
            items.unshift(['状态', '未保存，保存当前配置后生效']);
        }
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            wrap.appendChild(row);
        }
        return wrap;
    }

    async function selectConfigDatasetPreset(file) {
        selectedConfigDatasetFile = file || '';
        selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
        configDatasetPreviewState = {
            file: '',
            loading: false,
            payload: null,
            error: '',
        };
        renderConfigDatasetPicker();
        renderConfigDatasetPickerDialog();
        updateTomlDirtyState();
        await loadStepEstimate();
    }

    function datasetPresetByFile(file) {
        return (datasetPresetState.presets || []).find((item) => item.path === file) || null;
    }

    function datasetPresetSummaryByFile(file) {
        return datasetPresetByFile(file)?.summary || null;
    }

    function ensureConfigDatasetPreview() {
        if (!selectedConfigDatasetFile) return;
        if (configDatasetPreviewState.file === selectedConfigDatasetFile && (configDatasetPreviewState.loading || configDatasetPreviewState.payload || configDatasetPreviewState.error)) {
            return;
        }
        loadConfigDatasetPresetPreview(selectedConfigDatasetFile);
    }

    async function loadConfigDatasetPresetPreview(file) {
        if (!file || location.protocol === 'file:') return;
        const requestSeq = ++configDatasetPreviewRequestSeq;
        configDatasetPreviewState = {
            file,
            loading: true,
            payload: null,
            error: '',
        };
        renderConfigDatasetPreviewArea();
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: '0',
                source: 'source',
                limit: '1',
            });
            const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            configDatasetPreviewState = {
                file,
                loading: false,
                payload,
                error: '',
            };
        } catch (e) {
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            configDatasetPreviewState = {
                file,
                loading: false,
                payload: null,
                error: e.message || '读取数据集预览失败',
            };
        }
        renderConfigDatasetPreviewArea();
    }

    function renderConfigDatasetPreviewArea() {
        const previews = document.querySelectorAll('.config-dataset-preview');
        if (!previews.length) return;
        previews.forEach((preview) => {
            preview.innerHTML = '';
            const summary = document.createElement('div');
            summary.className = 'config-dataset-summary';
            summary.appendChild(createConfigDatasetSummary());
            preview.appendChild(summary);
            preview.appendChild(createConfigDatasetPreviewImage());
        });
    }

    function createDatasetEditor() {
        const panel = document.createElement('div');
        panel.id = 'dataset-editor';
        panel.className = 'dataset-editor';
        renderDatasetEditor(panel);
        return panel;
    }

    function renderDatasetPresetList() {
        const list = document.getElementById('dataset-preset-list');
        if (!list) return;
        list.innerHTML = '';
        const presets = datasetPresetState.presets || [];
        const showErrorAsEmptyState = datasetPresetState.error && !presets.length;
        if (datasetPresetState.loading && !presets.length) {
            const loading = document.createElement('p');
            loading.className = 'dataset-preset-empty';
            loading.textContent = '正在读取数据集预设...';
            list.appendChild(loading);
            return;
        }
        if (showErrorAsEmptyState) {
            const error = document.createElement('p');
            error.className = 'dataset-preset-empty error';
            error.textContent = datasetPresetState.error;
            list.appendChild(error);
        }
        if (!presets.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = datasetPresetState.error ? '读取数据集预设失败。' : '还没有数据集预设。';
            list.appendChild(empty);
            return;
        }
        for (const preset of presets) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = [
                'dataset-preset-item',
                preset.path === datasetPresetState.selectedFile ? 'active' : '',
                preset.readonly ? 'readonly' : '',
            ].filter(Boolean).join(' ');
            btn.dataset.file = preset.path;
            const summary = preset.summary || {};
            btn.innerHTML = [
                `<strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong>`,
                `<span>${escapeHtml(preset.path)}</span>`,
                `<small>${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}</small>`,
            ].join('');
            btn.addEventListener('click', () => loadDatasetPreset(preset.path));
            list.appendChild(btn);
        }
    }

    function renderDatasetPresetHeader() {
        const header = document.getElementById('dataset-preset-header');
        if (!header) return;
        const file = datasetPresetState.selectedFile;
        const preset = datasetPresetByFile(file);
        const summary = preset?.summary || {};
        header.innerHTML = '';
        const title = document.createElement('div');
        title.innerHTML = [
            `<strong>${escapeHtml(preset?.label || preset?.filename || file || '新数据集预设')}</strong>`,
            `<span>${escapeHtml(file || '尚未保存')}</span>`,
        ].join('');
        const meta = document.createElement('div');
        meta.className = 'dataset-preset-meta';
        const status = datasetPresetState.dirty ? '未保存' : (datasetPresetState.readonly ? '只读' : '已同步');
        meta.innerHTML = [
            `<span>${status}</span>`,
            `<span>${Number(summary.dataset_count || datasetPresetState.datasets.length || 0)} 组</span>`,
            `<span>重复 ${Number(summary.repeat_total || datasetPresetState.datasets.reduce((sum, row) => sum + Number(row.num_repeats || 1), 0) || 0)}</span>`,
        ].join('');
        header.append(title, meta);
        if (datasetPresetState.status) {
            const statusEl = document.createElement('span');
            statusEl.className = 'dataset-preset-status';
            statusEl.textContent = datasetPresetState.status;
            meta.appendChild(statusEl);
        }
        updateDatasetPresetActionState();
    }

    function updateDatasetPresetActionState() {
        const saveBtn = document.getElementById('btn-save-dataset-preset');
        if (saveBtn) {
            saveBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile || !datasetPresetState.dirty;
            saveBtn.title = datasetPresetState.readonly
                ? '系统数据集预设只读，请复制后编辑'
                : (datasetPresetState.dirty ? '保存当前数据集预设' : '当前数据集预设没有未保存修改');
        }
        const deleteBtn = document.getElementById('btn-delete-dataset-preset');
        if (deleteBtn) {
            deleteBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
            deleteBtn.title = datasetPresetState.readonly ? '系统数据集预设不能删除' : '只删除 TOML 预设，不删除图片或缓存目录';
        }
        const renameBtn = document.getElementById('btn-rename-dataset-preset');
        if (renameBtn) {
            renameBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
        }
        const copyBtn = document.getElementById('btn-copy-dataset-preset');
        if (copyBtn) copyBtn.disabled = !datasetPresetState.selectedFile;
        const exportBtn = document.getElementById('btn-export-dataset-preset');
        if (exportBtn) exportBtn.disabled = !datasetPresetState.selectedFile;
    }

    function renderDatasetEditor(existingPanel = null) {
        const panel = existingPanel || document.getElementById('dataset-editor');
        if (!panel) return;
        panel.innerHTML = '';
        const state = datasetEditorStateForActivePanel();

        const header = document.createElement('div');
        header.className = 'dataset-editor-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>多数据集路径</strong><span>每一行是一组数据：填写原始图路径、重复次数和分桶参数；缩放图与 LoRA 缓存会在训练运行目录中自动生成。</span>';
        const actions = document.createElement('div');
        actions.className = 'dataset-editor-actions';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '添加数据集';
        addBtn.title = '新增一组数据集路径。适合把多个角色、画风或批次一起训练，并给每组设置独立重复次数。';
        addBtn.addEventListener('click', addDatasetEditorRow);
        actions.append(addBtn);
        header.append(title, actions);
        panel.appendChild(header);

        if (state.loading) {
            const loading = document.createElement('p');
            loading.className = 'dataset-editor-message';
            loading.textContent = '正在读取数据集配置...';
            panel.appendChild(loading);
            return;
        }
        if (state.error) {
            const error = document.createElement('p');
            error.className = 'dataset-editor-message error';
            error.textContent = state.error;
            panel.appendChild(error);
        }

        const rows = state.datasets.length
            ? state.datasets
            : normalizeDatasetEditorRows([{
                source_dir: currentConfig.source_image_dir || '',
                image_dir: currentConfig.resized_image_dir || '',
                cache_dir: currentConfig.lora_cache_dir || '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults(state.defaults || {}),
            }]);
        if (!state.datasets.length) {
            setActiveDatasetRows(rows);
        }

        panel.appendChild(createDatasetDefaultsEditor());

        const list = document.createElement('div');
        list.className = 'dataset-editor-list';
        rows.forEach((row, index) => {
            list.appendChild(createDatasetEditorRow(row, index));
        });
        panel.appendChild(list);

        const footer = document.createElement('div');
        footer.className = 'dataset-editor-footer';
        const configPath = document.createElement('code');
        configPath.textContent = activeDatasetFileLabel();
        const dirty = document.createElement('span');
        dirty.className = activeDatasetDirty() ? 'dataset-editor-dirty active' : 'dataset-editor-dirty';
        dirty.textContent = activeDatasetDirty() ? '有未保存的数据集修改' : '数据集路径已同步';
        footer.append(configPath, dirty);
        panel.appendChild(footer);
        renderDatasetPresetHeader();
    }

    function datasetEditorStateForActivePanel() {
        return isDatasetTabActive() ? datasetPresetState : datasetEditorState;
    }

    function isDatasetTabActive() {
        return Boolean(document.getElementById('dataset-editor')?.closest('#tab-datasets'));
    }

    function setActiveDatasetRows(rows) {
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
    }

    function activeDatasetFileLabel() {
        if (isDatasetTabActive()) {
            return datasetPresetState.selectedFile || '保存后生成 configs/datasets/<名称>.toml';
        }
        return datasetEditorState.dataset_config || currentConfig.dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';
    }

    function activeDatasetDirty() {
        return isDatasetTabActive() ? datasetPresetState.dirty : datasetEditorState.dirty;
    }

    function createDatasetDefaultsEditor() {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (isDatasetTabActive()) {
            datasetPresetState.defaults = defaults;
        } else {
            datasetEditorState.defaults = defaults;
        }
        const wrap = document.createElement('div');
        wrap.className = 'dataset-defaults-list';

        const heading = document.createElement('div');
        heading.className = 'dataset-defaults-heading';
        heading.innerHTML = '<strong>通用标注设置</strong><span>分辨率、分桶和验证集参数已移到每个数据集路径卡片中，可按路径独立配置。</span>';
        wrap.appendChild(heading);

        const fields = [
            ['caption_extension', 'text'],
            ['keep_tokens', 'number'],
        ];

        for (const [key, type, layout] of fields) {
            const row = document.createElement('div');
            row.className = ['dataset-config-field', layout === 'wide' ? 'wide' : ''].filter(Boolean).join(' ');
            row.dataset.key = key;

            const label = document.createElement('label');
            label.className = 'dataset-config-label';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'field-name';
            nameSpan.textContent = datasetConfigLabel(key);
            nameSpan.title = key;
            label.appendChild(nameSpan);

            const btn = document.createElement('button');
            btn.className = 'info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                btn.classList.toggle('active');
                row.querySelector('.field-help')?.classList.toggle('visible');
            });
            label.appendChild(btn);
            row.appendChild(label);

            const input = createDatasetConfigInput(key, type, defaults);
            row.appendChild(input);

            const helpDiv = document.createElement('div');
            helpDiv.className = 'field-help';
            helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, defaults)));
            row.appendChild(helpDiv);
            wrap.appendChild(row);
        }
        return wrap;
    }

    function createDatasetConfigInput(key, type, defaults) {
        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(defaults[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
            input.dataset.valueType = 'boolean';
        } else {
            input = document.createElement('input');
            input.type = type;
            input.dataset.valueType = type === 'number' ? 'number' : 'string';
            input.value = datasetConfigValue(key, defaults);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-config-input';
        input.dataset.key = key;
        input.addEventListener('input', () => updateDatasetConfigValue(key, input));
        input.addEventListener('change', () => updateDatasetConfigValue(key, input));
        return input;
    }

    function datasetConfigLabel(key) {
        const labels = {
            resolution: '分辨率',
            batch_size: '数据集批大小',
            enable_bucket: '启用长宽比分桶',
            min_bucket_reso: '最小桶边长',
            max_bucket_reso: '最大桶边长',
            bucket_reso_steps: '桶尺寸步长',
            bucket_no_upscale: '小图放大',
            validation_split: '验证集比例',
            validation_split_num: '固定验证数量',
            validation_seed: '验证随机种子',
            caption_extension: '标注扩展名',
            keep_tokens: '保留前置 token',
        };
        return `${labels[key] || FIELD_LABEL_ZH[key] || key} / ${key}`;
    }

    function datasetConfigValue(key, defaults) {
        return defaults[key] ?? '';
    }

    function updateDatasetConfigValue(key, input) {
        updateDatasetDefault(key, input);
    }

    function createDatasetEditorRow(row, index) {
        const wrap = document.createElement('div');
        wrap.className = 'dataset-editor-row';
        wrap.dataset.index = String(index);
        const head = document.createElement('div');
        head.className = 'dataset-row-head';
        const titleBox = document.createElement('div');
        titleBox.className = 'dataset-row-title';
        const title = document.createElement('strong');
        title.textContent = `第 ${index + 1} 组数据集`;
        const subtitle = document.createElement('span');
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        subtitle.textContent = `${settings.resolution}px · 桶 ${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps} · 重复 ${row.num_repeats || 1}`;
        titleBox.append(title, subtitle);
        const headActions = document.createElement('div');
        headActions.className = 'dataset-row-head-actions';
        if (isDatasetTabActive()) {
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'btn btn-small';
            previewBtn.textContent = '预览图片和标注';
            previewBtn.disabled = !datasetPresetState.selectedFile || datasetPresetState.dirty;
            previewBtn.title = previewBtn.disabled
                ? '请先保存当前数据集预设，再预览磁盘中的图片和同名标注。'
                : '打开这一组数据集的原始图预览，并读取同名 caption 标注。';
            previewBtn.addEventListener('click', () => openDatasetPreview(index));
            headActions.appendChild(previewBtn);
        }
        head.append(titleBox, headActions);
        wrap.appendChild(head);

        const paths = document.createElement('div');
        paths.className = 'dataset-row-paths';
        paths.appendChild(createDatasetPathField(index, 'source_dir', '原始数据集路径', row.source_dir, 'image_dataset'));
        wrap.appendChild(paths);

        wrap.appendChild(createDatasetRowSettingsEditor(row, index));

        const repeat = document.createElement('label');
        repeat.className = 'dataset-repeat-field';
        const repeatText = document.createElement('span');
        repeatText.textContent = '重复次数';
        repeatText.title = '这一组图片在每轮里重复使用几次。小数据集或重点角色可以适当提高，但过高会更容易过拟合。';
        const repeatInput = document.createElement('input');
        repeatInput.type = 'number';
        repeatInput.min = '1';
        repeatInput.step = '1';
        repeatInput.value = String(row.num_repeats || 1);
        repeatInput.title = '每轮训练中这组数据的重复倍率。1 表示正常使用一次，2 表示等效看两遍。';
        repeatInput.addEventListener('input', () => updateDatasetEditorRow(index, 'num_repeats', repeatInput.value));
        repeat.append(repeatText, repeatInput);
        wrap.appendChild(repeat);

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'btn btn-small danger dataset-remove-btn';
        remove.textContent = '删除';
        remove.disabled = datasetEditorStateForActivePanel().datasets.length <= 1;
        remove.title = remove.disabled ? '至少保留一组数据集路径' : '从当前 dataset_config 中移除这一组路径，不会删除磁盘文件。';
        remove.addEventListener('click', () => removeDatasetEditorRow(index));
        wrap.appendChild(remove);
        return wrap;
    }

    function createDatasetRowSettingsEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-row-settings';
        const fields = [
            ['resolution', 'number'],
            ['batch_size', 'number'],
            ['enable_bucket', 'select'],
            ['min_bucket_reso', 'number'],
            ['max_bucket_reso', 'number'],
            ['bucket_reso_steps', 'number'],
            ['bucket_no_upscale', 'select'],
            ['validation_split', 'number'],
            ['validation_split_num', 'number'],
            ['validation_seed', 'number'],
        ];
        for (const [key, type] of fields) {
            const field = document.createElement('label');
            field.className = 'dataset-row-setting-field';
            const label = document.createElement('span');
            label.textContent = datasetConfigLabel(key);
            label.title = key;
            field.appendChild(label);
            field.appendChild(createDatasetRowSettingInput(index, key, type, settings));
            panel.appendChild(field);
        }
        return panel;
    }

    function createDatasetRowSettingInput(index, key, type, settings) {
        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(settings[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
        } else {
            input = document.createElement('input');
            input.type = type;
            input.value = datasetConfigValue(key, settings);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-row-setting-input';
        input.addEventListener('input', () => updateDatasetEditorRowSetting(index, key, input));
        input.addEventListener('change', () => updateDatasetEditorRowSetting(index, key, input));
        return input;
    }

    function createDatasetPathField(index, key, label, value, placeholder) {
        const field = document.createElement('label');
        field.className = 'dataset-path-field';
        const text = document.createElement('span');
        text.textContent = label;
        const titles = {
            source_dir: '原始图片和 caption 所在目录。预处理从这里读图；缩放图和 LoRA 缓存会写入本次训练运行目录。',
            image_dir: '缩放图目录。预处理会把图片按分辨率/分桶规则写到这里；训练从这里枚举训练图片。',
            cache_dir: 'LoRA 缓存目录。VAE latent、文本编码器缓存、PE 特征缓存会写到这里；训练用它加速。',
        };
        text.title = titles[key] || label;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = value || '';
        input.placeholder = placeholder;
        input.title = titles[key] || '';
        input.addEventListener('input', () => updateDatasetEditorRow(index, key, input.value));
        field.append(text, input);
        return field;
    }

    async function openDatasetPreview(index) {
        if (!datasetPresetState.selectedFile) {
            setDatasetPresetStatus('请先选择一个数据集预设', 'error');
            return;
        }
        if (datasetPresetState.dirty) {
            setDatasetPresetStatus('请先保存当前数据集预设，再打开预览', 'error');
            return;
        }
        datasetPreviewState.datasetIndex = index;
        datasetPreviewState.source = 'source';
        datasetPreviewState.payload = null;
        const dialog = document.getElementById('dataset-preview-dialog');
        renderDatasetPreviewDialog({ loading: true });
        if (dialog?.showModal && !dialog.open) {
            dialog.showModal();
        }
        await loadDatasetPreviewImages();
    }

    async function loadDatasetPreviewImages() {
        const file = datasetPresetState.selectedFile;
        if (!file) return;
        const requestSeq = ++datasetPreviewLoadSeq;
        renderDatasetPreviewDialog({ loading: true });
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: String(datasetPreviewState.datasetIndex || 0),
                source: 'source',
                limit: '120',
            });
            const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== datasetPreviewLoadSeq) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            datasetPreviewState.payload = payload;
            renderDatasetPreviewDialog();
        } catch (e) {
            if (requestSeq !== datasetPreviewLoadSeq) return;
            datasetPreviewState.payload = {
                ok: false,
                error: e.message || '读取数据集预览失败',
                images: [],
            };
            renderDatasetPreviewDialog();
        }
    }

    function renderDatasetPreviewDialog(options = {}) {
        const title = document.getElementById('dataset-preview-dialog-title');
        const meta = document.getElementById('dataset-preview-dialog-meta');
        const grid = document.getElementById('dataset-preview-grid');
        const details = document.getElementById('dataset-preview-details');
        const empty = document.getElementById('dataset-preview-empty');
        if (!title || !meta || !grid || !details || !empty) return;

        const datasetNo = Number(datasetPreviewState.datasetIndex || 0) + 1;
        title.textContent = `第 ${datasetNo} 组数据集预览`;
        if (options.loading) {
            meta.textContent = '正在读取图片和同名标注...';
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = '正在读取数据集图片...';
            empty.hidden = false;
            return;
        }

        const payload = datasetPreviewState.payload || {};
        if (payload.error) {
            meta.textContent = payload.error;
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = payload.error;
            empty.hidden = false;
            return;
        }

        const countText = `${payload.count || 0}/${payload.total || 0} 张`;
        meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · ${countText} · 标注 ${payload.caption_extension || '.txt'}`;
        renderDatasetPreviewDetails(payload);
        grid.innerHTML = '';
        const images = Array.isArray(payload.images) ? payload.images : [];
        if (!images.length) {
            empty.textContent = payload.message || '当前目录没有可预览图片。';
            empty.hidden = false;
            return;
        }
        empty.hidden = true;
        for (const image of images) {
            grid.appendChild(createDatasetPreviewCard(image));
        }
    }

    function renderDatasetPreviewDetails(payload) {
        const details = document.getElementById('dataset-preview-details');
        if (!details) return;
        details.innerHTML = '';
        const row = payload.row || {};
        const settings = normalizeDatasetDefaults(payload.settings || row.settings || {});
        const items = [
            ['数据集文件', payload.file || datasetPresetState.selectedFile || '-'],
            ['当前目录', payload.directory || '-'],
            ['原始路径', row.source_dir || '-'],
            ['重复次数', row.num_repeats ?? '-'],
            ['分辨率', settings.resolution || '-'],
            ['分桶', settings.enable_bucket ? `${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}` : '关闭'],
            ['验证集', datasetPreviewValidationText(settings)],
        ];
        for (const [label, value] of items) {
            details.appendChild(createPreviewDetailRow(label, String(value)));
        }
    }

    function datasetPreviewValidationText(settings) {
        if (Number(settings.validation_split_num || 0) > 0) return `固定 ${settings.validation_split_num} 张`;
        return `${settings.validation_split ?? 0}`;
    }

    function createDatasetPreviewCard(image) {
        const card = document.createElement('article');
        card.className = 'dataset-preview-card';
        const imageWrap = document.createElement('button');
        imageWrap.type = 'button';
        imageWrap.className = 'dataset-preview-image-btn';
        imageWrap.title = '点击在大图预览中查看。';
        imageWrap.addEventListener('click', () => openPreviewDialog(datasetPreviewImageToPreviewImage(image)));

        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name;
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            card.classList.add('dataset-preview-card-error');
            img.alt = '图片加载失败';
        });
        imageWrap.appendChild(img);

        const body = document.createElement('div');
        body.className = 'dataset-preview-card-body';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const file = document.createElement('span');
        file.textContent = image.file || '';
        body.append(name, file);

        const caption = image.caption || {};
        const captionBox = document.createElement('div');
        captionBox.className = ['dataset-preview-caption', caption.ok ? '' : 'missing'].filter(Boolean).join(' ');
        const captionHead = document.createElement('div');
        const captionTitle = document.createElement('span');
        captionTitle.textContent = caption.ok ? `标注 ${caption.extension || ''}` : `缺少标注 ${caption.extension || ''}`;
        captionHead.appendChild(captionTitle);
        if (caption.file) {
            const copyBtn = document.createElement('button');
            copyBtn.type = 'button';
            copyBtn.className = 'btn btn-small';
            copyBtn.textContent = '复制标注';
            copyBtn.addEventListener('click', () => copyDatasetCaptionText(caption.text || '', copyBtn));
            captionHead.appendChild(copyBtn);
        }
        const pre = document.createElement('pre');
        pre.textContent = caption.ok ? (caption.text || '(空标注)') : '未找到同名 caption 文件';
        captionBox.append(captionHead, pre);
        body.appendChild(captionBox);

        card.append(imageWrap, body);
        return card;
    }

    function datasetPreviewImageToPreviewImage(image) {
        return {
            ...image,
            detailContext: 'dataset',
            sample: {},
            source_task: null,
        };
    }

    async function copyDatasetCaptionText(text, button) {
        try {
            await copyText(text || '');
            const original = button.textContent;
            button.textContent = '已复制';
            button.classList.add('btn-primary');
            setTimeout(() => {
                button.textContent = original;
                button.classList.remove('btn-primary');
            }, 1000);
        } catch (e) {
            alert('复制标注失败: ' + e.message);
        }
    }

    function normalizeDatasetEditorRows(rows) {
        return (rows || [])
            .filter((row) => row && typeof row === 'object')
            .map((row) => ({
                source_dir: String(row.source_dir || row.source_image_dir || ''),
                image_dir: String(row.image_dir || row.resized_image_dir || ''),
                cache_dir: String(row.cache_dir || row.lora_cache_dir || ''),
                num_repeats: Math.max(1, Number.parseInt(row.num_repeats || 1, 10) || 1),
                settings: normalizeDatasetRowSettings(row),
            }));
    }

    function datasetRowsForPayload(rows) {
        return normalizeDatasetEditorRows(rows).map((row) => ({
            source_dir: row.source_dir,
            image_dir: row.image_dir,
            cache_dir: row.cache_dir,
            num_repeats: row.num_repeats,
            settings: normalizeDatasetDefaults(row.settings || {}),
        }));
    }

    function normalizeDatasetRowSettings(row) {
        if (row.settings && typeof row.settings === 'object') {
            return normalizeDatasetDefaults(row.settings);
        }
        if ([...DATASET_SETTING_KEYS].some((key) => key in row)) {
            return normalizeDatasetDefaults(row);
        }
        return {};
    }

    function normalizeDatasetDefaults(defaults) {
        const raw = defaults && typeof defaults === 'object' ? defaults : {};
        return {
            resolution: Math.max(1, Number.parseInt(raw.resolution || 1024, 10) || 1024),
            batch_size: Math.max(1, Number.parseInt(raw.batch_size || 1, 10) || 1),
            enable_bucket: raw.enable_bucket !== false && raw.enable_bucket !== 'false',
            min_bucket_reso: Math.max(1, Number.parseInt(raw.min_bucket_reso || 256, 10) || 256),
            max_bucket_reso: Math.max(1, Number.parseInt(raw.max_bucket_reso || 1024, 10) || 1024),
            bucket_reso_steps: Math.max(1, Number.parseInt(raw.bucket_reso_steps || 64, 10) || 64),
            bucket_no_upscale: raw.bucket_no_upscale === true || raw.bucket_no_upscale === 'true',
            validation_split: Math.max(0, Number(raw.validation_split ?? 0.025) || 0),
            validation_split_num: Math.max(0, Number.parseInt(raw.validation_split_num || 0, 10) || 0),
            validation_seed: Math.max(0, Number.parseInt(raw.validation_seed || 42, 10) || 42),
            caption_extension: String(raw.caption_extension || '.txt'),
            keep_tokens: Math.max(0, Number.parseInt(raw.keep_tokens || 3, 10) || 0),
        };
    }

    function updateDatasetDefault(key, input) {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (input.type === 'checkbox') {
            defaults[key] = input.checked;
        } else if (input.tagName === 'SELECT') {
            defaults[key] = input.value === 'true';
        } else if (input.type === 'number') {
            defaults[key] = key === 'validation_split' ? Math.max(0, Number(input.value) || 0) : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            defaults[key] = input.value;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.defaults = defaults;
        } else {
            datasetEditorState.defaults = defaults;
        }
        markDatasetEditorDirty();
    }

    function updateDatasetEditorRow(index, key, value) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        rows[index][key] = key === 'num_repeats'
            ? Math.max(1, Number.parseInt(value || '1', 10) || 1)
            : value;
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        if (!isDatasetTabActive() && index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    function updateDatasetEditorRowSetting(index, key, input) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        const settings = normalizeDatasetDefaults(rows[index].settings || state.defaults || {});
        if (input.tagName === 'SELECT') {
            settings[key] = input.value === 'true';
        } else if (input.type === 'number') {
            settings[key] = key === 'validation_split' ? Math.max(0, Number(input.value) || 0) : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            settings[key] = input.value;
        }
        rows[index].settings = settings;
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
    }

    function markDatasetEditorDirty() {
        if (isDatasetTabActive()) {
            datasetPresetState.dirty = true;
            datasetPresetState.status = '有未保存的数据集修改';
            renderDatasetPresetHeader();
        } else {
            datasetEditorState.dirty = true;
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
        const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
        if (dirty) {
            dirty.classList.add('active');
            dirty.textContent = '有未保存的数据集修改';
        }
    }

    function addDatasetEditorRow() {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        rows.push({
            source_dir: '',
            image_dir: '',
            cache_dir: '',
            num_repeats: 1,
            settings: normalizeDatasetDefaults(state.defaults || {}),
        });
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) updateTomlDirtyState();
    }

    function removeDatasetEditorRow(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (rows.length <= 1) return;
        rows.splice(index, 1);
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    function syncDatasetEditorToCompatFields() {
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const first = rows[0];
        if (!first) return;
        setFieldInputValue('source_image_dir', first.source_dir);
        setFieldInputValue('resized_image_dir', first.image_dir);
        setFieldInputValue('lora_cache_dir', first.cache_dir);
        if (datasetEditorState.dataset_config) {
            setFieldInputValue('dataset_config', datasetEditorState.dataset_config);
        }
    }

    function setFieldInputValue(key, value) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return;
        if (input.type === 'checkbox') {
            input.checked = Boolean(value);
        } else {
            input.value = value || '';
        }
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function setCurrentTrainingSourceFromVariant(variant) {
        if (!variant) return;
        if (val('method-select') === 'spd' || variant === 'spd') {
            currentTrainingSource = {
                method: 'spd',
                methods_subdir: 'methods',
                file: 'configs/methods/spd.toml',
            };
            return;
        }
        currentTrainingSource = {
            method: variant,
            methods_subdir: 'gui-methods',
            file: `configs/gui-methods/${variant}.toml`,
        };
    }

    function outputRunRuntimeFile(run = selectedOutputRun()) {
        const runtime = (run?.files || []).find((item) => item.kind === 'runtime');
        return runtime?.file || '';
    }

    function rememberSelectionSnapshot() {
        selectionSnapshot.method = val('method-select');
        selectionSnapshot.variant = val('variant-select');
        selectionSnapshot.preset = val('preset-select');
    }

    function restoreSelectionSnapshot() {
        const methodSelect = document.getElementById('method-select');
        const variantSelect = document.getElementById('variant-select');
        const presetSelect = document.getElementById('preset-select');
        if (methodSelect && selectionSnapshot.method && [...methodSelect.options].some((opt) => opt.value === selectionSnapshot.method)) {
            methodSelect.value = selectionSnapshot.method;
        }
        if (variantSelect && selectionSnapshot.variant && [...variantSelect.options].some((opt) => opt.value === selectionSnapshot.variant)) {
            variantSelect.value = selectionSnapshot.variant;
        }
        if (presetSelect && selectionSnapshot.preset && [...presetSelect.options].some((opt) => opt.value === selectionSnapshot.preset)) {
            presetSelect.value = selectionSnapshot.preset;
        }
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
    }

    async function confirmBeforeConfigSelectionChange(message) {
        const ok = await handlePendingConfigSwitch({
            targetLabel: '新的配置选择',
        });
        if (!ok) restoreSelectionSnapshot();
        return ok;
    }

    function updateChoiceGuide(config = currentConfig) {
        const container = document.getElementById('choice-guide');
        if (!container) return;
        container.innerHTML = '';
        const methodKey = activeMethodKey(config);
        container.appendChild(createChoiceCard('方法', methodKey, METHOD_GUIDE_ZH, defaultMethodGuide(), methodGuideFromConfig(methodKey, config)));
        const sourceKey = currentTrainingSource.method || val('variant-select');
        container.appendChild(createChoiceCard('配置', sourceKey, VARIANT_GUIDE_ZH, defaultVariantGuide(), configGuideFromCurrentSource(sourceKey, config)));
        const presetKey = val('preset-select');
        container.appendChild(createChoiceCard('预设', presetKey, PRESET_GUIDE_ZH, defaultPresetGuide(), presetGuideFromConfig(presetKey, config)));
    }

    function createChoiceCard(kind, key, guideMap, fallback, overrideGuide = null) {
        const guide = overrideGuide || guideMap[key] || fallback;
        const helpId = `choice-guide-hint-${++choiceGuideHintSeq}`;
        const card = document.createElement('article');
        card.className = 'choice-card';

        const heading = document.createElement('div');
        heading.className = 'choice-card-heading';
        const title = document.createElement('strong');
        title.textContent = `${kind}: ${key || '-'}`;
        const name = document.createElement('span');
        name.textContent = guide.title;
        heading.appendChild(title);
        heading.appendChild(name);
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'info-toggle choice-info-toggle';
        toggle.textContent = '?';
        toggle.title = `展开${kind}说明`;
        toggle.setAttribute('aria-label', `${kind}说明`);
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-controls', helpId);
        heading.appendChild(toggle);
        card.appendChild(heading);

        const body = document.createElement('div');
        body.id = helpId;
        body.className = 'choice-card-body';
        body.hidden = true;
        body.appendChild(choiceLine('说明', guide.summary));
        body.appendChild(choiceLine('取舍', guide.tradeoff));
        body.appendChild(choiceLine('推荐', guide.recommend, 'choice-recommend'));
        if (Array.isArray(guide.details) && guide.details.length) {
            const details = document.createElement('ul');
            details.className = 'choice-details';
            for (const detail of guide.details) {
                const item = document.createElement('li');
                item.textContent = detail;
                details.appendChild(item);
            }
            body.appendChild(details);
        }
        toggle.addEventListener('click', () => {
            const nextOpen = body.hidden;
            body.hidden = !nextOpen;
            toggle.classList.toggle('active', nextOpen);
            toggle.setAttribute('aria-expanded', String(nextOpen));
            toggle.title = nextOpen ? `收起${kind}说明` : `展开${kind}说明`;
        });
        card.appendChild(body);
        return card;
    }

    function choiceLine(label, text, extraClass = '') {
        const line = document.createElement('p');
        line.className = extraClass;
        const strong = document.createElement('strong');
        strong.textContent = `${label}: `;
        line.appendChild(strong);
        line.appendChild(document.createTextNode(text));
        return line;
    }

    function defaultMethodGuide() {
        return choiceHelp(
            '自定义方法',
            '当前方法没有专门说明，通常表示它来自后端方法列表。',
            '请结合变体 TOML 判断实际训练行为。',
            '不确定时使用 lora。'
        );
    }

    function defaultVariantGuide() {
        return choiceHelp(
            '自定义变体',
            '当前变体对应一个 gui-methods TOML 文件，里面才是实际训练参数。',
            '自定义变体灵活，但需要自行确认字段组合是否合理。',
            '不确定时从内置 lora 变体复制再改。'
        );
    }

    function defaultPresetGuide() {
        return choiceHelp(
            '自定义预设',
            '当前预设来自 presets.toml 或自定义配置。',
            '它会覆盖部分硬件、采样或性能参数。',
            '不确定时使用 default。'
        );
    }

    function activeMethodKey(config = currentConfig) {
        const inferred = inferMethodFromConfig(config);
        if (inferred) return inferred;
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') {
            return 'spd';
        }
        if (currentTrainingSource.methods_subdir === 'gui-methods') {
            return VARIANT_METHOD_FAMILY[currentTrainingSource.method] || val('method-select') || 'lora';
        }
        return val('method-select') || 'lora';
    }

    function inferMethodFromConfig(config) {
        if (!config || typeof config !== 'object') return '';
        const moduleName = String(config.network_module || '');
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') return 'spd';
        if ('dit_path' in config && 'iterations' in config && currentTrainingSource.method === 'spd') return 'spd';
        if (isTruthy(config.use_lokr)) return 'lokr';
        if (isTruthy(config.use_easycontrol) || moduleName.includes('easycontrol')) return 'easycontrol';
        if (isTruthy(config.use_ip_adapter) || moduleName.includes('ip_adapter')) return 'ip_adapter';
        if (moduleName.includes('soft_tokens')) return 'soft_tokens';
        if (isTruthy(config.add_reft) || ('reft_dim' in config && Number(config.reft_dim) > 0)) return 'reft';
        if (
            isTruthy(config.use_hydra) ||
            isTruthy(config.use_sigma_router) ||
            String(config.use_moe_style || 'false') !== 'false' ||
            moduleName.includes('chimera') ||
            moduleName.includes('hydra')
        ) {
            if (moduleName.includes('chimera') || 'content_router_source' in config) return 'chimera';
            return 'hydralora';
        }
        if (isTruthy(config.use_timestep_mask)) return 'tlora';
        if (isTruthy(config.use_ortho)) return 'ortholora';
        return '';
    }

    function methodGuideFromConfig(methodKey, config = currentConfig) {
        const base = METHOD_GUIDE_ZH[methodKey] || defaultMethodGuide();
        const details = compactList([
            flagDetail('use_lokr', 'LoKr', config.use_lokr),
            isTruthy(config.use_lokr) ? valueDetail('lokr_factor', config.lokr_factor) : '',
            valueDetail('network_dim', config.network_dim),
            valueDetail('network_alpha', config.network_alpha),
            valueDetail('learning_rate', config.learning_rate),
            valueDetail('max_train_epochs', config.max_train_epochs),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前表单已读取关键训练字段。`,
            details,
        };
    }

    function configGuideFromCurrentSource(sourceKey, config = currentConfig) {
        const isImported = currentTrainingSource.methods_subdir === 'imported';
        const base = isImported
            ? choiceHelp(
                '导入训练配置',
                `当前表单来自 ${currentTrainingSource.file || '导入配置'}。`,
                '它会按 base.toml → 当前预设 → 该 TOML 的顺序合并；不会强行加入变体下拉。',
                '适合把历史训练配置作为独立入口继续查看、预检测或训练。'
            )
            : (VARIANT_GUIDE_ZH[sourceKey] || defaultVariantGuide());
        const details = compactList([
            currentTrainingSource.file ? `文件: ${currentTrainingSource.file}` : '',
            config.dataset_config ? `数据集配置: ${config.dataset_config}` : '',
            config.output_name ? `输出名称: ${config.output_name}` : '',
            globalSettings?.output_root ? `Web 输出根目录: ${globalSettings.output_root}` : '',
            config.source_image_dir ? `原始数据集: ${config.source_image_dir}` : '',
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 已读取当前 TOML 的路径和输出信息。`,
            details,
        };
    }

    function presetGuideFromConfig(presetKey, config = currentConfig) {
        const base = PRESET_GUIDE_ZH[presetKey] || defaultPresetGuide();
        const details = compactList([
            valueDetail('mixed_precision', config.mixed_precision),
            valueDetail('optimizer_type', config.optimizer_type),
            valueDetail('lr_scheduler', config.lr_scheduler),
            valueDetail('train_batch_size', config.train_batch_size),
            valueDetail('gradient_accumulation_steps', config.gradient_accumulation_steps),
            valueDetail('sample_ratio', config.sample_ratio),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前已合并后的预设/配置值如下。`,
            details,
        };
    }

    function isTruthy(value) {
        return value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true';
    }

    function compactList(items) {
        return items.filter((item) => item !== undefined && item !== null && String(item).trim() !== '');
    }

    function valueDetail(key, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${FIELD_LABEL_ZH[key] || key}: ${formatChoiceValue(value)}`;
    }

    function flagDetail(key, label, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${label}: ${isTruthy(value) ? '开启' : '关闭'}`;
    }

    function formatChoiceValue(value) {
        if (Array.isArray(value)) return value.join(', ');
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    function createFieldRow(key, value) {
        const row = document.createElement('div');
        row.className = 'field-row';
        row.dataset.key = key;
        if (key === 'sample_prompts') row.classList.add('field-row-sample-prompts');

        const main = document.createElement('div');
        main.className = 'field-main';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'field-name';
        nameSpan.textContent = formatFieldName(key);
        nameSpan.title = key;

        const input = createFieldInput(key, value);
        input.dataset.key = key;
        input.dataset.valueType = fieldValueTypeForKey(key, value);
        input.addEventListener('input', handleFormFieldChange);
        input.addEventListener('change', handleFormFieldChange);

        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            const labelStack = document.createElement('div');
            labelStack.className = 'field-label-stack';
            labelStack.appendChild(nameSpan);

            const rowsWrap = input.querySelector('.sample-prompts-rows');
            if (rowsWrap) {
                const labelActions = document.createElement('div');
                labelActions.className = 'field-label-actions';
                labelActions.appendChild(createSamplePromptAddButton(rowsWrap));
                labelStack.appendChild(labelActions);
            }
            main.appendChild(labelStack);
        } else {
            main.appendChild(nameSpan);
        }
        main.appendChild(input);

        const btn = document.createElement('button');
        btn.className = 'info-toggle';
        btn.textContent = '?';
        btn.type = 'button';
        btn.title = '查看填写建议、好处、代价、风险和推荐';
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
            const helpDiv = row.querySelector('.field-help');
            if (helpDiv) helpDiv.classList.toggle('visible');
        });
        main.appendChild(btn);
        row.appendChild(main);

        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help';
        helpDiv.appendChild(createHelpContent(key, value));
        row.appendChild(helpDiv);

        return row;
    }

    function handleFormFieldChange() {
        updateTomlDirtyState();
        updateStepEstimatePanel();
        updateLoKrFieldState();
        updateChoiceGuideFromLiveForm();
    }

    function updateChoiceGuideFromLiveForm() {
        if (!currentConfig || Object.keys(currentConfig).length === 0) return;
        updateChoiceGuide(liveConfigFromForm());
    }

    function liveConfigFromForm() {
        const liveConfig = { ...(currentConfig || {}) };
        for (const input of document.querySelectorAll('#config-form .field-input[data-key]')) {
            const key = input.dataset.key;
            if (!key) continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (isActiveNetworkArgFieldKey(key)) continue;
            const original = key in liveConfig ? liveConfig[key] : FORM_UI_DEFAULTS[key];
            liveConfig[key] = readFieldInputValue(input, original);
        }
        liveConfig.network_args = collectNetworkArgsFromForm(liveConfig).networkArgs;
        return liveConfig;
    }

    function formatFieldName(key) {
        const label = FIELD_LABEL_ZH[key];
        return label ? `${label} / ${key}` : key;
    }

    function createFieldInput(key, value) {
        if (key === 'sample_prompts') {
            if (samplePromptsMode === 'path') {
                return createSamplePromptsPathInput(value);
            }
            return createSamplePromptsEditor(value);
        }
        const options = FIELD_OPTIONS[key];
        if (options && !Array.isArray(value)) {
            return createSelectInput(key, value, options);
        }

        let input;
        if (typeof value === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value;
        } else {
            input = document.createElement('input');
            input.type = isNumericField(key, value) ? 'number' : 'text';
            if (input.type === 'number') {
                input.step = isIntegerNumericField(key, value) ? '1' : '0.01';
                if (!allowsNegativeNumberField(key)) input.min = '0';
            }
            input.value = Array.isArray(value) ? JSON.stringify(value) : (value ?? '');
        }
        input.className = 'field-input';
        if (key === 'lokr_factor') {
            input.disabled = !readLoKrEnabled();
            input.title = input.disabled ? '启用 LoKr 后生效' : '';
        }
        return input;
    }

    function createSamplePromptsPathInput(value) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input';
        input.value = value ?? '';
        input.title = '当前 sample_prompts 指向非 .txt 文件，保留为文件路径。';
        return input;
    }

    function createSamplePromptsEditor(value) {
        const editor = document.createElement('div');
        editor.className = 'field-input sample-prompts-editor';
        editor.dataset.originalContent = value ?? '';
        editor.dataset.touched = '0';

        const rows = document.createElement('div');
        rows.className = 'sample-prompts-rows';

        editor.appendChild(rows);

        editor.addEventListener('input', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });
        editor.addEventListener('change', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });

        renderSamplePromptRows(editor, value ?? '');
        return editor;
    }

    function createSamplePromptAddButton(rowsWrap) {
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small sample-prompts-add-btn';
        addBtn.textContent = '添加行';
        addBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            appendSamplePromptRow(rowsWrap, blankSamplePromptRow());
            markSamplePromptsEditorTouched(editor);
            handleFormFieldChange();
        });
        return addBtn;
    }

    function setSamplePromptsEditorContent(editor, content) {
        if (!editor) return;
        editor.dataset.originalContent = content || '';
        editor.dataset.touched = '0';
        renderSamplePromptRows(editor, content || '');
    }

    function markSamplePromptsEditorTouched(editor) {
        if (editor) editor.dataset.touched = '1';
    }

    function renderSamplePromptRows(editor, content) {
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        rowsWrap.innerHTML = '';
        const rows = parseSamplePromptRows(content);
        for (const row of rows) {
            appendSamplePromptRow(rowsWrap, row);
        }
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    function appendSamplePromptRow(rowsWrap, row) {
        const item = document.createElement('div');
        item.className = 'sample-prompt-row';

        const promptField = createSamplePromptTextField('提示词', 'prompt', row.prompt || '');
        const heightField = createSamplePromptInputField('长 / h', 'height', row.height || '', 'number', '1');
        const widthField = createSamplePromptInputField('宽 / w', 'width', row.width || '', 'number', '1');
        const cfgField = createSamplePromptInputField('CFG / g', 'cfg', row.cfg || '', 'number', '0.1');
        const stepsField = createSamplePromptInputField('步数 / s', 'steps', row.steps || '', 'number', '1');
        const seedField = createSamplePromptInputField('种子 / d', 'seed', row.seed || '', 'number', '1');
        const extra = document.createElement('input');
        extra.type = 'hidden';
        extra.dataset.samplePromptField = 'extra';
        extra.value = row.extra || '';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-small btn-subtle-danger sample-prompt-remove';
        removeBtn.textContent = '删除';
        removeBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            const rowCount = rowsWrap.querySelectorAll('.sample-prompt-row').length;
            if (rowCount <= 1) {
                clearSamplePromptRow(item);
            } else {
                item.remove();
            }
            markSamplePromptsEditorTouched(editor);
            updateSamplePromptRemoveButtons(rowsWrap);
            handleFormFieldChange();
        });

        const rowActions = document.createElement('div');
        rowActions.className = 'sample-prompt-row-actions';
        rowActions.append(removeBtn);

        item.append(promptField, heightField, widthField, cfgField, stepsField, seedField, extra, rowActions);
        rowsWrap.appendChild(item);
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    function createSamplePromptTextField(labelText, field, value) {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field sample-prompt-field-text';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.samplePromptField = field;
        input.value = value || '';
        label.append(span, input);
        return label;
    }

    function createSamplePromptInputField(labelText, field, value, type = 'text', step = '') {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = type;
        input.dataset.samplePromptField = field;
        input.value = value || '';
        if (type === 'number') {
            input.min = '0';
            input.step = step || '1';
        }
        label.append(span, input);
        return label;
    }

    function clearSamplePromptRow(row) {
        row.querySelectorAll('[data-sample-prompt-field]').forEach((input) => {
            input.value = '';
        });
    }

    function updateSamplePromptRemoveButtons(rowsWrap) {
        const rows = rowsWrap.querySelectorAll('.sample-prompt-row');
        rows.forEach((row) => {
            const button = row.querySelector('.sample-prompt-remove');
            if (!button) return;
            button.textContent = rows.length <= 1 ? '清空' : '删除';
        });
    }

    function blankSamplePromptRow() {
        return { prompt: '', height: '', width: '', cfg: '', steps: '', seed: '', extra: '' };
    }

    function parseSamplePromptRows(content) {
        const rows = String(content || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line && !line.startsWith('#'))
            .map(parseSamplePromptLine);
        return rows.length ? rows : [blankSamplePromptRow()];
    }

    function parseSamplePromptLine(line) {
        const parts = String(line || '').trim().split(/\s+--/);
        const row = blankSamplePromptRow();
        row.prompt = (parts.shift() || '').trim();
        const extras = [];

        for (const rawPart of parts) {
            const part = rawPart.trim();
            let match = part.match(/^h\s+(\d+)$/i);
            if (match) {
                row.height = match[1];
                continue;
            }
            match = part.match(/^w\s+(\d+)$/i);
            if (match) {
                row.width = match[1];
                continue;
            }
            match = part.match(/^g\s+([\d.]+)$/i);
            if (match) {
                row.cfg = match[1];
                continue;
            }
            match = part.match(/^s\s+(\d+)$/i);
            if (match) {
                row.steps = match[1];
                continue;
            }
            match = part.match(/^d\s+(\d+)$/i);
            if (match) {
                row.seed = match[1];
                continue;
            }
            if (part) extras.push(`--${part}`);
        }
        row.extra = extras.join(' ');
        return row;
    }

    function serializeSamplePromptsEditor(editor) {
        const rows = [];
        for (const rowEl of editor.querySelectorAll('.sample-prompt-row')) {
            const row = samplePromptRowFromElement(rowEl);
            const line = serializeSamplePromptRow(row);
            if (line) rows.push(line);
        }
        return rows.join('\n');
    }

    function samplePromptRowFromElement(rowEl) {
        const value = (field) => rowEl.querySelector(`[data-sample-prompt-field="${field}"]`)?.value?.trim() || '';
        return {
            prompt: value('prompt'),
            height: value('height'),
            width: value('width'),
            cfg: value('cfg'),
            steps: value('steps'),
            seed: value('seed'),
            extra: value('extra'),
        };
    }

    function serializeSamplePromptRow(row) {
        if (!row.prompt) return '';
        const args = [];
        if (row.width) args.push(`--w ${positiveIntegerText(row.width)}`);
        if (row.height) args.push(`--h ${positiveIntegerText(row.height)}`);
        if (row.steps) args.push(`--s ${positiveIntegerText(row.steps)}`);
        if (row.cfg) args.push(`--g ${positiveNumberText(row.cfg)}`);
        if (row.seed) args.push(`--d ${positiveIntegerText(row.seed)}`);
        if (row.extra) args.push(row.extra.trim());
        return [row.prompt.trim(), ...args.filter(Boolean)].join(' ');
    }

    function positiveIntegerText(value) {
        const n = Math.max(0, Math.floor(Number(value)));
        return Number.isFinite(n) ? String(n) : '';
    }

    function positiveNumberText(value) {
        const n = Number(value);
        if (!Number.isFinite(n) || n < 0) return '';
        return String(n);
    }

    function isNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            return ['integer', 'number'].includes(networkArgSpec.valueType);
        }
        return typeof value === 'number' || [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_ratio',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'save_every_n_epochs',
            'checkpointing_epochs',
        ].includes(key);
    }

    function isIntegerNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) return networkArgSpec.valueType === 'integer';
        return [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'save_every_n_epochs',
            'checkpointing_epochs',
        ].includes(key) || Number.isInteger(value);
    }

    function allowsNegativeNumberField(key) {
        return ['b_cond_init', 'pe_lora_layer_from'].includes(key);
    }

    function createSelectInput(key, value, options) {
        const select = document.createElement('select');
        select.className = 'field-input field-select';
        select.dataset.valueType = fieldValueTypeForKey(key, value);
        const normalizedValue = optionValue(value);
        const normalizedOptions = options.map(optionValue);
        const displayOptions = [...options];
        if (!normalizedOptions.includes(normalizedValue)) {
            displayOptions.unshift(value);
        }

        for (const option of displayOptions) {
            const opt = document.createElement('option');
            opt.value = optionValue(option);
            opt.textContent = optionLabel(key, option);
            if (opt.value === normalizedValue) opt.selected = true;
            select.appendChild(opt);
        }
        return select;
    }

    Object.assign(FIELD_HELP_ZH, EXTRA_FIELD_HELP_ZH);

    function fieldValueType(value) {
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'boolean') return 'boolean';
        if (typeof value === 'number') return 'number';
        return 'string';
    }

    function fieldValueTypeForKey(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            if (networkArgSpec.valueType === 'boolean' || networkArgSpec.valueType === 'booleanInt') return 'boolean';
            if (networkArgSpec.valueType === 'integer' || networkArgSpec.valueType === 'number') return 'number';
            return 'string';
        }
        if (key === 'use_lokr') return 'boolean';
        if (key === 'lokr_factor') return 'number';
        if (isNumericField(key, value)) return 'number';
        return fieldValueType(value);
    }

    function optionValue(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    function optionLabel(key, value) {
        if (key === 'use_lokr') {
            return value === true || value === 'true' ? '启用 LoKr' : '普通 LoRA';
        }
        if (key === 'use_moe_style' && (value === false || value === 'false')) {
            return '关闭专家路由 / false';
        }
        if (key === 'splice_position') {
            return value === 'front_of_padding' ? 'Padding 前沿 / front_of_padding' : '序列末尾 / end_of_sequence';
        }
        if (key === 'contrastive_negative_mode') {
            return {
                shuffled: '随机负样本 / shuffled',
                jaccard: 'Jaccard 降权 / jaccard',
                hard: '困难负样本 / hard',
            }[value] || String(value);
        }
        if (key === 'contrastive_objective') {
            return {
                infonce: 'InfoNCE / infonce',
                agsm: 'AGSM / agsm',
            }[value] || String(value);
        }
        if (value === true) return '开启 / true';
        if (value === false) return '关闭 / false';
        return String(value);
    }

    function generateDefaultHelp(key, value) {
        const typeStr = Array.isArray(value) ? '数组' :
            typeof value === 'boolean' ? '布尔值 (true/false)' :
            typeof value === 'number' ? '数值' : '字符串';
        const label = FIELD_LABEL_ZH[key] || key;
        const section = sectionTitleForField(key);
        const currentText = value === undefined ? '未设置' : JSON.stringify(value);
        return help(
            `${label} 是当前配置里的${section}字段，WebUI 暂时没有为它写专门教程。`,
            `按 ${typeStr} 填写。当前值: ${currentText}。如果你只是想正常训练，不需要为了“看懂它”而主动修改。`,
            ['保留这个字段可以完整复现当前 TOML 的训练行为。'],
            ['它通常属于低频或方法内部参数，改动后效果不一定能从字段名直观看出来。'],
            ['不了解来源时修改，可能导致训练启动失败、缓存失效，或让训练结果和预期不一致。'],
            '新手建议保持当前值；要改之前先看右侧 TOML 所属变体，或复制一份新配置做实验。'
        );
    }

    function sectionTitleForField(key) {
        for (const section of FORM_SECTION_DEFS) {
            if ((section.keys || []).includes(key)) return section.title;
        }
        if (String(key).includes('cache')) return '缓存/预处理';
        if (String(key).includes('sample')) return '训练中预览图';
        if (String(key).includes('router') || String(key).includes('repa') || String(key).includes('reft')) return '方法内部';
        return '高级配置';
    }

    function createHelpContent(key, value) {
        const spec = getHelpSpec(key, value);
        const content = document.createElement('div');
        content.className = 'help-content';
        addHelpSection(content, '作用', spec.summary, 'summary');
        addHelpSection(content, '怎么填', spec.fill, 'fill');
        addHelpSection(content, '好处', spec.benefit, 'benefit');
        addHelpSection(content, '代价', spec.cost, 'cost');
        addHelpSection(content, '风险', spec.risk, 'risk');
        addHelpSection(content, '推荐', spec.recommend, 'recommend');
        return content;
    }

    function addHelpSection(parent, title, body, kind) {
        if (body === undefined || body === null || body === '') return;
        if (Array.isArray(body) && body.length === 0) return;

        const section = document.createElement('section');
        section.className = `help-section help-${kind}`;

        const heading = document.createElement('div');
        heading.className = 'help-heading';
        heading.textContent = title;
        section.appendChild(heading);

        if (Array.isArray(body)) {
            const list = document.createElement('ul');
            for (const item of body) {
                if (!item) continue;
                const li = document.createElement('li');
                li.textContent = item;
                list.appendChild(li);
            }
            section.appendChild(list);
        } else {
            const text = document.createElement('p');
            text.textContent = body;
            section.appendChild(text);
        }
        parent.appendChild(section);
    }

    function getHelpSpec(key, value) {
        // 优先使用内置中文说明
        if (FIELD_HELP_ZH[key]) return FIELD_HELP_ZH[key];
        // 其次从服务端获取的 field help 中取英文（作为兜底）
        const remote = fieldHelp[key];
        if (remote) {
            const remoteText = remote.en || remote.ko || '';
            if (remoteText) {
                const label = FIELD_LABEL_ZH[key] || key;
                return help(
                    `${label} 来自项目配置 schema 或方法配置，属于当前训练链路的一部分。`,
                    `${remoteText} 新手只需要确认当前值来自可信变体；不要为了试错随手改。`,
                    ['能保留上游配置说明，帮助你追踪字段来源。'],
                    ['英文说明通常偏开发者视角，仍需要结合当前方法和 TOML 判断。'],
                    ['如果字段和当前方法不匹配，可能训练启动后才暴露错误。'],
                    '不确定时保持当前变体默认值；需要实验时先另存为新配置。'
                );
            }
        }
        return generateDefaultHelp(key, value);
    }

    // ── TOML 编辑器 ──
    function setTomlManagerMode(mode) {
        const nextMode = mode === 'output' ? 'output' : 'project';
        tomlManagerMode = nextMode;
        document.querySelectorAll('.toml-mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.tomlMode === nextMode);
        });
        const projectManager = document.getElementById('toml-project-manager');
        const outputManager = document.getElementById('output-run-manager');
        const projectActions = document.querySelectorAll('.toml-primary-actions, .toml-secondary-actions');
        const outputActions = document.getElementById('output-run-actions');
        if (projectManager) projectManager.hidden = nextMode !== 'project';
        if (outputManager) outputManager.hidden = nextMode !== 'output';
        projectActions.forEach((el) => {
            el.hidden = nextMode !== 'project';
        });
        if (outputActions) outputActions.hidden = nextMode !== 'output';
        if (nextMode === 'output') {
            const label = document.getElementById('toml-current-file');
            if (label) label.textContent = outputRunState.file || outputRunState.selectedRun || '训练输出配置';
            setBadge('toml-current-badge', false, '当前训练');
            setBadge('toml-trainable-badge', Boolean(outputRunState.file), '只读快照');
            setBadge('toml-lock-badge', Boolean(outputRunState.file), '只读');
            setBadge('toml-dirty-badge', false, '未保存');
            updateOutputRunActionState();
            if (!outputRunState.runs.length && !outputRunState.loading) {
                loadOutputRuns();
            } else {
                renderOutputRunManager();
            }
            return;
        }
        updateTomlSelectionUI(currentTomlFile);
        updateTomlDirtyState();
    }

    async function switchTomlManagerMode(nextMode) {
        const normalizedMode = nextMode === 'output' ? 'output' : 'project';
        if (normalizedMode !== tomlManagerMode && normalizedMode === 'output' && hasPendingConfigChanges(currentTomlFile)) {
            if (!(await confirmDiscardTomlChanges('当前项目预设有未保存修改，切换到训练输出配置会暂时隐藏这些修改。是否继续？'))) {
                return false;
            }
        }
        setTomlManagerMode(normalizedMode);
        return true;
    }

    async function loadTomlFileList(preferredFile = '', options = {}) {
        const groups = await api('/api/config/file-groups');
        tomlFileGroups = Array.isArray(groups) ? groups : [];
        tomlFileMeta = {};
        tomlFiles = [];
        for (const group of tomlFileGroups) {
            for (const item of group.files || []) {
                tomlFiles.push(item.path);
                tomlFileMeta[item.path] = item;
            }
        }
        populateTomlFileSelect(reorderTomlFileGroups(tomlFileGroups));
        if (preferredFile && !tomlFiles.includes(preferredFile) && currentTomlFile === preferredFile) {
            await handleDeletedTomlSelection(preferredFile, '当前配置文件已不存在或已被删除');
            return;
        }
        if (preferredFile && tomlFiles.includes(preferredFile)) {
            await loadTomlFile(preferredFile, { force: options.force === true });
            return;
        }
        if (options.skipDefaultLoad) {
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            return;
        }
        // 默认加载当前变体对应的文件
        const variant = currentTrainingSource.method || val('variant-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const target = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlFiles.includes(target)) {
            await loadTomlFile(target);
        } else if (tomlFiles.length > 0) {
            await loadTomlFile(tomlFiles[0]);
        }
    }

    async function loadOutputRuns(options = {}) {
        if (location.protocol === 'file:') return;
        outputRunState = {
            ...outputRunState,
            loading: true,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api('/api/config/output-runs');
            if (!data.ok) throw new Error(data.error || '读取训练输出配置失败');
            const runs = Array.isArray(data.runs) ? data.runs : [];
            let selectedRun = outputRunState.selectedRun;
            if (selectedRun && !runs.some((item) => item.name === selectedRun)) selectedRun = '';
            if (!selectedRun && runs.length && options.keepSelection !== true) {
                selectedRun = runs[0].name || '';
            }
            outputRunState = {
                ...outputRunState,
                loading: false,
                runs,
                outputRoot: data.output_root || '',
                selectedRun,
                error: '',
            };
            renderOutputRunManager();
            if (selectedRun) {
                await loadOutputRunConfig(selectedRun, preferredOutputRunKind(selectedRun));
            } else {
                updateOutputRunSelectionUI();
            }
        } catch (e) {
            outputRunState = {
                ...outputRunState,
                loading: false,
                runs: [],
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取训练输出配置失败: ' + e.message);
        }
    }

    async function loadOutputRunConfig(runName, kind = 'original') {
        const run = outputRunState.runs.find((item) => item.name === runName);
        if (!run) {
            outputRunState = { ...outputRunState, selectedRun: '', content: '', file: '' };
            renderOutputRunManager();
            return;
        }
        const available = new Set((run.files || []).map((item) => item.kind));
        const selectedKind = available.has(kind) ? kind : preferredOutputRunKind(runName);
        outputRunState = {
            ...outputRunState,
            selectedRun: runName,
            selectedKind,
            content: '读取中...',
            file: '',
            saveAsOpen: false,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api(`/api/config/output-runs/read?run=${encodeURIComponent(runName)}&kind=${encodeURIComponent(selectedKind)}`);
            if (!data.ok) throw new Error(data.error || '读取运行配置失败');
            outputRunState = {
                ...outputRunState,
                selectedRun: data.run || runName,
                selectedKind: data.kind || selectedKind,
                content: data.content || '',
                file: data.file || '',
                error: '',
            };
            renderOutputRunManager();
            setTomlStatus('', '');
        } catch (e) {
            outputRunState = {
                ...outputRunState,
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取运行配置失败: ' + e.message);
        }
    }

    function preferredOutputRunKind(runName = outputRunState.selectedRun) {
        const run = outputRunState.runs.find((item) => item.name === runName);
        const kinds = (run?.files || []).map((item) => item.kind);
        if (kinds.includes(outputRunState.selectedKind)) return outputRunState.selectedKind;
        if (kinds.includes('original')) return 'original';
        if (kinds.includes('runtime')) return 'runtime';
        if (kinds.includes('dataset')) return 'dataset';
        return 'original';
    }

    function renderOutputRunManager() {
        renderOutputRunList();
        renderOutputRunDetail();
        updateOutputRunActionState();
        if (tomlManagerMode === 'output') {
            updateOutputRunSelectionUI();
        }
    }

    function renderOutputRunList() {
        const container = document.getElementById('output-run-list');
        if (!container) return;
        container.innerHTML = '';
        if (outputRunState.loading) {
            const loading = document.createElement('div');
            loading.className = 'output-run-empty';
            loading.textContent = '正在读取全局输出文件夹...';
            container.appendChild(loading);
            return;
        }
        if (outputRunState.error) {
            const error = document.createElement('div');
            error.className = 'output-run-empty error';
            error.textContent = outputRunState.error;
            container.appendChild(error);
            return;
        }
        const runs = filteredOutputRuns();
        if (!runs.length) {
            const empty = document.createElement('div');
            empty.className = 'output-run-empty';
            empty.textContent = outputRunState.search
                ? '没有匹配的训练输出配置。'
                : `没有在 ${outputRunState.outputRoot || '输出文件夹'} 找到训练配置。`;
            container.appendChild(empty);
            return;
        }
        for (const run of runs) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'output-run-item';
            btn.classList.toggle('active', run.name === outputRunState.selectedRun);
            btn.dataset.run = run.name;
            btn.title = run.path || run.name;
            btn.addEventListener('click', () => loadOutputRunConfig(run.name, preferredOutputRunKind(run.name)));

            const name = document.createElement('strong');
            name.textContent = run.name;
            const meta = document.createElement('span');
            const fileLabels = (run.files || []).map((item) => item.label).join(' / ');
            meta.textContent = [run.mtime_text, fileLabels].filter(Boolean).join(' · ');
            const path = document.createElement('small');
            path.textContent = run.path || '';
            btn.append(name, meta, path);
            container.appendChild(btn);
        }
    }

    function renderOutputRunDetail() {
        const run = selectedOutputRun();
        const title = document.getElementById('output-run-title');
        const meta = document.getElementById('output-run-meta');
        const tabs = document.getElementById('output-run-kind-tabs');
        const viewer = document.getElementById('output-run-config-viewer');
        const saveAs = document.getElementById('output-run-save-as');
        if (title) title.textContent = run?.name || '未选择运行目录';
        if (meta) {
            meta.textContent = run
                ? [run.path, run.mtime_text].filter(Boolean).join(' · ')
                : `从 ${outputRunState.outputRoot || '全局输出文件夹'} 读取训练快照配置。`;
        }
        if (tabs) {
            tabs.innerHTML = '';
            const files = run?.files || [];
            if (!files.length) {
                const empty = document.createElement('span');
                empty.className = 'output-run-kind-empty';
                empty.textContent = '无可读 TOML';
                tabs.appendChild(empty);
            }
            for (const file of files) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'output-run-kind-btn';
                btn.classList.toggle('active', file.kind === outputRunState.selectedKind);
                btn.textContent = file.label;
                btn.title = file.file || file.filename;
                btn.addEventListener('click', () => loadOutputRunConfig(run.name, file.kind));
                tabs.appendChild(btn);
            }
        }
        if (viewer) {
            viewer.value = outputRunState.content || '';
            viewer.placeholder = run ? '这个运行目录没有可显示的配置内容。' : '选择左侧运行目录后查看配置。';
        }
        if (saveAs) {
            saveAs.hidden = !outputRunState.saveAsOpen;
        }
        renderOutputRunSaveAsControls();
    }

    function renderOutputRunSaveAsControls() {
        const input = document.getElementById('output-run-save-name');
        const select = document.getElementById('output-run-save-group');
        if (input && !input.value && outputRunState.saveAsOpen) {
            input.value = outputRunSaveAsDefaultName();
        }
        if (!select) return;
        const current = select.value || 'imported';
        select.innerHTML = '';
        const groups = saveAsTargetGroups();
        for (const group of groups) {
            const opt = document.createElement('option');
            opt.value = group.id;
            opt.textContent = group.label || group.id;
            select.appendChild(opt);
        }
        select.value = groups.some((group) => group.id === current) ? current : (groups[0]?.id || 'imported');
    }

    function filteredOutputRuns() {
        const query = outputRunState.search.trim().toLowerCase();
        if (!query) return outputRunState.runs;
        return outputRunState.runs.filter((run) => {
            const haystack = [
                run.name,
                run.path,
                run.mtime_text,
                ...(run.files || []).map((item) => `${item.kind} ${item.label} ${item.file}`),
            ].join(' ').toLowerCase();
            return haystack.includes(query);
        });
    }

    function selectedOutputRun() {
        return outputRunState.runs.find((item) => item.name === outputRunState.selectedRun) || null;
    }

    function updateOutputRunSelectionUI() {
        const label = document.getElementById('toml-current-file');
        if (label) {
            label.textContent = outputRunState.file || outputRunState.selectedRun || '训练输出配置';
        }
        setBadge('toml-current-badge', false, '当前训练');
        setBadge('toml-trainable-badge', Boolean(outputRunState.file), '只读快照');
        setBadge('toml-lock-badge', Boolean(outputRunState.file), '只读');
        setBadge('toml-dirty-badge', false, '未保存');
    }

    function updateOutputRunActionState() {
        const run = selectedOutputRun();
        const hasContent = Boolean(outputRunState.content && outputRunState.file);
        const hasOriginal = Boolean(run?.has_original);
        setButtonDisabled('btn-copy-output-config', !hasContent);
        setButtonDisabled('btn-export-output-config', !hasContent);
        const saveBtn = document.getElementById('btn-save-output-config-as');
        if (saveBtn) {
            saveBtn.disabled = !run || !hasOriginal;
            saveBtn.title = !run
                ? '请先选择一个训练运行目录'
                : (hasOriginal ? '把 config.original.toml 复制到 configs/imported，随后可在项目预设中编辑。' : '这个运行目录没有 config.original.toml，不能复制为项目预设。');
        }
    }

    function setButtonDisabled(id, disabled) {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = Boolean(disabled);
    }

    async function copyOutputRunConfigContent() {
        const text = outputRunState.content || '';
        if (!text) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const viewer = document.getElementById('output-run-config-viewer');
                viewer?.focus();
                viewer?.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制训练输出配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    function exportOutputRunConfig() {
        if (!outputRunState.content) return;
        const filename = outputRunState.file
            ? outputRunState.file.split('/').pop()
            : `${outputRunState.selectedRun || 'output-run'}.${outputRunState.selectedKind}.toml`;
        const blob = new Blob([outputRunState.content], { type: 'application/toml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename || 'output-run-config.toml';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    function openOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) {
            setTomlStatus('error', '请先选择一个训练运行目录');
            return;
        }
        if (!run.has_original) {
            setTomlStatus('error', '这个运行目录没有 config.original.toml，不能复制为项目预设');
            return;
        }
        outputRunState = { ...outputRunState, saveAsOpen: true };
        renderOutputRunManager();
        const input = document.getElementById('output-run-save-name');
        if (input) {
            input.value = outputRunSaveAsDefaultName();
            input.focus();
            input.select();
        }
    }

    function closeOutputRunSaveAs() {
        outputRunState = { ...outputRunState, saveAsOpen: false };
        renderOutputRunManager();
    }

    function outputRunSaveAsDefaultName() {
        const run = selectedOutputRun();
        const stem = String(run?.name || 'output_run')
            .replace(/-\d{8}-\d{6}(?:-\d+)?$/i, '')
            .replace(/[^A-Za-z0-9_.-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'output_run';
        return `${stem}_from_output`;
    }

    async function confirmOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) return;
        const name = val('output-run-save-name') || outputRunSaveAsDefaultName();
        const group = val('output-run-save-group') || 'imported';
        try {
            const res = await api('/api/config/output-runs/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    run: run.name,
                    name,
                    target_group: group,
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '复制为项目预设失败');
                return;
            }
            outputRunState = { ...outputRunState, saveAsOpen: false };
            if (hasPendingConfigChanges(currentTomlFile)) {
                if (!(await confirmDiscardTomlChanges('复制完成后会回到项目预设并加载新文件，当前项目预设里已有未保存修改。是否继续？'))) {
                    return;
                }
            }
            await switchTomlManagerMode('project');
            await loadTomlFileList(res.file, { force: true });
            await loadTomlFile(res.file, { force: true });
            setTomlStatus('ok', `已复制为新项目预设: ${res.file}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '复制为项目预设失败: ' + e.message);
        }
    }

    async function selectAndApplyTomlFile(filePath) {
        if (!filePath) return false;
        const previousFile = currentTomlFile;
        const previousSelect = document.getElementById('toml-file-select');
        const targetLabel = tomlFileDisplayName(filePath);
        const canSwitch = await handlePendingConfigSwitch({ targetLabel });
        if (!canSwitch) {
            if (previousSelect) previousSelect.value = previousFile || '';
            updateTomlSelectionUI(previousFile);
            return false;
        }
        await loadTomlFile(filePath, { force: true });
        const meta = tomlFileMeta[filePath];
        if (!meta?.trainable) {
            setTomlStatus('error', '已打开该配置文件，但它不是完整训练配置，不能加载为当前训练入口');
            return false;
        }
        await applyTomlToConfig({ silent: true });
        rememberSelectionSnapshot();
        setTomlStatus('ok', `已加载选中配置: ${meta.path || filePath}`);
        return true;
    }

    async function loadTomlFile(filePath, options = {}) {
        if (!options.force && !(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，切换文件会丢失这些修改。是否继续？'))) {
            const select = document.getElementById('toml-file-select');
            if (select) select.value = currentTomlFile || '';
            return;
        }
        resetTomlDeleteConfirm();
        resetTomlSaveConfirm();
        const data = await api(`/api/config/raw?file=${encodeURIComponent(filePath)}`);
        if (data?.ok === false) {
            if (isMissingTomlFileResponse(data)) {
                await handleDeletedTomlSelection(filePath, data.error || '配置文件不存在或已被删除');
                return;
            }
            setTomlStatus('error', data.error || '读取配置文件失败');
            return;
        }
        currentTomlFile = filePath;
        document.getElementById('toml-file-select').value = filePath;
        tomlSavedContent = data.content || '';
        document.getElementById('toml-editor').value = tomlSavedContent;
        if (data.meta) tomlFileMeta[filePath] = data.meta;
        updateTomlSelectionUI(filePath);
        applyTomlLockState(filePath);
        updateTomlDirtyState();
        setTomlStatus('', '');
    }

    async function saveTomlFile(options = {}) {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件，或使用“另存新配置”保存导入内容');
            return false;
        }
        if (isTomlLocked(file)) {
            setTomlStatus('error', '该配置文件已锁定，请使用“另存新配置”创建可编辑配置');
            return false;
        }
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(file);
        const directEditorSave = options.mode === 'editor';
        if (directEditorSave) {
            if (formDirty) {
                setTomlStatus('error', '左侧表单或数据集预设选择有未保存修改，请先使用“保存更新当前选中配置”处理后再直接保存 TOML');
                updateTomlActionState(file);
                return false;
            }
            if (!editorDirty) {
                setTomlStatus('error', '直接编辑器没有未保存的 TOML 文本修改');
                return false;
            }
            if (!options.skipConfirm && tomlSaveConfirmFile !== file) {
                armTomlSaveConfirm(file);
                return false;
            }
            resetTomlSaveConfirm({ update: false });
            return await saveRawTomlContent(file, document.getElementById('toml-editor').value, { reloadConfig: currentTrainingSource.file === file });
        }
        if (editorDirty && !formDirty && !options.skipConfirm && tomlSaveConfirmFile !== file) {
            armTomlSaveConfirm(file);
            return false;
        }
        resetTomlSaveConfirm({ update: false });
        if (currentTrainingSource.file === file) {
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) return false;
            const datasetWasDirty = datasetEditorState.dirty;
            if (datasetWasDirty) {
                const datasetSaved = await saveDatasetEditor({ trainFile: file, reloadList: false });
                if (!datasetSaved) return false;
            }
            const changedValues = collectChangedFormValues({ persistDefaultFields: true });
            if (Object.keys(changedValues).length > 0) {
                return await saveFormPatchToToml(file, changedValues);
            }
            if (datasetApplied.applied || datasetWasDirty) {
                await loadConfig();
                await loadTomlFileList(file);
                updateTomlDirtyState();
                setTomlStatus('ok', datasetWasDirty ? '✓ 已保存数据集修改' : '✓ 已应用数据集预设');
                return true;
            }
        }
        const content = document.getElementById('toml-editor').value;
        return await saveRawTomlContent(file, content, { reloadConfig: currentTrainingSource.file === file });
    }

    async function saveRawTomlContent(file, content, options = {}) {
        try {
            const res = await api('/api/config/raw', {
                method: 'PUT',
                body: JSON.stringify({ file, content }),
            });
            if (res.ok) {
                tomlSavedContent = content;
                resetTomlSaveConfirm({ update: false });
                updateTomlDirtyState();
                setTomlStatus('ok', '✓ 已保存');
                await loadTomlFileList(file);
                if (options.reloadConfig) {
                    await loadConfig(); // 仅当前训练源被保存时刷新左侧表单
                }
                return true;
            } else {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }

    async function saveFormPatchToToml(file, values) {
        try {
            const content = document.getElementById('toml-editor').value;
            const preparedValues = await prepareFormPatchValues(values);
            const res = await api('/api/config/raw', {
                method: 'PATCH',
                body: JSON.stringify({ file, values: preparedValues, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }

            if (typeof res.content === 'string') {
                document.getElementById('toml-editor').value = res.content;
                tomlSavedContent = res.content;
            }
            resetTomlSaveConfirm({ update: false });
            await loadConfig();
            await loadTomlFileList(file);
            updateTomlDirtyState();
            setTomlStatus('ok', `✓ 已保存 ${res.changed?.length || Object.keys(preparedValues).length} 个表单修改`);
            return true;
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }

    async function applySelectedDatasetPresetToCurrentConfig(file) {
        const nextDataset = selectedConfigDatasetFile || '';
        const currentDataset = currentConfig.dataset_config || '';
        if (!nextDataset || nextDataset === currentDataset) {
            if (!nextDataset && currentDataset) {
                const res = await api('/api/config/raw', {
                    method: 'PATCH',
                    body: JSON.stringify({
                        file,
                        values: { dataset_config: '' },
                        content: document.getElementById('toml-editor')?.value || '',
                    }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '清除数据集预设失败');
                    return null;
                }
                if (typeof res.content === 'string') {
                    const editor = document.getElementById('toml-editor');
                    if (editor) {
                        editor.value = res.content;
                        tomlSavedContent = res.content;
                    }
                }
                currentConfig.dataset_config = '';
                return { applied: true, response: res };
            }
            return { applied: false };
        }
        try {
            const res = await api('/api/config/dataset-presets/apply', {
                method: 'POST',
                body: JSON.stringify({
                    dataset_file: nextDataset,
                    train_file: file,
                    train_content: document.getElementById('toml-editor')?.value || '',
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '应用数据集预设失败');
                return null;
            }
            if (typeof res.train_content === 'string') {
                const editor = document.getElementById('toml-editor');
                if (editor) {
                    editor.value = res.train_content;
                    tomlSavedContent = res.train_content;
                }
            }
            currentConfig.dataset_config = res.dataset_config || nextDataset;
            const values = res.values || {};
            for (const [key, value] of Object.entries(values)) {
                currentConfig[key] = value;
            }
            return { applied: true, response: res };
        } catch (e) {
            setTomlStatus('error', '应用数据集预设失败: ' + e.message);
            return null;
        }
    }

    async function saveDatasetPresetEditor() {
        if (datasetPresetState.readonly) {
            setDatasetPresetStatus('系统数据集预设只读，请复制后编辑', 'error');
            return null;
        }
        let file = datasetPresetState.selectedFile || '';
        const wasUnnamedPreset = !file;
        if (!file) {
            const name = await showDatasetPresetNameDialog({
                title: '保存数据集预设',
                description: '当前预设还没有文件名。请输入一个名称，保存到 configs/datasets/。',
                confirmText: '保存预设',
            });
            if (name === null) return null;
            file = datasetPresetPathFromName(name);
            datasetPresetState.selectedFile = file;
        }
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setDatasetPresetStatus('请至少填写一个原始数据集路径', 'error');
            return null;
        }
        try {
            const res = await api('/api/config/dataset-presets', {
                method: 'PUT',
                body: JSON.stringify({
                    file,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                    overwrite: !(datasetPresetState.isNew || wasUnnamedPreset),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '保存数据集预设失败', 'error');
                return null;
            }
            datasetPresetState = {
                ...datasetPresetState,
                selectedFile: res.file || file,
                datasets: normalizeDatasetEditorRows(res.datasets || rows),
                defaults: normalizeDatasetDefaults(res.defaults || datasetPresetState.defaults || {}),
                dirty: false,
                isNew: false,
                readonly: false,
                status: res.message || '已保存数据集预设',
            };
            await loadDatasetPresets({ selectCurrent: false });
            await loadDatasetPreset(datasetPresetState.selectedFile);
            setDatasetPresetStatus(res.message || '已保存数据集预设', 'ok');
            if (selectedConfigDatasetFile === datasetPresetState.selectedFile) {
                selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
                await loadStepEstimate();
            }
            return res;
        } catch (e) {
            setDatasetPresetStatus('保存数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    async function createNewDatasetPreset() {
        if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，新建会丢弃这些修改。是否继续？'))) return;
        const name = await showDatasetPresetNameDialog({
            title: '新建数据集预设',
            description: '输入新预设名称，稍后保存时会写入 configs/datasets/。',
            confirmText: '创建预设',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (datasetPresetByFile(nextFile)) {
            setDatasetPresetStatus('数据集预设已存在，请换一个名称或使用复制/重命名', 'error');
            return;
        }
        datasetPresetState = {
            ...datasetPresetState,
            selectedFile: nextFile,
            datasets: normalizeDatasetEditorRows([{
                source_dir: '',
                image_dir: '',
                cache_dir: '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults({}),
            }]),
            defaults: normalizeDatasetDefaults({}),
            dirty: true,
            isNew: true,
            readonly: false,
            error: '',
            status: '新预设尚未保存',
        };
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    async function copyDatasetPreset() {
        if (!datasetPresetState.selectedFile) return;
        const name = await showDatasetPresetNameDialog({
            title: '复制数据集预设',
            description: '使用当前编辑器中的内容复制为新的数据集预设。',
            value: `${datasetPresetState.selectedFile.split('/').pop().replace(/\.toml$/i, '')}_copy`,
            confirmText: '复制预设',
        });
        if (name === null) return;
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        try {
            const res = await api('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '复制数据集预设失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false });
            await loadDatasetPreset(res.file);
            setDatasetPresetStatus('已复制数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('复制数据集预设失败: ' + e.message, 'error');
        }
    }

    async function renameDatasetPreset() {
        const oldFile = datasetPresetState.selectedFile;
        if (!oldFile || datasetPresetState.readonly) return;
        const name = await showDatasetPresetNameDialog({
            title: '重命名数据集预设',
            description: '会先保存为新预设，再删除旧 TOML；图片、缩放图和缓存目录不受影响。',
            value: oldFile.split('/').pop().replace(/\.toml$/i, ''),
            confirmText: '重命名',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (nextFile === oldFile) return;
        const saved = await copyDatasetPresetToName(name);
        if (!saved) return;
        try {
            const del = await api(`/api/config/dataset-presets?file=${encodeURIComponent(oldFile)}`, { method: 'DELETE' });
            if (!del.ok) {
                setDatasetPresetStatus(del.error || '新预设已保存，但旧预设删除失败', 'error');
                return;
            }
            if (selectedConfigDatasetFile === oldFile) selectedConfigDatasetFile = nextFile;
            await loadDatasetPresets({ selectCurrent: false });
            await loadDatasetPreset(nextFile);
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已重命名数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('重命名数据集预设失败: ' + e.message, 'error');
        }
    }

    async function copyDatasetPresetToName(name) {
        try {
            const res = await api('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: datasetRowsForPayload(datasetPresetState.datasets),
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '保存新数据集预设失败', 'error');
                return null;
            }
            return res;
        } catch (e) {
            setDatasetPresetStatus('保存新数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    async function deleteDatasetPreset() {
        const file = datasetPresetState.selectedFile;
        if (!file || datasetPresetState.readonly) return;
        const ok = await showAppConfirmDialog({
            title: '删除数据集预设',
            description: file,
            message: '只删除数据集预设 TOML，不删除图片、缩放图或缓存目录。',
            confirmText: '删除预设',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/config/dataset-presets?file=${encodeURIComponent(file)}`, { method: 'DELETE' });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集预设失败', 'error');
                return;
            }
            if (selectedConfigDatasetFile === file) {
                selectedConfigDatasetFile = '';
                selectedConfigDatasetSummary = null;
            }
            datasetPresetState.selectedFile = '';
            datasetPresetState.dirty = false;
            await loadDatasetPresets({ selectCurrent: false });
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已删除数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集预设失败: ' + e.message, 'error');
        }
    }

    function importDatasetPreset() {
        document.getElementById('dataset-import-input')?.click();
    }

    async function handleDatasetPresetImport(event) {
        const fileInput = event.target;
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            const content = await file.text();
            const name = await showDatasetPresetNameDialog({
                title: '导入数据集预设',
                description: '输入导入后的预设名称，文件会保存到 configs/datasets/。',
                value: file.name.replace(/\.toml$/i, ''),
                confirmText: '导入预设',
            });
            if (name === null) return;
            const target = datasetPresetPathFromName(name);
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file: target, content }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '导入数据集预设失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false });
            await loadDatasetPreset(target);
            setDatasetPresetStatus('已导入数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('导入数据集预设失败: ' + e.message, 'error');
        } finally {
            fileInput.value = '';
        }
    }

    async function exportDatasetPreset() {
        const file = datasetPresetState.selectedFile;
        if (!file) return;
        try {
            const data = await api(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
            if (!data.ok) {
                setDatasetPresetStatus(data.error || '导出数据集预设失败', 'error');
                return;
            }
            const blob = new Blob([data.content || ''], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = file.split('/').pop() || 'dataset.toml';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            setDatasetPresetStatus('已导出数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('导出数据集预设失败: ' + e.message, 'error');
        }
    }

    function datasetPresetPathFromName(name) {
        const stem = String(name || '')
            .replace(/\.toml$/i, '')
            .replace(/\\/g, '/')
            .split('/')
            .pop()
            .replace(/[^A-Za-z0-9_-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'dataset';
        return `configs/datasets/${stem}.toml`;
    }

    async function showDatasetPresetNameDialog(options = {}) {
        const name = await showHistoryTaskInputDialog({
            title: options.title || '数据集预设名称',
            description: options.description || '请输入数据集预设名称。',
            label: options.label || '预设名称',
            value: options.value || '',
            placeholder: options.placeholder || '例如 rokkotsu_goddess_v2',
            confirmText: options.confirmText || '确认',
        });
        if (name === null) return null;
        const clean = name.trim();
        if (!clean) {
            setDatasetPresetStatus('请输入数据集预设名称', 'error');
            return null;
        }
        return clean;
    }

    function setDatasetPresetStatus(message, level = '') {
        datasetPresetState.status = message || '';
        const header = document.getElementById('dataset-preset-header');
        if (!header) return;
        let status = header.querySelector('.dataset-preset-status');
        if (!status) {
            status = document.createElement('div');
            status.className = 'dataset-preset-status';
            header.appendChild(status);
        }
        status.textContent = message || '';
        status.className = ['dataset-preset-status', level].filter(Boolean).join(' ');
    }

    async function saveDatasetEditor(options = {}) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || currentTomlFile || '';
        const targetContent = options.trainContent ?? (document.getElementById('toml-editor')?.value || '');
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setTomlStatus('error', '请至少填写一个原始数据集路径');
            return null;
        }
        try {
            const res = await api('/api/config/datasets', {
                method: 'PUT',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    train_file: targetFile,
                    train_content: targetContent,
                    prefer_existing_dataset_config: options.preferExistingDatasetConfig !== false,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存数据集配置失败');
                return null;
            }
            if (typeof res.train_content === 'string' && res.train_content) {
                const editor = document.getElementById('toml-editor');
                if (editor && targetFile === (currentTomlFile || val('toml-file-select'))) {
                    editor.value = res.train_content;
                    tomlSavedContent = res.train_content;
                }
            }
        datasetEditorState = {
            loading: false,
            loaded: true,
            dirty: false,
            dataset_config: res.dataset_config || datasetEditorState.dataset_config,
            datasets: normalizeDatasetEditorRows(res.datasets || rows),
            defaults: normalizeDatasetDefaults(res.defaults || datasetEditorState.defaults || {}),
            error: '',
        };
        currentConfig.dataset_config = datasetEditorState.dataset_config;
        if (datasetEditorState.datasets[0]) {
            currentConfig.source_image_dir = datasetEditorState.datasets[0].source_dir;
            currentConfig.resized_image_dir = datasetEditorState.datasets[0].image_dir;
            currentConfig.lora_cache_dir = datasetEditorState.datasets[0].cache_dir;
            }
            syncDatasetEditorToCompatFields();
            renderDatasetEditor();
            updateTomlDirtyState();
            await loadStepEstimate();
            if (options.reloadList !== false) {
                await loadTomlFileList(targetFile);
            }
            return res;
        } catch (e) {
            setTomlStatus('error', '保存数据集配置失败: ' + e.message);
            return null;
        }
    }

    function collectChangedFormValues(options = {}) {
        const values = {};
        let networkArgsTouched = false;
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            const key = input.dataset.key;
            if (!key) return;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) return;
            if (isActiveNetworkArgFieldKey(key)) {
                if (networkArgInputChanged(input)) networkArgsTouched = true;
                return;
            }
            if (key === 'sample_prompts') {
                if (samplePromptsMode === 'path') {
                    const original = typeof currentConfig.sample_prompts === 'string' ? currentConfig.sample_prompts : '';
                    const next = readFieldInputValue(input, original);
                    if (!valuesEqual(next, original)) {
                        values[key] = next;
                    }
                    return;
                }
                const nextPrompts = readFieldInputValue(input, samplePromptsContent);
                if (nextPrompts !== samplePromptsContent) {
                    values[key] = nextPrompts;
                }
                return;
            }
            const hasOriginal = key in currentConfig;
            const original = hasOriginal ? currentConfig[key] : FORM_UI_DEFAULTS[key];
            const next = readFieldInputValue(input, original);
            if (!hasOriginal) {
                if (shouldSkipUiDefaultField(key, next, options)) return;
                values[key] = next;
                return;
            }
            if (!valuesEqual(next, original)) {
                values[key] = next;
            }
        });
        if (networkArgsTouched) {
            const merged = collectNetworkArgsFromForm({ network_args: values.network_args ?? currentConfig.network_args });
            if (!valuesEqual(merged.networkArgs, currentConfig.network_args || [])) {
                values.network_args = merged.networkArgs;
            } else if ('network_args' in values) {
                delete values.network_args;
            }
        }
        if (values.use_lokr === true && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        return values;
    }

    function networkArgInputChanged(input) {
        const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
        if (!spec) return false;
        const original = networkArgFieldValueFromConfig(spec, currentConfig);
        const next = readFieldInputValue(input, original);
        return !valuesEqual(next, original);
    }

    function networkArgFieldValueFromConfig(spec, config = currentConfig) {
        const argMap = parseNetworkArgMap(config?.network_args);
        return coerceNetworkArgValue(argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default, spec);
    }

    function collectNetworkArgsFromForm(baseConfig = currentConfig) {
        const baseArgs = normalizeNetworkArgArray(baseConfig?.network_args);
        const inputs = [...document.querySelectorAll('#config-form .field-input[data-key]')]
            .filter((input) => isActiveNetworkArgFieldKey(input.dataset.key));
        if (!inputs.length) {
            return { networkArgs: baseArgs, changed: !valuesEqual(baseArgs, currentConfig.network_args || []) };
        }

        const formValues = new Map();
        const changedKeys = new Set();
        for (const input of inputs) {
            const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
            const original = networkArgFieldValueFromConfig(spec, currentConfig);
            const next = readFieldInputValue(input, original);
            formValues.set(spec.arg, { spec, value: next });
            if (!valuesEqual(next, original)) changedKeys.add(spec.key);
        }

        const result = [];
        const seenArgs = new Set();
        for (const raw of baseArgs) {
            const parsed = parseNetworkArgEntry(raw);
            if (!parsed || !formValues.has(parsed.arg)) {
                result.push(raw);
                continue;
            }
            seenArgs.add(parsed.arg);
            const { spec, value } = formValues.get(parsed.arg);
            result.push(formatNetworkArg(spec, value));
        }

        for (const { spec, value } of formValues.values()) {
            if (seenArgs.has(spec.arg)) continue;
            if (!changedKeys.has(spec.key)) continue;
            result.push(formatNetworkArg(spec, value));
        }

        return {
            networkArgs: result,
            changed: !valuesEqual(result, currentConfig.network_args || []),
        };
    }

    function formatNetworkArg(spec, value) {
        return `${spec.arg}=${formatNetworkArgValue(spec, value)}`;
    }

    function formatNetworkArgValue(spec, value) {
        if (spec.valueType === 'booleanInt') return parseBooleanNetworkArg(value, spec.default) ? '1' : '0';
        if (spec.valueType === 'boolean') return parseBooleanNetworkArg(value, spec.default) ? 'true' : 'false';
        if (spec.valueType === 'integer') {
            const n = Number(value);
            return Number.isFinite(n) ? String(Math.trunc(n)) : String(spec.default);
        }
        if (spec.valueType === 'number') {
            const n = Number(value);
            return Number.isFinite(n) ? String(n) : String(spec.default);
        }
        return String(value ?? '').trim();
    }

    async function prepareFormPatchValues(values) {
        const nextValues = { ...values };
        if ('sample_prompts' in nextValues && samplePromptsMode !== 'path') {
            const promptText = String(nextValues.sample_prompts || '');
            if (promptText.trim()) {
                const saved = await saveSamplePrompts(promptText);
                nextValues.sample_prompts = saved.file || samplePromptsPath;
            } else {
                await saveSamplePrompts('');
                nextValues.sample_prompts = '';
            }
        }
        return nextValues;
    }

    function shouldSkipUiDefaultField(key, value, options = {}) {
        if (!(key in FORM_UI_DEFAULTS)) return false;
        if (options.persistDefaultFields && FORM_UI_PERSIST_DEFAULT_FIELDS.has(key)) return false;
        if (OPTIONAL_EMPTY_FIELDS.has(key) && value === '') return true;
        return valuesEqual(value, FORM_UI_DEFAULTS[key]);
    }

    function readFieldInputValue(input, originalValue) {
        if (input.classList?.contains('sample-prompts-editor')) {
            if (input.dataset.touched !== '1') return input.dataset.originalContent || '';
            return serializeSamplePromptsEditor(input);
        }
        if (input.tagName === 'TEXTAREA') return normalizeMultilineText(input.value);
        if (input.type === 'checkbox') return input.checked;
        const raw = input.value;
        switch (input.dataset.valueType || fieldValueType(originalValue)) {
            case 'number':
                return parseNumberValue(raw, originalValue);
            case 'boolean':
                return raw === 'true';
            case 'array':
                return parseArrayValue(raw);
            default:
                return raw;
        }
    }

    function readLoKrEnabled() {
        const input = document.querySelector('#config-form .field-input[data-key="use_lokr"]');
        if (!input) return currentConfig.use_lokr === true;
        return readFieldInputValue(input, currentConfig.use_lokr ?? FORM_UI_DEFAULTS.use_lokr) === true;
    }

    function updateLoKrFieldState() {
        const factorInput = document.querySelector('#config-form .field-input[data-key="lokr_factor"]');
        if (!factorInput) return;
        const enabled = readLoKrEnabled();
        factorInput.disabled = !enabled;
        factorInput.title = enabled ? '' : '启用 LoKr 后生效';
        const row = factorInput.closest('.field-row');
        if (row) row.classList.toggle('field-row-disabled', !enabled);
    }

    function parseNumberValue(raw, fallback) {
        const trimmed = String(raw).trim();
        if (trimmed === '' && fallback === '') return '';
        if (trimmed === '') return fallback;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : fallback;
    }

    function parseArrayValue(raw) {
        const trimmed = String(raw).trim();
        if (!trimmed) return [];
        try {
            const parsed = JSON.parse(trimmed);
            return Array.isArray(parsed) ? parsed : [parsed];
        } catch {
            return trimmed.split(',').map((item) => item.trim()).filter(Boolean);
        }
    }

    function valuesEqual(a, b) {
        if (isBooleanLikeValue(a) && isBooleanLikeValue(b)) {
            return normalizeBooleanLikeValue(a) === normalizeBooleanLikeValue(b);
        }
        if (isNumberLikeValue(a) && isNumberLikeValue(b)) {
            return Number(a) === Number(b);
        }
        return JSON.stringify(a) === JSON.stringify(b);
    }

    function isBooleanLikeValue(value) {
        return value === true || value === false || value === 'true' || value === 'false';
    }

    function normalizeBooleanLikeValue(value) {
        return value === true || value === 'true';
    }

    function isNumberLikeValue(value) {
        if (typeof value === 'number') return Number.isFinite(value);
        if (typeof value !== 'string') return false;
        const trimmed = value.trim();
        return trimmed !== '' && Number.isFinite(Number(trimmed));
    }

    function normalizeMultilineText(value) {
        return String(value || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .join('\n');
    }

    function currentSamplePromptText(config) {
        const raw = typeof config.sample_prompts === 'string' ? config.sample_prompts.trim() : '';
        samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
        samplePromptsContent = '';

        if (!raw) {
            samplePromptsMode = 'editor-inline';
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isEditableSamplePromptsTextFilePath(raw)) {
            samplePromptsMode = 'editor-file';
            samplePromptsPath = normalizeSamplePromptsPath(raw);
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isSamplePromptsFilePath(raw)) {
            samplePromptsMode = 'path';
            return raw;
        }

        samplePromptsMode = 'editor-inline';
        samplePromptsContent = raw;
        return raw;
    }

    function normalizeSamplePromptsPath(value) {
        return String(value || '').replace(/\\/g, '/').trim();
    }

    function isEditableSamplePromptsTextFilePath(value) {
        const text = normalizeSamplePromptsPath(value);
        if (!text.toLowerCase().endsWith('.txt')) return false;
        if (!text.startsWith('configs/')) return false;
        return !text.split('/').includes('..');
    }

    function isSamplePromptsFilePath(value) {
        const text = normalizeSamplePromptsPath(value).toLowerCase();
        return text.endsWith('.txt') || text.endsWith('.toml') || text.endsWith('.json');
    }

    async function loadSamplePrompts(filePath = samplePromptsPath, parentSeq = configLoadSeq) {
        if (location.protocol === 'file:') return;
        const requestSeq = ++samplePromptsLoadSeq;
        try {
            const data = await api(`/api/config/sample-prompts?file=${encodeURIComponent(filePath || samplePromptsPath)}`);
            if (parentSeq !== configLoadSeq || requestSeq !== samplePromptsLoadSeq) return;
            if (data?.ok === false) {
                throw new Error(data.error || '读取预览提示词失败');
            }
            samplePromptsPath = data.file || samplePromptsPath;
            samplePromptsContent = data.content || '';
            const input = document.querySelector('#config-form .field-input[data-key="sample_prompts"]');
            if (input) {
                if (input.classList?.contains('sample-prompts-editor')) {
                    setSamplePromptsEditorContent(input, samplePromptsContent);
                } else {
                    input.value = samplePromptsContent;
                }
            }
        } catch (e) {
            console.warn('读取预览提示词失败:', e);
        }
    }

    async function saveSamplePrompts(content) {
        const res = await api('/api/config/sample-prompts', {
            method: 'PUT',
            body: JSON.stringify({ file: samplePromptsPath, content }),
        });
        if (!res.ok) {
            throw new Error(res.error || '保存预览提示词失败');
        }
        samplePromptsPath = res.file || samplePromptsPath;
        samplePromptsContent = res.content || '';
        return res;
    }

    async function importTomlFile() {
        if (!(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，导入会覆盖编辑器内容。是否继续？'))) {
            return;
        }
        const input = document.getElementById('toml-import-input');
        if (!input) return;
        input.value = '';
        input.click();
    }

    function handleTomlImport(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = () => {
            currentTomlFile = '';
            tomlSavedContent = '';
            document.getElementById('toml-current-file').textContent = `未保存导入: ${file.name}`;
            document.getElementById('toml-file-select').value = '';
            document.getElementById('toml-editor').value = reader.result || '';
            setTomlEditorLocked(false);
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            setTomlStatus('ok', `已导入 ${file.name}，点击保存或另存为写入项目`, { persist: true });
        };
        reader.onerror = () => {
            setTomlStatus('error', '导入失败: 无法读取本地文件');
        };
        reader.readAsText(file, 'utf-8');
    }

    function exportTomlFile() {
        const content = document.getElementById('toml-editor').value;
        const file = currentTomlFile || val('toml-file-select');
        const filename = exportTomlFilename(file);
        const blob = new Blob([content], { type: 'application/toml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    async function saveTomlAs() {
        const editor = document.getElementById('toml-editor');
        const currentFile = currentTomlFile;
        const target = await showTomlSaveAsDialog(currentFile);
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '另存新配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === currentFile) {
            setTomlStatus('error', '另存新配置失败: 新配置不能和当前选中文件同名');
            return;
        }
        if (tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        try {
            const baseContent = editor.value;
            const preparedValues = await prepareFormPatchValues(collectChangedFormValues({ persistDefaultFields: true }));
            const content = Object.keys(preparedValues).length
                ? await previewPatchedTomlContent(file, baseContent, preparedValues)
                : baseContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存为失败');
                return;
            }

            currentTrainingSource = {
                method: file.split('/').pop().replace(/\.toml$/i, ''),
                methods_subdir: 'imported',
                file,
            };
            currentTomlFile = file;
            tomlSavedContent = content;
            editor.value = content;
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) {
                setTomlStatus('error', `新配置已创建: ${file}，但数据集预设应用失败，请修正后再次保存更新当前选中配置`, { persist: true });
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            if (datasetApplied.applied) {
                const editorAfterDataset = document.getElementById('toml-editor');
                tomlSavedContent = editorAfterDataset?.value || tomlSavedContent;
            }
            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file);
            await applyTomlToConfig({ silent: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已另存新配置: ${file} → ${groupLabel}`);
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function createBlankPresetFromLoraTemplate() {
        let templateContent = '';
        try {
            const data = await api(`/api/config/raw?file=${encodeURIComponent(BLANK_PRESET_TEMPLATE_FILE)}`);
            if (data?.ok === false) {
                setTomlStatus('error', data.error || '读取 LoRA 模板失败');
                return;
            }
            templateContent = typeof data.content === 'string' ? data.content : '';
        } catch (e) {
            setTomlStatus('error', '读取 LoRA 模板失败: ' + e.message);
            return;
        }
        if (!templateContent.trim()) {
            setTomlStatus('error', `读取 LoRA 模板失败: ${BLANK_PRESET_TEMPLATE_FILE} 内容为空或不存在`);
            return;
        }

        const target = await showTomlSaveAsDialog(BLANK_PRESET_TEMPLATE_FILE, {
            title: '创建空白预设配置',
            description: `以 ${BLANK_PRESET_TEMPLATE_LABEL} 为模板，并套用全局基础模型路径，创建一个新的可编辑项目预设。`,
            confirmText: '创建空白预设配置',
            hint: '新文件默认创建到 configs/imported/；分组只影响右侧列表归类。全局模型路径只作为初始默认值，创建后仍可在配置页覆盖。',
            currentText: `模板: ${BLANK_PRESET_TEMPLATE_LABEL} (${BLANK_PRESET_TEMPLATE_FILE})`,
        });
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '创建空白预设配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === BLANK_PRESET_TEMPLATE_FILE) {
            setTomlStatus('error', '创建空白预设配置失败: 不能覆盖 LoRA 标准模板');
            return;
        }
        if (tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        const canSwitch = await handlePendingConfigSwitch({ targetLabel: `新的空白预设配置 ${file.split('/').pop() || file}` });
        if (!canSwitch) return;

        try {
            // 空白预设先复用模板，再把全局默认基础模型路径灌进去，减少新建后手工改三项的次数。
            const globalModelPathOverrides = getGlobalModelPathOverrides();
            const content = Object.keys(globalModelPathOverrides).length
                ? await previewPatchedTomlContent(file, templateContent, globalModelPathOverrides)
                : templateContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建空白预设配置失败');
                return;
            }

            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file, { force: true });
            await applyTomlToConfig({ silent: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已创建空白预设配置: ${file} → ${groupLabel}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '创建空白预设配置失败: ' + e.message);
        }
    }

    async function previewPatchedTomlContent(file, content, values) {
        const res = await api('/api/config/raw/patch-preview', {
            method: 'POST',
            body: JSON.stringify({ file, content, values }),
        });
        if (!res.ok) {
            throw new Error(res.error || '应用表单修改失败');
        }
        return typeof res.content === 'string' ? res.content : content;
    }

    async function showTomlSaveAsDialog(currentFile, options = {}) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-save-as-dialog-body';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const labelText = document.createElement('span');
        labelText.textContent = options.nameLabel || '新配置名称或 configs/ 路径';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = '';
        input.placeholder = options.placeholder || '例如 rokkotsu_v2 或 configs/imported/rokkotsu_v2.toml';
        input.className = 'history-task-dialog-input';
        label.append(labelText, input);

        const groups = saveAsTargetGroups();
        const groupWrap = document.createElement('div');
        groupWrap.className = 'toml-save-as-group-list';
        const groupTitle = document.createElement('span');
        groupTitle.className = 'toml-save-as-group-title';
        groupTitle.textContent = '保存到分组';
        groupWrap.appendChild(groupTitle);

        const radios = [];
        for (const group of groups) {
            const option = document.createElement('label');
            option.className = 'toml-move-option';

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'toml-save-as-target-group';
            radio.value = group.id;
            radio.checked = group.id === 'imported' || (!radios.length && !groups.some((item) => item.id === 'imported'));
            radios.push(radio);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            detail.textContent = `${(group.files || []).length} 个配置`;
            text.append(title, detail);

            option.append(radio, text);
            groupWrap.appendChild(option);
        }

        const hint = document.createElement('p');
        hint.className = 'toml-save-as-hint';
        hint.textContent = options.hint || '只填写文件名时会创建到 configs/imported/，分组只影响右侧列表归类；必须使用新名称，不会覆盖当前选中配置。';

        const current = document.createElement('p');
        current.className = 'toml-save-as-current';
        current.textContent = options.currentText
            || (currentFile ? `当前选中配置: ${currentFile}` : '当前没有选中的配置文件，将使用编辑器内容创建新配置。');

        wrap.append(label, groupWrap, hint, current);

        return showHistoryTaskDialog({
            title: options.title || '另存新配置',
            description: options.description || '输入一个新名称，并选择它在右侧配置列表中的目标分组。',
            body: wrap,
            confirmText: options.confirmText || '创建配置文件',
            onOpen: () => input.focus(),
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-save-as-target-group"]:checked');
                return {
                    name: input.value,
                    group: checked?.value || 'imported',
                };
            },
        });
    }

    function saveAsTargetGroups() {
        const groups = reorderTomlFileGroups(tomlFileGroups)
            .filter((group) => group.trainable && group.movable && !group.locked && !group.user_group_locked);
        if (groups.some((group) => group.id === 'imported')) return groups;
        const imported = tomlFileGroups.find((group) => group.id === 'imported');
        if (imported && imported.trainable && !imported.locked && !imported.user_group_locked) {
            return [imported, ...groups];
        }
        return groups.length ? groups : [{
            id: 'imported',
            label: '导入配置',
            files: [],
        }];
    }

    async function moveTomlFileToGroup(file, groupId) {
        if (!groupId || groupId === 'imported') return true;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: groupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存成功，但移动到指定分组失败');
                return false;
            }
            return true;
        } catch (e) {
            setTomlStatus('error', '另存成功，但移动分组请求失败: ' + e.message);
            return false;
        }
    }

    function normalizeTomlSaveAsPath(rawPath) {
        let file = String(rawPath || '').trim().replace(/\\/g, '/');
        file = file.replace(/^\/+/, '');
        if (!file) return '';
        if (!file.startsWith('configs/')) {
            file = `configs/imported/${file}`;
        }
        if (!file.toLowerCase().endsWith('.toml')) {
            file += '.toml';
        }
        return file;
    }

    function exportTomlFilename(filePath) {
        const base = String(filePath || '').split('/').filter(Boolean).pop();
        if (!base) return 'anima-config.toml';
        return base.toLowerCase().endsWith('.toml') ? base : `${base}.toml`;
    }

    function isFixedSystemTomlGroup(group) {
        return Boolean(
            group.id === 'web_config' ||
            group.id === 'presets' ||
            group.id === 'methods' ||
            group.id === 'gui_methods' ||
            group.system_locked
        );
    }

    function shouldShowTomlGroup(group) {
        return !isFixedSystemTomlGroup(group);
    }

    function reorderTomlFileGroups(groups) {
        return [...(groups || [])]
            .map((group, index) => ({ group, index }))
            .filter(({ group }) => group.user_managed || group.lockable || (group.files || []).length > 0)
            .sort((a, b) => {
                const aFixed = isFixedSystemTomlGroup(a.group);
                const bFixed = isFixedSystemTomlGroup(b.group);
                if (aFixed !== bFixed) return aFixed ? 1 : -1;
                return a.index - b.index;
            })
            .map((item) => item.group);
    }

    function getSortableTomlGroups() {
        return [...(tomlFileGroups || [])]
            .filter((group) => !isFixedSystemTomlGroup(group) && (group.user_managed || group.lockable || (group.files || []).length > 0));
    }

    function populateTomlFileSelect(groups) {
        const sel = document.getElementById('toml-file-select');
        const prev = sel.value;
        sel.innerHTML = '';
        for (const group of groups) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = group.label || group.id || '配置文件';
            for (const item of group.files || []) {
                const opt = document.createElement('option');
                opt.value = item.path;
                opt.textContent = [tomlLockLabel(item), tomlFileDisplayName(item)].filter(Boolean).join(' / ');
                opt.dataset.locked = item.locked ? '1' : '0';
                optgroup.appendChild(opt);
            }
            sel.appendChild(optgroup);
        }
        if (tomlFiles.includes(prev)) {
            sel.value = prev;
        }
        renderTomlFileGroups(groups);
    }

    function renderTomlFileGroups(groups) {
        const container = document.getElementById('toml-file-groups');
        if (!container) return;
        container.innerHTML = '';
        const stored = readTomlGroupState();

        const toolbar = document.createElement('div');
        toolbar.className = 'toml-group-toolbar';
        const createBtn = document.createElement('button');
        createBtn.type = 'button';
        createBtn.className = 'toml-group-action-btn';
        createBtn.textContent = '新建分组';
        createBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            runTomlGroupAction(createTomlGroup, createBtn);
        });
        toolbar.appendChild(createBtn);
        container.appendChild(toolbar);

        const visibleGroups = (groups || []).filter(shouldShowTomlGroup);
        if (visibleGroups.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'toml-file-group-empty';
            empty.textContent = '系统分组已隐藏。可点击“新建分组”创建自己的配置分组。';
            container.appendChild(empty);
        }

        for (const group of visibleGroups) {
            const details = document.createElement('details');
            details.className = 'toml-file-group';
            if (group.locked) details.classList.add('readonly');
            details.dataset.groupId = group.id;
            details.open = stored[group.id] ?? Boolean(group.open);
            details.addEventListener('toggle', () => {
                const next = readTomlGroupState();
                next[group.id] = details.open;
                writeTomlGroupState(next);
            });

            const summary = document.createElement('summary');
            const orderActions = createTomlGroupOrderActions(group);
            if (orderActions) summary.appendChild(orderActions);
            const title = document.createElement('span');
            title.className = 'toml-group-title';
            title.textContent = `${group.label || group.id} (${(group.files || []).length})`;
            summary.appendChild(title);
            const actions = createTomlGroupActions(group);
            if (actions) summary.appendChild(actions);
            if (group.lockable) {
                const groupLockBtn = document.createElement('button');
                groupLockBtn.type = 'button';
                groupLockBtn.className = 'toml-group-lock-btn';
                groupLockBtn.textContent = group.user_group_locked ? '解除分组锁定' : '锁定分组';
                groupLockBtn.title = group.user_group_locked
                    ? '解除该分组的用户锁定'
                    : '锁定该分组内所有文件，防止误保存';
                groupLockBtn.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    runTomlGroupAction(() => toggleTomlGroupLock(group), groupLockBtn);
                });
                summary.appendChild(groupLockBtn);
            }
            if (group.locked) {
                const badge = document.createElement('em');
                badge.textContent = group.user_group_locked ? '分组锁定' : '锁定';
                summary.appendChild(badge);
            }
            details.appendChild(summary);

            const list = document.createElement('div');
            list.className = 'toml-file-list';
            const files = group.files || [];
            if (!files.length) {
                const empty = document.createElement('div');
                empty.className = 'toml-file-group-empty';
                empty.textContent = group.user_managed ? '空分组，可使用“移动”放入当前配置。' : '暂无配置文件。';
                list.appendChild(empty);
            }
            files.forEach((item, index) => {
                list.appendChild(createTomlFileButton(item, group, index, files.length));
            });
            details.appendChild(list);
            container.appendChild(details);
        }
        updateTomlSelectionUI(currentTomlFile);
    }

    function createTomlGroupOrderActions(group) {
        if (isFixedSystemTomlGroup(group)) return null;
        const sortableGroups = getSortableTomlGroups();
        const groupIndex = sortableGroups.findIndex((item) => item.id === group.id);
        if (groupIndex < 0) return null;

        const wrap = document.createElement('span');
        wrap.className = 'toml-group-order-actions';
        wrap.appendChild(createTomlGroupActionButton('上移', () => reorderTomlGroup(group, 'up'), {
            disabled: groupIndex <= 0,
            title: groupIndex <= 0 ? '已经是最上面的可移动分组' : '把这个分组上移一位',
            variant: 'order',
        }));
        wrap.appendChild(createTomlGroupActionButton('下移', () => reorderTomlGroup(group, 'down'), {
            disabled: groupIndex >= sortableGroups.length - 1,
            title: groupIndex >= sortableGroups.length - 1 ? '已经是最下面的可移动分组' : '把这个分组下移一位',
            variant: 'order',
        }));
        return wrap;
    }

    function createTomlGroupActions(group) {
        const wrap = document.createElement('span');
        wrap.className = 'toml-group-actions';

        if (group.renamable) {
            wrap.appendChild(createTomlGroupActionButton('重命名', () => renameTomlGroup(group), {
                title: '重命名这个配置分组',
            }));
        }
        wrap.appendChild(createTomlGroupActionButton('删除分组', () => deleteTomlGroup(group), {
            title: deleteTomlGroupButtonTitle(group),
            danger: true,
            disabled: !canDeleteTomlGroup(group),
        }));
        return wrap;
    }

    function createTomlGroupActionButton(label, handler, options = {}) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'toml-group-action-btn',
            options.variant ? `toml-group-action-btn-${options.variant}` : '',
            options.danger ? 'danger' : '',
        ].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(options.disabled);
        btn.title = options.title || label;
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (!btn.disabled) runTomlGroupAction(handler, btn);
        });
        return btn;
    }

    function runTomlGroupAction(handler, button = null) {
        if (tomlGroupActionBusy) return;
        tomlGroupActionBusy = true;
        if (button) button.disabled = true;
        Promise.resolve()
            .then(handler)
            .catch((e) => {
                setTomlStatus('error', '分组操作失败: ' + e.message);
            })
            .finally(() => {
                tomlGroupActionBusy = false;
                if (button?.isConnected) button.disabled = false;
            });
    }

    function createTomlFileButton(item, group = null, index = 0, total = 1) {
        const row = document.createElement('div');
        row.className = 'toml-file-row-wrap';
        row.dataset.file = item.path;
        row.dataset.groupId = group?.id || item.group || '';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'toml-file-item';
        if (item.locked) btn.classList.add('readonly');
        btn.dataset.file = item.path;
        btn.title = tomlFileDisplayName(item);
        btn.addEventListener('click', () => selectAndApplyTomlFile(item.path));

        const name = document.createElement('span');
        name.className = 'toml-file-name';
        name.textContent = item.label || item.path;
        btn.appendChild(name);

        const meta = document.createElement('span');
        meta.className = 'toml-file-meta';
        const tags = [];
        if (item.filename && item.filename !== item.label) tags.push(item.filename);
        if (currentTrainingSource.file === item.path) tags.push('当前训练');
        const lockLabel = tomlLockLabel(item);
        if (lockLabel) tags.push(lockLabel);
        tags.push(item.trainable ? '可训练' : '非训练');
        tags.push(item.path);
        meta.textContent = tags.join(' / ');
        btn.appendChild(meta);
        row.appendChild(btn);

        if (group && total > 1) {
            const actions = document.createElement('div');
            actions.className = 'toml-file-order-actions';
            actions.appendChild(createTomlFileOrderButton('↑', '上移', item, group, 'up', index <= 0));
            actions.appendChild(createTomlFileOrderButton('↓', '下移', item, group, 'down', index >= total - 1));
            row.appendChild(actions);
        }
        return row;
    }

    function createTomlFileOrderButton(label, title, item, group, direction, disabled) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'toml-file-order-btn';
        btn.textContent = label;
        btn.title = `${title}: ${tomlFileDisplayName(item)}`;
        btn.disabled = Boolean(disabled);
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (!btn.disabled) reorderTomlFileInGroup(item.path, group.id, direction);
        });
        return btn;
    }

    function updateTomlSelectionUI(filePath) {
        document.querySelectorAll('.toml-file-item').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.file === filePath);
        });
        const label = document.getElementById('toml-current-file');
        if (label) label.textContent = filePath ? tomlFileDisplayName(filePath) : '未保存导入内容';
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            const meta = tomlFileMeta[filePath];
            const dirty = hasPendingConfigChanges(filePath);
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存或另存为'
                : (meta?.trainable ? '将该配置作为当前表单和训练入口' : '该文件不是完整训练配置');
        }
        updateTomlBadges(filePath);
    }

    function isTomlDirty() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return false;
        return editor.value !== tomlSavedContent;
    }

    function hasUnsavedFormChanges(filePath = currentTomlFile) {
        if (!filePath || currentTrainingSource.file !== filePath) return false;
        if (!currentConfig || Object.keys(currentConfig).length === 0) return false;
        return datasetEditorState.dirty
            || selectedConfigDatasetFile !== (currentConfig.dataset_config || '')
            || Object.keys(collectChangedFormValues()).length > 0;
    }

    function hasPendingConfigChanges(filePath = currentTomlFile) {
        return isTomlDirty() || hasUnsavedFormChanges(filePath);
    }

    async function confirmDiscardTomlChanges(message) {
        if (!hasPendingConfigChanges(currentTomlFile)) return true;
        return confirmUnsavedDiscard(message);
    }

    function confirmUnsavedDiscard(message) {
        return showAppConfirmDialog({
            title: '未保存更改',
            description: '当前页面有尚未保存的修改',
            message,
            confirmText: '继续并丢弃',
            cancelText: '留在当前页面',
            danger: true,
        });
    }

    function collectPendingConfigChangeDetails(pending = pendingConfigSwitchState()) {
        const changes = [];
        if (pending.formDirty) {
            if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
                changes.push({
                    label: '数据集预设 / dataset_config',
                    original: currentConfig.dataset_config || '未设置',
                    next: selectedConfigDatasetFile || '未设置',
                });
            }
            for (const [key, nextValue] of Object.entries(collectChangedFormValues())) {
                changes.push({
                    label: formatFieldName(key),
                    original: originalValueForChange(key),
                    next: nextValue,
                });
            }
            if (datasetEditorState.dirty) {
                changes.push({
                    label: '多数据集路径与参数',
                    original: currentConfig.dataset_config || '当前配置内的数据集字段',
                    next: summarizeDatasetEditorState(datasetEditorState),
                });
            }
        }
        if (pending.editorDirty) {
            const editorValue = document.getElementById('toml-editor')?.value || '';
            changes.push({
                label: '直接编辑 TOML',
                original: summarizeTextChange(tomlSavedContent),
                next: summarizeTextChange(editorValue),
            });
        }
        return changes;
    }

    function originalValueForChange(key) {
        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            return samplePromptsContent || '';
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return networkArgFieldValueFromConfig(NETWORK_ARG_FIELD_MAP.get(key), currentConfig);
        }
        if (key in currentConfig) return currentConfig[key];
        return FORM_UI_DEFAULTS[key];
    }

    function summarizeDatasetEditorState(state) {
        const rows = normalizeDatasetEditorRows(state.datasets || []);
        const parts = rows.map((row, index) => {
            const settings = normalizeDatasetDefaults(row.settings || state.defaults || {});
            return [
                `第 ${index + 1} 组`,
                row.source_dir || '未设置原始路径',
                `重复 ${row.num_repeats || 1}`,
                `${settings.resolution}px`,
            ].join(' · ');
        });
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        parts.push(`通用标注 ${defaults.caption_extension} · keep_tokens ${defaults.keep_tokens}`);
        return parts.join('\n');
    }

    function summarizeTextChange(text) {
        const value = String(text || '');
        const lines = value.split(/\r?\n/).length;
        const chars = value.length;
        const preview = value.split(/\r?\n/).find((line) => line.trim()) || '空内容';
        return `${lines} 行 / ${chars} 字符\n${preview}`;
    }

    function formatConfigChangeValue(value) {
        let text;
        if (typeof value === 'string') {
            text = value;
        } else {
            try {
                text = JSON.stringify(value, null, 2);
            } catch {
                text = String(value);
            }
        }
        if (text === '') text = '空';
        return text.length > 600 ? `${text.slice(0, 600)}\n...` : text;
    }

    function showConfigSwitchToast(filePath, stateText) {
        const toast = document.getElementById('config-switch-toast');
        if (!toast) return;
        if (configSwitchToastTimer) {
            clearTimeout(configSwitchToastTimer);
            configSwitchToastTimer = null;
        }
        const file = (filePath || currentTomlFile || '当前配置').split('/').pop() || '当前配置';
        toast.textContent = `${file}，${stateText}`;
        toast.hidden = false;
        configSwitchToastTimer = setTimeout(() => {
            toast.hidden = true;
            configSwitchToastTimer = null;
        }, 2000);
    }

    async function handlePendingConfigSwitch({ targetLabel = '' } = {}) {
        const pending = pendingConfigSwitchState();
        if (!pending.hasChanges) return true;
        const action = await showUnsavedConfigSwitchDialog({ pending, targetLabel });
        if (action === 'cancel') return false;
        if (action === 'discard') {
            showConfigSwitchToast(pendingToastLabel(pending), '更改未保存');
            return true;
        }
        const saved = await savePendingConfigSwitchChanges(pending);
        if (!saved) return false;
        showConfigSwitchToast(pendingToastLabel(pending), '更改保存完成');
        return true;
    }

    function pendingConfigSwitchState() {
        const editorFile = currentTomlFile || val('toml-file-select') || '';
        const formFile = currentTrainingSource.file || '';
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(formFile);
        const dirtyFiles = [];
        if (editorDirty && editorFile) dirtyFiles.push(editorFile);
        if (formDirty && formFile && !dirtyFiles.includes(formFile)) dirtyFiles.push(formFile);
        const canSave = dirtyFiles.length > 0 && dirtyFiles.every((file) => file && !isTomlLocked(file));
        return {
            editorDirty,
            editorFile,
            formDirty,
            formFile,
            dirtyFiles,
            sourceFile: dirtyFiles[0] || formFile || editorFile || '',
            hasChanges: editorDirty || formDirty,
            canSave,
        };
    }

    function pendingToastLabel(pending) {
        const files = pending?.dirtyFiles || [];
        if (files.length > 1) {
            const first = files[0].split('/').pop() || files[0];
            return `${first} 等 ${files.length} 个配置`;
        }
        return pending?.sourceFile || currentTomlFile || '当前配置';
    }

    async function savePendingConfigSwitchChanges(pending) {
        if (pending.editorDirty) {
            const savedEditor = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedEditor) return false;
        }
        if (pending.formDirty && (!pending.editorDirty || pending.formFile !== pending.editorFile)) {
            if (currentTomlFile !== pending.formFile) {
                await loadTomlFile(pending.formFile, { force: true });
            }
            const savedForm = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedForm) return false;
        }
        return true;
    }

    function showUnsavedConfigSwitchDialog({ pending = pendingConfigSwitchState(), targetLabel = '' } = {}) {
        const dialog = document.getElementById('history-task-dialog');
        const title = document.getElementById('history-task-dialog-title');
        const desc = document.getElementById('history-task-dialog-desc');
        const body = document.getElementById('history-task-dialog-body');
        const cancelBtn = document.getElementById('history-task-dialog-cancel');
        const confirmBtn = document.getElementById('history-task-dialog-confirm');
        const closeBtn = dialog?.querySelector('.history-task-dialog-header button[value="cancel"]');
        if (!dialog || !title || !desc || !body || !cancelBtn || !confirmBtn) {
            return Promise.resolve('cancel');
        }
        if (sharedDialogBusy || dialog.open) {
            return Promise.resolve('cancel');
        }
        sharedDialogBusy = true;

        title.textContent = '有更改待保存';
        desc.textContent = targetLabel ? `即将切换到 ${targetLabel}` : '即将切换配置';
        body.innerHTML = '';
        body.appendChild(createConfigSwitchDialogBody(pending));
        cancelBtn.textContent = '放弃未保存的更改';
        cancelBtn.value = 'discard';
        confirmBtn.textContent = '保存更改并切换';
        confirmBtn.value = 'save';
        confirmBtn.disabled = !pending.canSave;
        confirmBtn.title = pending.canSave ? '' : '存在只读配置，不能直接保存；请先另存为可编辑配置，或放弃未保存更改后切换。';
        confirmBtn.classList.remove('btn-danger');
        confirmBtn.classList.add('btn-primary');
        dialog.returnValue = '';

        return new Promise((resolve) => {
            const cleanup = () => {
                dialog.removeEventListener('close', handleClose);
                sharedDialogBusy = false;
                cancelBtn.value = 'cancel';
                confirmBtn.value = 'confirm';
                confirmBtn.title = '';
                if (closeBtn) closeBtn.value = 'cancel';
            };
            const handleClose = () => {
                const action = dialog.returnValue === 'save'
                    ? 'save'
                    : (dialog.returnValue === 'discard' ? 'discard' : 'cancel');
                cleanup();
                resolve(action);
            };
            dialog.addEventListener('close', handleClose);
            try {
                if (dialog.showModal) {
                    dialog.showModal();
                } else {
                    dialog.setAttribute('open', 'open');
                }
            } catch {
                cleanup();
                resolve('cancel');
                return;
            }
            requestAnimationFrame(() => cancelBtn.focus());
        });
    }

    function createConfigSwitchDialogBody(pending = pendingConfigSwitchState()) {
        const wrap = document.createElement('div');
        wrap.className = 'config-switch-dialog-body';

        const intro = document.createElement('p');
        intro.textContent = '当前配置有未保存修改。请先选择保存后切换，或放弃这些修改后继续切换。';
        wrap.appendChild(intro);

        const list = document.createElement('div');
        list.className = 'config-switch-change-list';
        const changes = collectPendingConfigChangeDetails(pending);
        if (!changes.length) {
            const empty = document.createElement('p');
            empty.textContent = '检测到未保存状态，但没有可展示的字段差异。';
            list.appendChild(empty);
        }
        for (const change of changes) {
            const item = document.createElement('article');
            item.className = 'config-switch-change-item';

            const label = document.createElement('strong');
            label.textContent = change.label;
            item.appendChild(label);

            const values = document.createElement('div');
            values.className = 'config-switch-change-values';
            values.appendChild(createConfigSwitchChangeValue('原始', change.original));
            values.appendChild(createConfigSwitchChangeValue('未保存的更改', change.next));
            item.appendChild(values);
            list.appendChild(item);
        }
        wrap.appendChild(list);
        return wrap;
    }

    function createConfigSwitchChangeValue(labelText, value) {
        const box = document.createElement('div');
        box.className = 'config-switch-change-value';
        const label = document.createElement('span');
        label.textContent = labelText;
        const code = document.createElement('code');
        code.textContent = formatConfigChangeValue(value);
        box.append(label, code);
        return box;
    }

    function showAppConfirmDialog(options) {
        return showHistoryTaskConfirmDialog({
            title: options.title || '确认操作',
            description: options.description || '',
            message: options.message || '',
            confirmText: options.confirmText || '确认',
            cancelText: options.cancelText || '取消',
            danger: options.danger,
        }).then(Boolean);
    }

    function updateTomlDirtyState() {
        if (!hasPendingConfigChanges(currentTomlFile)) {
            resetTomlSaveConfirm({ update: false });
        }
        updateTomlBadges(currentTomlFile);
        updateTomlActionState(currentTomlFile);
    }

    function updateTomlBadges(filePath) {
        const meta = tomlFileMeta[filePath];
        setBadge('toml-current-badge', Boolean(filePath && currentTrainingSource.file === filePath), '当前训练');
        setBadge('toml-trainable-badge', Boolean(filePath), meta?.trainable ? '可训练' : '非训练');
        setBadge('toml-lock-badge', Boolean(meta?.locked), tomlLockLabel(meta) || '只读');
        setBadge('toml-dirty-badge', hasPendingConfigChanges(filePath), '未保存');
    }

    function setBadge(id, visible, text) {
        const badge = document.getElementById(id);
        if (!badge) return;
        badge.hidden = !visible;
        badge.textContent = text;
    }

    function updateTomlActionState(filePath) {
        const meta = tomlFileMeta[filePath];
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(filePath);
        const dirty = editorDirty || formDirty;
        const saveBtn = document.getElementById('btn-save-toml');
        if (saveBtn) {
            saveBtn.disabled = Boolean(meta?.locked) || !filePath || !dirty;
            saveBtn.textContent = '保存更新当前选中配置';
            saveBtn.classList.remove('btn-confirm-danger');
            saveBtn.title = meta?.locked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (dirty
                    ? (formDirty
                        ? '把左侧表单、数据集预设选择和采样提示词等修改写回当前选中的 TOML；保存后训练会使用这些新值。'
                        : '把直接编辑器里的 TOML 文本写回当前文件。')
                    : '当前配置没有未保存修改，不需要保存。');
        }
        updateTomlEditorPanelState(filePath);
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (meta?.trainable
                    ? '把右侧选中的 TOML 加载到左侧表单，并把它设为“开始训练”使用的配置。'
                    : '该文件不是完整训练配置，不能作为训练入口。');
        }
        const moveBtn = document.getElementById('btn-move-toml-group');
        if (moveBtn) {
            const canMove = Boolean(filePath && meta && !meta.locked && !dirty && getMovableTomlGroups(meta.group).length > 0);
            moveBtn.disabled = !canMove;
            moveBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再移动分组位置'
                : (meta?.locked
                    ? `${tomlLockLabel(meta) || '只读'}配置不能移动分组位置`
                    : (canMove ? '只调整右侧配置文件列表里的分组归属，不会改 TOML 内容或磁盘路径。' : '当前没有其他可移入的分组'));
        }
        const reloadBtn = document.getElementById('btn-reload-toml');
        if (reloadBtn) {
            reloadBtn.disabled = !filePath;
            reloadBtn.title = '从磁盘重新读取当前配置文件；未保存的编辑会被丢弃，但不会切换训练入口。';
        }
        const lockBtn = document.getElementById('btn-lock-toml');
        if (lockBtn) {
            const hasFile = Boolean(filePath && meta);
            const isSystemOrGroupLocked = Boolean(meta?.system_locked || meta?.group_locked);
            lockBtn.disabled = !hasFile || isSystemOrGroupLocked || dirty;
            lockBtn.textContent = meta?.user_locked ? '解除锁定' : '锁定当前文件';
            lockBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : lockTomlButtonTitle(meta);
        }
        const deleteBtn = document.getElementById('btn-delete-toml');
        if (deleteBtn) {
            const canDelete = Boolean(filePath && meta && !meta.locked && !dirty);
            if (!canDelete) resetTomlDeleteConfirm({ update: false });
            const confirming = canDelete && tomlDeleteConfirmFile === filePath;
            deleteBtn.disabled = !canDelete;
            deleteBtn.textContent = confirming ? '确认删除配置' : '删除当前配置';
            deleteBtn.classList.toggle('btn-confirm-danger', confirming);
            deleteBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再删除'
                : (confirming ? '再次点击才会真正删除当前配置文件' : deleteTomlButtonTitle(meta));
        }
        const restoreBtn = document.getElementById('btn-restore-system-toml');
        if (restoreBtn) {
            restoreBtn.disabled = dirty;
            restoreBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : '从项目内置版本还原系统预设；会覆盖系统预设文件，还原前会自动备份。用户导入配置不会被还原。';
        }
    }

    function readTomlGroupState() {
        try {
            return JSON.parse(localStorage.getItem('anima.tomlGroupOpen') || '{}') || {};
        } catch {
            return {};
        }
    }

    function writeTomlGroupState(state) {
        localStorage.setItem('anima.tomlGroupOpen', JSON.stringify(state));
    }

    function isTomlLocked(filePath) {
        return Boolean(tomlFileMeta[filePath]?.locked);
    }

    function applyTomlLockState(filePath) {
        const locked = isTomlLocked(filePath);
        setTomlEditorLocked(locked);
        updateTomlActionState(filePath);
    }

    function setTomlEditorLocked(locked) {
        const editor = document.getElementById('toml-editor');
        editor.readOnly = locked;
        editor.title = locked ? '该配置文件已锁定，只能导出或使用新名称另存新配置' : '';
    }

    function updateTomlEditorPanelState(filePath = currentTomlFile) {
        const panel = document.getElementById('toml-edit-panel');
        const manager = document.querySelector('.toml-manager');
        const toggleBtn = document.getElementById('btn-toggle-toml-editor');
        const saveDirectBtn = document.getElementById('btn-save-toml-direct');
        const copyBtn = document.getElementById('btn-copy-toml');
        const meta = tomlFileMeta[filePath];
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(filePath);
        const dirty = editorDirty || formDirty;
        const locked = Boolean(meta?.locked);
        const confirming = Boolean(filePath && tomlSaveConfirmFile === filePath);
        if (toggleBtn) {
            const open = Boolean(panel && !panel.hidden);
            if (manager) manager.classList.toggle('toml-edit-open', open);
            toggleBtn.disabled = !filePath;
            toggleBtn.textContent = open ? '收起配置文件编辑' : '直接编辑配置文件';
            toggleBtn.classList.toggle('active', open);
            toggleBtn.title = open
                ? '收起二级配置文件编辑界面；不会自动保存修改。'
                : '展开二级界面，查看、复制或直接编辑当前 TOML。适合批量改字段，保存时需要二次确认。';
        }
        if (saveDirectBtn) {
            saveDirectBtn.disabled = locked || !filePath || !editorDirty || formDirty;
            saveDirectBtn.textContent = confirming ? '确认保存配置文件' : '保存配置文件';
            saveDirectBtn.classList.toggle('btn-confirm-danger', confirming);
            saveDirectBtn.title = locked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (formDirty
                    ? '左侧表单或数据集还有未保存修改，请先用“保存更新当前选中配置”保存。'
                    : (editorDirty
                    ? (confirming ? '再次点击才会真正写入磁盘；请确认 TOML 内容没有语法错误。' : '第一次点击进入确认，第二次点击保存，防止误覆盖配置文件。')
                    : '直接编辑器没有未保存的 TOML 文本修改'));
        }
        if (copyBtn) {
            copyBtn.disabled = !filePath && !document.getElementById('toml-editor')?.value;
            copyBtn.title = '复制当前编辑器里的 TOML 内容，方便备份、对比或发给别人排查。';
        }
    }

    function toggleTomlEditorPanel() {
        const panel = document.getElementById('toml-edit-panel');
        if (!panel) return;
        if (!currentTomlFile) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        panel.hidden = !panel.hidden;
        updateTomlEditorPanelState(currentTomlFile);
        if (!panel.hidden) {
            document.getElementById('toml-editor')?.focus();
        }
    }

    async function copyTomlEditorContent() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(editor.value);
            } else {
                editor.focus();
                editor.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制当前配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    function tomlLockLabel(meta) {
        if (!meta?.locked) return '';
        if (meta.system_locked) return '系统只读';
        if (meta.user_locked) return '用户锁定';
        if (meta.user_group_locked) return '分组锁定';
        if (meta.group_locked) return '分组只读';
        return meta.lock_reason_label || '只读';
    }

    function tomlFileDisplayParts(fileOrMeta) {
        const meta = typeof fileOrMeta === 'string'
            ? (tomlFileMeta[fileOrMeta] || { path: fileOrMeta })
            : (fileOrMeta || {});
        const path = meta.path || '';
        const filename = meta.filename || (path ? path.split('/').pop() : '');
        const label = meta.label || '';
        const parts = [];
        if (label && label !== filename && label !== path) parts.push(label);
        if (filename) parts.push(filename);
        if (path && path !== filename) parts.push(path);
        return parts;
    }

    function tomlFileDisplayName(fileOrMeta) {
        const parts = tomlFileDisplayParts(fileOrMeta);
        return parts.length ? parts.join(' / ') : '未命名配置文件';
    }

    function lockTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.system_locked) return '系统预设已内置锁定，不能手动解除';
        if (meta.group_locked) return '该文件属于只读分组，不能手动解除';
        if (meta.user_group_locked) return '该文件所在分组已锁定，请在分组标题解除锁定';
        if (meta.user_locked) return '解除你为该文件设置的锁定';
        return '锁定当前文件，防止误保存';
    }

    function deleteTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.locked) return `${tomlLockLabel(meta) || '只读'}配置不能删除`;
        return '删除当前选中的配置文件';
    }

    function resetTomlDeleteConfirm(options = {}) {
        if (tomlDeleteConfirmTimer) {
            clearTimeout(tomlDeleteConfirmTimer);
            tomlDeleteConfirmTimer = null;
        }
        if (!tomlDeleteConfirmFile) return;
        tomlDeleteConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(currentTomlFile);
        }
    }

    function armTomlDeleteConfirm(file) {
        resetTomlDeleteConfirm({ update: false });
        tomlDeleteConfirmFile = file;
        tomlDeleteConfirmTimer = setTimeout(() => {
            resetTomlDeleteConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认删除配置”才会删除: ${file}`);
    }

    function resetTomlSaveConfirm(options = {}) {
        if (tomlSaveConfirmTimer) {
            clearTimeout(tomlSaveConfirmTimer);
            tomlSaveConfirmTimer = null;
        }
        if (!tomlSaveConfirmFile) return;
        tomlSaveConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(currentTomlFile);
        }
    }

    function armTomlSaveConfirm(file) {
        resetTomlSaveConfirm({ update: false });
        tomlSaveConfirmFile = file;
        tomlSaveConfirmTimer = setTimeout(() => {
            resetTomlSaveConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认保存”才会写入当前配置: ${file}`);
    }

    function setTomlStatus(cls, text, options = {}) {
        const el = document.getElementById('toml-status');
        if (tomlStatusTimer) {
            clearTimeout(tomlStatusTimer);
            tomlStatusTimer = null;
        }
        el.className = cls;
        el.textContent = text;
        if (cls === 'ok' && !options.persist) {
            tomlStatusTimer = setTimeout(() => {
                el.textContent = '';
                tomlStatusTimer = null;
            }, 3000);
        }
    }

    async function applyTomlToConfig(options = {}) {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再加载选中配置');
            updateTomlActionState(file);
            return;
        }
        if (!meta?.trainable) {
            setTomlStatus('error', '该文件不是完整训练配置，不能加载选中配置');
            return;
        }

        currentTrainingSource = {
            method: meta.method,
            methods_subdir: meta.methods_subdir || 'gui-methods',
            file: meta.path,
        };

        if (meta.methods_subdir === 'methods' && meta.method === 'spd') {
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === 'spd')) {
                methodSelect.value = 'spd';
            }
            const variantSelect = document.getElementById('variant-select');
            const variants = await api('/api/methods/spd/variants');
            populateSelect('variant-select', variants, 'spd');
        } else if (meta.methods_subdir === 'gui-methods') {
            const methodFamily = VARIANT_METHOD_FAMILY[meta.method] || meta.method || 'lora';
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === methodFamily)) {
                methodSelect.value = methodFamily;
            }
            const variantSelect = document.getElementById('variant-select');
            if (![...variantSelect.options].some((opt) => opt.value === meta.method)) {
                const variants = await api(`/api/methods/${encodeURIComponent(methodFamily)}/variants`);
                populateSelect('variant-select', variants, meta.method);
            }
            if ([...variantSelect.options].some((opt) => opt.value === meta.method)) {
                variantSelect.value = meta.method;
            }
        }

        await loadConfig();
        renderTomlFileGroups(reorderTomlFileGroups(tomlFileGroups));
        updateTomlDirtyState();
        rememberSelectionSnapshot();
        if (!options.silent) {
            setTomlStatus('ok', `已应用 ${meta.path} 到表单`);
        }
    }

    async function toggleTomlUserLock() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (!file || !meta) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再调整锁定');
            updateTomlActionState(file);
            return;
        }
        if (meta.system_locked) {
            setTomlStatus('error', '系统预设已内置锁定，不能手动解除');
            return;
        }
        if (meta.group_locked) {
            setTomlStatus('error', '该文件属于只读分组，不能手动解除');
            return;
        }
        if (meta.user_group_locked) {
            setTomlStatus('error', '该文件所在分组已锁定，请在分组标题解除锁定');
            return;
        }

        const nextLocked = !meta.user_locked;
        const message = nextLocked
            ? `锁定 ${file}？锁定后不能直接保存，仍可使用新名称另存新配置。`
            : `解除 ${file} 的用户锁定？解除后可以直接编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置文件' : '解除配置锁定',
            description: file,
            message,
            confirmText: nextLocked ? '锁定' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            const res = await api('/api/config/lock', {
                method: 'POST',
                body: JSON.stringify({ file, locked: nextLocked }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '锁定操作失败');
                if (res.meta) tomlFileMeta[file] = res.meta;
                updateTomlDirtyState();
                return;
            }
            if (res.meta) tomlFileMeta[file] = res.meta;
            await loadTomlFileList(file);
            applyTomlLockState(file);
            updateTomlDirtyState();
            setTomlStatus('ok', res.message || (nextLocked ? '已锁定当前文件' : '已解除用户锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function toggleTomlGroupLock(groupOrId) {
        const group = typeof groupOrId === 'string'
            ? tomlFileGroups.find((item) => item.id === groupOrId)
            : groupOrId;
        if (!group) {
            setTomlStatus('error', '分组不存在，请先刷新文件列表');
            return;
        }
        if (!group.lockable) {
            setTomlStatus('error', '该分组不能手动锁定或解锁');
            return;
        }

        const nextLocked = !group.user_group_locked;
        const sourceGroupIds = group.sourceGroupIds?.length ? group.sourceGroupIds : [group.id];
        const message = nextLocked
            ? `锁定分组“${group.label || group.id}”？该分组内文件将不能直接保存，仍可使用新名称另存新配置。`
            : `解除分组“${group.label || group.id}”的锁定？解除后该分组内文件可恢复编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置分组' : '解除分组锁定',
            description: group.label || group.id,
            message,
            confirmText: nextLocked ? '锁定分组' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            let lastResponse = null;
            for (const groupId of sourceGroupIds) {
                const res = await api('/api/config/group-lock', {
                    method: 'POST',
                    body: JSON.stringify({ group: groupId, locked: nextLocked }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '分组锁定操作失败');
                    return;
                }
                lastResponse = res;
            }
            await loadTomlFileList(currentTomlFile || '');
            applyTomlLockState(currentTomlFile);
            updateTomlDirtyState();
            setTomlStatus('ok', lastResponse?.message || (nextLocked ? '已锁定当前分组' : '已解除分组锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function createTomlGroup() {
        const label = await showHistoryTaskInputDialog({
            title: '新建配置分组',
            description: '用于整理右侧 TOML 配置文件。新分组默认可训练，可移入 imported 配置。',
            label: '分组名称',
            placeholder: '例如：角色配置 / 试验配置 / 正式配置',
            confirmText: '创建分组',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api('/api/config/file-groups', {
                method: 'POST',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已创建');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function renameTomlGroup(group) {
        const label = await showHistoryTaskInputDialog({
            title: '重命名配置分组',
            description: '只修改分组显示名称，不会改动配置文件路径。',
            label: '分组名称',
            value: group.label || group.id,
            placeholder: '例如：正式配置',
            confirmText: '保存名称',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '重命名分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已重命名');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function reorderTomlGroup(group, direction) {
        if (!group?.id) return;
        try {
            const res = await api('/api/config/file-groups/reorder-group', {
                method: 'POST',
                body: JSON.stringify({ group: group.id, direction }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '调整分组顺序失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组顺序已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function moveCurrentTomlToGroup() {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再移动分组');
            updateTomlActionState(file);
            return;
        }
        const meta = tomlFileMeta[file];
        if (meta?.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能移动分组`);
            return;
        }

        const groups = getMovableTomlGroups(meta?.group);
        if (!groups.length) {
            setTomlStatus('error', '当前没有其他可移入的分组，请先新建分组或解除目标分组锁定');
            return;
        }
        const targetGroupId = await showMoveTomlDialog(file, meta, groups);
        if (!targetGroupId) return;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: targetGroupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '移动分组失败');
                return;
            }
            await loadTomlFileList(file);
            setTomlStatus('ok', res.message || '配置已移动到分组');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    function getMovableTomlGroups(currentGroupId = '') {
        return reorderTomlFileGroups(tomlFileGroups)
            .filter((group) => group.movable && !group.locked && !group.user_group_locked && group.id !== currentGroupId);
    }

    function deleteTomlGroupButtonTitle(group) {
        if (!group) return '配置分组不可用';
        if (group.user_group_locked) return '该分组已锁定，请先解除分组锁定后再删除';
        if (!group.deletable) return '系统固定分组或只读分组不能删除';
        const count = (group.files || []).length;
        return count > 0
            ? `删除当前分组“${group.label || group.id}”；不会删除其中 ${count} 个 TOML 文件`
            : `删除当前空分组“${group.label || group.id}”`;
    }

    function canDeleteTomlGroup(group) {
        return Boolean(group?.deletable && !group.user_group_locked);
    }

    function showMoveTomlDialog(file, meta, groups) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-move-dialog-body';

        const current = document.createElement('p');
        current.className = 'toml-move-current';
        current.textContent = `当前配置: ${file}`;
        wrap.appendChild(current);

        const list = document.createElement('div');
        list.className = 'toml-move-option-list';
        const radios = [];
        for (const group of groups) {
            const label = document.createElement('label');
            label.className = 'toml-move-option';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'toml-move-target-group';
            input.value = group.id;
            input.checked = group.id !== meta?.group && !radios.some((item) => item.checked);
            radios.push(input);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            const count = (group.files || []).length;
            detail.textContent = `${count} 个配置`;
            text.append(title, detail);

            label.append(input, text);
            list.appendChild(label);
        }
        wrap.appendChild(list);

        return showHistoryTaskDialog({
            title: '移动配置',
            description: '选择目标分组后确认，配置文件路径不会改变，只调整右侧分组归属。',
            body: wrap,
            confirmText: '移动到分组',
            onOpen: () => {
                const checked = radios.find((item) => item.checked) || radios[0];
                checked?.focus();
            },
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-move-target-group"]:checked');
                return checked?.value || '';
            },
        });
    }

    async function reorderTomlFileInGroup(file, groupId, direction) {
        if (!file || !groupId) return;
        if (hasPendingConfigChanges(currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再调整排序');
            updateTomlActionState(currentTomlFile);
            return;
        }
        try {
            const res = await api('/api/config/file-groups/reorder-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: groupId, direction }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '排序失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || file);
            setTomlStatus('ok', res.message || '配置排序已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function deleteTomlGroup(group) {
        if (!canDeleteTomlGroup(group)) {
            setTomlStatus('error', deleteTomlGroupButtonTitle(group));
            return;
        }
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除配置分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个 TOML 文件；这些文件会回到导入配置或数据集配置等默认分组。`
                : '只删除这个分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        const reallyOk = await showHistoryTaskConfirmDialog({
            title: '你真的确认吗？',
            description: group.label || group.id,
            message: count > 0
                ? `确认后会删除这个分组，分组内 ${count} 个 TOML 文件会回到默认分组。`
                : '确认后会删除这个空分组。',
            confirmText: '我确认',
            cancelText: '我觉得不对',
            cancelPrimary: true,
            danger: true,
        });
        if (!reallyOk) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '删除分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已删除');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function deleteTomlFile() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再删除');
            updateTomlActionState(file);
            return;
        }
        if (!meta) {
            await handleDeletedTomlSelection(file, '当前配置已不在列表中，已刷新配置列表');
            return;
        }
        if (meta.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能删除`);
            updateTomlActionState(file);
            return;
        }

        if (tomlDeleteConfirmFile !== file) {
            armTomlDeleteConfirm(file);
            return;
        }
        resetTomlDeleteConfirm({ update: false });

        try {
            const res = await api(`/api/config/raw?file=${encodeURIComponent(file)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                if (isMissingTomlFileResponse(res)) {
                    await handleDeletedTomlSelection(file, res.error || '配置文件不存在或已被删除');
                    return;
                }
                setTomlStatus('error', res.error || '删除失败');
                return;
            }

            await handleDeletedTomlSelection(file, `已删除配置: ${file}`, { ok: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    function isMissingTomlFileResponse(res) {
        return String(res?.error || '').includes('不存在') || String(res?.error || '').includes('已被删除');
    }

    async function handleDeletedTomlSelection(file, message, options = {}) {
        if (currentTrainingSource.file === file) {
            setCurrentTrainingSourceFromVariant(val('variant-select') || 'lora');
        }
        delete tomlFileMeta[file];
        tomlFiles = tomlFiles.filter((item) => item !== file);
        clearCurrentTomlSelection();
        await loadTomlFileList('', { skipDefaultLoad: true });
        clearCurrentTomlSelection();
        updateTomlDirtyState();
        setTomlStatus(options.ok ? 'ok' : 'error', message, { persist: true });
    }

    function clearCurrentTomlSelection() {
        resetTomlDeleteConfirm({ update: false });
        resetTomlSaveConfirm({ update: false });
        currentTomlFile = '';
        tomlSavedContent = '';
        const editor = document.getElementById('toml-editor');
        if (editor) {
            editor.value = '';
            editor.readOnly = false;
            editor.title = '';
        }
        const select = document.getElementById('toml-file-select');
        if (select) select.value = '';
        updateTomlSelectionUI('');
        applyTomlLockState('');
    }

    async function restoreSystemTomlPresets() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再还原系统预设');
            updateTomlActionState(file);
            return;
        }

        const currentHint = meta?.restorable ? `\n当前文件 ${file} 也会一起还原。` : '';
        const ok = await showAppConfirmDialog({
            title: '还原系统预设',
            description: 'base、presets、methods、gui-methods',
            message: `还原会覆盖系统预设文件，但会先自动备份当前内容。用户导入/副本和数据集配置不会被还原。${currentHint}`,
            confirmText: '还原系统预设',
            danger: true,
        });
        if (!ok) return;

        try {
            const res = await api('/api/config/restore-system', {
                method: 'POST',
                body: JSON.stringify({}),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '还原失败');
                return;
            }

            const preferredFile = file && tomlFiles.includes(file) ? file : '';
            await loadTomlFileList(preferredFile);
            const restoredCount = res.restored?.length || 0;
            const skippedCount = res.skipped?.length || 0;
            const backupText = res.backup_dir ? `，备份在 ${res.backup_dir}` : '';
            setTomlStatus('ok', `已还原 ${restoredCount} 个系统预设，跳过 ${skippedCount} 个${backupText}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    // ── 训练控制 ──
    async function startTraining() {
        const selectedTrainingConfigFile = currentTrainingConfigFile();
        if (tomlManagerMode !== 'output' || !outputRunState.file) {
            if (hasPendingConfigChanges(currentTomlFile)) {
                setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或另存新配置，再开始训练');
                updateTomlActionState(currentTomlFile);
                document.querySelector('[data-tab="config"]')?.click();
                return;
            }
        }
        if (!selectedTrainingConfigFile) {
            const message = tomlManagerMode === 'output' && outputRunState.selectedRun
                ? '这个训练输出没有可直接继续训练的 config.runtime.toml，请先另存原始配置或选择其他运行目录'
                : '请选择要训练的配置文件';
            setTomlStatus('error', message);
            return;
        }
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            const message = 'SPD 是 CLI 实验配置，只能通过 tasks.py exp-spd / scripts/distill_spd.py 运行；Web 普通训练入口已拦截，避免误用 train.py。';
            setTomlStatus('error', message, { persist: true });
            alert(message);
            return;
        }
        if (continueTrainingSource && !(await refreshContinueTrainingSourceCompatibility())) {
            setTomlStatus('error', continueTrainingSource.message || '继续训练权重与当前配置不兼容', { persist: true });
            return;
        }
        const preflight = await runPreflight(variant, preset, methodsSubdir);
        if (!preflight) {
            if (isPreflightDialogOpen()) await waitForPreflightDialogClose();
            return;
        }
        const willAutoPreprocess = !currentTrainingConfigIsRuntime();
        if (!preflight.ok) {
            const action = await showPreflightDialog(preflight, false, { willAutoPreprocess });
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
            }
            return;
        }
        const action = await showPreflightDialog(preflight, true, { willAutoPreprocess });
        if (action === 'preprocess') {
            await startPreprocessFromPreflight(preflight);
            return;
        }
        if (action !== 'continue') return;
        await startTrainingUnchecked(variant, preset, methodsSubdir, { willAutoPreprocess });
    }

    async function runPreflight(variant, preset, methodsSubdir) {
        const pending = showPreflightPendingDialog({
            title: '训练前预检测',
            message: '正在检查模型路径、数据集路径和预处理启动环境...',
            detail: '这一步可能需要几秒钟；窗口保持打开表示仍在检查。',
        });
        try {
            const res = await api('/api/training/preflight', {
                method: 'POST',
                signal: pending.signal,
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                }),
            });
            pending.resolve();
            return res;
        } catch (e) {
            pending.resolve();
            if (e.name === 'AbortError') {
                return null;
            }
            showPreflightRequestError(`预检测请求失败: ${e.message}`);
            return null;
        }
    }

    function isCliOnlySpdSource(variant, methodsSubdir) {
        return String(methodsSubdir || '') === 'methods' && String(variant || '') === 'spd';
    }

    function currentTrainingConfigIsRuntime() {
        return currentTrainingConfigFile().replace(/\\/g, '/').endsWith('/config.runtime.toml');
    }

    async function chooseTrainingLaunchMode(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        if (isRunning) {
            const ok = await showAppConfirmDialog({
                title: '加入训练队列',
                description: '当前已有任务在运行',
                message: `确认后会冻结当前配置，并加入队列等待自动执行。${sourceDetail}`,
                confirmText: '加入队列',
                cancelText: '取消',
            });
            return ok ? 'queue' : 'cancel';
        }
        const startNow = await showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '可以立即启动，也可以先加入队列',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '立即预处理并训练' : '立即开始训练',
            cancelText: '不立即启动',
        });
        if (startNow) return 'start';
        const queue = await showAppConfirmDialog({
            title: '加入训练队列',
            description: '冻结当前配置并等待自动执行',
            message: `确认后会创建独立运行配置并加入队列。${sourceDetail}`,
            confirmText: '加入队列',
            cancelText: '取消',
        });
        return queue ? 'queue' : 'cancel';
    }

    async function confirmTrainingLaunch(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        return showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '训练启动前的最后一步',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理；预处理完成后会自动开始训练。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '确认预处理并训练' : '确认开始训练',
            cancelText: '返回检查',
        });
    }

    async function startTrainingUnchecked(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess });
            return;
        }
        renderPreflightPending({
            title: willAutoPreprocess ? '启动预处理后训练' : '启动训练',
            message: willAutoPreprocess
                ? '正在创建运行目录并启动预处理...'
                : '正在创建运行目录并启动训练...',
            detail: willAutoPreprocess
                ? '预处理完成后会自动开始训练；成功后会自动切换到训练页。'
                : '后端正在准备训练进程；启动成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    gpu_whitelist: selectedGpuPayload(),
                    confirmed: true,
                    confirm_preprocess: willAutoPreprocess,
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (res.ok) {
                const dialog = document.getElementById('preflight-dialog');
                if (dialog?.open) dialog.close('training-started');
                enterLiveTrainingForNewRun();
                appendLog(`[状态] ${res.message || '任务已启动'}`);
            } else {
                if (res.preflight) {
                    const action = await showPreflightDialog(res.preflight, false);
                    if (action === 'preprocess') {
                        await startPreprocessFromPreflight(res.preflight);
                    }
                } else {
                    showPreflightRequestError(res.error || '启动失败');
                }
            }
        } catch (e) {
            showPreflightRequestError('请求失败: ' + e.message);
        }
    }

    async function enqueueTrainingFromConfig(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        renderPreflightPending({
            title: '加入训练队列',
            message: '正在冻结当前配置并加入队列...',
            detail: '队列会保存独立运行配置；之后修改当前 TOML 不会影响这个队列任务。',
        });
        try {
            const res = await api('/api/training/queue/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    gpu_whitelist: selectedGpuPayload(),
                    confirmed: true,
                    confirm_preprocess: willAutoPreprocess,
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (!res.ok) {
                if (res.preflight) {
                    await showPreflightDialog(res.preflight, false, { willAutoPreprocess });
                } else {
                    showPreflightRequestError(res.error || '加入队列失败');
                }
                return;
            }
            const dialog = document.getElementById('preflight-dialog');
            if (dialog?.open) dialog.close('queued');
            updateTrainingQueueFromPayload(res);
            document.querySelector('[data-tab="training"]')?.click();
            appendLog(`[状态] ${res.message || '已加入训练队列'}`);
        } catch (e) {
            showPreflightRequestError('加入队列失败: ' + e.message);
        }
    }

    function enterLiveTrainingForNewRun() {
        returnToLiveTraining({ refresh: false });
        document.querySelector('[data-tab="training"]')?.click();
        pollStatus();
        replayTrainingLogs();
    }

    function showPreflightDialog(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog) {
            if (!allowContinue) return Promise.resolve('cancel');
            const confirmText = options.willAutoPreprocess ? '确认预处理并训练' : '确认开始训练';
            return showAppConfirmDialog({
                title: '训练前预检测',
                description: '检测到训练前提示',
                message: `${preflightPlainText(result)}\n\n是否继续训练？`,
                confirmText,
            }).then((ok) => ok ? 'continue' : 'cancel');
        }
        renderPreflightResult(result, allowContinue, options);
        if (!dialog.open) dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => {
                resolve(dialog.returnValue || 'cancel');
            }, { once: true });
        });
    }

    function showPreflightPendingDialog(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const controller = new AbortController();
        if (!dialog) {
            return { signal: controller.signal, resolve: () => {} };
        }
        renderPreflightPending(options);
        let settled = false;
        const cleanup = () => {
            dialog.removeEventListener('close', handleClose);
        };
        const handleClose = () => {
            cleanup();
            if (!settled) {
                controller.abort();
            }
        };
        dialog.addEventListener('close', handleClose);
        if (!dialog.open) {
            try {
                dialog.showModal();
            } catch (e) {
                dialog.setAttribute('open', 'open');
            }
        }
        return {
            signal: controller.signal,
            resolve: () => {
                settled = true;
                cleanup();
            },
        };
    }

    function renderPreflightPending(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        if (heading) heading.textContent = options.title || '训练前预检测';
        if (summary) {
            summary.className = 'preflight-summary pending';
            summary.setAttribute('aria-live', 'polite');
            summary.textContent = options.message || '正在预检测...';
        }
        if (list) {
            list.innerHTML = '';
            const row = document.createElement('div');
            row.className = 'preflight-item pending';
            row.setAttribute('aria-busy', 'true');

            const badge = document.createElement('span');
            badge.className = 'preflight-badge preflight-spinner';
            badge.setAttribute('aria-label', '正在检查');
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = options.detail || '正在连接后端并执行轻量检查...';
            const path = document.createElement('div');
            path.className = 'preflight-path';
            path.textContent = '请稍等，预检测返回后会在这里显示每一项结果。';
            body.append(title, path);
            row.appendChild(body);
            list.appendChild(row);
        }
        if (preprocessBtn) {
            preprocessBtn.hidden = true;
            preprocessBtn.disabled = true;
        }
        if (continueBtn) {
            continueBtn.hidden = false;
            continueBtn.disabled = true;
            continueBtn.textContent = '正在检查...';
        }
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    function showPreflightRequestError(message) {
        const result = {
            ok: false,
            summary: { errors: 1, warnings: 0, checks: 1 },
            checks: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            errors: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            warnings: [],
        };
        const dialog = document.getElementById('preflight-dialog');
        if (dialog) {
            renderPreflightResult(result, false);
            if (!dialog.open) dialog.showModal();
        } else {
            alert(message);
        }
    }

    function isPreflightDialogOpen() {
        const dialog = document.getElementById('preflight-dialog');
        return Boolean(dialog?.open);
    }

    function waitForPreflightDialogClose() {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog?.open) return Promise.resolve();
        return new Promise((resolve) => {
            dialog.addEventListener('close', resolve, { once: true });
        });
    }

    function renderPreflightResult(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        const errors = result.summary?.errors || 0;
        const warnings = result.summary?.warnings || 0;
        const checks = result.summary?.checks || 0;
        const canPreprocess = preflightCanStartPreprocess(result);
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);

        if (heading) heading.textContent = '训练前预检测';
        summary.className = `preflight-summary ${errors ? 'error' : warnings ? 'warning' : 'ok'}`;
        summary.removeAttribute('aria-live');
        if (errors && canPreprocess) {
            summary.textContent = `发现 ${errors} 个错误：当前数据需要先预处理。点击下方按钮后，还会出现最终确认；确认后才会启动预处理并在完成后训练。`;
        } else {
            summary.textContent = errors
                ? `发现 ${errors} 个错误，已阻止训练。`
                : warnings
                    ? (willAutoPreprocess
                        ? `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会预处理并训练。`
                        : `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会开始训练。`)
                    : willAutoPreprocess
                        ? `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会创建运行目录、预处理并自动训练。`
                        : `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会开始训练。`;
        }

        list.innerHTML = '';
        for (const item of result.checks || []) {
            const row = document.createElement('div');
            row.className = `preflight-item ${item.level}`;

            const badge = document.createElement('span');
            badge.className = 'preflight-badge';
            badge.textContent = item.level === 'ok' ? '通过' :
                item.level === 'warning' ? '警告' : '错误';
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = `${FIELD_LABEL_ZH[item.key] || item.key}: ${item.message}`;
            body.appendChild(title);
            if (item.path) {
                const path = document.createElement('div');
                path.className = 'preflight-path';
                path.textContent = item.path;
                body.appendChild(path);
            }
            row.appendChild(body);
            list.appendChild(row);
        }

        preprocessBtn.hidden = !canPreprocess;
        preprocessBtn.disabled = !canPreprocess;
        continueBtn.hidden = !allowContinue;
        continueBtn.disabled = !allowContinue;
        continueBtn.textContent = warnings
            ? (willAutoPreprocess ? '查看最终确认' : '查看最终确认')
            : (willAutoPreprocess ? '下一步：最终确认' : '下一步：最终确认');
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    function preflightCanStartPreprocess(result) {
        const checks = result.checks || [];
        const errors = result.errors || [];
        const allowedErrorKeys = new Set(['training_images', 'resized_image_dir']);
        if (errors.some((item) => !allowedErrorKeys.has(item.key))) return false;
        const sourceOk = checks.some((item) => item.key === 'source_image_dir' && item.level === 'ok');
        if (!sourceOk) return false;
        return checks.some((item) =>
            ['training_images', 'resized_image_dir', 'lora_cache_dir', 'latent_cache', 'text_cache'].includes(item.key)
            && ['error', 'warning'].includes(item.level)
        );
    }

    async function startPreprocessFromPreflight(result) {
        const variant = result.variant || currentTrainingSource.method || val('variant-select');
        const preset = result.preset || val('preset-select');
        const methodsSubdir = result.methods_subdir || currentTrainingSource.methods_subdir || 'gui-methods';
        if (continueTrainingSource && !(await refreshContinueTrainingSourceCompatibility())) {
            showPreflightRequestError(continueTrainingSource.message || '继续训练权重与当前配置不兼容');
            return;
        }
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess: true });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess: true });
            return;
        }
        renderPreflightPending({
            title: '启动预处理',
            message: '正在创建运行目录并启动预处理...',
            detail: '正在把任务交给后端；成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/preprocess', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    train_after: true,
                    confirmed: true,
                    confirm_train_after: true,
                    confirm_preprocess: true,
                    gpu_whitelist: selectedGpuPayload(),
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (!res.ok) {
                showPreflightRequestError(res.error || '预处理启动失败');
                return;
            }
            const dialog = document.getElementById('preflight-dialog');
            if (dialog?.open) dialog.close('preprocess-started');
            enterLiveTrainingForNewRun();
            appendLog(`[状态] ${res.message || '预处理已启动'}`);
        } catch (e) {
            showPreflightRequestError('预处理请求失败: ' + e.message);
        }
    }

    function currentTrainingConfigFile() {
        if (tomlManagerMode === 'output') {
            return outputRunRuntimeFile();
        }
        return currentTrainingSource.file || currentTomlFile || val('toml-file-select') || '';
    }

    function preflightPlainText(result) {
        return (result.checks || [])
            .map((item) => `[${item.level}] ${item.key}: ${item.message}${item.path ? ` (${item.path})` : ''}`)
            .join('\n');
    }

    async function stopTraining() {
        const ok = await showAppConfirmDialog({
            title: '停止训练',
            description: '当前运行中的训练任务',
            message: '确定要停止训练吗？停止后当前训练过程会立即中断。',
            confirmText: '停止训练',
            danger: true,
        });
        if (!ok) return;
        await api('/api/training/stop', { method: 'POST' });
    }

    // ── WebSocket ──
    function connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setLogStatus('连接中', 'warning');
        ws = new WebSocket(`${proto}//${location.host}/ws/training`);
        ws.onopen = () => {
            setLogStatus('已连接', 'ok');
            replayTrainingLogs();
        };
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            handleWsMessage(msg);
        };
        ws.onclose = () => {
            setLogStatus('已断开，准备重连', 'warning');
            setTimeout(connectWebSocket, 3000);
        };
        ws.onerror = () => {
            setLogStatus('连接异常', 'error');
            ws.close();
        };
    }

    function handleWsMessage(msg) {
        switch (msg.type) {
            case 'log':
                if (isHistoryReviewMode()) break;
                markTrainingActivity(msg.ts);
                appendLogRecord(msg);
                break;
            case 'progress':
                if (isHistoryReviewMode()) break;
                updateProgress(msg);
                break;
            case 'metrics':
                if (isHistoryReviewMode()) break;
                updateMetrics(msg);
                break;
            case 'status':
                if (isHistoryReviewMode()) {
                    loadTrainingHistoryList();
                    renderResumePanelState();
                    break;
                }
                updateStatus(msg);
                loadTrainingQueue();
                loadTrainingHistoryList();
                break;
            case 'queue':
                updateTrainingQueueFromPayload(msg);
                loadTrainingHistoryList();
                break;
            case 'system':
                if (isHistoryReviewMode()) break;
                updateSystem(msg);
                break;
        }
    }

    function appendLog(line) {
        appendLogRecord({ line });
    }

    function appendLogRecord(record) {
        if (record?.id && record.id <= trainingRuntime.lastLogId) return;
        if (record?.id) trainingRuntime.lastLogId = record.id;

        const el = document.getElementById('log-output');
        const line = record?.line ?? '';
        const prefix = record?.kind === 'progress' ? '[进度] ' : '';
        el.textContent += prefix + line + '\n';
        trainingRuntime.logLineCount += 1;

        if (trainingRuntime.logLineCount > MAX_LOG_LINES) {
            const lines = el.textContent.split('\n').filter(Boolean).slice(-MAX_LOG_LINES);
            el.textContent = lines.join('\n') + '\n';
            trainingRuntime.logLineCount = lines.length;
        }
        el.scrollTop = el.scrollHeight;
        updateLogStatusText();
    }

    async function replayTrainingLogs() {
        if (isHistoryReviewMode()) return;
        try {
            const payload = await api(`/api/training/logs?after=${trainingRuntime.lastLogId}&limit=1000`);
            for (const record of payload.records || []) {
                if (record.ts) markTrainingActivity(record.ts);
                appendLogRecord(record);
                replayMetricsFromLogRecord(record);
            }
            await replayMetricsHistory();
            updateLogStatusText();
        } catch (e) {
            setLogStatus('日志回放失败', 'error');
        }
    }

    async function replayMetricsHistory() {
        if (isHistoryReviewMode()) return;
        try {
            const records = await api('/api/training/metrics');
            for (const record of records || []) {
                updateMetrics(record);
            }
        } catch (e) {
            // 历史指标不是训练控制关键路径，失败时保留日志回放。
        }
    }

    function replayMetricsFromLogRecord(record) {
        const line = record?.line || '';
        const parsed = parseMetricsFromProgressLine(line);
        if (!parsed || parsed.loss === undefined) return;
        updateMetrics({ ...parsed, ts: record.ts });
    }

    function setLogStatus(text, state = '') {
        const el = document.getElementById('log-status');
        if (!el) return;
        el.textContent = text;
        el.className = `log-status ${state}`.trim();
    }

    function updateLogStatusText() {
        const state = ws?.readyState === WebSocket.OPEN ? 'ok' : 'warning';
        const text = ws?.readyState === WebSocket.OPEN
            ? `已连接 · ${trainingRuntime.logLineCount} 行`
            : `${trainingRuntime.logLineCount} 行`;
        setLogStatus(text, state);
    }

    function updateProgress(msg) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts);
        const pct = msg.total > 0 ? (msg.current / msg.total * 100) : 0;
        document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
        let text = `${msg.label}: ${msg.current}/${msg.total} (${pct.toFixed(1)}%)`;
        if (msg.rate) text += ` — ${msg.rate}`;
        document.getElementById('progress-text').textContent = text;
        document.getElementById('metric-step').textContent = msg.current;
        if (msg.rate) document.getElementById('metric-rate').textContent = msg.rate;
    }

    function updateMetrics(msg) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts);
        if (msg.loss !== undefined) {
            document.getElementById('metric-loss').textContent = msg.loss.toFixed(5);
            const step = msg.step || ++stepCounter;
            lossChart?.push(step, msg.loss, { rawStep: msg.step ?? step });
        }
        if (msg.lr !== undefined) {
            document.getElementById('metric-lr').textContent = msg.lr.toExponential(2);
        }
        if (msg.step !== undefined) {
            document.getElementById('metric-step').textContent = msg.step;
        }
        if (msg.rate) {
            document.getElementById('metric-rate').textContent = msg.rate;
        }
    }

    function updateStatus(msg) {
        if (isHistoryReviewMode()) return;
        const dot = document.querySelector('.dot');
        const text = document.getElementById('status-text');
        const stopBtn = document.getElementById('btn-stop-training');

        dot.className = 'dot ' + msg.state;
        const stateMap = { idle: '空闲', running: '训练中', error: '错误', compiling: '编译中' };
        const jobLabel = msg.job === 'preprocess' ? '预处理中' : (stateMap[msg.state] || msg.state);
        text.textContent = msg.state === 'running' ? jobLabel : (stateMap[msg.state] || msg.state);
        trainingRuntime.state = msg.state;
        trainingRuntime.job = msg.job || trainingRuntime.job || '';
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at);
        }
        if (msg.state !== 'running' && msg.state !== 'compiling') {
            trainingRuntime.lastOutputAt = 0;
            trainingRuntime.lastUiActivityAt = 0;
        }
        if (msg.output_dir !== undefined) {
            trainingRuntime.outputDir = msg.output_dir || '';
        }
        if (msg.sample_dir !== undefined) {
            trainingRuntime.sampleDir = msg.sample_dir || '';
            if (previewSettings) {
                previewSettings.current_task_sample_dir = trainingRuntime.sampleDir;
                previewSettings.effective_training_dir = trainingRuntime.sampleDir || previewSettings.training_dir;
                updatePreviewDirectorySummary();
            }
        }
        if (msg.sample_config !== undefined) {
            trainingRuntime.sampleConfig = msg.sample_config || null;
            trainingSampleState = trainingRuntime.sampleConfig;
        }
        applyRuntimeInfoToState(msg);

        stopBtn.disabled = msg.state !== 'running';

        if (msg.variant) document.getElementById('train-variant').textContent = msg.variant;
        if (msg.preset) document.getElementById('train-preset').textContent = msg.preset;

        if (msg.message) appendLog(`[状态] ${msg.message}`);

        if (msg.state === 'idle' || msg.state === 'error') {
            document.getElementById('progress-bar').style.width = '0%';
            trainingRuntime.quietHintShown = false;
            trainingRuntime.job = '';
            if (!msg.output_dir) {
                clearRuntimeInfo();
            }
        }
        renderCurrentRuntimePaths();
        refreshTrainingHealth();
    }

    function clearRuntimeInfo() {
        trainingRuntime.runDir = '';
        trainingRuntime.runtimeConfigFile = '';
        trainingRuntime.originalConfigFile = '';
        trainingRuntime.datasetConfigFile = '';
        trainingRuntime.modelCacheDir = '';
        trainingRuntime.datasetCacheDir = '';
        trainingRuntime.trainingOutputDir = '';
        trainingRuntime.logsDir = '';
    }

    function applyRuntimeInfoToState(msg) {
        const fields = {
            run_dir: 'runDir',
            runtime_config_file: 'runtimeConfigFile',
            original_config_file: 'originalConfigFile',
            dataset_config_file: 'datasetConfigFile',
            model_cache_dir: 'modelCacheDir',
            dataset_cache_dir: 'datasetCacheDir',
            training_output_dir: 'trainingOutputDir',
            logs_dir: 'logsDir',
        };
        for (const [wireKey, stateKey] of Object.entries(fields)) {
            if (msg[wireKey] !== undefined) {
                trainingRuntime[stateKey] = msg[wireKey] || '';
            }
        }
    }

    function renderCurrentRuntimePaths() {
        if (isHistoryReviewMode()) return;
        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        const task = currentRuntimeTaskInfo();
        const hasRuntimePaths = runtimePathItems(task, { includeHistory: false }).length > 0;
        if (configPanel) configPanel.hidden = !hasRuntimePaths;
        if (!hasRuntimePaths) {
            const paths = document.getElementById('history-paths');
            if (paths) paths.innerHTML = '';
            if (configOutput) configOutput.textContent = '';
            return;
        }
        if (configTitle) {
            configTitle.textContent = trainingRuntime.job === 'preprocess'
                ? '当前预处理运行目录'
                : '当前任务运行目录';
        }
        if (configOutput) {
            configOutput.textContent = [
                task.runtime_config_file ? `实际运行配置: ${task.runtime_config_file}` : '',
                task.original_config_file ? `原始配置: ${task.original_config_file}` : '',
            ].filter(Boolean).join('\n');
        }
        renderHistoryPaths(task, { includeHistory: false });
    }

    function currentRuntimeTaskInfo() {
        return {
            run_dir: trainingRuntime.runDir,
            runtime_config_file: trainingRuntime.runtimeConfigFile,
            original_config_file: trainingRuntime.originalConfigFile,
            dataset_config_file: trainingRuntime.datasetConfigFile,
            model_cache_dir: trainingRuntime.modelCacheDir,
            dataset_cache_dir: trainingRuntime.datasetCacheDir,
            training_output_dir: trainingRuntime.trainingOutputDir,
            logs_dir: trainingRuntime.logsDir,
            output_dir: trainingRuntime.outputDir,
            sample_dir: trainingRuntime.sampleDir,
        };
    }

    function updateSystem(msg) {
        if (isHistoryReviewMode()) return;
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at);
        }
        if (msg.vram_used_gb !== undefined) {
            document.getElementById('metric-vram').textContent =
                `${msg.vram_used_gb}/${msg.vram_total_gb} GB`;
        }
        if (msg.gpu_util !== undefined) {
            trainingRuntime.lastGpuUtil = Number(msg.gpu_util);
            let gpuText = `${msg.gpu_util}%`;
            if (msg.gpu_temp) gpuText += ` ${msg.gpu_temp}°C`;
            document.getElementById('metric-gpu').textContent = gpuText;
        }
        refreshTrainingHealth();
    }

    function markTrainingActivity(ts) {
        const value = Number(ts);
        const ms = value > 100000000000 ? value : value * 1000;
        if (Number.isFinite(ms) && ms > 0) {
            trainingRuntime.lastOutputAt = Math.max(trainingRuntime.lastOutputAt, ms);
        } else {
            trainingRuntime.lastOutputAt = Date.now();
        }
        trainingRuntime.lastUiActivityAt = Date.now();
        trainingRuntime.quietHintShown = false;
    }

    function refreshTrainingHealth() {
        const el = document.getElementById('training-health');
        const ageEl = document.getElementById('metric-log-age');
        if (!el || !ageEl) return;

        if (isHistoryReviewMode()) {
            el.className = 'training-health';
            return;
        }

        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        if (!isRunning) {
            ageEl.textContent = '-';
            el.className = 'training-health';
            el.textContent = '未运行任务。';
            return;
        }

        const ageSeconds = trainingRuntime.lastOutputAt
            ? Math.max(0, Math.floor((Date.now() - trainingRuntime.lastOutputAt) / 1000))
            : null;
        ageEl.textContent = ageSeconds == null ? '-' : formatDuration(ageSeconds);

        const jobName = trainingRuntime.job === 'preprocess' ? '预处理' : '训练';

        const gpu = trainingRuntime.lastGpuUtil;
        const gpuActive = gpu != null && gpu >= 15;
        if (ageSeconds == null) {
            el.className = 'training-health';
            el.textContent = gpuActive
                ? `${jobName}运行中，GPU ${gpu}% 活跃，等待第一条日志。`
                : `${jobName}运行中，等待日志和系统指标。`;
            return;
        }

        if (ageSeconds >= 180 && gpuActive) {
            el.className = 'training-health warning';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，但 GPU ${gpu}% 仍在工作；通常是单步较慢或任务脚本未输出进度。`;
            if (!trainingRuntime.quietHintShown) {
                appendLog(`[提示] ${el.textContent}`);
                trainingRuntime.quietHintShown = true;
            }
            return;
        }

        if (ageSeconds >= 180) {
            el.className = 'training-health error';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，且 GPU 活跃度不高；建议观察进程或检查终端输出。`;
            return;
        }

        el.className = 'training-health ok';
        el.textContent = gpu == null
            ? `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出。`
            : `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出，GPU ${gpu}%。`;
    }

    function parseMetricsFromProgressLine(line) {
        const text = String(line || '');
        const stepMatch = text.match(/\|\s*(\d+)\/\d+\s*\[/) || text.match(/step[=:/\s]+(\d+)/i);
        const lossMatch = text.match(/(?:avr_)?loss[=:/\s]+([\d.eE\-+]+)/i);
        const lrMatch = text.match(/(?:^|[\s,])(?:lr|learning_rate)[=:/\s]+([\d.eE\-+]+)/i);
        const rateMatch = text.match(/([\d.]+\s*(?:s\/it|it\/s|s\/step))/i);
        const out = {};
        if (stepMatch) out.step = Number(stepMatch[1]);
        if (lossMatch) out.loss = Number(lossMatch[1]);
        if (lrMatch) out.lr = Number(lrMatch[1]);
        if (rateMatch) out.rate = rateMatch[1].replace(/\s+/g, '');
        if (Object.keys(out).length === 0) return null;
        if (out.step !== undefined && !Number.isFinite(out.step)) delete out.step;
        if (out.loss !== undefined && !Number.isFinite(out.loss)) delete out.loss;
        if (out.lr !== undefined && !Number.isFinite(out.lr)) delete out.lr;
        return Object.keys(out).length ? out : null;
    }

    function lastValue(records, key) {
        for (let i = records.length - 1; i >= 0; i -= 1) {
            const value = records[i]?.[key];
            if (value !== undefined && value !== null && value !== '') return value;
        }
        return undefined;
    }

    function readConfigNumber(configText, key) {
        const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const match = String(configText || '').match(new RegExp(`^\\s*${escapedKey}\\s*=\\s*([^\\n#]+)`, 'm'));
        if (!match) return undefined;
        const value = Number(match[1].trim().replace(/^["']|["']$/g, ''));
        return Number.isFinite(value) ? value : undefined;
    }

    function formatLr(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n.toExponential(2) : '-';
    }

    function formatDuration(totalSeconds) {
        const seconds = Math.max(0, Number(totalSeconds) || 0);
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        const rest = seconds % 60;
        if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
        const hours = Math.floor(minutes / 60);
        const restMinutes = minutes % 60;
        return restMinutes ? `${hours}h ${restMinutes}m` : `${hours}h`;
    }

    // ── 全局设置 ──
    async function loadGlobalSettings() {
        if (location.protocol === 'file:') return;
        try {
            const data = await api('/api/settings/global');
            if (!data.ok) throw new Error(data.error || '读取全局设置失败');
            globalSettings = data;
            applyGlobalSettingsToInputs(data);
            updateChoiceGuide();
            setGlobalSettingsStatus('', '');
            if (tomlManagerMode === 'output') {
                await loadOutputRuns({ keepSelection: true });
            }
        } catch (e) {
            setGlobalSettingsStatus('读取全局设置失败: ' + e.message, 'error');
        }
    }

    async function saveGlobalSettings() {
        try {
            const payload = collectGlobalSettingsPayload();
            const res = await api('/api/settings/global', {
                method: 'PUT',
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                setGlobalSettingsStatus(res.error || '保存失败', 'error');
                return;
            }
            globalSettings = {
                ...(globalSettings || {}),
                ...res,
            };
            applyGlobalSettingsToInputs(globalSettings);
            updateChoiceGuide();
            setGlobalSettingsStatus(res.message || '全局设置已保存', 'ok');
        } catch (e) {
            setGlobalSettingsStatus('保存失败: ' + e.message, 'error');
        }
    }

    async function resetGlobalSettings() {
        applyGlobalSettingsToInputs({
            defaults: globalSettings?.defaults || {},
            output_root: globalSettings?.defaults?.output_root || 'output/runs',
            ...Object.fromEntries(GLOBAL_MODEL_PATH_FIELDS.map(([key]) => [key, globalSettings?.defaults?.[key] || ''])),
        });
        await saveGlobalSettings();
    }

    function setGlobalSettingsStatus(text, state = '') {
        const el = document.getElementById('global-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

    function applyGlobalSettingsToInputs(data) {
        const snapshot = data || globalSettings || {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            if (!input) continue;
            const fallback = snapshot?.defaults?.[key] || '';
            input.value = snapshot?.[key] ?? fallback;
        }
    }

    function collectGlobalSettingsPayload() {
        const payload = {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            payload[key] = input ? input.value : (globalSettings?.[key] || '');
        }
        return payload;
    }

    function getGlobalModelPathOverrides() {
        const overrides = {};
        const source = globalSettings || {};
        for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
            const value = source[key] ?? source.defaults?.[key] ?? '';
            if (String(value || '').trim()) {
                overrides[key] = String(value).trim();
            }
        }
        return overrides;
    }

    function toggleGlobalSettingHelp(button) {
        if (!button) return;
        const helpId = button.getAttribute('aria-controls');
        const help = helpId ? document.getElementById(helpId) : null;
        if (!help) return;
        const visible = help.classList.toggle('visible');
        button.classList.toggle('active', visible);
        button.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    // ── 预览图 ──
    async function loadPreviewSettings() {
        if (location.protocol === 'file:') return;
        try {
            const taskQuery = selectedPreviewTaskId && !selectedPreviewGroup
                ? `?task_id=${encodeURIComponent(selectedPreviewTaskId)}`
                : '';
            previewSettings = await api('/api/preview/settings' + taskQuery);
            document.getElementById('preview-training-dir').value = previewSettings.training_dir || '';
            document.getElementById('preview-inference-dir').value = previewSettings.inference_dir || '';
            document.getElementById('preview-custom-dir').value = previewSettings.custom_dir || '';
            updatePreviewDirectorySummary();
            renderPreviewTaskSelect();
        } catch (e) {
            setPreviewStatus('读取路径设置失败: ' + e.message, 'error');
        }
    }

    async function savePreviewSettings() {
        try {
            const res = await api('/api/preview/settings', {
                method: 'PUT',
                body: JSON.stringify({
                    training_dir: val('preview-training-dir'),
                    inference_dir: val('preview-inference-dir'),
                    custom_dir: val('preview-custom-dir'),
                }),
            });
            if (!res.ok) {
                setPreviewStatus(res.error || '保存失败', 'error');
                return;
            }
            setPreviewStatus(res.message || '路径设置已保存', 'ok');
            await loadPreviewSettings();
            await loadPreviewImages();
        } catch (e) {
            setPreviewStatus('保存失败: ' + e.message, 'error');
        }
    }

    async function resetPreviewSettings() {
        if (!previewSettings?.defaults) return;
        document.getElementById('preview-training-dir').value = previewSettings.defaults.training_dir || 'output/ckpt/sample';
        document.getElementById('preview-inference-dir').value = previewSettings.defaults.inference_dir || 'output/tests';
        document.getElementById('preview-custom-dir').value = previewSettings.defaults.custom_dir || '';
        await savePreviewSettings();
    }

    async function loadPreviewImages() {
        if (location.protocol === 'file:') {
            setPreviewEmpty('静态打开没有后端 API，无法读取项目预览图。');
            return;
        }
        const requestSeq = ++previewRequestSeq;
        setPreviewLoading();
        try {
            if (!historyTasks.length) {
                await loadTrainingHistoryList();
            }
            if (!previewSettings) {
                await loadPreviewSettings();
            }
            const params = new URLSearchParams({ source: currentPreviewSource });
            if (currentPreviewSource === 'training' && selectedPreviewGroup) {
                params.set('mode', 'config_group');
                params.set('methods_subdir', selectedPreviewGroup.methods_subdir);
                params.set('variant', selectedPreviewGroup.variant);
                params.set('preset', selectedPreviewGroup.preset || 'default');
                if (selectedPreviewGroup.history_group_key) {
                    params.set('group_key', selectedPreviewGroup.history_group_key);
                }
                params.set('include_archived', showArchivedHistory ? '1' : '0');
            } else if (currentPreviewSource === 'training' && selectedPreviewTaskId) {
                params.set('task_id', selectedPreviewTaskId);
            }
            const payload = await api(`/api/preview/images?${params.toString()}`);
            if (requestSeq !== previewRequestSeq) return;
            if (!payload.ok) {
                setPreviewEmpty(payload.error || '读取预览图失败');
                return;
            }
            renderPreviewImages(payload);
            trainingSampleState = payload.sample_config || trainingSampleState;
            loadPreviewWeights();
        } catch (e) {
            if (requestSeq !== previewRequestSeq) return;
            setPreviewEmpty('读取预览图失败: ' + e.message);
        }
    }

    async function loadPreviewWeights() {
        const requestSeq = ++previewWeightRequestSeq;
        if (location.protocol === 'file:') {
            if (requestSeq === previewWeightRequestSeq) {
                renderPreviewWeights({ ok: true, weights: [], message: '静态打开没有后端 API。' });
            }
            return;
        }
        if (currentPreviewSource !== 'training') {
            if (requestSeq === previewWeightRequestSeq) {
                renderPreviewWeights({
                    ok: true,
                    weights: [],
                    message: '权重文件只随训练来源显示。',
                });
            }
            return;
        }
        try {
            const params = new URLSearchParams({ source: 'training' });
            if (selectedPreviewGroup) {
                params.set('mode', 'config_group');
                params.set('methods_subdir', selectedPreviewGroup.methods_subdir);
                params.set('variant', selectedPreviewGroup.variant);
                params.set('preset', selectedPreviewGroup.preset || 'default');
                if (selectedPreviewGroup.history_group_key) {
                    params.set('group_key', selectedPreviewGroup.history_group_key);
                }
                params.set('include_archived', showArchivedHistory ? '1' : '0');
            } else if (selectedPreviewTaskId) {
                params.set('task_id', selectedPreviewTaskId);
            }
            const payload = await api(`/api/preview/weights?${params.toString()}`);
            if (requestSeq !== previewWeightRequestSeq) return;
            renderPreviewWeights(payload);
        } catch (e) {
            if (requestSeq !== previewWeightRequestSeq) return;
            renderPreviewWeights({ ok: false, weights: [], error: '读取权重文件失败: ' + e.message });
        }
    }

    function setPreviewSource(source) {
        currentPreviewSource = source || 'training';
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.previewSource === currentPreviewSource);
        });
        updatePreviewTaskVisibility();
        updatePreviewDirectorySummary();
        previewWeightRequestSeq += 1;
        loadPreviewImages();
    }

    function renderPreviewTaskSelect() {
        const select = document.getElementById('preview-training-task');
        if (!select) return;
        const previousValue = selectedPreviewSelectValue();
        select.innerHTML = '';
        const liveOption = document.createElement('option');
        liveOption.value = '';
        select.appendChild(liveOption);

        const trainingTasks = historyTasks
            .filter((task) => task.job === 'training' && (showArchivedHistory || !historyTaskIsArchived(task)))
            .sort((a, b) => Number(b.started_at || 0) - Number(a.started_at || 0));
        liveOption.textContent = trainingTasks.length
            ? `当前任务或最新运行目录 · ${trainingTasks.length} 个历史训练`
            : '当前任务或最新运行目录 · 暂无历史训练';

        const groups = previewTrainingGroups(trainingTasks);
        if (groups.length) {
            const groupOptions = document.createElement('optgroup');
            groupOptions.label = '训练分组合并';
            for (const group of groups) {
                const option = document.createElement('option');
                option.value = encodePreviewGroupValue(group);
                option.textContent = `${group.label} · ${group.tasks.length} 次训练`;
                groupOptions.appendChild(option);
            }
            select.appendChild(groupOptions);
        }

        if (trainingTasks.length) {
            const taskOptions = document.createElement('optgroup');
            taskOptions.label = '单个训练任务';
            for (const task of trainingTasks) {
                const option = document.createElement('option');
                option.value = encodePreviewTaskValue(task.id);
                option.textContent = [
                    task.name || `${task.methods_subdir || '-'} / ${task.variant || '-'}`,
                    task.started_at_text || task.id,
                    historyStateLabel(task.state),
                ].filter(Boolean).join(' · ');
                taskOptions.appendChild(option);
            }
            select.appendChild(taskOptions);
        }

        const values = Array.from(select.options).map((option) => option.value);
        const nextValue = values.includes(previousValue) ? previousValue : '';
        applyPreviewSelectionValue(nextValue);
        select.value = nextValue;
        select.disabled = false;
        updatePreviewTaskVisibility();
    }

    function previewTrainingGroups(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            if (!map.has(group.key)) {
                map.set(group.key, { ...group, tasks: [] });
            }
            map.get(group.key).tasks.push(task);
        }
        return Array.from(map.values())
            .filter((group) => group.tasks.length > 0)
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || a.label.localeCompare(b.label, 'zh-CN');
            });
    }

    function selectedPreviewSelectValue() {
        if (selectedPreviewGroup) return encodePreviewGroupValue(selectedPreviewGroup);
        if (selectedPreviewTaskId) return encodePreviewTaskValue(selectedPreviewTaskId);
        return '';
    }

    function encodePreviewTaskValue(taskId) {
        return `task:${taskId || ''}`;
    }

    function encodePreviewGroupValue(group) {
        const payload = [
            group.methods_subdir || '',
            group.variant || '',
            group.preset || 'default',
            group.history_group_key || '',
            group.history_group_label || '',
            group.history_source_config_file || '',
        ].map((value) => encodeURIComponent(value)).join('|');
        return `group:${payload}`;
    }

    function decodePreviewGroupValue(value) {
        if (!String(value || '').startsWith('group:')) return null;
        const parts = String(value).slice(6).split('|').map((item) => decodeURIComponent(item));
        if (!parts[0] || !parts[1]) return null;
        return {
            methods_subdir: parts[0],
            variant: parts[1],
            preset: parts[2] || 'default',
            history_group_key: parts[3] || '',
            history_group_label: parts[4] || '',
            history_source_config_file: parts[5] || '',
            label: parts[4] || parts[5] || `${parts[0]} / ${parts[1]} / ${parts[2] || 'default'}`,
        };
    }

    function applyPreviewSelectionValue(value) {
        const group = decodePreviewGroupValue(value);
        if (group) {
            selectedPreviewGroup = group;
            selectedPreviewTaskId = '';
            return;
        }
        selectedPreviewGroup = null;
        selectedPreviewTaskId = String(value || '').startsWith('task:') ? String(value).slice(5) : '';
    }

    function updatePreviewTaskVisibility() {
        const field = document.getElementById('preview-training-task-field');
        if (field) field.hidden = currentPreviewSource !== 'training';
    }

    async function changePreviewTask(taskId) {
        applyPreviewSelectionValue(taskId || '');
        previewSettings = null;
        previewWeightRequestSeq += 1;
        await loadPreviewSettings();
        await loadPreviewImages();
    }

    function renderPreviewImages(payload) {
        const grid = document.getElementById('preview-grid');
        const empty = document.getElementById('preview-empty');
        const title = document.getElementById('preview-title');
        const subtitle = document.getElementById('preview-subtitle');
        const count = document.getElementById('preview-count');

        title.textContent = payload.label || previewSourceLabel(currentPreviewSource);
        subtitle.textContent = payload.directory
            ? `目录: ${payload.directory}${previewDirectoryHint(payload)}`
            : '尚未设置目录。';
        count.textContent = `${payload.count || 0} 张`;
        document.getElementById('preview-current-dir').textContent = payload.directory || '-';

        grid.innerHTML = '';
        if (!payload.images?.length) {
            setPreviewEmpty(previewEmptyMessage(payload));
            return;
        }
        empty.hidden = true;
        for (const image of payload.images) {
            grid.appendChild(createPreviewCard(image));
        }
    }

    function renderPreviewWeights(payload) {
        const list = document.getElementById('preview-weights-list');
        const empty = document.getElementById('preview-weights-empty');
        const subtitle = document.getElementById('preview-weights-subtitle');
        if (!list || !empty || !subtitle) return;

        const weights = payload.weights || [];
        const sortedWeights = sortPreviewWeights(weights);
        subtitle.textContent = payload.directory
            ? `目录: ${payload.directory}${payload.mode === 'config_group' && payload.group_task_count != null
                ? ` · ${payload.group_task_count} 次训练`
                : payload.task_count
                    ? ` · 本任务 ${payload.task_count} 个`
                    : ''}`
            : '选择训练任务后显示保存轮次、步数和对应权重。';
        updatePreviewWeightSortButton();
        list.innerHTML = '';
        if (!sortedWeights.length) {
            empty.textContent = payload.error || payload.message || '未找到权重文件。';
            empty.hidden = false;
            return;
        }
        empty.hidden = true;
        for (const item of sortedWeights) {
            list.appendChild(createPreviewWeightItem(item));
        }
    }

    function sortPreviewWeights(weights) {
        const direction = previewWeightSortDirection === 'desc' ? -1 : 1;
        return [...(weights || [])].sort((a, b) => comparePreviewWeight(a, b, direction));
    }

    function comparePreviewWeight(a, b, direction) {
        const epochDiff = compareOptionalNumber(a.epoch, b.epoch, direction);
        if (epochDiff !== 0) return epochDiff;
        const stepDiff = compareOptionalNumber(a.steps, b.steps, direction);
        if (stepDiff !== 0) return stepDiff;
        return String(a.name || '').localeCompare(String(b.name || ''), 'zh-CN');
    }

    function compareOptionalNumber(a, b, direction) {
        const aNumber = sortableNumber(a);
        const bNumber = sortableNumber(b);
        if (aNumber === null && bNumber === null) return 0;
        if (aNumber === null) return 1;
        if (bNumber === null) return -1;
        return direction * (aNumber - bNumber);
    }

    function sortableNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function updatePreviewWeightSortButton() {
        const btn = document.getElementById('btn-sort-weights');
        if (!btn) return;
        const isDesc = previewWeightSortDirection === 'desc';
        btn.textContent = isDesc ? '反序' : '正序';
        btn.title = isDesc ? '当前按 Epoch/Step 从大到小排列，点击切换为正序。' : '当前按 Epoch/Step 从小到大排列，点击切换为反序。';
    }

    function togglePreviewWeightSort() {
        previewWeightSortDirection = previewWeightSortDirection === 'asc' ? 'desc' : 'asc';
        updatePreviewWeightSortButton();
        loadPreviewWeights();
    }

    function createPreviewWeightItem(item) {
        const row = document.createElement('article');
        row.className = `preview-weight-item preview-weight-${item.kind || 'weight'}`;
        if (item.scope === 'task') {
            row.classList.add('preview-weight-task');
        }
        if (item.source_task?.id) {
            row.dataset.sourceTaskId = item.source_task.id;
        }

        const main = document.createElement('div');
        main.className = 'preview-weight-main';
        const name = document.createElement('strong');
        name.textContent = item.name;
        const file = document.createElement('span');
        file.textContent = item.file || '';
        const badge = document.createElement('em');
        badge.textContent = item.scope_label || '';
        main.append(name, file, badge);

        const actions = document.createElement('div');
        actions.className = 'preview-weight-actions';
        const download = document.createElement('a');
        download.className = 'btn btn-small btn-primary preview-weight-download';
        download.href = previewWeightDownloadUrl(item);
        download.download = item.name || 'weight.safetensors';
        download.textContent = '下载';
        download.title = '通过浏览器下载这个权重文件。';
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'btn btn-small preview-weight-copy';
        copy.textContent = '复制路径';
        copy.title = '复制这个权重文件的完整路径。';
        copy.addEventListener('click', () => copyPreviewWeightPath(item, copy));
        const continueBtn = document.createElement('button');
        continueBtn.type = 'button';
        continueBtn.className = 'btn btn-small preview-weight-continue';
        continueBtn.textContent = '继续训练';
        continueBtn.title = '把这个权重设置为新的 LoRA/LoKr 补充训练来源。';
        continueBtn.addEventListener('click', () => selectContinueLoraWeight(item.abs_path || item.file || ''));
        actions.append(continueBtn, download, copy);

        const stats = document.createElement('div');
        stats.className = 'preview-weight-stats';
        stats.append(
            createWeightStat('Epoch', item.epoch ?? '-'),
            createWeightStat('Step', item.steps ?? '-'),
            createWeightStat('计划', weightPlanText(item)),
            createWeightStat('保存', item.mtime_text || '-'),
            createWeightStat('大小', formatBytes(item.size_bytes)),
            createWeightStat('类型', weightKindLabel(item.kind)),
        );
        const source = createPreviewWeightSource(item);
        if (source) stats.append(source);

        row.append(main, stats, actions);
        return row;
    }

    function previewWeightDownloadUrl(item) {
        if (item.download_url) return item.download_url;
        const params = new URLSearchParams({ file: item.file || '' });
        const taskId = item.source_task?.id || '';
        if (taskId) params.set('task_id', taskId);
        return `/api/preview/weight?${params.toString()}`;
    }

    function createPreviewWeightSource(item) {
        if (!item.source_task?.label) return null;
        const box = document.createElement('div');
        box.className = 'preview-weight-source';
        const sourceText = `来源 ${item.source_task.label}`;
        const text = document.createElement('span');
        text.className = 'preview-weight-source-text';
        text.textContent = sourceText;
        text.title = '双击复制来源文本；也可以像普通文本一样拖选。';
        text.addEventListener('dblclick', async () => {
            await copyPreviewWeightSource(sourceText, text);
        });
        box.appendChild(text);
        return box;
    }

    async function copyPreviewWeightSource(text, el) {
        selectElementText(el);
        try {
            await copyText(text);
            selectElementText(el);
            el.classList.add('copied');
            const originalTitle = el.title;
            el.title = '已复制来源文本。';
            setTimeout(() => {
                el.classList.remove('copied');
                el.title = originalTitle;
            }, 1000);
        } catch (e) {
            selectElementText(el);
            alert('复制来源失败: ' + e.message);
        }
    }

    function selectElementText(el) {
        if (!el || !window.getSelection || !document.createRange) return;
        const range = document.createRange();
        range.selectNodeContents(el);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }

    async function copyPreviewWeightPath(item, button) {
        const path = item.abs_path || item.file || '';
        if (!path) return;
        try {
            await copyText(path);
            const original = button.textContent;
            button.textContent = '已复制';
            button.classList.add('btn-primary');
            setTimeout(() => {
                button.textContent = original;
                button.classList.remove('btn-primary');
            }, 1200);
        } catch (e) {
            alert('复制权重路径失败: ' + e.message);
        }
    }

    function createWeightStat(label, value) {
        const box = document.createElement('div');
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('strong');
        valEl.textContent = value;
        box.append(key, valEl);
        return box;
    }

    function weightKindLabel(kind) {
        return {
            epoch: '按轮保存',
            step: '按步保存',
            resume: '续训检查点',
            final: '最终权重',
            weight: '权重',
        }[kind] || '权重';
    }

    function weightPlanText(item) {
        const epochs = item.num_epochs ? `${item.num_epochs}ep` : '';
        const steps = item.max_steps ? `${item.max_steps}步` : '';
        return [epochs, steps].filter(Boolean).join(' / ') || '-';
    }

    function createPreviewCard(image) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'preview-card';
        button.addEventListener('click', () => openPreviewDialog(image));

        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name;
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            button.classList.add('preview-card-error');
            img.alt = '图片加载失败';
        });

        const meta = document.createElement('div');
        meta.className = 'preview-card-meta';
        const title = document.createElement('strong');
        title.textContent = image.name;
        const detail = document.createElement('span');
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        detail.textContent = previewCardPrimaryMeta(image);
        const sub = document.createElement('span');
        sub.textContent = previewCardSecondaryMeta(image, dims);
        meta.append(title, detail, sub);
        if (image.source_task?.label) {
            const source = document.createElement('span');
            source.textContent = `来源: ${image.source_task.label}`;
            meta.appendChild(source);
        }

        button.append(img, meta);
        return button;
    }

    function openPreviewDialog(image) {
        const dialog = document.getElementById('preview-dialog');
        const img = document.getElementById('preview-dialog-image');
        document.getElementById('preview-dialog-title').textContent = image.name;
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        document.getElementById('preview-dialog-meta').textContent =
            `${image.file} · ${dims} · ${formatBytes(image.size_bytes)} · ${image.mtime_text || ''}`;
        renderPreviewDialogDetails(image, dims);
        img.src = image.url;
        img.alt = image.name;
        if (dialog?.showModal) {
            dialog.showModal();
        }
    }

    function setPreviewLoading() {
        document.getElementById('preview-count').textContent = '读取中';
        document.getElementById('preview-grid').innerHTML = '';
        setPreviewEmpty('正在读取预览图...');
    }

    function setPreviewEmpty(message) {
        const empty = document.getElementById('preview-empty');
        if (!empty) return;
        empty.textContent = message;
        empty.hidden = false;
        document.getElementById('preview-grid').innerHTML = '';
    }

    function previewEmptyMessage(payload) {
        const base = payload.message || '暂无预览图。';
        if (currentPreviewSource !== 'training') return base;
        const cfg = payload.sample_config || trainingSampleState || trainingRuntime.sampleConfig || {};
        const msg = cfg.message || '';
        if (!msg || base.includes(msg)) return base;
        if (cfg.enabled) {
            const samplingDelayHint = '如果训练刚开始，可能还没到达采样频率。';
            return base.includes(samplingDelayHint) ? base : `${base} ${samplingDelayHint}`;
        }
        if (payload.preview_settings?.effective_training_source === 'latest_run') {
            const latestRunHint = '最新运行目录里还没有可显示的样张。';
            return base.includes(latestRunHint) ? base : `${base} ${latestRunHint}`;
        }
        return `${base} ${msg}。`;
    }

    function updatePreviewDirectorySummary() {
        const el = document.getElementById('preview-current-dir');
        if (!el || !previewSettings) return;
        if (currentPreviewSource === 'training') {
            el.textContent = previewSettings.effective_training_dir || previewSettings.training_dir || '-';
        } else if (currentPreviewSource === 'inference') {
            el.textContent = previewSettings.inference_dir || '-';
        } else {
            el.textContent = previewSettings.custom_dir || '-';
        }
    }

    function previewDirectoryHint(payload) {
        const source = payload.preview_settings?.effective_training_source || '';
        const latestRun = payload.preview_settings?.latest_run_dir || '';
        if (source === 'current_task') return ' · 当前任务';
        if (source === 'latest_run') {
            return latestRun ? ` · 最新运行 ${latestRun}` : ' · 最新运行';
        }
        return '';
    }

    function setPreviewStatus(text, state = '') {
        const el = document.getElementById('preview-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

    function previewSourceLabel(source) {
        return {
            training: '训练过程中采样结果',
            inference: '推理预览',
            custom: '自定义路径',
        }[source] || '预览图';
    }

    function previewCardPrimaryMeta(image) {
        const sample = image.sample || {};
        const parts = [];
        if (sample.epoch != null) parts.push(`Epoch ${sample.epoch}`);
        if (sample.step != null) parts.push(`Step ${sample.step}`);
        if (sample.seed != null) parts.push(`seed ${sample.seed}`);
        return parts.length ? parts.join(' · ') : (image.mtime_text || '无采样元信息');
    }

    function previewCardSecondaryMeta(image, dims) {
        const sample = image.sample || {};
        const params = sample.parameters || {};
        const renderSize = params.width && params.height ? `${params.width}x${params.height}` : dims;
        const steps = params.sample_steps ? `${params.sample_steps} steps` : '';
        const sampler = sample.sampler || params.sample_sampler || '';
        return [renderSize, steps, sampler, formatBytes(image.size_bytes)].filter(Boolean).join(' · ');
    }

    function renderPreviewDialogDetails(image, dims) {
        const box = document.getElementById('preview-dialog-details');
        if (!box) return;
        box.innerHTML = '';
        if (image.detailContext === 'dataset') {
            renderDatasetImageDialogDetails(box, image, dims);
            return;
        }
        const sample = image.sample || {};
        const params = sample.parameters || {};
        const promptNo = sample.prompt_index != null ? Number(sample.prompt_index) + 1 : null;

        const rows = [
            ['轮次', sample.epoch != null ? `Epoch ${sample.epoch}` : '-'],
            ['步数', sample.step != null ? `Step ${sample.step}` : '-'],
            ['来源任务', image.source_task?.label || '-'],
            ['任务时间', image.source_task?.started_at_text || '-'],
            ['提示词序号', promptNo ? `第 ${promptNo} 条` : '-'],
            ['生成时间', sample.generated_at_text || image.mtime_text || '-'],
            ['种子', sample.seed ?? params.seed ?? '-'],
            ['采样器', sample.sampler || params.sample_sampler || '-'],
            ['生成步数', params.sample_steps ?? '-'],
            ['CFG', params.guidance_scale ?? params.scale ?? '-'],
            ['Flow Shift', params.flow_shift ?? '-'],
            ['尺寸', params.width && params.height ? `${params.width}x${params.height}` : dims],
            ['文件大小', formatBytes(image.size_bytes)],
            ['提示词文件', sample.source?.prompt_file || '-'],
        ];
        for (const [label, value] of rows) {
            box.appendChild(createPreviewDetailRow(label, value));
        }
        if (sample.prompt) {
            box.appendChild(createPreviewDetailBlock('提示词', sample.prompt));
        }
        if (sample.negative_prompt) {
            box.appendChild(createPreviewDetailBlock('负面提示词', sample.negative_prompt));
        }
        if (sample.raw_prompt) {
            box.appendChild(createPreviewDetailBlock('原始参数行', sample.raw_prompt));
        }
        box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
    }

    function renderDatasetImageDialogDetails(box, image, dims) {
        const caption = image.caption || {};
        const rows = [
            ['文件时间', image.mtime_text || '-'],
            ['尺寸', dims],
            ['长', image.height ? `${image.height} px` : '-'],
            ['宽', image.width ? `${image.width} px` : '-'],
            ['总像素', formatTotalPixels(image.total_pixels)],
            ['文件大小', formatBytes(image.size_bytes)],
        ];
        for (const [label, value] of rows) {
            box.appendChild(createPreviewDetailRow(label, value));
        }
        box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
        box.appendChild(createPreviewDetailBlock('标注文件', caption.file || '未找到同名标注文件'));
        const captionText = caption.ok ? (caption.text || '(空标注)') : '未找到同名 caption 文件';
        box.appendChild(createPreviewDetailBlock('标注内容', captionText, true));
    }

    function formatTotalPixels(totalPixels) {
        const count = Number(totalPixels);
        if (!Number.isFinite(count) || count <= 0) return '-';
        return `${count.toLocaleString('zh-CN')} px (${(count / 1000000).toFixed(2)} MP)`;
    }

    function createPreviewDetailRow(label, value) {
        const row = document.createElement('div');
        row.className = 'preview-detail-row';
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('strong');
        valEl.textContent = value;
        row.append(key, valEl);
        return row;
    }

    function createPreviewDetailBlock(label, value, preformatted = false) {
        const block = document.createElement('div');
        block.className = 'preview-detail-block';
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('p');
        if (preformatted) valEl.className = 'preview-detail-preformatted';
        valEl.textContent = value;
        block.append(key, valEl);
        return block;
    }

    async function copyText(text) {
        if (navigator.clipboard?.writeText) {
            try {
                await navigator.clipboard.writeText(text);
                return;
            } catch (_) {
                // 浏览器可能因权限或焦点拒绝 Clipboard API，继续使用 textarea 兜底。
            }
        }
        const area = document.createElement('textarea');
        area.value = text;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.left = '-9999px';
        document.body.appendChild(area);
        area.focus();
        area.select();
        try {
            if (!document.execCommand('copy')) {
                throw new Error('浏览器拒绝复制操作');
            }
        } finally {
            area.remove();
        }
    }

    function formatBytes(bytes) {
        const n = Number(bytes) || 0;
        if (n < 1024) return `${n} B`;
        if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
        return `${(n / 1024 / 1024).toFixed(1)} MB`;
    }

    // ── 训练队列 ──
    async function loadTrainingQueue() {
        if (location.protocol === 'file:') return;
        trainingQueueState = { ...trainingQueueState, loading: true, error: '' };
        renderTrainingQueue();
        try {
            const payload = await api('/api/training/queue');
            updateTrainingQueueFromPayload(payload);
        } catch (e) {
            trainingQueueState = { ...trainingQueueState, loading: false, error: '读取队列失败: ' + e.message };
            renderTrainingQueue();
        }
    }

    function updateTrainingQueueFromPayload(payload = {}) {
        trainingQueueState = {
            loading: false,
            paused: Boolean(payload.paused),
            items: Array.isArray(payload.items) ? payload.items : [],
            error: payload.ok === false ? (payload.error || '队列状态异常') : '',
            currentItemId: String(payload.current_item_id || ''),
        };
        renderTrainingQueue();
    }

    function renderTrainingQueue() {
        const list = document.getElementById('training-queue-list');
        const summary = document.getElementById('training-queue-summary');
        const pauseBtn = document.getElementById('btn-toggle-queue-pause');
        if (!list || !summary) return;
        list.innerHTML = '';
        const activeItems = trainingQueueState.items.filter((item) =>
            ['queued', 'running'].includes(String(item.state || ''))
        );
        const queuedCount = activeItems.filter((item) => item.state === 'queued').length;
        const running = activeItems.find((item) => item.state === 'running');
        summary.className = [
            'training-queue-summary',
            trainingQueueState.paused ? 'paused' : '',
            running ? 'running' : '',
        ].filter(Boolean).join(' ');
        if (trainingQueueState.loading) {
            summary.textContent = '正在读取队列...';
        } else if (trainingQueueState.error) {
            summary.textContent = trainingQueueState.error;
        } else if (running) {
            summary.textContent = `正在运行：${queueItemTitle(running)} · 等待 ${queuedCount} 个`;
        } else if (queuedCount) {
            summary.textContent = trainingQueueState.paused
                ? `队列已暂停 · 等待 ${queuedCount} 个任务`
                : `空闲时会自动启动 · 等待 ${queuedCount} 个任务`;
        } else {
            summary.textContent = trainingQueueState.paused ? '队列已暂停，暂无等待任务。' : '暂无等待任务。';
        }
        if (pauseBtn) {
            pauseBtn.textContent = trainingQueueState.paused ? '继续' : '暂停';
            pauseBtn.disabled = trainingQueueState.loading;
        }
        const visible = activeItems.length
            ? activeItems
            : trainingQueueState.items
                .filter((item) => ['done', 'error', 'canceled'].includes(String(item.state || '')))
                .slice(-3)
                .reverse();
        if (!visible.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = '从配置页开始训练时，可以选择加入队列。';
            list.appendChild(empty);
            return;
        }
        for (const item of visible) {
            list.appendChild(createTrainingQueueItem(item));
        }
    }

    function createTrainingQueueItem(item) {
        const card = document.createElement('article');
        card.className = ['training-queue-item', item.state || 'queued'].join(' ');
        const state = document.createElement('span');
        state.className = 'training-queue-state';
        state.textContent = queueStateLabel(item.state);

        const main = document.createElement('div');
        main.className = 'training-queue-main';
        const title = document.createElement('strong');
        title.textContent = queueItemTitle(item);
        const meta = document.createElement('span');
        meta.textContent = [
            item.requires_preprocess ? '预处理后训练' : (item.kind === 'resume' ? '续训' : '训练'),
            `${item.methods_subdir || '-'} / ${item.variant || '-'}`,
            `GPU: ${queueGpuLabel(item.gpu_whitelist)}`,
        ].filter(Boolean).join(' · ');
        const path = document.createElement('em');
        path.textContent = item.runtime_config_file || item.source_config_file || '';
        const message = document.createElement('em');
        message.textContent = [
            item.message || '',
            item.created_at_text ? `入队: ${item.created_at_text}` : '',
        ].filter(Boolean).join(' · ');
        main.append(title, meta, path, message);

        const actions = document.createElement('div');
        actions.className = 'training-queue-item-actions';
        if (item.state === 'queued') {
            actions.append(
                createQueueActionButton('上移', () => moveQueueItem(item.id, 'up')),
                createQueueActionButton('下移', () => moveQueueItem(item.id, 'down')),
                createQueueActionButton('取消', () => cancelQueueItem(item.id), 'danger'),
            );
        }
        card.append(state, main);
        if (actions.childNodes.length) card.appendChild(actions);
        return card;
    }

    function createQueueActionButton(label, handler, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.addEventListener('click', handler);
        return btn;
    }

    function queueItemTitle(item) {
        const resumeName = item?.resume_info?.checkpoint_name || '';
        const source = item?.source_config_file || item?.runtime_config_file || '';
        const fallback = runLabelFromPath(source) || `${item?.variant || '训练'} / ${item?.preset || 'default'}`;
        return item?.kind === 'resume' && resumeName ? `续训 · ${resumeName}` : fallback;
    }

    function queueGpuLabel(value) {
        const list = Array.isArray(value) ? value : [];
        return list.length ? list.join(',') : '全部';
    }

    function queueStateLabel(state) {
        return {
            queued: '等待',
            running: '运行中',
            done: '完成',
            error: '异常',
            canceled: '已取消',
        }[state] || state || '未知';
    }

    async function moveQueueItem(itemId, direction) {
        try {
            const payload = await api(`/api/training/queue/${encodeURIComponent(itemId)}/move`, {
                method: 'POST',
                body: JSON.stringify({ direction }),
            });
            updateTrainingQueueFromPayload(payload);
            if (!payload.ok) appendLog(`[状态] ${payload.error || '移动队列任务失败'}`);
        } catch (e) {
            appendLog(`[状态] 移动队列任务失败: ${e.message}`);
        }
    }

    async function cancelQueueItem(itemId) {
        const ok = await showAppConfirmDialog({
            title: '取消队列任务',
            description: '等待中的任务会从自动调度中移除',
            message: '确定要取消这个队列任务吗？已创建的运行目录会保留，方便排查。',
            confirmText: '取消任务',
            danger: true,
        });
        if (!ok) return;
        try {
            const payload = await api(`/api/training/queue/${encodeURIComponent(itemId)}`, { method: 'DELETE' });
            updateTrainingQueueFromPayload(payload);
            if (!payload.ok) appendLog(`[状态] ${payload.error || '取消队列任务失败'}`);
        } catch (e) {
            appendLog(`[状态] 取消队列任务失败: ${e.message}`);
        }
    }

    async function toggleTrainingQueuePause() {
        try {
            const payload = await api('/api/training/queue/pause', {
                method: 'POST',
                body: JSON.stringify({ paused: !trainingQueueState.paused }),
            });
            updateTrainingQueueFromPayload(payload);
        } catch (e) {
            appendLog(`[状态] 切换队列暂停失败: ${e.message}`);
        }
    }

    // ── 状态轮询 ──
    async function pollStatus() {
        if (isHistoryReviewMode()) return;
        try {
            const status = await api('/api/training/status');
            updateStatus({
                state: status.status,
                variant: status.variant,
                preset: status.preset,
                job: status.job,
                last_output_at: status.last_output_at,
                last_log_id: status.last_log_id,
                output_dir: status.output_dir,
                sample_dir: status.sample_dir,
                sample_config: status.sample_config,
                run_dir: status.run_dir,
                runtime_config_file: status.runtime_config_file,
                original_config_file: status.original_config_file,
                dataset_config_file: status.dataset_config_file,
                model_cache_dir: status.model_cache_dir,
                dataset_cache_dir: status.dataset_cache_dir,
                training_output_dir: status.training_output_dir,
                logs_dir: status.logs_dir,
            });
            if ((status.last_log_id || 0) > trainingRuntime.lastLogId) {
                await replayTrainingLogs();
            }
        } catch (e) { /* ignore */ }
    }

    async function loadTrainingHistoryList() {
        if (location.protocol === 'file:') return;
        try {
            const params = new URLSearchParams();
            if (showArchivedHistory) params.set('include_archived', '1');
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/training/history${suffix}`);
            historyTasks = payload.tasks || [];
            renderTrainingHistoryList();
            renderPreviewTaskSelect();
            setPreviewStatus('', '');
        } catch (e) {
            const list = document.getElementById('task-history-list');
            if (list) list.textContent = '读取任务列表失败';
            renderPreviewTaskSelect();
            setPreviewStatus('读取训练任务列表失败: ' + e.message, 'error');
        }
    }

    function renderTrainingHistoryList() {
        const list = document.getElementById('task-history-list');
        if (!list) return;
        list.innerHTML = '';
        const visibleTasks = historyTasks.filter((task) => showArchivedHistory || !historyTaskIsArchived(task));
        if (!visibleTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = historyTasks.length
                ? '没有未归档任务。勾选“显示归档”可查看已归档记录。'
                : '暂无历史任务。下一次训练启动后会自动记录。';
            list.appendChild(empty);
            return;
        }
        const groups = groupHistoryTasks(visibleTasks);
        const selectedTimelineTasks = new Set(currentHistoryTimelineSelection || []);
        for (const group of groups) {
            const section = document.createElement('section');
            section.className = 'task-history-group';
            const groupSelected = group.tasks.some((task) => selectedTimelineTasks.has(task.id));
            if (historyViewMode === 'config_group' && currentHistoryConfigGroup && (configGroupKey(group) === configGroupKey(currentHistoryConfigGroup) || groupSelected)) {
                section.classList.add('active');
            }
            section.appendChild(createHistoryGroupHeading(group));
            for (const task of group.tasks) {
                section.appendChild(createHistoryTaskItem(task));
            }
            list.appendChild(section);
        }
    }

    function groupHistoryTasks(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            if (!map.has(group.key)) {
                map.set(group.key, { ...group, tasks: [] });
            }
            map.get(group.key).tasks.push(task);
        }
        return Array.from(map.values())
            .map(enrichHistoryGroup)
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
            });
    }

    function historyConfigGroupFromTask(task) {
        const methodsSubdir = String(task.methods_subdir || '-');
        const variant = String(task.variant || '-');
        const preset = String(task.preset || 'default');
        const legacyLabel = `${methodsSubdir} / ${variant} / ${preset}`;
        const historyKey = String(task.history_group_key || '').trim();
        const sourceConfig = String(task.history_source_config_file || '').trim();
        const groupLabel = String(task.history_group_label || '').trim() || sourceConfig || legacyLabel;
        return {
            key: historyKey || [methodsSubdir, variant, preset].join('\u0001'),
            history_group_key: historyKey || '',
            history_group_label: groupLabel,
            history_source_config_file: sourceConfig,
            methods_subdir: methodsSubdir,
            variant,
            preset,
            label: groupLabel,
            legacy_label: legacyLabel,
        };
    }

    function configGroupKey(group) {
        if (group?.key) return group.key;
        if (group?.history_group_key) return group.history_group_key;
        return [group.methods_subdir || '-', group.variant || '-', group.preset || 'default'].join('\u0001');
    }

    function enrichHistoryGroup(group) {
        const tasks = [...(group.tasks || [])].sort((a, b) => {
            const aTime = Number(a.started_at || 0);
            const bTime = Number(b.started_at || 0);
            return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
        });
        const latestTask = tasks[0] || {};
        const runDirs = new Set(tasks.map((task) => historyTaskRunPath(task)).filter(Boolean));
        return {
            ...group,
            tasks,
            latestTask,
            run_count: runDirs.size,
            display_label: historyTaskDisplayName(latestTask) || group.label,
            source_label: group.history_source_config_file || '',
            fallback_group_label: group.history_group_label || group.legacy_label || group.label,
        };
    }

    function historyTaskDisplayName(task) {
        if (!task) return '';
        const customName = String(task.name || '').trim();
        if (task.training_mode === 'continue_lora') {
            const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
            const name = String(task.continue_from_weight_name || '').trim();
            const continueName = `继续训练 ${kind}${name ? ` · ${name}` : ''}`;
            return customName && customName !== continueName ? customName : continueName;
        }
        if (customName) return customName;
        const defaultName = String(
            task.history_run_label
            || runLabelFromPath(task.run_dir || task.training_output_dir || task.output_dir)
            || task.id
            || ''
        ).trim();
        return defaultName;
    }

    function historyTaskIsArchived(task) {
        if (Boolean(task?.archived)) return true;
        return task?.job === 'preprocess' && !task?.updated_at;
    }

    function historyTaskRunPath(task) {
        return String(task?.run_dir || task?.training_output_dir || task?.output_dir || '').trim();
    }

    function historyResumeLabel(task) {
        const resume = task?.resume_from || {};
        if (!resume || typeof resume !== 'object') return '';
        const checkpoint = String(resume.checkpoint_name || '').trim();
        const step = resume.checkpoint_step !== undefined && resume.checkpoint_step !== null
            ? String(resume.checkpoint_step).trim()
            : '';
        if (checkpoint && step) return `从检查点恢复: ${checkpoint} · step ${step}`;
        if (checkpoint) return `从检查点恢复: ${checkpoint}`;
        if (step) return `从检查点恢复: step ${step}`;
        return resume.source_task_id ? '从检查点恢复' : '';
    }

    function historyContinueLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
        const name = String(task.continue_from_weight_name || '').trim();
        return `继续训练 ${kind}${name ? `: ${name}` : ''}`;
    }

    function historyContinuePathLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const path = String(task.continue_from_weight_abs_path || '').trim();
        return path ? `基于: ${path}` : '';
    }

    function runLabelFromPath(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        if (parts[parts.length - 1] === 'training_output' && parts.length > 1) {
            return parts[parts.length - 2];
        }
        return parts[parts.length - 1];
    }

    function historyGroupDisplayLabel(group) {
        return String(group?.display_label || group?.history_run_label || group?.label || configGroupLabel(group) || '').trim();
    }

    function createHistoryGroupHeading(group) {
        const heading = document.createElement('div');
        heading.className = 'task-history-group-title';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = group.tasks.filter((task) => task.job === 'preprocess').length;

        const title = document.createElement('span');
        const name = document.createElement('strong');
        name.textContent = historyGroupDisplayLabel(group);
        const meta = document.createElement('em');
        meta.textContent = [
            group.source_label ? `源配置: ${group.source_label}` : `配置分组: ${group.fallback_group_label || group.label}`,
            `${trainingCount} 次训练`,
            preprocessCount ? `${preprocessCount} 个预处理` : '',
            group.run_count ? `${group.run_count} 个运行目录` : '',
        ].filter(Boolean).join(' · ');
        title.append(name, meta);
        heading.appendChild(title);

        if (trainingCount) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'task-history-group-action';
            btn.textContent = '选择合并查看';
            btn.title = '手动选择要合并查阅的训练任务，再合并查看日志和 Loss 曲线。';
            btn.addEventListener('click', () => chooseTimelineTasksForMerge(group));
            heading.appendChild(btn);
        }
        return heading;
    }

    function createHistoryTaskItem(task) {
        const card = document.createElement('article');
        card.className = 'task-history-item';
        if (historyViewMode === 'single' && task.id === viewingHistoryTaskId) card.classList.add('active');
        const archived = historyTaskIsArchived(task);
        if (archived) card.classList.add('archived');

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'task-history-main';
        main.addEventListener('click', () => loadHistoryTask(task.id));

        const title = document.createElement('strong');
        title.textContent = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const meta = document.createElement('span');
        meta.textContent = [
            task.job === 'preprocess' ? '预处理' : '训练',
            historyContinueLabel(task),
            historyResumeLabel(task),
            historyStateLabel(task.state),
            task.started_at_text || task.id,
            archived ? '已归档' : '',
        ].filter(Boolean).join(' · ');
        const paths = document.createElement('em');
        paths.textContent = [
            `目录: ${task.run_dir || task.training_output_dir || task.output_dir || task.history_dir || task.id}`,
            historyContinuePathLabel(task),
        ].filter(Boolean).join(' · ');
        const counts = document.createElement('em');
        counts.textContent = `${task.metric_count || 0} loss点 / ${task.log_count || 0} 日志`;
        main.append(title, meta, paths, counts);

        const actions = document.createElement('div');
        actions.className = 'task-history-actions';
        actions.append(
            createHistoryActionButton('重命名', () => renameHistoryTask(task)),
            createHistoryActionButton(archived ? '取消归档' : '归档', () => archiveHistoryTask(task)),
            createHistoryActionButton('删除', () => deleteHistoryTask(task), 'danger'),
        );

        card.append(main, actions);
        return card;
    }

    function createHistoryActionButton(label, handler, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            handler();
        });
        return btn;
    }

    async function renameHistoryTask(task) {
        const fallback = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const name = await showHistoryTaskInputDialog({
            title: '重命名任务',
            description: '只修改任务列表中的显示名称，不会改动磁盘目录。',
            label: '任务名称',
            value: fallback,
            placeholder: '例如：肋骨女神 5.14 第一次训练',
            confirmText: '保存名称',
        });
        if (name === null) return;
        await updateHistoryTaskMeta(task.id, { name: name.trim() });
    }

    async function regroupHistoryTask(task) {
        const group = await showHistoryTaskInputDialog({
            title: '设置任务分组',
            description: '相同分组名的任务会在左侧任务列表中归到一起。留空表示未分组。',
            label: '分组名称',
            value: task.group || '',
            placeholder: '例如：肋骨女神 / 测试组 / 正式训练',
            confirmText: '保存分组',
        });
        if (group === null) return;
        await updateHistoryTaskMeta(task.id, { group: group.trim() });
    }

    async function archiveHistoryTask(task) {
        const archived = historyTaskIsArchived(task);
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '取消归档任务' : '归档任务',
            description: historyTaskLabel(task),
            message: archived
                ? '取消归档后，这个任务会重新出现在默认任务列表中。'
                : '归档后默认会隐藏这个任务，可勾选“显示归档”再次查看。',
            confirmText: archived ? '取消归档' : '确认归档',
        });
        if (!ok) return;
        await updateHistoryTaskMeta(task.id, { archived: !archived });
    }

    async function deleteHistoryTask(task) {
        const deletesLinkedPreprocess = task?.job === 'training';
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除历史任务',
            description: historyTaskLabel(task),
            message: deletesLinkedPreprocess
                ? '会删除该训练任务的日志、loss 指标、TOML 快照，并一并删除同一运行目录下对应的预处理任务。此操作不可撤销。'
                : '会删除该任务的日志、loss 指标和 TOML 快照。此操作不可撤销。',
            confirmText: '确认删除',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/training/history/${encodeURIComponent(task.id)}`, { method: 'DELETE' });
            if (!res.ok) {
                alert(res.error || '删除失败');
                return;
            }
            if (viewingHistoryTaskId === task.id) {
                clearResumeOptions();
                returnToLiveTraining();
            }
            await loadTrainingHistoryList();
        } catch (e) {
            alert('删除失败: ' + e.message);
        }
    }

    async function updateHistoryTaskMeta(taskId, patch) {
        try {
            const res = await api(`/api/training/history/${encodeURIComponent(taskId)}`, {
                method: 'PATCH',
                body: JSON.stringify(patch),
            });
            if (!res.ok) {
                alert(res.error || '更新任务失败');
                return;
            }
            await loadTrainingHistoryList();
            if (viewingHistoryTaskId === taskId) {
                const payload = await api(`/api/training/history/${encodeURIComponent(taskId)}`);
                if (payload.ok) renderHistoryTask(payload);
            }
        } catch (e) {
            alert('更新任务失败: ' + e.message);
        }
    }

    function historyTaskLabel(task) {
        return historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || task.id}`;
    }

    function showHistoryTaskInputDialog(options) {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = options.value || '';
        input.placeholder = options.placeholder || '';
        input.className = 'history-task-dialog-input';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const span = document.createElement('span');
        span.textContent = options.label || '输入内容';
        label.append(span, input);

        return showHistoryTaskDialog({
            title: options.title,
            description: options.description,
            body: label,
            confirmText: options.confirmText || '确认',
            onOpen: () => {
                input.focus();
                input.select();
            },
            getValue: () => input.value,
        });
    }

    function showHistoryTaskConfirmDialog(options) {
        const wrap = document.createElement('div');
        wrap.className = 'history-task-dialog-message';
        const strong = document.createElement('strong');
        strong.textContent = options.description || '';
        const p = document.createElement('p');
        p.textContent = options.message || '';
        wrap.append(strong, p);
        return showHistoryTaskDialog({
            title: options.title,
            description: '',
            body: wrap,
            confirmText: options.confirmText || '确认',
            cancelText: options.cancelText || '取消',
            cancelPrimary: options.cancelPrimary,
            danger: options.danger,
            getValue: () => true,
        });
    }

    function showHistoryTaskDialog(options) {
        const dialog = document.getElementById('history-task-dialog');
        const title = document.getElementById('history-task-dialog-title');
        const desc = document.getElementById('history-task-dialog-desc');
        const body = document.getElementById('history-task-dialog-body');
        const cancelBtn = document.getElementById('history-task-dialog-cancel');
        const confirmBtn = document.getElementById('history-task-dialog-confirm');
        if (!dialog || !title || !desc || !body || !cancelBtn || !confirmBtn) {
            return Promise.resolve(null);
        }
        if (sharedDialogBusy || dialog.open) {
            return Promise.resolve(null);
        }
        sharedDialogBusy = true;

        title.textContent = options.title || '任务操作';
        desc.textContent = options.description || '';
        body.innerHTML = '';
        if (options.body) body.appendChild(options.body);
        cancelBtn.textContent = options.cancelText || '取消';
        cancelBtn.classList.toggle('btn-primary', Boolean(options.cancelPrimary));
        confirmBtn.textContent = options.confirmText || '确认';
        confirmBtn.disabled = false;
        confirmBtn.classList.toggle('btn-danger', Boolean(options.danger));
        confirmBtn.classList.toggle('btn-primary', !options.danger);
        dialog.returnValue = '';

        return new Promise((resolve) => {
            const cleanup = () => {
                dialog.removeEventListener('close', handleClose);
                sharedDialogBusy = false;
            };
            const handleClose = () => {
                cleanup();
                if (dialog.returnValue === 'confirm') {
                    resolve(options.getValue ? options.getValue() : true);
                } else {
                    resolve(null);
                }
            };
            dialog.addEventListener('close', handleClose);
            try {
                if (dialog.showModal) {
                    dialog.showModal();
                } else {
                    dialog.setAttribute('open', 'open');
                }
            } catch (e) {
                cleanup();
                resolve(null);
                return;
            }
            requestAnimationFrame(() => {
                if (options.onOpen) {
                    options.onOpen();
                } else {
                    confirmBtn.focus();
                }
            });
        });
    }

    async function loadHistoryTask(taskId) {
        try {
            const payload = await api(`/api/training/history/${encodeURIComponent(taskId)}`);
            if (!payload.ok) {
                alert(payload.error || '读取历史任务失败');
                return;
            }
            viewingHistoryTaskId = taskId;
            historyViewMode = 'single';
            currentHistoryConfigGroup = null;
            currentHistoryTimelineSelection = [];
            resumeOptionsState = {
                loading: true,
                taskId,
                checkpoints: [],
                defaultCheckpoint: '',
                error: '',
                message: '正在读取可续训检查点...',
            };
            renderTrainingHistoryList();
            renderHistoryTask(payload);
            await loadResumeOptionsForTask(taskId);
        } catch (e) {
            alert('读取历史任务失败: ' + e.message);
        }
    }

    async function refreshHistoryView() {
        if (historyViewMode === 'config_group' && currentHistoryConfigGroup) {
            await loadConfigGroupTimeline(currentHistoryConfigGroup, {
                taskIds: currentHistoryTimelineSelection,
                skipSelectionDialog: true,
            });
            return;
        }
        if (!viewingHistoryTaskId) return;
        await loadHistoryTask(viewingHistoryTaskId);
    }

    async function chooseTimelineTasksForMerge(group) {
        const selection = await showTimelineTaskSelectionDialog(group);
        if (!selection) return;
        if (selection.group && !selection.taskIds?.length) {
            await loadConfigGroupTimeline(selection.group, { skipSelectionDialog: true });
            return;
        }
        await loadConfigGroupTimeline(group, {
            taskIds: selection.taskIds || [],
            skipSelectionDialog: true,
        });
    }

    async function loadConfigGroupTimeline(group, options = {}) {
        if (!group?.history_group_key && (!group?.methods_subdir || !group?.variant)) return;
        let taskIds = Array.isArray(options.taskIds) ? options.taskIds.filter(Boolean) : [];
        if (!taskIds.length && !options.skipSelectionDialog) {
            const selection = await showTimelineTaskSelectionDialog(group);
            if (!selection) return;
            if (selection.group && !selection.taskIds?.length) {
                group = selection.group;
            } else {
                taskIds = selection.taskIds || [];
            }
        }
        const query = new URLSearchParams({
            methods_subdir: group.methods_subdir || '',
            variant: group.variant || '',
            preset: group.preset || 'default',
            include_archived: showArchivedHistory ? '1' : '0',
        });
        if (!taskIds.length && group.history_group_key) {
            query.set('group_key', group.history_group_key);
        }
        for (const taskId of taskIds) {
            query.append('task_id', taskId);
        }
        try {
            const payload = await api(`/api/training/history/config-group/timeline?${query.toString()}`);
            if (!payload.ok) {
                alert(payload.error || '读取配置分组合并日志失败');
                return;
            }
            historyViewMode = 'config_group';
            viewingHistoryTaskId = '';
            currentHistoryConfigGroup = payload.group || group;
            currentHistoryTimelineSelection = (payload.summary?.selected_task_ids || taskIds || []).filter(Boolean);
            currentHistoryTaskForResume = null;
            clearResumeOptions();
            renderTrainingHistoryList();
            renderConfigGroupTimeline(payload);
        } catch (e) {
            alert('读取配置分组合并日志失败: ' + e.message);
        }
    }

    function showTimelineTaskSelectionDialog(group) {
        const visibleTasks = historyTasks.filter((task) => showArchivedHistory || !historyTaskIsArchived(task));
        const candidates = groupHistoryTasks(visibleTasks)
            .map((item) => ({
                ...item,
                trainingTasks: item.tasks.filter((task) => task.job === 'training'),
            }))
            .filter((item) => item.trainingTasks.length);

        if (!candidates.length) {
            alert('没有可合并的训练分组');
            return Promise.resolve(null);
        }

        const body = document.createElement('div');
        body.className = 'history-merge-dialog-body';

        const toolbar = document.createElement('div');
        toolbar.className = 'history-merge-dialog-toolbar';
        const hint = document.createElement('span');
        hint.textContent = `已列出 ${candidates.length} 个训练分组`;
        const selectAll = document.createElement('button');
        selectAll.type = 'button';
        selectAll.className = 'btn btn-small';
        selectAll.textContent = '全选';
        const clearAll = document.createElement('button');
        clearAll.type = 'button';
        clearAll.className = 'btn btn-small';
        clearAll.textContent = '清空';
        toolbar.append(hint, selectAll, clearAll);
        body.appendChild(toolbar);

        const list = document.createElement('div');
        list.className = 'history-merge-task-list';
        const selectedTaskSet = new Set(currentHistoryTimelineSelection || []);
        const selectedGroupSet = new Set(
            currentHistoryTimelineSelection.length
                ? candidates
                    .filter((item) => item.trainingTasks.some((task) => selectedTaskSet.has(task.id)))
                    .map((item) => item.key)
                : [configGroupKey(group)]
        );
        for (const item of candidates) {
            const label = document.createElement('label');
            label.className = 'history-merge-task-option';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = item.key || '';
            checkbox.checked = selectedGroupSet.has(item.key);
            const content = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = historyGroupDisplayLabel(item);
            const meta = document.createElement('em');
            const lossCount = item.trainingTasks.reduce((sum, task) => sum + Number(task.metric_count || 0), 0);
            const logCount = item.trainingTasks.reduce((sum, task) => sum + Number(task.log_count || 0), 0);
            meta.textContent = [
                item.source_label ? `源配置: ${item.source_label}` : `配置分组: ${item.fallback_group_label || item.label}`,
                `${item.trainingTasks.length} 次训练`,
                `${lossCount} loss点`,
                `${logCount} 日志`,
                item.trainingTasks.some((task) => historyTaskIsArchived(task)) ? '含归档' : '',
            ].filter(Boolean).join(' · ');
            content.append(title, meta);
            label.append(checkbox, content);
            list.appendChild(label);
        }
        body.appendChild(list);

        const checkedValues = () => Array.from(list.querySelectorAll('input[type="checkbox"]:checked'))
            .map((input) => input.value)
            .filter(Boolean);
        const syncConfirm = () => {
            const confirmBtn = document.getElementById('history-task-dialog-confirm');
            if (confirmBtn) confirmBtn.disabled = checkedValues().length === 0;
        };
        list.addEventListener('change', syncConfirm);
        selectAll.addEventListener('click', () => {
            list.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = true; });
            syncConfirm();
        });
        clearAll.addEventListener('click', () => {
            list.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = false; });
            syncConfirm();
        });

        return showHistoryTaskDialog({
            title: '选择要合并查看的训练分组',
            description: '勾选一个或多个分组后合并查看 Loss 和日志',
            body,
            confirmText: '合并查看',
            onOpen: syncConfirm,
            getValue: () => {
                const selectedGroupKeys = new Set(checkedValues());
                const selectedGroups = candidates.filter((item) => selectedGroupKeys.has(item.key));
                if (!selectedGroups.length) return null;
                if (selectedGroups.length === 1) {
                    return { group: selectedGroups[0], taskIds: [] };
                }
                const taskIds = selectedGroups.flatMap((item) => item.trainingTasks.map((task) => task.id).filter(Boolean));
                return taskIds.length ? { group: null, taskIds } : null;
            },
        });
    }

    function historyTaskStepOffset(task) {
        const resume = task?.resume_from || {};
        const step = Number(resume.checkpoint_step || 0);
        return Number.isFinite(step) && step > 0 ? step : 0;
    }

    function historyLossChartPoints(lossPoints, task) {
        const offset = historyTaskStepOffset(task);
        const out = [];
        let maxStep = null;
        for (const item of lossPoints || []) {
            const rawStep = Number(item.step);
            if (!Number.isFinite(rawStep)) continue;
            const step = rawStep + offset;
            if (maxStep !== null && step < maxStep) continue;
            if (maxStep === null || step > maxStep) maxStep = step;
            out.push({
                step,
                loss: item.loss,
                rawStep,
                displayStepOffset: offset,
            });
        }
        return out;
    }

    function renderHistoryTask(payload) {
        const task = payload.task || {};
        currentHistoryTaskForResume = task;
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `历史任务: ${historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`} · ${historyStateLabel(task.state)}`;
        }
        document.getElementById('train-variant').textContent = task.variant || '-';
        document.getElementById('train-preset').textContent = task.preset || '-';
        document.getElementById('progress-bar').style.width = task.state === 'idle' ? '100%' : '0%';
        document.getElementById('progress-text').textContent = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
        document.getElementById('metric-vram').textContent = '-';
        document.getElementById('metric-gpu').textContent = '-';
        document.getElementById('metric-log-age').textContent = task.finished_at_text ? '已结束' : '历史';
        document.getElementById('metric-rate').textContent = '-';

        const logs = payload.logs || [];
        const metrics = metricsWithProgressFallback(payload.metrics || [], logs);
        const lossPoints = metrics.filter((item) => item.loss !== undefined);
        const chartPoints = historyLossChartPoints(lossPoints, task);
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('step', {
            xRange: {
                min: chartPoints[0]?.step,
                max: chartPoints[chartPoints.length - 1]?.step,
            },
        });
        lossChart?.setData(chartPoints, { keepAll: true });
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        const lastChartPoint = chartPoints[chartPoints.length - 1] || {};
        const configLr = readConfigNumber(payload.config_toml, 'learning_rate');
        const system = payload.system || [];
        const lastSystem = system[system.length - 1] || {};
        document.getElementById('metric-loss').textContent = lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : '-';
        document.getElementById('metric-lr').textContent = formatLr(lastValue(metrics, 'lr') ?? configLr);
        document.getElementById('metric-step').textContent = lastChartPoint.step ?? lastValue(metrics, 'step') ?? lastLossMetric.step ?? '-';
        document.getElementById('metric-rate').textContent = lastValue(metrics, 'rate') || '-';
        document.getElementById('metric-vram').textContent =
            lastSystem.vram_used_gb !== undefined ? `${lastSystem.vram_used_gb}/${lastSystem.vram_total_gb} GB` : '-';
        if (lastSystem.gpu_util !== undefined) {
            document.getElementById('metric-gpu').textContent =
                `${lastSystem.gpu_util}%${lastSystem.gpu_temp ? ` ${lastSystem.gpu_temp}°C` : ''}`;
        } else {
            document.getElementById('metric-gpu').textContent = '-';
        }

        const logEl = document.getElementById('log-output');
        logEl.textContent = logs
            .map((record) => `${record.kind === 'progress' ? '[进度] ' : ''}${record.line || ''}`)
            .join('\n');
        if (logEl.textContent) logEl.textContent += '\n';
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`历史 · ${(payload.logs || []).length} 行`, 'warning');

        const health = document.getElementById('training-health');
        health.className = 'training-health';
        health.textContent = [
            task.message || '历史任务记录',
            task.history_dir ? `历史目录: ${task.history_dir}` : '',
            task.output_dir ? `输出目录: ${task.output_dir}` : '',
            task.sample_dir ? `样张目录: ${task.sample_dir}` : '',
        ].filter(Boolean).join(' · ');

        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        if (configPanel) configPanel.hidden = false;
        if (configTitle) configTitle.textContent = '任务配置快照';
        if (configOutput) configOutput.textContent = payload.config_toml || '# 无配置快照';
        renderHistoryPaths(task);
        renderResumePanelState();
    }

    function renderConfigGroupTimeline(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `合并查看: ${configGroupLabel(group)} · ${summary.task_count || 0} 次训练`;
        }

        document.getElementById('train-variant').textContent = group.variant || '-';
        document.getElementById('train-preset').textContent = group.preset || '-';
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').textContent =
            `${summary.started_at_text || '-'} → ${summary.finished_at_text || '持续/未结束'}`;
        document.getElementById('metric-vram').textContent = '-';
        document.getElementById('metric-gpu').textContent = '-';
        document.getElementById('metric-log-age').textContent = '分组合并';
        document.getElementById('metric-rate').textContent = '-';

        const metrics = payload.metrics || [];
        const lossPoints = metrics.filter((item) => item.loss !== undefined);
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('step', {
            xRange: {
                min: summary.start_display_step ?? lossPoints[0]?.display_step ?? lossPoints[0]?.step,
                max: summary.end_display_step ?? lossPoints[lossPoints.length - 1]?.display_step ?? lossPoints[lossPoints.length - 1]?.step,
            },
        });
        lossChart?.setData(lossPoints.map((item) => ({
            step: item.display_step || item.step || 0,
            loss: item.loss,
            rawStep: item.step,
            displayStepOffset: item.display_step_offset || 0,
            sourceTaskLabel: item.source_task_label || '',
            sourceTaskIndex: item.source_task_index || 0,
            stageBreakBefore: Boolean(item.stage_break_before),
            stageLabel: item.stage_break_before ? `任务${item.source_task_index || ''}` : '',
        })), { keepAll: true });
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        document.getElementById('metric-loss').textContent =
            lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : '-';
        document.getElementById('metric-lr').textContent = formatLr(lastValue(metrics, 'lr'));
        document.getElementById('metric-step').textContent =
            lastMetric.display_step ?? lastMetric.step ?? lastLossMetric.display_step ?? lastLossMetric.step ?? '-';

        const logs = payload.logs || [];
        const logEl = document.getElementById('log-output');
        logEl.textContent = logs.map(formatGroupTimelineLogRecord).join('\n');
        if (logEl.textContent) logEl.textContent += '\n';
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`手动合并 · ${logs.length} 行日志 · ${summary.loss_count || 0} Loss 点 · 已隐藏 ${summary.progress_count || 0} 条进度记录`, 'warning');

        const health = document.getElementById('training-health');
        health.className = 'training-health ok';
        health.textContent = [
            `已手动合并 ${summary.task_count || 0} 次训练`,
            `${summary.loss_count || 0} 个 Loss 点`,
            `${summary.log_count || 0} 行日志`,
            `${summary.progress_count || 0} 条进度记录未显示`,
            summary.include_archived ? '包含归档任务' : '',
        ].filter(Boolean).join(' · ');

        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        if (configPanel) configPanel.hidden = false;
        if (configTitle) configTitle.textContent = '分组训练明细';
        if (configOutput) configOutput.textContent = configGroupTimelineSummary(payload);
        renderConfigGroupPaths(payload);
        renderResumePanelState();
    }

    function formatGroupTimelineLogRecord(record) {
        const taskPrefix = record.source_task_index ? `[任务${record.source_task_index}] ` : '';
        const kindPrefix = record.kind === 'progress' ? '[进度] ' : '';
        return `${taskPrefix}${kindPrefix}${record.line || ''}`;
    }

    function configGroupTimelineSummary(payload) {
        const group = payload.group || {};
        const lines = [`# 手动合并查看: ${configGroupLabel(group)}`, ''];
        for (const segment of payload.segments || []) {
            const task = segment.task || {};
            const segmentLines = [
                `任务 ${segment.index}: ${task.label || task.id || '-'}`,
                `  ID: ${task.id || '-'}`,
                `  状态: ${historyStateLabel(task.state)}`,
                `  时间: ${task.started_at_text || '-'} -> ${task.finished_at_text || '未结束'}`,
                `  输出目录: ${task.output_dir || '-'}`,
                `  真实 Step: ${formatStepRange(segment.start_display_step, segment.end_display_step)}`,
                segment.display_step_offset ? `  续训偏移: +${segment.display_step_offset}` : '',
                `  日志: ${segment.log_count || 0} 行`,
                `  进度记录: ${segment.progress_count || 0} 条`,
                `  Loss/指标: ${segment.metric_count || 0} 条`,
                '',
            ].filter(Boolean);
            lines.push(...segmentLines);
        }
        return lines.join('\n');
    }

    function formatStepRange(start, end) {
        if (start === undefined || start === null || end === undefined || end === null) return '-';
        return `${start} -> ${end}`;
    }

    function renderConfigGroupPaths(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = [
            ['配置文件', configGroupLabel(group)],
            ['源配置', group.history_source_config_file || '-'],
            ['合并训练数', `${summary.task_count || 0}`],
            ['时间范围', `${summary.started_at_text || '-'} -> ${summary.finished_at_text || '未结束'}`],
            ['真实步数', formatStepRange(summary.start_display_step, summary.end_display_step)],
            ['归档任务', summary.include_archived ? '已包含' : '未包含'],
        ];
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    function configGroupLabel(group) {
        if (group.methods_subdir === '手动选择') {
            return group.variant || '手动选择';
        }
        return group.history_run_label || group.history_group_label || group.label || `${group.methods_subdir || '-'} / ${group.variant || '-'} / ${group.preset || 'default'}`;
    }

    function metricsWithProgressFallback(metrics, logs) {
        const out = [...metrics];
        const seen = new Set(out.map(metricIdentity));
        for (const record of logs || []) {
            if (record.kind !== 'progress') continue;
            const parsed = parseMetricsFromProgressLine(record.line);
            if (!parsed) continue;
            const item = { ...parsed, ts: record.ts };
            const key = metricIdentity(item);
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(item);
        }
        out.sort((a, b) => (Number(a.ts || 0) - Number(b.ts || 0)) || (Number(a.step || 0) - Number(b.step || 0)));
        return out;
    }

    function metricIdentity(item) {
        return [
            item.step ?? '',
            item.loss != null ? Number(item.loss).toFixed(8) : '',
            item.lr != null ? Number(item.lr).toFixed(12) : '',
        ].join('|');
    }

    function returnToLiveTraining(options = {}) {
        const refresh = options.refresh !== false;
        viewingHistoryTaskId = '';
        historyViewMode = 'live';
        currentHistoryConfigGroup = null;
        currentHistoryTimelineSelection = [];
        currentHistoryTaskForResume = null;
        clearResumeOptions();
        const banner = document.getElementById('history-view-banner');
        if (banner) banner.hidden = true;
        const resumePanel = document.getElementById('history-resume-panel');
        if (resumePanel) resumePanel.hidden = true;
        const configPanel = document.getElementById('history-config-panel');
        if (configPanel) configPanel.hidden = true;
        const configOutput = document.getElementById('history-config-output');
        if (configOutput) configOutput.textContent = '';
        const paths = document.getElementById('history-paths');
        if (paths) paths.innerHTML = '';
        document.getElementById('log-output').textContent = '';
        trainingRuntime.lastLogId = 0;
        trainingRuntime.logLineCount = 0;
        stepCounter = 0;
        lossChart?.clear();
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('index');
        renderTrainingHistoryList();
        if (refresh) {
            pollStatus();
            replayTrainingLogs();
        }
    }

    async function loadResumeOptionsForTask(taskId = viewingHistoryTaskId) {
        if (!taskId) {
            clearResumeOptions();
            return;
        }
        resumeOptionsState = {
            loading: true,
            taskId,
            checkpoints: [],
            defaultCheckpoint: '',
            error: '',
            message: '正在读取可续训检查点...',
        };
        renderResumePanelState();
        try {
            const payload = await api(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`);
            if (taskId !== viewingHistoryTaskId) return;
            if (!payload.ok) {
                resumeOptionsState = {
                    loading: false,
                    taskId,
                    checkpoints: [],
                    defaultCheckpoint: '',
                    error: payload.error || '读取续训检查点失败',
                    message: '',
                };
                renderResumePanelState();
                return;
            }
            resumeOptionsState = {
                loading: false,
                taskId,
                checkpoints: payload.checkpoints || [],
                defaultCheckpoint: payload.default_checkpoint || '',
                error: '',
                message: payload.message || '',
            };
            renderResumePanelState();
        } catch (e) {
            if (taskId !== viewingHistoryTaskId) return;
            resumeOptionsState = {
                loading: false,
                taskId,
                checkpoints: [],
                defaultCheckpoint: '',
                error: '读取续训检查点失败: ' + e.message,
                message: '',
            };
            renderResumePanelState();
        }
    }

    function clearResumeOptions() {
        resumeOptionsState = {
            loading: false,
            taskId: '',
            checkpoints: [],
            defaultCheckpoint: '',
            error: '',
            message: '',
        };
        currentHistoryTaskForResume = null;
        renderResumePanelState();
    }

    function renderResumePanelState() {
        const panel = document.getElementById('history-resume-panel');
        const select = document.getElementById('resume-checkpoint-select');
        const btn = document.getElementById('btn-resume-training');
        const queueBtn = document.getElementById('btn-queue-resume-training');
        const summary = document.getElementById('resume-checkpoint-summary');
        const status = document.getElementById('resume-training-status');
        if (!panel || !select || !btn || !summary || !status) return;

        const isTrainingTask = historyViewMode === 'single' && viewingHistoryTaskId && currentHistoryTaskForResume?.job === 'training';
        panel.hidden = !isTrainingTask;
        if (!isTrainingTask) {
            select.innerHTML = '<option value="">选择历史训练任务后读取</option>';
            select.disabled = true;
            btn.disabled = true;
            if (queueBtn) queueBtn.disabled = true;
            summary.textContent = '';
            status.textContent = '';
            status.className = 'resume-status';
            return;
        }

        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        select.innerHTML = '';
        if (resumeOptionsState.loading) {
            select.appendChild(optionNode('', '正在读取检查点...'));
        } else if (resumeOptionsState.checkpoints.length) {
            for (const item of resumeOptionsState.checkpoints) {
                select.appendChild(optionNode(item.path, resumeCheckpointOptionLabel(item)));
            }
            select.value = resumeOptionsState.defaultCheckpoint || resumeOptionsState.checkpoints[0]?.path || '';
        } else {
            select.appendChild(optionNode('', '未找到可续训状态目录'));
        }

        const hasCheckpoint = Boolean(select.value);
        select.disabled = resumeOptionsState.loading || !hasCheckpoint || isRunning;
        btn.disabled = resumeOptionsState.loading || !hasCheckpoint || isRunning;
        if (queueBtn) queueBtn.disabled = resumeOptionsState.loading || !hasCheckpoint;
        summary.innerHTML = '';
        const selected = selectedResumeCheckpoint();
        if (selected) {
            summary.append(
                resumeSummaryLine('状态目录', selected.path),
                resumeSummaryLine('已训练到', resumeCheckpointProgressText(selected)),
                resumeSummaryLine('保存时间', selected.mtime_text || '-'),
                resumeSummaryLine('关联权重', selected.paired_weight || '无或未找到'),
            );
        } else {
            const note = document.createElement('p');
            note.textContent = resumeOptionsState.message || resumeOptionsState.error || '该任务还没有可续训状态。需要训练配置启用 checkpointing_epochs，训练中才会写出状态目录。';
            summary.appendChild(note);
        }

        status.textContent = isRunning
            ? '当前已有训练或预处理在运行，续训按钮暂不可用。'
            : (resumeOptionsState.error || resumeOptionsState.message || '');
        status.className = [
            'resume-status',
            isRunning ? 'warning' : (resumeOptionsState.error ? 'error' : ''),
        ].filter(Boolean).join(' ');
    }

    function optionNode(value, text) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        return option;
    }

    function selectedResumeCheckpoint() {
        const select = document.getElementById('resume-checkpoint-select');
        const value = select?.value || '';
        if (!value) return null;
        return resumeOptionsState.checkpoints.find((item) => item.path === value) || null;
    }

    function resumeCheckpointOptionLabel(item) {
        return [
            item.kind_label || '训练状态',
            resumeCheckpointProgressText(item),
            item.scope_label || '',
            item.name || '',
        ].filter(Boolean).join(' · ');
    }

    function resumeCheckpointProgressText(item) {
        const parts = [];
        if (item.epoch != null) parts.push(`Epoch ${item.epoch}`);
        if (item.step != null) parts.push(`Step ${item.step}`);
        return parts.join(' / ') || '步数未知';
    }

    function resumeSummaryLine(label, value) {
        const row = document.createElement('div');
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('code');
        valEl.textContent = value || '-';
        row.append(key, valEl);
        return row;
    }

    async function resumeTrainingFromCheckpoint() {
        if (!viewingHistoryTaskId) return;
        const selected = selectedResumeCheckpoint();
        if (!selected) {
            setResumeStatus('请先选择一个可续训状态目录。', 'error');
            return;
        }
        const taskName = historyTaskLabel(currentHistoryTaskForResume || {});
        const ok = await showHistoryTaskConfirmDialog({
            title: '从检查点继续训练',
            description: taskName,
            message: `将使用这个历史任务的配置快照，并从 ${selected.name} 继续训练。训练会恢复优化器、学习率调度器和已完成步数；启动后会生成一个新的训练任务记录。`,
            confirmText: '确认开始续训',
        });
        if (!ok) return;

        setResumeStatus('正在启动续训...', '');
        try {
            const res = await api('/api/training/resume', {
                method: 'POST',
                body: JSON.stringify({
                    task_id: viewingHistoryTaskId,
                    checkpoint: selected.path,
                    gpu_whitelist: selectedGpuPayload(),
                }),
            });
            if (!res.ok) {
                setResumeStatus(res.error || '续训启动失败', 'error');
                return;
            }
            const message = res.message || '续训已启动';
            setResumeStatus(message, 'ok');
            await loadTrainingHistoryList();
            returnToLiveTraining();
            appendLog(`[状态] ${message}: ${selected.name}`);
        } catch (e) {
            setResumeStatus('续训启动失败: ' + e.message, 'error');
        }
    }

    async function queueResumeTrainingFromCheckpoint() {
        if (!viewingHistoryTaskId) return;
        const selected = selectedResumeCheckpoint();
        if (!selected) {
            setResumeStatus('请先选择一个可续训状态目录。', 'error');
            return;
        }
        const taskName = historyTaskLabel(currentHistoryTaskForResume || {});
        const ok = await showHistoryTaskConfirmDialog({
            title: '续训加入队列',
            description: taskName,
            message: `将使用这个历史任务的配置快照，并从 ${selected.name} 续训。任务会排队等待当前训练结束后自动启动。`,
            confirmText: '加入队列',
        });
        if (!ok) return;

        setResumeStatus('正在加入队列...', '');
        try {
            const res = await api('/api/training/queue/resume', {
                method: 'POST',
                body: JSON.stringify({
                    task_id: viewingHistoryTaskId,
                    checkpoint: selected.path,
                    gpu_whitelist: selectedGpuPayload(),
                }),
            });
            if (!res.ok) {
                setResumeStatus(res.error || '续训加入队列失败', 'error');
                return;
            }
            updateTrainingQueueFromPayload(res);
            setResumeStatus(res.message || '续训任务已加入队列', 'ok');
            document.querySelector('[data-tab="training"]')?.click();
        } catch (e) {
            setResumeStatus('续训加入队列失败: ' + e.message, 'error');
        }
    }

    function setResumeStatus(text, state = '') {
        const el = document.getElementById('resume-training-status');
        if (!el) return;
        el.textContent = text || '';
        el.className = ['resume-status', state].filter(Boolean).join(' ');
    }

    function renderHistoryPaths(task, options = {}) {
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = runtimePathItems(task, options);
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    function runtimePathItems(task, options = {}) {
        const includeHistory = options.includeHistory !== false;
        return [
            includeHistory ? ['历史目录', task.history_dir_abs || task.history_dir] : null,
            task.training_mode === 'continue_lora' ? ['基于权重', task.continue_from_weight_abs_path] : null,
            ['本次运行目录', task.run_dir],
            ['实际运行配置', task.runtime_config_file],
            ['原始配置副本', task.original_config_file],
            ['运行时数据集配置', task.dataset_config_file],
            ['模型缓存目录', task.model_cache_dir],
            ['数据集缓存目录', task.dataset_cache_dir],
            ['训练结果目录', task.training_output_dir || task.output_dir],
            ['样张目录', task.sample_dir],
            ['日志目录', task.logs_dir],
            includeHistory ? ['历史日志文件', task.logs_path] : null,
            includeHistory ? ['历史指标文件', task.metrics_path] : null,
            includeHistory ? ['系统指标文件', task.system_path] : null,
            includeHistory ? ['历史 TOML 快照', task.config_snapshot] : null,
        ].filter((item) => item && item[1]);
    }

    function historyStateLabel(state) {
        return {
            running: '运行中',
            idle: '完成',
            error: '异常',
            interrupted: '已中断',
        }[state] || state || '未知';
    }

    // ── 事件绑定 ──
    function setupEventListeners() {
        installBeginnerTooltips();
        document.getElementById('method-select').addEventListener('change', async () => {
            if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换方法会重新加载表单并丢弃这些修改。是否继续？'))) {
                return;
            }
            updateChoiceGuide();
            await loadVariants({ reset: true });
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('variant-select').addEventListener('change', async () => {
            if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换变体会重新加载表单并丢弃这些修改。是否继续？'))) {
                return;
            }
            setCurrentTrainingSourceFromVariant(val('variant-select'));
            updateChoiceGuide();
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('preset-select').addEventListener('change', async () => {
            if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换预设会重新加载表单并丢弃这些修改。是否继续？'))) {
                return;
            }
            updateChoiceGuide();
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('btn-load-config').addEventListener('click', reloadCurrentConfig);
        document.getElementById('btn-start-from-config').addEventListener('click', startTraining);
        document.getElementById('btn-open-continue-lora-dialog').addEventListener('click', openContinueLoraDialog);
        document.getElementById('btn-clear-continue-lora-source').addEventListener('click', clearContinueTrainingSource);
        document.getElementById('btn-inspect-continue-lora-path').addEventListener('click', () => {
            selectContinueLoraWeight(document.getElementById('continue-lora-path-input')?.value || '');
        });
        document.getElementById('continue-lora-history-task').addEventListener('change', (event) => {
            continueLoraDialogState.taskId = event.target.value || '';
            loadContinueLoraWeights();
        });
        document.getElementById('btn-refresh-continue-lora-weights').addEventListener('click', loadContinueLoraWeights);
        document.getElementById('btn-open-tutorial').addEventListener('click', openTutorialDialog);
        document.getElementById('btn-stop-training').addEventListener('click', stopTraining);
        document.getElementById('btn-refresh-queue').addEventListener('click', loadTrainingQueue);
        document.getElementById('btn-toggle-queue-pause').addEventListener('click', toggleTrainingQueuePause);
        document.getElementById('btn-apply-toml').addEventListener('click', applyTomlToConfig);
        document.getElementById('btn-move-toml-group').addEventListener('click', moveCurrentTomlToGroup);
        document.getElementById('btn-create-blank-preset').addEventListener('click', createBlankPresetFromLoraTemplate);
        document.getElementById('btn-save-toml').addEventListener('click', saveTomlFile);
        document.getElementById('btn-toggle-toml-editor').addEventListener('click', toggleTomlEditorPanel);
        document.getElementById('btn-copy-toml').addEventListener('click', copyTomlEditorContent);
        document.getElementById('btn-save-toml-direct').addEventListener('click', () => saveTomlFile({ mode: 'editor' }));
        document.getElementById('btn-import-toml').addEventListener('click', importTomlFile);
        document.getElementById('btn-export-toml').addEventListener('click', exportTomlFile);
        document.getElementById('btn-save-as-toml').addEventListener('click', saveTomlAs);
        document.getElementById('btn-lock-toml').addEventListener('click', toggleTomlUserLock);
        document.getElementById('btn-delete-toml').addEventListener('click', deleteTomlFile);
        document.getElementById('btn-restore-system-toml').addEventListener('click', restoreSystemTomlPresets);
        document.getElementById('toml-import-input').addEventListener('change', handleTomlImport);
        document.getElementById('btn-toml-mode-project').addEventListener('click', () => switchTomlManagerMode('project'));
        document.getElementById('btn-toml-mode-output').addEventListener('click', () => switchTomlManagerMode('output'));
        document.getElementById('btn-refresh-output-runs').addEventListener('click', () => loadOutputRuns({ keepSelection: true }));
        document.getElementById('btn-copy-output-config').addEventListener('click', copyOutputRunConfigContent);
        document.getElementById('btn-export-output-config').addEventListener('click', exportOutputRunConfig);
        document.getElementById('btn-save-output-config-as').addEventListener('click', openOutputRunSaveAs);
        document.getElementById('btn-confirm-output-config-save-as').addEventListener('click', confirmOutputRunSaveAs);
        document.getElementById('btn-cancel-output-config-save-as').addEventListener('click', closeOutputRunSaveAs);
        document.getElementById('output-run-search').addEventListener('input', (event) => {
            outputRunState = { ...outputRunState, search: event.target.value || '' };
            renderOutputRunList();
        });
        document.getElementById('btn-new-dataset-preset').addEventListener('click', createNewDatasetPreset);
        document.getElementById('btn-copy-dataset-preset').addEventListener('click', copyDatasetPreset);
        document.getElementById('btn-rename-dataset-preset').addEventListener('click', renameDatasetPreset);
        document.getElementById('btn-import-dataset-preset').addEventListener('click', importDatasetPreset);
        document.getElementById('dataset-import-input').addEventListener('change', handleDatasetPresetImport);
        document.getElementById('btn-export-dataset-preset').addEventListener('click', exportDatasetPreset);
        document.getElementById('btn-delete-dataset-preset').addEventListener('click', deleteDatasetPreset);
        document.getElementById('btn-save-dataset-preset').addEventListener('click', saveDatasetPresetEditor);
        document.getElementById('btn-refresh-dataset-preview').addEventListener('click', loadDatasetPreviewImages);
        document.getElementById('btn-config-dataset-dialog-refresh').addEventListener('click', () => loadDatasetPresets({ selectCurrent: false }));
        document.getElementById('btn-config-dataset-dialog-manage').addEventListener('click', () => {
            closeConfigDatasetPickerDialog();
            document.querySelector('[data-tab="datasets"]')?.click();
        });
        document.getElementById('btn-reload-toml').addEventListener('click', async () => {
            const file = currentTomlFile || val('toml-file-select');
            if (file && (await confirmDiscardTomlChanges('当前 TOML 有未保存修改，重新读取文件会丢失这些修改。是否继续？'))) {
                loadTomlFile(file, { force: true });
            }
        });
        document.getElementById('toml-file-select').addEventListener('change', (e) => {
            selectAndApplyTomlFile(e.target.value);
        });
        document.getElementById('toml-editor').addEventListener('input', updateTomlDirtyState);
        document.getElementById('btn-clear-log').addEventListener('click', () => {
            if (isHistoryReviewMode()) return;
            document.getElementById('log-output').textContent = '';
            trainingRuntime.logLineCount = 0;
            updateLogStatusText();
        });
        document.getElementById('btn-refresh-history').addEventListener('click', loadTrainingHistoryList);
        document.getElementById('btn-live-training').addEventListener('click', returnToLiveTraining);
        document.getElementById('btn-refresh-history-view').addEventListener('click', refreshHistoryView);
        document.getElementById('btn-close-history').addEventListener('click', returnToLiveTraining);
        document.getElementById('btn-refresh-resume-options').addEventListener('click', () => loadResumeOptionsForTask());
        document.getElementById('btn-resume-training').addEventListener('click', resumeTrainingFromCheckpoint);
        document.getElementById('btn-queue-resume-training').addEventListener('click', queueResumeTrainingFromCheckpoint);
        document.getElementById('resume-checkpoint-select').addEventListener('change', renderResumePanelState);
        document.getElementById('history-show-archived').addEventListener('change', (e) => {
            showArchivedHistory = e.target.checked;
            loadTrainingHistoryList();
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.addEventListener('click', () => setPreviewSource(btn.dataset.previewSource));
        });
        document.getElementById('btn-refresh-preview').addEventListener('click', loadPreviewImages);
        document.getElementById('btn-refresh-weights').addEventListener('click', loadPreviewWeights);
        document.getElementById('btn-sort-weights').addEventListener('click', togglePreviewWeightSort);
        document.getElementById('btn-save-preview-settings').addEventListener('click', savePreviewSettings);
        document.getElementById('btn-reset-preview-settings').addEventListener('click', resetPreviewSettings);
        document.getElementById('btn-save-global-settings').addEventListener('click', saveGlobalSettings);
        document.getElementById('btn-reset-global-settings').addEventListener('click', resetGlobalSettings);
        document.querySelectorAll('.global-setting-help-toggle').forEach((btn) => {
            btn.addEventListener('click', () => toggleGlobalSettingHelp(btn));
        });
        document.getElementById('preview-training-task').addEventListener('change', (e) => changePreviewTask(e.target.value));

        setTomlManagerMode('project');
    }

    function installBeginnerTooltips() {
        const tips = {
            'method-select': '选择训练方法家族。新手通常选择 lora；LoKr、Hydra、ReFT 等属于进阶或实验方法。',
            'variant-select': '选择具体训练配置文件。它决定默认学习率、rank、缓存、方法开关等实际训练参数。',
            'preset-select': '选择预设覆盖项。default 最稳；低显存或快速试跑时再选择其他预设。',
            'gpu-picker-toggle': '选择训练时允许使用的 GPU 白名单。默认“全部 GPU”表示不限制；选择会保存在本机浏览器。',
            'btn-load-config': '重新读取当前方法、变体和预设合并后的配置；不会启动训练，也不会保存当前未保存修改。',
            'btn-start-from-config': '先做训练前预检测，再用当前左侧表单/当前训练配置启动训练。若数据缺缓存，会提示先预处理。',
            'btn-open-continue-lora-dialog': '选择已有 LoRA 或 LoKr safetensors 权重作为新训练任务的初始化来源。',
            'btn-clear-continue-lora-source': '清除继续训练来源，下一次启动会按从零开始训练。',
            'btn-inspect-continue-lora-path': '检查这个 safetensors 是否为 LoRA/LoKr，并确认是否兼容当前变体。',
            'continue-lora-history-task': '从历史训练任务中选择一个输出目录，读取其中保存的权重文件。',
            'btn-refresh-continue-lora-weights': '重新扫描所选历史训练任务的 safetensors 权重。',
            'btn-open-tutorial': '打开基础教程，按顺序了解全局设置、数据集、配置保存、预处理和训练启动。',
            'btn-stop-training': '停止当前正在运行的训练或预处理任务；已经写出的日志、样张和权重文件会保留。',
            'btn-refresh-queue': '重新读取训练队列状态，包括等待、运行、异常和已取消任务。',
            'btn-toggle-queue-pause': '暂停或继续队列自动启动。暂停不会停止当前正在运行的任务。',
            'btn-refresh-resume-options': '重新扫描这个历史任务输出目录里的训练状态目录，例如 output_name-checkpoint-state。',
            'btn-resume-training': '从选中的训练状态目录恢复训练。它不是加载普通权重热启动，而是恢复 optimizer、scheduler、随机状态和步数。',
            'btn-queue-resume-training': '把选中的续训状态加入队列，等待当前任务结束后自动启动。',
            'resume-checkpoint-select': '只有包含 train_state.json 的状态目录才会出现在这里；普通 safetensors 权重不能完整恢复训练进度。',
            'btn-import-toml': '从本地选择 TOML 文件导入到 WebUI 管理区；导入后仍需要加载或保存为配置才能训练。',
            'btn-export-toml': '下载/导出当前选中的 TOML 内容，适合训练前备份或分享配置。',
            'btn-save-toml': '保存左侧表单或当前 TOML 的未保存修改。保存后，“开始训练”才会使用这些新值。',
            'btn-toggle-toml-editor': '展开二级界面，查看、复制或直接编辑当前 TOML；适合批量改字段。',
            'btn-save-as-toml': '把当前配置另存为新 TOML，适合从系统预设复制出自己的可编辑版本。',
            'btn-apply-toml': '把右侧选中的 TOML 加载到左侧表单，并设为当前训练入口。',
            'btn-move-toml-group': '移动右侧配置文件所在分组，只改变列表归类，不改变 TOML 内容。',
            'btn-create-blank-preset': '以 LoRA 标准训练变体 lora.toml 为模板，并套用全局基础模型路径创建新的可编辑项目预设。',
            'btn-fill-global-model-paths': '用全局设置里的基础 DiT、Qwen3、VAE 路径覆盖当前配置表单；覆盖前会要求确认。',
            'btn-reload-toml': '从磁盘重新读取当前 TOML；会丢弃未保存编辑。',
            'btn-copy-toml': '复制当前编辑器里的 TOML 内容，方便备份或排查。',
            'btn-save-toml-direct': '保存直接编辑器里的 TOML 文本。需要连续点击两次确认写入。',
            'btn-lock-toml': '锁定当前配置文件，防止误改；系统预设或分组锁定的文件可能无法手动解锁。',
            'btn-delete-toml': '删除当前选中的可编辑 TOML。需要二次确认；不会删除训练输出目录。',
            'btn-restore-system-toml': '把项目内置系统预设恢复到项目版本。会自动备份，但不影响用户导入配置。',
            'btn-toml-mode-project': '管理 configs 下的项目预设，可编辑、另存、分组和锁定。',
            'btn-toml-mode-output': '查看全局输出文件夹里的训练快照配置；只读，可复制为新的项目预设后编辑。',
            'btn-refresh-output-runs': '重新扫描全局输出文件夹下的训练运行目录。',
            'btn-copy-output-config': '复制当前只读训练快照 TOML 内容。',
            'btn-export-output-config': '导出当前只读训练快照 TOML。',
            'btn-save-output-config-as': '把当前运行目录的原始配置复制到项目预设中，再切回项目预设编辑。',
            'output-run-search': '按运行目录名、时间或配置文件路径筛选训练输出配置。',
            'btn-live-training': '从历史任务视图回到当前正在监控的训练/预处理状态。',
            'btn-refresh-history': '重新读取训练任务历史列表，包括日志、loss、输出目录和样张目录记录。',
            'btn-refresh-history-view': '重新读取当前正在查看的历史日志和 Loss；适合训练仍在写日志时手动更新。',
            'btn-merge-config-group-history': '按同一个配置文件分组合并查看训练日志和 Loss 曲线；预处理任务不会参与合并。',
            'btn-clear-log': '清空当前页面显示的日志文本；不会删除磁盘上的历史日志。',
            'history-show-archived': '显示已归档任务。归档只是隐藏列表项，不会删除训练记录。',
            'btn-refresh-preview': '重新扫描当前预览来源目录，读取最新生成的样张图片。',
            'btn-refresh-weights': '重新扫描选中训练任务的权重文件，显示保存轮次和步数。',
            'btn-sort-weights': '按 Epoch/Step 切换权重文件正序或反序排列。',
            'btn-save-preview-settings': '保存预览图路径设置，只影响预览结果页面读取目录，不会改训练配置。',
            'btn-reset-preview-settings': '恢复预览图目录默认值，例如旧版训练样张兼容目录 output/ckpt/sample。',
            'btn-save-global-settings': '保存 Web 训练输出根目录。每次训练都会在这里创建独立运行目录。',
            'btn-reset-global-settings': '恢复 Web 训练输出根目录默认值 output/runs。',
            'global-output-root': 'Web 训练输出根目录。支持项目相对路径或绝对路径。',
            'global-pretrained-model-path': '新建空白预设时默认写入的基础 DiT 模型路径；单个配置仍可覆盖。',
            'global-qwen3-path': '新建空白预设时默认写入的 Qwen3 文本编码器路径；单个配置仍可覆盖。',
            'global-vae-path': '新建空白预设时默认写入的 VAE 路径；单个配置仍可覆盖。',
            'preview-training-task': '选择一个历史训练任务后，预览图会读取该任务记录的 sample_dir；不选时会优先看当前任务和最新运行目录。',
            'preview-training-dir': '训练中采样的兼容兜底目录；新 Web 运行通常会优先读取全局输出根目录下的最新运行目录。',
            'preview-inference-dir': '推理预览来源目录，通常存放手动推理或测试生成的图片。',
            'preview-custom-dir': '自定义预览目录。填任意项目内或绝对路径后，可在“自定义路径”来源中查看图片。',
            'btn-new-dataset-preset': '新建一个 configs/datasets 下的数据集预设。',
            'btn-copy-dataset-preset': '把当前数据集预设复制成可编辑的新文件。',
            'btn-rename-dataset-preset': '重命名当前数据集预设，会保留图片和缓存目录不变。',
            'btn-import-dataset-preset': '导入外部 TOML 为新的数据集预设。',
            'btn-export-dataset-preset': '导出当前数据集预设 TOML。',
            'btn-delete-dataset-preset': '只删除当前数据集预设 TOML，不删除图片或缓存目录。',
            'btn-save-dataset-preset': '保存当前数据集预设编辑器中的路径和蓝图参数。',
            'btn-refresh-dataset-preview': '重新扫描当前数据集路径，读取最新图片和同名 caption 标注。',
            'btn-config-dataset-dialog-refresh': '重新读取可选的数据集预设列表，并保留当前配置页的选择状态。',
            'btn-config-dataset-dialog-manage': '切换到数据集页，编辑或新增可复用的数据集预设。',
            'btn-open-unnamed-dataset-dialog': '打开待命名弹窗。',
        };
        for (const [id, title] of Object.entries(tips)) {
            const el = document.getElementById(id);
            if (el && !el.title) el.title = title;
        }
        document.querySelectorAll('.tab-btn').forEach((btn) => {
            const labels = {
                config: '配置页：选择方法/变体/预设，编辑训练参数并引用数据集预设。',
                datasets: '数据集页：管理可复用的多数据集预设。',
                training: '训练页：查看当前任务、历史任务、loss 曲线、日志和显存状态。',
                preview: '预览结果页：查看训练中样张、推理输出或自定义目录图片。',
                settings: '全局设置页：设置 Web 训练输出根目录和新建预设默认模型路径。',
            };
            const key = btn.dataset.tab;
            if (labels[key]) btn.title = labels[key];
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            const labels = {
                training: '读取训练任务的 sample_dir，或优先读取最新 Web 运行目录里的训练样张。',
                inference: '读取推理预览目录，适合查看手动测试生成图。',
                custom: '读取你填写的自定义目录，适合临时检查任意图片文件夹。',
            };
            const key = btn.dataset.previewSource;
            if (labels[key]) btn.title = labels[key];
        });
    }

    // ── 工具函数 ──
    async function api(url, opts = {}) {
        const headers = { 'Content-Type': 'application/json' };
        const res = await fetch(url, { headers, ...opts });
        const text = await res.text();
        let data;
        try {
            data = text ? JSON.parse(text) : {};
        } catch {
            data = { ok: false, error: text || `HTTP ${res.status}` };
        }
        if (!res.ok && data && !Object.prototype.hasOwnProperty.call(data, 'ok')) {
            data.ok = false;
        }
        return data;
    }

    function val(id) {
        return document.getElementById(id)?.value || '';
    }

    function populateSelect(id, items, preferred = '') {
        const sel = document.getElementById(id);
        const prev = sel.value;
        sel.innerHTML = '';
        for (const item of items) {
            const opt = document.createElement('option');
            opt.value = item;
            opt.textContent = item;
            sel.appendChild(opt);
        }
        if (items.includes(prev)) {
            sel.value = prev;
        } else if (preferred && items.includes(preferred)) {
            sel.value = preferred;
        }
    }
})();

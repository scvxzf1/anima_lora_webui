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
    let configLoadSeq = 0;
    let datasetLoadSeq = 0;
    let stepEstimateSeq = 0;
    let samplePromptsLoadSeq = 0;
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
    let trainingSampleState = null;
    let samplePromptsPath = 'configs/sample_prompts.txt';
    let samplePromptsContent = '';
    let viewingHistoryTaskId = '';
    let historyViewMode = 'live';
    let currentHistoryTaskForResume = null;
    let currentHistoryConfigGroup = null;
    let resumeOptionsState = {
        loading: false,
        taskId: '',
        checkpoints: [],
        defaultCheckpoint: '',
        error: '',
        message: '',
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
    const FORM_UI_DEFAULTS = {
        train_batch_size: 1,
        gradient_accumulation_steps: 1,
        sample_prompts: '',
        sample_every_n_epochs: '',
        sample_every_n_steps: '',
        sample_at_first: false,
        sample_sampler: 'ddim',
        use_lokr: false,
        lokr_factor: 8,
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
    ]);
    const DATASET_EDITOR_COMPAT_FIELDS = new Set([
        'source_image_dir',
        'resized_image_dir',
        'lora_cache_dir',
        'dataset_config',
    ]);
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
    };
    const MAX_LOG_LINES = 2000;
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
            description: '最常改：输出命名、训练时长、学习率和保存频率。',
            open: true,
            className: 'config-group-primary',
            keys: [
                'output_name',
                'output_dir',
                'learning_rate',
            ],
        },
        {
            title: '步数与训练量',
            description: '集中设置训练轮数、批大小、梯度累积和保存间隔；下方会实时估算总步数。',
            open: true,
            className: 'config-group-steps',
            keys: [
                'max_train_epochs',
                'train_batch_size',
                'gradient_accumulation_steps',
                'sample_ratio',
                'save_every_n_epochs',
                'checkpointing_epochs',
            ],
        },
        {
            title: '数据集设置',
            description: '训练读取缩放图目录和 LoRA 缓存目录；原始数据集路径主要用于预处理和生成缓存路径。',
            open: true,
            className: 'config-group-data',
            keys: [
                'source_image_dir',
                'resized_image_dir',
                'lora_cache_dir',
                'dataset_config',
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
            title: 'LoRA / LoKr 核心参数',
            description: '控制适配器类型、容量和热启动；不确定时保持默认。',
            open: true,
            className: 'lora-tuning-group',
            keys: [
                'network_dim',
                'network_alpha',
                'use_lokr',
                'lokr_factor',
                'network_weights',
                'dim_from_weights',
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
                'compile_mode',
                'compile_inductor_mode',
                'static_token_count',
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
            title: '优化器与采样',
            description: '进阶训练动态；默认值通常已经足够。',
            open: false,
            keys: [
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
            title: '输出格式与训练范围',
            description: '模型保存格式、保存精度和训练目标范围。',
            open: false,
            keys: [
                'save_model_as',
                'save_precision',
                'network_train_unet_only',
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
            ],
        },
    ];

    const VARIANT_METHOD_FAMILY = {
        lora: 'lora',
        lora_longer: 'lora',
        'lora-8gb': 'lora',
        lora_repa: 'lora',
        lokr: 'lokr',
        ortholora: 'ortholora',
        tlora: 'tlora',
        tlora_ortho: 'tlora',
        hydralora_sigma: 'hydralora',
        hydralora_experimental: 'hydralora',
        hydralora_fei: 'hydralora',
        fera: 'hydralora',
        reft: 'reft',
        tlora_ortho_reft: 'reft',
        postfix_ortho_cond: 'postfix',
        ip_adapter: 'ip_adapter',
        easycontrol: 'easycontrol',
        soft_tokens: 'lora',
    };

    function help(summary, fill, benefit, cost, risk, recommend) {
        return { summary, fill, benefit, cost, risk, recommend };
    }

    const FIELD_LABEL_ZH = {
        add_reft: '启用 ReFT',
        alpha_rank_scale: '按秩缩放 Alpha',
        attn_mode: '注意力后端',
        balance_loss_warmup_ratio: '均衡损失预热比例',
        balance_loss_weight: '均衡损失权重',
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
        checkpointing_epochs: '训练状态保存间隔',
        compile_inductor_mode: 'Inductor 编译模式',
        compile_mode: '编译模式',
        dataloader_pin_memory: '固定内存加载',
        dataset_config: '数据集配置',
        discrete_flow_shift: 'Flow 偏移',
        easycontrol_cond_noise_max: 'EasyControl 条件噪声上限',
        easycontrol_drop_p: 'EasyControl 条件丢弃率',
        enable_bucket: '启用长宽比分桶',
        fei_feature_dim: 'FEI 特征维度',
        fei_sigma_low_div: 'FEI 低 Sigma 除数',
        fera_fecl_weight: 'FeRA FECL 权重',
        fera_num_bands: 'FeRA 分带数',
        train_batch_size: '批大小',
        gradient_accumulation_steps: '梯度累积步数',
        gradient_checkpointing: '梯度检查点',
        ip_features_cache_to_disk: 'IP 特征写入磁盘',
        ip_image_drop_p: 'IP 图像条件丢弃率',
        keep_tokens: '保留 Token 数',
        learning_rate: '学习率',
        log_every_n_steps: '日志记录间隔',
        log_with: '日志后端',
        logging_dir: '日志目录',
        lora_cache_dir: 'LoRA 缓存目录',
        lr_scheduler: '学习率调度',
        masked_loss: '遮罩损失',
        max_train_epochs: '最大训练轮数',
        max_bucket_reso: '最大桶边长',
        min_rank: '最小秩',
        min_bucket_reso: '最小桶边长',
        mixed_precision: '混合精度',
        network_alpha: 'LoRA Alpha',
        network_args: '网络额外参数',
        network_dim: 'LoRA 秩',
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
        per_bucket_balance_weight: '分桶均衡权重',
        persistent_data_loader_workers: '常驻数据加载进程',
        pretrained_model_name_or_path: '基础模型路径',
        qwen3: 'Qwen3 文本编码器路径',
        reft_alpha: 'ReFT Alpha',
        reft_dim: 'ReFT 秩',
        reft_layers: 'ReFT 层范围',
        repa_layer: 'REPA 层',
        repa_lr_scale: 'REPA 学习率倍率',
        repa_weight: 'REPA 权重',
        resized_image_dir: '缩放图像目录',
        resolution: '数据集分辨率',
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
        sigma_bucket_boundaries: 'Sigma 桶边界',
        sigma_feature_dim: 'Sigma 特征维度',
        skip_cache_check: '跳过缓存检查',
        source_image_dir: '源图像目录',
        specialize_experts_by_sigma_buckets: '按 Sigma 桶专门化专家',
        static_token_count: '固定 Token 数',
        timestep_sampling: '时间步采样',
        torch_compile: '启用 torch.compile',
        trim_crossattn_kv: '裁剪交叉注意力 KV',
        unsloth_offload_checkpointing: 'Unsloth 检查点卸载',
        use_custom_down_autograd: '自定义 Down 反向',
        use_easycontrol: '启用 EasyControl',
        use_ip_adapter: '启用 IP-Adapter',
        use_moe_style: 'MoE 结构',
        use_ortho: '启用 OrthoLoRA',
        use_repa: '启用 REPA',
        use_shuffled_caption_variants: '使用打乱标题变体',
        use_timestep_mask: '启用 T-LoRA',
        validation_seed: '验证随机种子',
        validation_split: '验证集比例',
        validation_split_num: '固定验证数量',
        vae: 'VAE 路径',
        vae_chunk_size: 'VAE 分块大小',
        vae_disable_cache: '禁用 VAE 缓存',
    };

    const FIELD_OPTIONS = {
        attn_mode: ['flash', 'flex'],
        compile_inductor_mode: ['default', 'reduce-overhead', 'max-autotune'],
        compile_mode: ['blocks', 'full'],
        dataset_config: [
            'configs/datasets/ip_adapter.toml',
            'configs/datasets/easycontrol.toml',
            'configs/datasets/rokkotsu_goddess.toml',
        ],
        train_batch_size: [1, 2, 4, 8],
        gradient_accumulation_steps: [1, 2, 4, 8],
        log_with: ['tensorboard'],
        use_lokr: [false, true],
        lokr_factor: [2, 4, 8, 16],
        lr_scheduler: ['constant', 'cosine', 'cosine_with_restarts', 'polynomial'],
        max_train_epochs: [1, 2, 4, 8, 12, 16, 24],
        min_rank: [1, 2, 4, 8],
        mixed_precision: ['bf16', 'fp16', 'no'],
        network_alpha: [4, 8, 16, 32, 64],
        network_dim: [4, 8, 16, 32, 64, 128],
        network_module: [
            'networks.lora_anima',
            'networks.methods.postfix',
            'networks.methods.ip_adapter',
            'networks.methods.easycontrol',
            'networks.methods.soft_tokens',
        ],
        num_experts: [2, 4, 6, 8],
        num_sigma_buckets: [2, 3, 4],
        optimizer_type: ['AdamW', 'AdamW8bit', 'Lion', 'Prodigy'],
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
        save_every_n_epochs: [1, 2, 4, 8, 12],
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
        postfix: choiceHelp(
            'Postfix',
            '训练附加文本条件 token 或条件后缀，属于实验性文本条件方法。',
            '参数少、学习快；风险是适用场景更窄，和普通 LoRA 体验不同。',
            '用于特定文本条件实验。'
        ),
        ip_adapter: choiceHelp(
            'IP-Adapter',
            '图像条件适配器，用参考图像特征参与训练。',
            '能学习图像条件控制；代价是需要专用数据和图像特征缓存。',
            '需要参考图/图像条件时选。'
        ),
        easycontrol: choiceHelp(
            'EasyControl',
            '图像控制条件方法，使用专用数据集和缓存目录。',
            '控制信号更直接；代价是数据准备和训练路径更专门。',
            '需要图像控制训练时选。'
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
        postfix_ortho_cond: choiceHelp(
            'Postfix 条件正交版',
            'caption 条件后缀方法，使用专用 network_args。',
            '参数少、适合文本条件实验；不是普通 LoRA 的直接替代。',
            '做 Postfix 实验时选。'
        ),
        ip_adapter: choiceHelp(
            'IP-Adapter',
            '训练图像条件 cross-attention 适配器。',
            '能使用参考图像条件；需要专用 dataset_config 和图像特征缓存。',
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
            "LoRA 秩，也就是低秩矩阵的通道数。",
            "常用 16/32/64。数据少或低显存用 16-32；复杂画风、角色细节多时再升到 64。",
            ["容量更高，能记住更细的风格、角色和构图差异。"],
            ["显存、检查点体积和训练时间都会增加。"],
            ["过高容易过拟合，小数据集会把噪声和偶然构图也记进去。"],
            "默认 32 是稳妥起点；8GB 或小数据集优先 16-32。"
        ),
        network_alpha: help(
            "LoRA 缩放因子，实际更新强度约等于 alpha / dim。",
            "通常填得和 network_dim 一样。想让更新更保守时，填 dim 的一半。",
            ["控制更新幅度，能减少 LoRA 对底模的冲击。"],
            ["alpha 太低会让训练显得迟钝，需要更多步数。"],
            ["alpha 太高可能导致风格过冲、颜色发脏或提示词服从性下降。"],
            "推荐 alpha = dim；只有明显过拟合或过强时再降低。"
        ),
        network_module: help(
            "训练时加载的网络实现模块。",
            "普通 LoRA 家族保持 networks.lora_anima；Postfix、IP-Adapter 等变体由对应 TOML 自动填写。",
            ["允许不同训练方法共用同一个 Web 表单。"],
            ["改错模块会导致启动失败，或加载到不匹配的参数。"],
            ["不要手动改成未注册/不存在的 Python 模块。"],
            "除非你在开发新方法，否则保持变体默认值。"
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
            "推荐 true。"
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
            "新手先用普通 LoRA；需要 Hydra/FeRA 时直接选对应变体。"
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
            "需要图像条件时选 ip_adapter 变体；普通 LoRA 关闭。"
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
            "需要 EasyControl 时选 easycontrol 变体；普通训练关闭。"
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
            "优化器基础学习率。",
            "LoRA 家族常用 2e-5；ReFT/IP-Adapter/Postfix 可能更高，跟随变体默认。",
            ["直接影响收敛速度。"],
            ["越高越容易震荡，越低训练越慢。"],
            ["过高会损坏画面质量或让 loss 不稳定。"],
            "普通 LoRA 从 2e-5 开始；只在欠拟合/过拟合时小步调整。"
        ),
        max_train_epochs: help(
            "总训练轮数，一轮是完整遍历一次数据集。",
            "小数据集从 4-12 轮观察；实验方法按变体默认。",
            ["增加轮数能让模型看更多次数据。"],
            ["训练更久，过拟合风险更高。"],
            ["轮数太高会记住训练图构图，泛化下降。"],
            "先用默认，样张仍欠拟合再增加。"
        ),
        train_batch_size: help(
            "每个训练 step 一次送入 GPU 的样本数量。",
            "显存足够时可试 2/4；显存紧张或 1024 分辨率训练保持 1。",
            ["批大小越大，单次更新看到的数据越多，梯度更稳定。"],
            ["显存占用会明显上升，可能触发 OOM。"],
            ["调大批大小会减少每轮 step 数，常需要重新理解总步数和学习率。"],
            "默认 1 最稳；想要更大有效批大小时优先配合梯度累积。"
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
            "累积多少步梯度后再更新一次参数。",
            "显存不足但想要更大有效批大小时升高。",
            ["有效批大小变大，训练更稳。"],
            ["每次参数更新变慢，日志步数理解更复杂。"],
            ["过高会让反馈变慢，也可能改变最佳学习率。"],
            "默认即可；Postfix 等变体按 TOML 使用 4。"
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
            "默认 AdamW；低显存或实验可用 AdamW8bit、Lion、Prodigy 等。",
            ["不同优化器适合不同内存和收敛偏好。"],
            ["非默认优化器可能需要重新调学习率。"],
            ["随意切换会让历史经验不再适用。"],
            "先用 AdamW；显存紧张再考虑 8bit。"
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
            "训练中生成预览图时使用的提示词，一行一个。",
            "直接在输入框里每行填写一条提示词；保存时 WebUI 会自动写入 configs/sample_prompts.txt。",
            ["能在训练过程中直接看到当前模型效果。"],
            ["采样会额外占用训练时间和显存。"],
            ["提示词越多，每次采样生成的图片越多，训练被打断的时间越长。"],
            "想看训练中预览图时，至少填一行提示词，再设置按轮或按步采样。"
        ),
        sample_every_n_epochs: help(
            "每隔多少轮生成一次训练样张。",
            "填 1 表示每轮结束出图；留空表示不按轮采样。",
            ["最容易理解，适合按 epoch 保存模型一起观察。"],
            ["每轮都会额外跑一次推理，训练总耗时会增加。"],
            ["数据集很小或采样图很多时，出图会比较频繁。"],
            "新手建议填 1 或 2；只需要按步采样时留空。"
        ),
        sample_every_n_steps: help(
            "每隔多少优化步生成一次训练样张。",
            "例如 500；留空表示不按步采样。设置按轮采样时，按轮逻辑优先。",
            ["长 epoch 训练时能更早看到趋势。"],
            ["步数过小会频繁打断训练。"],
            ["需要结合预计总步数理解频率。"],
            "多数情况用按轮采样即可；只有单轮很长时再填。"
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
            "注意力计算后端。",
            "Linux 高性能环境通常用 flash；跨平台或兼容优先用 flex。",
            ["正确后端能显著影响速度和显存。"],
            ["某些后端依赖特定 CUDA/PyTorch/显卡支持。"],
            ["不兼容会启动失败或回退变慢。"],
            "当前环境支持 flash 时用 flash；出错再切 flex。"
        ),
        gradient_checkpointing: help(
            "反向传播时重算激活，而不是全部存显存。",
            "低显存训练开启；显存充足且追求速度可关闭。",
            ["显著降低显存占用。"],
            ["训练变慢，因为会重复计算。"],
            ["和 full compile 等模式可能存在兼容限制。"],
            "8GB/低显存推荐 true；高速训练可测试 false。"
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
            "训练时在 GPU 和 CPU 间交换的 DiT 块数。",
            "0 表示全放 GPU；显存不足时提高。",
            ["降低 GPU 显存峰值。"],
            ["CPU/GPU 交换会明显拖慢训练。"],
            ["设置过高会让训练时间变得很长。"],
            "显存够用保持 0；OOM 时先用 low_vram 或 lora-8gb。"
        ),
        torch_compile: help(
            "启用 torch.compile 编译前向图。",
            "Linux/新 PyTorch 环境可开启；调试兼容问题时关闭。",
            ["编译完成后训练更快。"],
            ["首次启动有编译等待，缓存也会占空间。"],
            ["与某些动态形状、block swap、checkpoint 组合可能不兼容。"],
            "默认开启；遇到编译报错再关闭。"
        ),
        compile_mode: help(
            "torch.compile 的编译范围。",
            "blocks 表示逐块编译；full 表示整模型大图编译。",
            ["full 可能带来更强跨块优化。"],
            ["full 对兼容性要求更高。"],
            ["full 与 gradient checkpointing、block swap 通常不兼容。"],
            "高显存和稳定环境用 full；低显存/排错用 blocks。"
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
            "混合精度模式。",
            "现代 NVIDIA GPU 优先 bf16；旧卡不支持 bf16 时用 fp16。",
            ["降低显存并提升吞吐。"],
            ["依赖硬件支持。"],
            ["fp16 更容易数值不稳定；bf16 在旧卡上可能不可用。"],
            "支持 bf16 就用 bf16。"
        ),
        static_token_count: help(
            "把所有 batch 固定到 4096 token。",
            "配合本项目 bucket 设计保持默认。",
            ["减少 torch.compile 因宽高比变化反复编译。"],
            ["会对较小图像做 padding，存在少量无效计算。"],
            ["关闭后可能触发多形状编译和性能抖动。"],
            "推荐保持 4096。"
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
            "基础 DiT 模型权重路径。",
            "填写 .safetensors 文件路径，通常在 models/diffusion_models 下。",
            ["决定 LoRA 训练依附的底模。"],
            ["模型文件大，需要本地存在。"],
            ["换底模后旧 LoRA 可能不兼容。"],
            "使用项目默认 Anima base 路径。"
        ),
        qwen3: help(
            "Qwen3 文本编码器权重路径。",
            "保持下载脚本放置的默认路径。",
            ["提供文本条件编码。"],
            ["模型较大，占用磁盘和加载时间。"],
            ["路径错误会导致预处理或训练启动失败。"],
            "不要手动改，除非你替换了文本编码器权重。"
        ),
        vae: help(
            "VAE 模型路径。",
            "保持 models/vae 下的默认权重路径。",
            ["负责图像和 latent 的互相转换。"],
            ["更换 VAE 会影响缓存兼容性。"],
            ["旧 latent 缓存可能与新 VAE 不匹配。"],
            "使用默认 qwen_image_vae。"
        ),
        output_dir: help(
            "训练输出目录。",
            "默认 output/ckpt；想分项目保存可改成子目录。",
            ["方便管理不同实验输出。"],
            ["目录过多时需要自己清理。"],
            ["路径写错可能让你找不到产物，或没有写权限。"],
            "默认 output/ckpt。"
        ),
        output_name: help(
            "输出检查点文件名前缀。",
            "用简短英文/数字/下划线命名，避免空格和特殊字符。",
            ["方便区分不同训练实验。"],
            ["同名训练会让目录里文件难以区分。"],
            ["命名不清会增加回溯成本。"],
            "建议包含方法和数据集简称，例如 anima_styleA。"
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
            "兼容旧配置的原始数据集路径；在多数据集模式中，它只是下方第 1 组原始路径的镜像。",
            "不要在这里编辑多组路径；请在“多数据集路径”里逐行填写原始数据集、缩放图和缓存目录。",
            ["旧脚本仍能读到 source_image_dir，同时 WebUI 可以使用 dataset_config 管理多组数据。"],
            ["只代表第 1 组，不能表达第 2 组及之后的数据集。"],
            ["误以为它是全局路径，可能会忽略下方真正参与训练的多组路径。"],
            "多数据集训练时以下方每一行为准；这里保持自动同步即可。"
        ),
        resized_image_dir: help(
            "预处理后 resize 图像的目录。",
            "默认 post_image_dataset/resized。",
            ["训练读取统一尺寸/bucket 后的数据。"],
            ["占额外磁盘。"],
            ["源图变更后旧 resize 文件可能过期。"],
            "保持默认；换数据后重新预处理。"
        ),
        lora_cache_dir: help(
            "LoRA 训练缓存目录，保存 latent 和文本嵌入。",
            "默认 post_image_dataset/lora。",
            ["让训练阶段少做重复编码。"],
            ["缓存会占磁盘。"],
            ["旧缓存与新 caption/图像不一致时会污染训练。"],
            "保持默认；改数据或 tokenizer 后重建缓存。"
        ),
        dataset_config: help(
            "替代默认数据集蓝图的 TOML 路径。",
            "IP-Adapter/EasyControl 等方法按变体填写专用数据集配置。",
            ["支持不同方法使用不同数据布局。"],
            ["需要维护额外 TOML 文件。"],
            ["路径错误或字段不匹配会导致数据加载失败。"],
            "普通 LoRA 不需要手动设置；实验方法保持变体默认。"
        ),
        resolution: help(
            "数据集蓝图的目标分辨率。",
            "常用 1024；显存紧张时降低，想训练更高细节时谨慎升高。",
            ["决定预处理/训练桶的基础像素规模。"],
            ["越高显存和训练时间越大。"],
            ["和已有缩放图、缓存不一致时需要重新预处理。"],
            "默认 1024。"
        ),
        batch_size: help(
            "dataset_config 内的 batch_size。",
            "通常保持 1；整体有效批大小主要看训练设置里的批大小和梯度累积。",
            ["数据蓝图里保留完整批大小信息。"],
            ["调大可能增加显存占用。"],
            ["与训练批大小混用时容易误判总步数。"],
            "不确定就保持 1。"
        ),
        enable_bucket: help(
            "启用长宽比分桶。",
            "开启后，预处理会给每张图选择最接近的桶尺寸，缩放并中心裁切后写入缩放图目录；训练读取这些缩放图和对应缓存。",
            ["更好保留原图构图比例，避免所有图片被硬塞进同一个正方形尺寸。"],
            ["换分桶参数后，缩放图目录和 LoRA 缓存目录都需要重新生成，旧缓存不能混用。"],
            ["不会额外生成“桶文件夹”；如果以为训练还会读原图，可能会误判数据是否已经更新。"],
            "推荐开启。"
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
            "TensorBoard 等日志输出目录。",
            "默认 output/logs。",
            ["便于查看训练曲线和历史记录。"],
            ["长时间训练会积累日志文件。"],
            ["同目录多实验混在一起会影响追踪。"],
            "默认即可；多项目可改成子目录。"
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

    // ── Tab 切换 ──
    function setupTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
                if (btn.dataset.tab === 'training' && lossChart?.resize) {
                    lossChart.resize();
                }
                if (btn.dataset.tab === 'preview') {
                    loadPreviewImages();
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
            await loadConfig();
            await loadTomlFileList();
            rememberSelectionSnapshot();
            await loadTrainingHistoryList();
            await loadPreviewSettings();
            await loadSamplePrompts();
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
        datasetEditorState = {
            loading: false,
            loaded: false,
            dirty: false,
            dataset_config: currentConfig.dataset_config || '',
            datasets: [],
            error: '',
        };
        renderConfigForm(currentConfig);
        loadDatasetEditor(requestSeq);
        loadSamplePrompts(samplePromptsPath, requestSeq);
        loadStepEstimate(requestSeq);
        updateChoiceGuide();
        // 同步加载对应的 TOML 文件到右侧编辑器
        const tomlFile = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlFiles.includes(tomlFile) && currentTomlFile !== tomlFile) {
            loadTomlFile(tomlFile, { force: true });
        }
    }

    async function reloadCurrentConfig() {
        if (!confirmDiscardTomlChanges('当前配置有未保存修改，刷新会重新读取表单和数据集设置并丢弃这些修改。是否继续？')) {
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
            if (key === 'general' || key === 'datasets') continue;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) continue;
            fieldsByKey[key] = value;
        }
        for (const [key, value] of Object.entries(FORM_UI_DEFAULTS)) {
            if (!(key in fieldsByKey)) fieldsByKey[key] = value;
        }
        fieldsByKey.sample_prompts = currentSamplePromptText(config);

        const consumed = new Set();
        for (const section of FORM_SECTION_DEFS) {
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

    function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    async function loadStepEstimate(parentSeq = configLoadSeq) {
        const requestSeq = ++stepEstimateSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        try {
            const data = await api(`/api/config/steps?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}`);
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

        const epochs = readLiveNumber('max_train_epochs', currentStepEstimate.max_train_epochs || 1);
        const batchSize = readLiveNumber('train_batch_size', currentStepEstimate.train_batch_size || 1);
        const gradAccum = readLiveNumber('gradient_accumulation_steps', currentStepEstimate.gradient_accumulation_steps || 1);
        const sampleRatio = readLiveNumber('sample_ratio', currentStepEstimate.sample_ratio || 1);
        const datasets = liveDatasetRowsForEstimate();
        const trainImages = datasets.reduce((sum, row) => sum + Number(row.train_image_count || 0), 0);
        const weightedImages = datasets.reduce((sum, row) => sum + (Number(row.train_image_count || 0) * Number(row.num_repeats || 1)), 0);
        const effectiveBatch = Math.max(1, batchSize * gradAccum);
        const repeatedImages = Math.max(0, Math.floor(weightedImages * sampleRatio));
        const stepsPerEpoch = repeatedImages ? Math.ceil(repeatedImages / effectiveBatch) : 0;
        const totalSteps = stepsPerEpoch * epochs;

        setText('step-dataset-count', String(datasets.length || 0));
        setText('step-train-images', String(trainImages));
        setText('step-repeated-images', `${repeatedImages} = ${weightedImages} x ${sampleRatio}`);
        setText('step-effective-batch', `${effectiveBatch} = ${batchSize} x ${gradAccum}`);
        setText('step-per-epoch', String(stepsPerEpoch));
        setText('step-total', String(totalSteps));
        renderStepDatasetBreakdown(datasets);
        setText('step-estimate-note', `公式: Σ(每组训练图片 x 重复次数) x sample_ratio / (train_batch_size x gradient_accumulation_steps) x max_train_epochs。缩放图目录为空时会暂按原始数据集图片数估算。`);
    }

    function liveDatasetRowsForEstimate() {
        const baseRows = Array.isArray(currentStepEstimate?.datasets) ? currentStepEstimate.datasets : [];
        if (!datasetEditorState.dirty || !datasetEditorState.datasets.length) {
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
        return datasetEditorState.datasets.map((row, idx) => {
            const old = baseRows.find((item) => item.source_dir === row.source_dir && item.image_dir === row.image_dir) || baseRows[idx] || {};
            const sourceCount = Number(old.source_image_count || 0);
            const resizedCount = Number(old.resized_image_count || 0);
            const trainCount = resizedCount || sourceCount;
            const repeats = Math.max(1, Number(row.num_repeats || old.num_repeats || 1));
            return {
                ...old,
                index: idx + 1,
                source_dir: row.source_dir,
                image_dir: row.image_dir,
                cache_dir: row.cache_dir,
                source_image_count: sourceCount,
                resized_image_count: resizedCount,
                train_image_count: trainCount,
                num_repeats: repeats,
                weighted_image_count: trainCount * repeats,
                uses_preprocessed_images: resizedCount > 0,
            };
        });
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

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function createGroup(name, fields, open, extraClass = '', description = '') {
        const details = document.createElement('details');
        details.className = ['config-group', extraClass].filter(Boolean).join(' ');
        if (open) details.open = true;

        const summary = document.createElement('summary');
        const title = document.createElement('span');
        title.textContent = `${name} (${fields.length} 项)`;
        summary.appendChild(title);
        details.appendChild(summary);

        const content = document.createElement('div');
        if (description) {
            const hint = document.createElement('p');
            hint.className = 'config-group-hint';
            hint.textContent = description;
            content.appendChild(hint);
        }
        if (extraClass === 'config-group-data') {
            content.appendChild(createDatasetEditor());
        }
        for (const [key, value] of fields) {
            content.appendChild(createFieldRow(key, value));
        }
        if (extraClass === 'config-group-steps') {
            content.appendChild(createStepEstimatePanel());
            updateStepEstimatePanel();
        }
        details.appendChild(content);
        return details;
    }

    function createDatasetEditor() {
        const panel = document.createElement('div');
        panel.id = 'dataset-editor';
        panel.className = 'dataset-editor';
        renderDatasetEditor(panel);
        return panel;
    }

    function renderDatasetEditor(existingPanel = null) {
        const panel = existingPanel || document.getElementById('dataset-editor');
        if (!panel) return;
        panel.innerHTML = '';

        const header = document.createElement('div');
        header.className = 'dataset-editor-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>多数据集路径</strong><span>每一行是一组数据：原始图用于预处理，缩放图和缓存用于训练；重复次数只影响这一组权重。</span>';
        const actions = document.createElement('div');
        actions.className = 'dataset-editor-actions';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '添加数据集';
        addBtn.title = '新增一组数据集路径。适合把多个角色、画风或批次一起训练，并给每组设置独立重复次数。';
        addBtn.addEventListener('click', addDatasetEditorRow);
        const suggestBtn = document.createElement('button');
        suggestBtn.type = 'button';
        suggestBtn.className = 'btn btn-small secondary';
        suggestBtn.textContent = '自动填入缩放图和缓存目录';
        suggestBtn.title = '根据每一行的原始数据集路径，自动填入对应的缩放图目录和 LoRA 缓存目录；不会创建文件或启动预处理。';
        suggestBtn.addEventListener('click', applySuggestedDataDirs);
        actions.append(addBtn, suggestBtn);
        header.append(title, actions);
        panel.appendChild(header);

        if (datasetEditorState.loading) {
            const loading = document.createElement('p');
            loading.className = 'dataset-editor-message';
            loading.textContent = '正在读取数据集配置...';
            panel.appendChild(loading);
            return;
        }
        if (datasetEditorState.error) {
            const error = document.createElement('p');
            error.className = 'dataset-editor-message error';
            error.textContent = datasetEditorState.error;
            panel.appendChild(error);
        }

        const rows = datasetEditorState.datasets.length
            ? datasetEditorState.datasets
            : normalizeDatasetEditorRows([{
                source_dir: currentConfig.source_image_dir || '',
                image_dir: currentConfig.resized_image_dir || '',
                cache_dir: currentConfig.lora_cache_dir || '',
                num_repeats: 1,
            }]);
        if (!datasetEditorState.datasets.length) {
            datasetEditorState.datasets = rows;
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
        configPath.textContent = datasetEditorState.dataset_config || currentConfig.dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';
        const dirty = document.createElement('span');
        dirty.className = datasetEditorState.dirty ? 'dataset-editor-dirty active' : 'dataset-editor-dirty';
        dirty.textContent = datasetEditorState.dirty ? '有未保存的数据集修改' : '数据集路径已同步';
        footer.append(configPath, dirty);
        panel.appendChild(footer);
    }

    function createDatasetDefaultsEditor() {
        const defaults = normalizeDatasetDefaults(datasetEditorState.defaults || {});
        datasetEditorState.defaults = defaults;
        const wrap = document.createElement('div');
        wrap.className = 'dataset-defaults-list';

        const heading = document.createElement('div');
        heading.className = 'dataset-defaults-heading';
        heading.innerHTML = '<strong>训练数据集配置</strong><span>这些是本 dataset_config 共用的规则：预处理会按分桶参数生成缩放图，训练读取缩放图和 LoRA 缓存。</span>';
        wrap.appendChild(heading);

        const fields = [
            ['source_image_dir', 'text', 'wide'],
            ['resolution', 'number'],
            ['enable_bucket', 'select'],
            ['min_bucket_reso', 'number'],
            ['max_bucket_reso', 'number'],
            ['bucket_reso_steps', 'number'],
            ['bucket_no_upscale', 'select'],
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
                input.step = key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1';
            }
        }
        input.className = 'field-input dataset-config-input';
        input.dataset.key = key;
        if (key === 'source_image_dir') {
            input.readOnly = true;
            input.title = '只显示下方第 1 组“原始数据集路径”的镜像；多数据集请在下面逐行编辑。';
        }
        input.addEventListener('input', () => updateDatasetConfigValue(key, input));
        input.addEventListener('change', () => updateDatasetConfigValue(key, input));
        return input;
    }

    function datasetConfigLabel(key) {
        const labels = {
            source_image_dir: '第 1 组原始路径镜像',
            resolution: '分辨率',
            enable_bucket: '启用长宽比分桶',
            min_bucket_reso: '最小桶边长',
            max_bucket_reso: '最大桶边长',
            bucket_reso_steps: '桶尺寸步长',
            bucket_no_upscale: '小图放大',
        };
        return `${labels[key] || FIELD_LABEL_ZH[key] || key} / ${key}`;
    }

    function datasetConfigValue(key, defaults) {
        if (key === 'source_image_dir') {
            return datasetEditorState.datasets[0]?.source_dir || currentConfig.source_image_dir || '';
        }
        return defaults[key] ?? '';
    }

    function updateDatasetConfigValue(key, input) {
        if (key === 'source_image_dir') {
            return;
        }
        updateDatasetDefault(key, input);
    }

    function createDatasetEditorRow(row, index) {
        const wrap = document.createElement('div');
        wrap.className = 'dataset-editor-row';
        wrap.dataset.index = String(index);
        wrap.appendChild(createDatasetPathField(index, 'source_dir', '原始数据集路径', row.source_dir, 'image_dataset'));
        wrap.appendChild(createDatasetPathField(index, 'image_dir', '缩放图目录', row.image_dir, 'post_image_dataset/resized'));
        wrap.appendChild(createDatasetPathField(index, 'cache_dir', 'LoRA 缓存目录', row.cache_dir, 'post_image_dataset/lora'));

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
        remove.disabled = datasetEditorState.datasets.length <= 1;
        remove.title = remove.disabled ? '至少保留一组数据集路径' : '从当前 dataset_config 中移除这一组路径，不会删除磁盘文件。';
        remove.addEventListener('click', () => removeDatasetEditorRow(index));
        wrap.appendChild(remove);
        return wrap;
    }

    function createDatasetPathField(index, key, label, value, placeholder) {
        const field = document.createElement('label');
        field.className = 'dataset-path-field';
        const text = document.createElement('span');
        text.textContent = label;
        const titles = {
            source_dir: '原始图片和 caption 所在目录。预处理从这里读图；训练通常不直接读原始图。',
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

    function normalizeDatasetEditorRows(rows) {
        return (rows || [])
            .filter((row) => row && typeof row === 'object')
            .map((row) => ({
                source_dir: String(row.source_dir || row.source_image_dir || ''),
                image_dir: String(row.image_dir || row.resized_image_dir || ''),
                cache_dir: String(row.cache_dir || row.lora_cache_dir || ''),
                num_repeats: Math.max(1, Number.parseInt(row.num_repeats || 1, 10) || 1),
            }));
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
        const defaults = normalizeDatasetDefaults(datasetEditorState.defaults || {});
        if (input.type === 'checkbox') {
            defaults[key] = input.checked;
        } else if (input.tagName === 'SELECT') {
            defaults[key] = input.value === 'true';
        } else if (input.type === 'number') {
            defaults[key] = key === 'validation_split' ? Math.max(0, Number(input.value) || 0) : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            defaults[key] = input.value;
        }
        datasetEditorState.defaults = defaults;
        markDatasetEditorDirty();
    }

    function updateDatasetEditorRow(index, key, value) {
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        if (!rows[index]) return;
        rows[index][key] = key === 'num_repeats'
            ? Math.max(1, Number.parseInt(value || '1', 10) || 1)
            : value;
        datasetEditorState.datasets = rows;
        if (index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    function markDatasetEditorDirty() {
        datasetEditorState.dirty = true;
        updateTomlDirtyState();
        updateStepEstimatePanel();
        const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
        if (dirty) {
            dirty.classList.add('active');
            dirty.textContent = '有未保存的数据集修改';
        }
    }

    function addDatasetEditorRow() {
        datasetEditorState.datasets = normalizeDatasetEditorRows(datasetEditorState.datasets);
        datasetEditorState.datasets.push({
            source_dir: '',
            image_dir: '',
            cache_dir: '',
            num_repeats: 1,
        });
        datasetEditorState.dirty = true;
        renderDatasetEditor();
        updateTomlDirtyState();
    }

    function removeDatasetEditorRow(index) {
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        if (rows.length <= 1) return;
        rows.splice(index, 1);
        datasetEditorState.datasets = rows;
        datasetEditorState.dirty = true;
        renderDatasetEditor();
        updateTomlDirtyState();
        updateStepEstimatePanel();
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

    async function applySuggestedDataDirs() {
        if (datasetEditorState.datasets.length) {
            const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
            const sourceDirs = rows.map((row) => row.source_dir.trim());
            if (!sourceDirs.some(Boolean)) {
                alert('请先填写至少一个原始数据集路径');
                return;
            }
            try {
                const result = await api('/api/config/datasets/suggest', {
                    method: 'POST',
                    body: JSON.stringify({ source_dirs: sourceDirs }),
                });
                if (!result.ok) {
                    alert(result.error || '生成路径失败');
                    return;
                }
                const suggestions = result.datasets || [];
                let cursor = 0;
                datasetEditorState.datasets = rows.map((row) => {
                    if (!row.source_dir.trim()) return row;
                    const next = suggestions[cursor++] || {};
                    return {
                        ...row,
                        source_dir: next.source_dir || row.source_dir,
                        image_dir: next.image_dir || row.image_dir,
                        cache_dir: next.cache_dir || row.cache_dir,
                    };
                });
                datasetEditorState.dirty = true;
                syncDatasetEditorToCompatFields();
                renderDatasetEditor();
                handleFormFieldChange();
                updateTomlDirtyState();
                setTomlStatus('ok', '已根据原始数据集填入每组缩放图目录和 LoRA 缓存目录，请保存后再训练', { persist: true });
                return;
            } catch (e) {
                alert('生成路径失败: ' + e.message);
                return;
            }
        }

        const sourceInput = document.querySelector('#config-form .field-input[data-key="source_image_dir"]');
        const resizedInput = document.querySelector('#config-form .field-input[data-key="resized_image_dir"]');
        const cacheInput = document.querySelector('#config-form .field-input[data-key="lora_cache_dir"]');
        const source = sourceInput?.value?.trim() || '';
        if (!source) {
            alert('请先填写源图像目录 / source_image_dir');
            return;
        }
        try {
            const result = await api(`/api/config/data-dirs/suggest?source_image_dir=${encodeURIComponent(source)}`);
            if (!result.ok) {
                alert(result.error || '生成路径失败');
                return;
            }
            if (resizedInput) resizedInput.value = result.resized_image_dir || '';
            if (cacheInput) cacheInput.value = result.lora_cache_dir || '';
            handleFormFieldChange();
            setTomlStatus('ok', '已根据原始数据集填入缩放图像目录和 LoRA 缓存目录，请保存更新当前选中配置后再训练', { persist: true });
        } catch (e) {
            alert('生成路径失败: ' + e.message);
        }
    }

    function setCurrentTrainingSourceFromVariant(variant) {
        if (!variant) return;
        currentTrainingSource = {
            method: variant,
            methods_subdir: 'gui-methods',
            file: `configs/gui-methods/${variant}.toml`,
        };
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

    function confirmBeforeConfigSelectionChange(message) {
        if (!hasPendingConfigChanges(currentTomlFile)) return true;
        const ok = confirm(message);
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
        card.appendChild(heading);

        card.appendChild(choiceLine('说明', guide.summary));
        card.appendChild(choiceLine('取舍', guide.tradeoff));
        card.appendChild(choiceLine('推荐', guide.recommend, 'choice-recommend'));
        if (Array.isArray(guide.details) && guide.details.length) {
            const details = document.createElement('ul');
            details.className = 'choice-details';
            for (const detail of guide.details) {
                const item = document.createElement('li');
                item.textContent = detail;
                details.appendChild(item);
            }
            card.appendChild(details);
        }
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
        if (currentTrainingSource.methods_subdir === 'gui-methods') {
            return VARIANT_METHOD_FAMILY[currentTrainingSource.method] || val('method-select') || 'lora';
        }
        return val('method-select') || 'lora';
    }

    function inferMethodFromConfig(config) {
        if (!config || typeof config !== 'object') return '';
        const moduleName = String(config.network_module || '');
        if (isTruthy(config.use_lokr)) return 'lokr';
        if (isTruthy(config.use_easycontrol) || moduleName.includes('easycontrol')) return 'easycontrol';
        if (isTruthy(config.use_ip_adapter) || moduleName.includes('ip_adapter')) return 'ip_adapter';
        if (moduleName.includes('postfix')) return 'postfix';
        if (isTruthy(config.add_reft) || ('reft_dim' in config && Number(config.reft_dim) > 0)) return 'reft';
        if (
            isTruthy(config.use_hydra) ||
            isTruthy(config.use_sigma_router) ||
            String(config.use_moe_style || 'false') !== 'false' ||
            moduleName.includes('chimera') ||
            moduleName.includes('hydra')
        ) {
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
            config.output_dir ? `输出目录: ${config.output_dir}` : '',
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

        const main = document.createElement('div');
        main.className = 'field-main';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'field-name';
        nameSpan.textContent = formatFieldName(key);
        nameSpan.title = key;
        main.appendChild(nameSpan);

        const input = createFieldInput(key, value);
        input.dataset.key = key;
        input.dataset.valueType = fieldValueTypeForKey(key, value);
        input.addEventListener('input', handleFormFieldChange);
        input.addEventListener('change', handleFormFieldChange);
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
            const original = key in liveConfig ? liveConfig[key] : FORM_UI_DEFAULTS[key];
            liveConfig[key] = readFieldInputValue(input, original);
        }
        return liveConfig;
    }

    function formatFieldName(key) {
        const label = FIELD_LABEL_ZH[key];
        return label ? `${label} / ${key}` : key;
    }

    function createFieldInput(key, value) {
        if (key === 'sample_prompts') {
            const textarea = document.createElement('textarea');
            textarea.rows = 5;
            textarea.placeholder = '一行一个提示词，例如:\nmasterpiece, best quality, 1girl --w 1024 --h 1024 --d 42';
            textarea.value = value ?? '';
            textarea.className = 'field-input field-textarea sample-prompts-input';
            return textarea;
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
                input.min = '0';
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

    function isNumericField(key, value) {
        return typeof value === 'number' || [
            'max_train_epochs',
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
        return [
            'max_train_epochs',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'save_every_n_epochs',
            'checkpointing_epochs',
        ].includes(key) || Number.isInteger(value);
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

    function fieldValueType(value) {
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'boolean') return 'boolean';
        if (typeof value === 'number') return 'number';
        return 'string';
    }

    function fieldValueTypeForKey(key, value) {
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
        if (preferredFile && tomlFiles.includes(preferredFile)) {
            await loadTomlFile(preferredFile, { force: options.force === true });
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

    async function loadTomlFile(filePath, options = {}) {
        if (!options.force && !confirmDiscardTomlChanges('当前 TOML 有未保存修改，切换文件会丢失这些修改。是否继续？')) {
            return;
        }
        resetTomlDeleteConfirm();
        resetTomlSaveConfirm();
        const data = await api(`/api/config/raw?file=${encodeURIComponent(filePath)}`);
        if (data?.ok === false) {
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
            setTomlStatus('error', '请先选择一个配置文件，或使用“保存新配置”保存导入内容');
            return;
        }
        if (isTomlLocked(file)) {
            setTomlStatus('error', '该配置文件已锁定，请使用“保存新配置”创建可编辑配置');
            return;
        }
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(file);
        const directEditorSave = options.mode === 'editor';
        if (directEditorSave) {
            if (formDirty) {
                setTomlStatus('error', '左侧表单或数据集有未保存修改，请先使用“保存更新当前选中配置”处理后再直接保存 TOML');
                updateTomlActionState(file);
                return;
            }
            if (!editorDirty) {
                setTomlStatus('error', '直接编辑器没有未保存的 TOML 文本修改');
                return;
            }
            if (tomlSaveConfirmFile !== file) {
                armTomlSaveConfirm(file);
                return;
            }
            resetTomlSaveConfirm({ update: false });
            await saveRawTomlContent(file, document.getElementById('toml-editor').value, { reloadConfig: currentTrainingSource.file === file });
            return;
        }
        if (editorDirty && !formDirty && tomlSaveConfirmFile !== file) {
            armTomlSaveConfirm(file);
            return;
        }
        resetTomlSaveConfirm({ update: false });
        if (currentTrainingSource.file === file) {
            if (datasetEditorState.dirty) {
                const saved = await saveDatasetEditor();
                if (!saved) return;
            }
            const changedValues = collectChangedFormValues();
            if (Object.keys(changedValues).length > 0) {
                await saveFormPatchToToml(file, changedValues);
                return;
            }
        }
        const content = document.getElementById('toml-editor').value;
        await saveRawTomlContent(file, content, { reloadConfig: currentTrainingSource.file === file });
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
            } else {
                setTomlStatus('error', res.error || '保存失败');
            }
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function saveFormPatchToToml(file, values) {
        try {
            if (datasetEditorState.dirty) {
                const saved = await saveDatasetEditor();
                if (!saved) return;
            }
            const content = document.getElementById('toml-editor').value;
            const preparedValues = await prepareFormPatchValues(values);
            const res = await api('/api/config/raw', {
                method: 'PATCH',
                body: JSON.stringify({ file, values: preparedValues, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存失败');
                return;
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
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function saveDatasetEditor(options = {}) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || currentTomlFile || '';
        const targetContent = options.trainContent ?? (document.getElementById('toml-editor')?.value || '');
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
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
                    datasets: rows,
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

    function collectChangedFormValues() {
        const values = {};
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            const key = input.dataset.key;
            if (!key) return;
            if (datasetEditorState.dirty && DATASET_EDITOR_COMPAT_FIELDS.has(key)) return;
            if (key === 'sample_prompts') {
                const nextPrompts = readFieldInputValue(input, samplePromptsContent);
                if (nextPrompts !== samplePromptsContent) {
                    values[key] = nextPrompts;
                }
                return;
            }
            const hasOriginal = key in currentConfig;
            const original = hasOriginal ? currentConfig[key] : FORM_UI_DEFAULTS[key];
            const next = readFieldInputValue(input, original);
            if (!hasOriginal && shouldSkipUiDefaultField(key, next)) return;
            if (!valuesEqual(next, original)) {
                values[key] = next;
            }
        });
        if (values.use_lokr === true && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        return values;
    }

    async function prepareFormPatchValues(values) {
        const nextValues = { ...values };
        if ('sample_prompts' in nextValues) {
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

    function shouldSkipUiDefaultField(key, value) {
        if (!(key in FORM_UI_DEFAULTS)) return false;
        if (OPTIONAL_EMPTY_FIELDS.has(key) && value === '') return true;
        return valuesEqual(value, FORM_UI_DEFAULTS[key]);
    }

    function readFieldInputValue(input, originalValue) {
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
        return JSON.stringify(a) === JSON.stringify(b);
    }

    function normalizeMultilineText(value) {
        return String(value || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .join('\n');
    }

    function currentSamplePromptText(config) {
        const raw = config.sample_prompts;
        if (typeof raw === 'string' && raw.endsWith('.txt')) {
            samplePromptsPath = raw;
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        return typeof raw === 'string' ? raw : FORM_UI_DEFAULTS.sample_prompts;
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
                input.value = samplePromptsContent;
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

    function importTomlFile() {
        if (!confirmDiscardTomlChanges('当前 TOML 有未保存修改，导入会覆盖编辑器内容。是否继续？')) {
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

        const file = normalizeTomlSaveAsPath(target);
        if (!file) {
            setTomlStatus('error', '保存新配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === currentFile) {
            setTomlStatus('error', '保存新配置失败: 新配置不能和当前选中文件同名');
            return;
        }
        if (tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        try {
            const content = editor.value;
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
            if (datasetEditorState.dirty) {
                const savedDataset = await saveDatasetEditor({
                    trainFile: file,
                    trainContent: content,
                    reloadList: false,
                    preferExistingDatasetConfig: false,
                });
                if (!savedDataset) {
                    setTomlStatus('error', `新配置已创建: ${file}，但数据集配置保存失败，请修正后再次保存更新当前选中配置`, { persist: true });
                    await loadTomlFileList(file, { force: true });
                    updateTomlDirtyState();
                    return;
                }
                tomlSavedContent = document.getElementById('toml-editor').value;
            }
            await loadTomlFileList(file);
            await applyTomlToConfig({ silent: true });
            updateTomlDirtyState();
            setTomlStatus('ok', `已保存新配置: ${file}`);
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function showTomlSaveAsDialog(currentFile) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-save-as-dialog-body';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const labelText = document.createElement('span');
        labelText.textContent = '新配置名称或 configs/ 路径';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = '';
        input.placeholder = '例如 rokkotsu_v2 或 configs/imported/rokkotsu_v2.toml';
        input.className = 'history-task-dialog-input';
        label.append(labelText, input);

        const hint = document.createElement('p');
        hint.className = 'toml-save-as-hint';
        hint.textContent = '只填写文件名时会保存到 configs/imported/；必须使用新名称，不会覆盖当前选中配置。';

        const current = document.createElement('p');
        current.className = 'toml-save-as-current';
        current.textContent = currentFile ? `当前选中配置: ${currentFile}` : '当前没有选中的配置文件，将使用编辑器内容创建新配置。';

        wrap.append(label, hint, current);

        return showHistoryTaskDialog({
            title: '保存新配置',
            description: '输入一个新名称，确认后创建新的 TOML 配置文件。',
            body: wrap,
            confirmText: '创建配置文件',
            onOpen: () => input.focus(),
            getValue: () => input.value,
        });
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
            group.system_locked
        );
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
        createBtn.addEventListener('click', createTomlGroup);
        toolbar.appendChild(createBtn);
        container.appendChild(toolbar);

        for (const group of groups) {
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
                    toggleTomlGroupLock(group);
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
        if (!group.renamable && !group.deletable) return null;
        const wrap = document.createElement('span');
        wrap.className = 'toml-group-actions';

        if (group.renamable) {
            wrap.appendChild(createTomlGroupActionButton('重命名', () => renameTomlGroup(group), {
                title: '重命名这个配置分组',
            }));
        }
        if (group.deletable) {
            wrap.appendChild(createTomlGroupActionButton('删除分组', () => deleteTomlGroup(group), {
                title: (group.files || []).length > 0
                    ? '只删除分组，不删除里面的 TOML 文件；文件会回到默认分组'
                    : '删除这个自定义分组',
                danger: true,
            }));
        }
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
            if (!btn.disabled) handler();
        });
        return btn;
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
        btn.addEventListener('click', () => loadTomlFile(item.path));

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
        return datasetEditorState.dirty || Object.keys(collectChangedFormValues()).length > 0;
    }

    function hasPendingConfigChanges(filePath = currentTomlFile) {
        return isTomlDirty() || hasUnsavedFormChanges(filePath);
    }

    function confirmDiscardTomlChanges(message) {
        if (!hasPendingConfigChanges(currentTomlFile)) return true;
        return confirm(message);
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
                ? '该配置文件已锁定，请使用新名称保存新配置后编辑'
                : (dirty
                    ? (formDirty
                        ? '把左侧表单、数据集路径和采样提示词等修改写回当前选中的 TOML；保存后训练会使用这些新值。'
                        : '把直接编辑器里的 TOML 文本写回当前文件。')
                    : '当前配置没有未保存修改，不需要保存。');
        }
        updateTomlEditorPanelState(filePath);
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或保存新配置'
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
                ? '当前配置尚未保存，请先保存更新当前选中配置或保存新配置'
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
                ? '当前配置尚未保存，请先保存更新当前选中配置或保存新配置'
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
        editor.title = locked ? '该配置文件已锁定，只能导出或使用新名称保存新配置' : '';
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
                ? '该配置文件已锁定，请使用新名称保存新配置后编辑'
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
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或保存新配置，再加载选中配置');
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

        if (meta.methods_subdir === 'gui-methods') {
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
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或保存新配置，再调整锁定');
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
            ? `锁定 ${file}？锁定后不能直接保存，仍可使用新名称保存新配置。`
            : `解除 ${file} 的用户锁定？解除后可以直接编辑保存。`;
        if (!confirm(message)) return;

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
        if (!confirmDiscardTomlChanges('当前 TOML 有未保存修改，调整分组锁定前会丢失这些修改。是否继续？')) {
            return;
        }
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
            ? `锁定分组“${group.label || group.id}”？该分组内文件将不能直接保存，仍可使用新名称保存新配置。`
            : `解除分组“${group.label || group.id}”的锁定？解除后该分组内文件可恢复编辑保存。`;
        if (!confirm(message)) return;

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
        if (!confirmDiscardTomlChanges('当前 TOML 有未保存修改，重命名分组前会重新读取文件列表。是否继续？')) {
            return;
        }
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
        if (!confirmDiscardTomlChanges('当前 TOML 有未保存修改，调整分组顺序前会重新读取文件列表。是否继续？')) {
            return;
        }
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
        if (!confirmDiscardTomlChanges('当前 TOML 有未保存修改，删除分组前会重新读取文件列表。是否继续？')) {
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
        if (!file || !meta) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再删除');
            updateTomlActionState(file);
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
                setTomlStatus('error', res.error || '删除失败');
                return;
            }

            if (currentTrainingSource.file === file) {
                currentTrainingSource = {
                    method: val('variant-select') || 'lora',
                    methods_subdir: 'gui-methods',
                    file: `configs/gui-methods/${val('variant-select') || 'lora'}.toml`,
                };
            }
            currentTomlFile = '';
            tomlSavedContent = '';
            document.getElementById('toml-editor').value = '';
            document.getElementById('toml-current-file').textContent = '未选择配置';
            await loadTomlFileList('');
            updateTomlDirtyState();
            setTomlStatus('ok', `已删除配置: ${file}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function restoreSystemTomlPresets() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或保存新配置，再还原系统预设');
            updateTomlActionState(file);
            return;
        }

        const currentHint = meta?.restorable ? `\n当前文件 ${file} 也会一起还原。` : '';
        const ok = confirm(
            `即将还原全部系统预设：base、presets、methods、gui-methods。${currentHint}\n\n还原会覆盖系统预设文件，但会先自动备份当前内容。\n用户导入/副本和数据集配置不会被还原。\n\n是否继续？`
        );
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
        if (hasPendingConfigChanges(currentTomlFile)) {
            setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或保存新配置，再开始训练');
            updateTomlActionState(currentTomlFile);
            document.querySelector('[data-tab="config"]')?.click();
            return;
        }
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        const preflight = await runPreflight(variant, preset, methodsSubdir);
        if (!preflight) return;
        if (!preflight.ok) {
            const action = await showPreflightDialog(preflight, false);
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
            }
            return;
        }
        if ((preflight.summary?.warnings || 0) > 0) {
            const action = await showPreflightDialog(preflight, true);
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
                return;
            }
            if (action !== 'continue') return;
        }
        await startTrainingUnchecked(variant, preset, methodsSubdir);
    }

    async function runPreflight(variant, preset, methodsSubdir) {
        try {
            return await api('/api/training/preflight', {
                method: 'POST',
                body: JSON.stringify({ variant, preset, methods_subdir: methodsSubdir }),
            });
        } catch (e) {
            alert('预检测请求失败: ' + e.message);
            return null;
        }
    }

    async function startTrainingUnchecked(variant, preset, methodsSubdir) {
        try {
            const res = await api('/api/training/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    extra_args: [],
                    gpu_whitelist: selectedGpuPayload(),
                }),
            });
            if (res.ok) {
                document.querySelector('[data-tab="training"]').click();
            } else {
                if (res.preflight) {
                    const action = await showPreflightDialog(res.preflight, false);
                    if (action === 'preprocess') {
                        await startPreprocessFromPreflight(res.preflight);
                    }
                } else {
                    alert(res.error || '启动失败');
                }
            }
        } catch (e) {
            alert('请求失败: ' + e.message);
        }
    }

    function showPreflightDialog(result, allowContinue) {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog) {
            return Promise.resolve(allowContinue && confirm(preflightPlainText(result) + '\n\n是否继续训练?') ? 'continue' : 'cancel');
        }
        renderPreflightResult(result, allowContinue);
        dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => {
                resolve(dialog.returnValue || 'cancel');
            }, { once: true });
        });
    }

    function renderPreflightResult(result, allowContinue) {
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const errors = result.summary?.errors || 0;
        const warnings = result.summary?.warnings || 0;
        const checks = result.summary?.checks || 0;
        const canPreprocess = preflightCanStartPreprocess(result);

        summary.className = `preflight-summary ${errors ? 'error' : warnings ? 'warning' : 'ok'}`;
        if (errors && canPreprocess) {
            summary.textContent = `发现 ${errors} 个错误：当前数据需要先预处理。点击“开始预处理”后，完成再启动训练。`;
        } else {
            summary.textContent = errors
                ? `发现 ${errors} 个错误，已阻止训练。`
                : warnings
                    ? `通过基础检查，但有 ${warnings} 个警告。`
                    : `基础路径检查通过，共 ${checks} 项。`;
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
        continueBtn.textContent = warnings ? '忽略警告并继续训练' : '继续训练';
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
        try {
            const res = await api('/api/training/preprocess', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    extra_args: [],
                    train_after: true,
                    gpu_whitelist: selectedGpuPayload(),
                }),
            });
            if (!res.ok) {
                alert(res.error || '预处理启动失败');
                return;
            }
            document.querySelector('[data-tab="training"]').click();
            appendLog(`[状态] ${res.message || '预处理已启动'}`);
        } catch (e) {
            alert('预处理请求失败: ' + e.message);
        }
    }

    function preflightPlainText(result) {
        return (result.checks || [])
            .map((item) => `[${item.level}] ${item.key}: ${item.message}${item.path ? ` (${item.path})` : ''}`)
            .join('\n');
    }

    async function stopTraining() {
        if (!confirm('确定停止训练?')) return;
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

        stopBtn.disabled = msg.state !== 'running';

        if (msg.variant) document.getElementById('train-variant').textContent = msg.variant;
        if (msg.preset) document.getElementById('train-preset').textContent = msg.preset;

        if (msg.message) appendLog(`[状态] ${msg.message}`);

        if (msg.state === 'idle' || msg.state === 'error') {
            document.getElementById('progress-bar').style.width = '0%';
            trainingRuntime.quietHintShown = false;
            trainingRuntime.job = '';
        }
        refreshTrainingHealth();
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
        if (currentPreviewSource !== 'training' || (!selectedPreviewTaskId && !selectedPreviewGroup)) {
            if (requestSeq === previewWeightRequestSeq) {
                renderPreviewWeights({
                    ok: true,
                    weights: [],
                    message: currentPreviewSource === 'training'
                        ? '选择一个训练任务或训练分组后显示权重文件。'
                        : '权重文件只随训练任务显示。',
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
                params.set('include_archived', showArchivedHistory ? '1' : '0');
            } else {
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
            .filter((task) => task.job === 'training' && (showArchivedHistory || !task.archived))
            .sort((a, b) => Number(b.started_at || 0) - Number(a.started_at || 0));
        liveOption.textContent = trainingTasks.length
            ? `当前任务或默认目录 · ${trainingTasks.length} 个历史训练`
            : '当前任务或默认目录 · 暂无历史训练';

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
            label: `${parts[0]} / ${parts[1]} / ${parts[2] || 'default'}`,
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
            ? `目录: ${payload.directory}`
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

        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'btn btn-small preview-weight-copy';
        copy.textContent = '复制路径';
        copy.title = '复制这个权重文件的完整路径。';
        copy.addEventListener('click', () => copyPreviewWeightPath(item, copy));

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

        row.append(main, stats, copy);
        return row;
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
        const path = item.file || '';
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
        if (cfg.enabled) return `${base} 如果训练刚开始，可能还没到达采样频率。`;
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

    function createPreviewDetailBlock(label, value) {
        const block = document.createElement('div');
        block.className = 'preview-detail-block';
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('p');
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
            });
            if ((status.last_log_id || 0) > trainingRuntime.lastLogId) {
                await replayTrainingLogs();
            }
        } catch (e) { /* ignore */ }
    }

    async function loadTrainingHistoryList() {
        if (location.protocol === 'file:') return;
        try {
            const payload = await api('/api/training/history');
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
        const visibleTasks = historyTasks.filter((task) => showArchivedHistory || !task.archived);
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
        for (const group of groups) {
            const section = document.createElement('section');
            section.className = 'task-history-group';
            if (historyViewMode === 'config_group' && currentHistoryConfigGroup && configGroupKey(group) === configGroupKey(currentHistoryConfigGroup)) {
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
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || a.label.localeCompare(b.label, 'zh-CN');
            });
    }

    function historyConfigGroupFromTask(task) {
        const methodsSubdir = String(task.methods_subdir || '-');
        const variant = String(task.variant || '-');
        const preset = String(task.preset || 'default');
        return {
            key: [methodsSubdir, variant, preset].join('\u0001'),
            methods_subdir: methodsSubdir,
            variant,
            preset,
            label: `${methodsSubdir} / ${variant} / ${preset}`,
        };
    }

    function configGroupKey(group) {
        return [group.methods_subdir || '-', group.variant || '-', group.preset || 'default'].join('\u0001');
    }

    function createHistoryGroupHeading(group) {
        const heading = document.createElement('div');
        heading.className = 'task-history-group-title';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = group.tasks.filter((task) => task.job === 'preprocess').length;

        const title = document.createElement('span');
        title.textContent = [
            group.label,
            `${trainingCount} 次训练`,
            preprocessCount ? `${preprocessCount} 个预处理` : '',
        ].filter(Boolean).join(' · ');
        heading.appendChild(title);

        if (trainingCount) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'task-history-group-action';
            btn.textContent = '合并查看 Loss/日志';
            btn.title = '按这个配置文件分组，合并查看组内所有训练任务的日志和 Loss 曲线。';
            btn.addEventListener('click', () => loadConfigGroupTimeline(group));
            heading.appendChild(btn);
        }
        return heading;
    }

    function createHistoryTaskItem(task) {
        const card = document.createElement('article');
        card.className = 'task-history-item';
        if (historyViewMode === 'single' && task.id === viewingHistoryTaskId) card.classList.add('active');
        if (task.archived) card.classList.add('archived');

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'task-history-main';
        main.addEventListener('click', () => loadHistoryTask(task.id));

        const title = document.createElement('strong');
        title.textContent = task.name || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const meta = document.createElement('span');
        meta.textContent = [
            task.job === 'preprocess' ? '预处理' : '训练',
            historyStateLabel(task.state),
            task.started_at_text || task.id,
            task.archived ? '已归档' : '',
        ].filter(Boolean).join(' · ');
        const paths = document.createElement('em');
        paths.textContent = `目录: ${task.history_dir || task.id}`;
        const counts = document.createElement('em');
        counts.textContent = `${task.metric_count || 0} loss点 / ${task.log_count || 0} 日志`;
        main.append(title, meta, paths, counts);

        const actions = document.createElement('div');
        actions.className = 'task-history-actions';
        actions.append(
            createHistoryActionButton('重命名', () => renameHistoryTask(task)),
            createHistoryActionButton('分组', () => regroupHistoryTask(task)),
            createHistoryActionButton(task.archived ? '取消归档' : '归档', () => archiveHistoryTask(task)),
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
        const fallback = task.name || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
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
        const ok = await showHistoryTaskConfirmDialog({
            title: task.archived ? '取消归档任务' : '归档任务',
            description: historyTaskLabel(task),
            message: task.archived
                ? '取消归档后，这个任务会重新出现在默认任务列表中。'
                : '归档后默认会隐藏这个任务，可勾选“显示归档”再次查看。',
            confirmText: task.archived ? '取消归档' : '确认归档',
        });
        if (!ok) return;
        await updateHistoryTaskMeta(task.id, { archived: !task.archived });
    }

    async function deleteHistoryTask(task) {
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除历史任务',
            description: historyTaskLabel(task),
            message: '会删除该任务的日志、loss 指标和 TOML 快照。此操作不可撤销。',
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
        return task.name || `${task.methods_subdir || '-'} / ${task.variant || task.id}`;
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
            danger: options.danger,
            getValue: () => true,
        });
    }

    function showHistoryTaskDialog(options) {
        const dialog = document.getElementById('history-task-dialog');
        const title = document.getElementById('history-task-dialog-title');
        const desc = document.getElementById('history-task-dialog-desc');
        const body = document.getElementById('history-task-dialog-body');
        const confirmBtn = document.getElementById('history-task-dialog-confirm');
        if (!dialog || !title || !desc || !body || !confirmBtn) {
            return Promise.resolve(null);
        }

        title.textContent = options.title || '任务操作';
        desc.textContent = options.description || '';
        body.innerHTML = '';
        if (options.body) body.appendChild(options.body);
        confirmBtn.textContent = options.confirmText || '确认';
        confirmBtn.classList.toggle('btn-danger', Boolean(options.danger));
        confirmBtn.classList.toggle('btn-primary', !options.danger);

        return new Promise((resolve) => {
            const cleanup = () => {
                dialog.removeEventListener('close', handleClose);
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
            if (dialog.showModal) {
                dialog.showModal();
            } else {
                dialog.setAttribute('open', 'open');
            }
            requestAnimationFrame(() => options.onOpen?.());
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
            await loadConfigGroupTimeline(currentHistoryConfigGroup);
            return;
        }
        if (!viewingHistoryTaskId) return;
        await loadHistoryTask(viewingHistoryTaskId);
    }

    async function loadConfigGroupTimeline(group) {
        if (!group?.methods_subdir || !group?.variant) return;
        const query = new URLSearchParams({
            methods_subdir: group.methods_subdir,
            variant: group.variant,
            preset: group.preset || 'default',
            include_archived: showArchivedHistory ? '1' : '0',
        });
        try {
            const payload = await api(`/api/training/history/config-group/timeline?${query.toString()}`);
            if (!payload.ok) {
                alert(payload.error || '读取配置分组合并日志失败');
                return;
            }
            historyViewMode = 'config_group';
            viewingHistoryTaskId = '';
            currentHistoryConfigGroup = payload.group || group;
            currentHistoryTaskForResume = null;
            clearResumeOptions();
            renderTrainingHistoryList();
            renderConfigGroupTimeline(payload);
        } catch (e) {
            alert('读取配置分组合并日志失败: ' + e.message);
        }
    }

    function renderHistoryTask(payload) {
        const task = payload.task || {};
        currentHistoryTaskForResume = task;
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `历史任务: ${task.name || `${task.methods_subdir || '-'} / ${task.variant || '-'}`} · ${historyStateLabel(task.state)}`;
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
        lossChart?.setXLabel?.('step');
        lossChart?.setData(lossPoints.map((item) => ({
            step: item.step || 0,
            loss: item.loss,
            rawStep: item.step,
        })), { keepAll: true });
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        const configLr = readConfigNumber(payload.config_toml, 'learning_rate');
        const system = payload.system || [];
        const lastSystem = system[system.length - 1] || {};
        document.getElementById('metric-loss').textContent = lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : '-';
        document.getElementById('metric-lr').textContent = formatLr(lastValue(metrics, 'lr') ?? configLr);
        document.getElementById('metric-step').textContent = lastValue(metrics, 'step') ?? lastLossMetric.step ?? '-';
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
            bannerTitle.textContent = `配置分组合并: ${configGroupLabel(group)} · ${summary.task_count || 0} 次训练`;
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
        lossChart?.setXLabel?.('loss点');
        lossChart?.setData(lossPoints.map((item) => ({
            step: item.visual_step || item.step || 0,
            loss: item.loss,
            rawStep: item.step,
            sourceTaskLabel: item.source_task_label || '',
        })), { keepAll: true });
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        document.getElementById('metric-loss').textContent =
            lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : '-';
        document.getElementById('metric-lr').textContent = formatLr(lastValue(metrics, 'lr'));
        document.getElementById('metric-step').textContent =
            lastMetric.visual_step ?? lastMetric.step ?? lastLossMetric.visual_step ?? lastLossMetric.step ?? '-';

        const logs = payload.logs || [];
        const logEl = document.getElementById('log-output');
        logEl.textContent = logs.map(formatGroupTimelineLogRecord).join('\n');
        if (logEl.textContent) logEl.textContent += '\n';
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`配置分组合并 · ${logs.length} 行日志 · ${summary.loss_count || 0} Loss 点 · 已隐藏 ${summary.progress_count || 0} 条进度记录`, 'warning');

        const health = document.getElementById('training-health');
        health.className = 'training-health ok';
        health.textContent = [
            `已合并 ${summary.task_count || 0} 次训练`,
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
        const lines = [`# 配置分组合并: ${configGroupLabel(group)}`, ''];
        for (const segment of payload.segments || []) {
            const task = segment.task || {};
            lines.push(
                `任务 ${segment.index}: ${task.label || task.id || '-'}`,
                `  ID: ${task.id || '-'}`,
                `  状态: ${historyStateLabel(task.state)}`,
                `  时间: ${task.started_at_text || '-'} -> ${task.finished_at_text || '未结束'}`,
                `  输出目录: ${task.output_dir || '-'}`,
                `  日志: ${segment.log_count || 0} 行`,
                `  进度记录: ${segment.progress_count || 0} 条`,
                `  Loss/指标: ${segment.metric_count || 0} 条`,
                '',
            );
        }
        return lines.join('\n');
    }

    function renderConfigGroupPaths(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = [
            ['配置文件', configGroupLabel(group)],
            ['合并训练数', `${summary.task_count || 0}`],
            ['时间范围', `${summary.started_at_text || '-'} -> ${summary.finished_at_text || '未结束'}`],
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
        return `${group.methods_subdir || '-'} / ${group.variant || '-'} / ${group.preset || 'default'}`;
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
        const summary = document.getElementById('resume-checkpoint-summary');
        const status = document.getElementById('resume-training-status');
        if (!panel || !select || !btn || !summary || !status) return;

        const isTrainingTask = historyViewMode === 'single' && viewingHistoryTaskId && currentHistoryTaskForResume?.job === 'training';
        panel.hidden = !isTrainingTask;
        if (!isTrainingTask) {
            select.innerHTML = '<option value="">选择历史训练任务后读取</option>';
            select.disabled = true;
            btn.disabled = true;
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

    function setResumeStatus(text, state = '') {
        const el = document.getElementById('resume-training-status');
        if (!el) return;
        el.textContent = text || '';
        el.className = ['resume-status', state].filter(Boolean).join(' ');
    }

    function renderHistoryPaths(task) {
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = [
            ['历史目录', task.history_dir_abs || task.history_dir],
            ['源图像目录', task.source_image_dir],
            ['缩放图像目录', task.resized_image_dir],
            ['LoRA 缓存目录', task.lora_cache_dir],
            ['输出目录', task.output_dir],
            ['样张目录', task.sample_dir],
            ['日志文件', task.logs_path],
            ['指标文件', task.metrics_path],
            ['系统指标文件', task.system_path],
            ['TOML 快照', task.config_snapshot],
        ].filter(([, value]) => value);
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
            if (!confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换方法会重新加载表单并丢弃这些修改。是否继续？')) {
                return;
            }
            updateChoiceGuide();
            await loadVariants({ reset: true });
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('variant-select').addEventListener('change', async () => {
            if (!confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换变体会重新加载表单并丢弃这些修改。是否继续？')) {
                return;
            }
            setCurrentTrainingSourceFromVariant(val('variant-select'));
            updateChoiceGuide();
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('preset-select').addEventListener('change', async () => {
            if (!confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换预设会重新加载表单并丢弃这些修改。是否继续？')) {
                return;
            }
            updateChoiceGuide();
            await loadConfig();
            rememberSelectionSnapshot();
        });
        document.getElementById('btn-load-config').addEventListener('click', reloadCurrentConfig);
        document.getElementById('btn-start-from-config').addEventListener('click', startTraining);
        document.getElementById('btn-stop-training').addEventListener('click', stopTraining);
        document.getElementById('btn-apply-toml').addEventListener('click', applyTomlToConfig);
        document.getElementById('btn-move-toml-group').addEventListener('click', moveCurrentTomlToGroup);
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
        document.getElementById('btn-reload-toml').addEventListener('click', () => {
            const file = currentTomlFile || val('toml-file-select');
            if (file && confirmDiscardTomlChanges('当前 TOML 有未保存修改，重新读取文件会丢失这些修改。是否继续？')) {
                loadTomlFile(file, { force: true });
            }
        });
        document.getElementById('toml-file-select').addEventListener('change', (e) => {
            loadTomlFile(e.target.value);
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
        document.getElementById('resume-checkpoint-select').addEventListener('change', renderResumePanelState);
        document.getElementById('history-show-archived').addEventListener('change', (e) => {
            showArchivedHistory = e.target.checked;
            renderTrainingHistoryList();
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.addEventListener('click', () => setPreviewSource(btn.dataset.previewSource));
        });
        document.getElementById('btn-refresh-preview').addEventListener('click', loadPreviewImages);
        document.getElementById('btn-refresh-weights').addEventListener('click', loadPreviewWeights);
        document.getElementById('btn-sort-weights').addEventListener('click', togglePreviewWeightSort);
        document.getElementById('btn-save-preview-settings').addEventListener('click', savePreviewSettings);
        document.getElementById('btn-reset-preview-settings').addEventListener('click', resetPreviewSettings);
        document.getElementById('preview-training-task').addEventListener('change', (e) => changePreviewTask(e.target.value));
    }

    function installBeginnerTooltips() {
        const tips = {
            'method-select': '选择训练方法家族。新手通常选择 lora；LoKr、Hydra、ReFT 等属于进阶或实验方法。',
            'variant-select': '选择具体训练配置文件。它决定默认学习率、rank、缓存、方法开关等实际训练参数。',
            'preset-select': '选择预设覆盖项。default 最稳；低显存或快速试跑时再选择其他预设。',
            'gpu-picker-toggle': '选择训练时允许使用的 GPU 白名单。默认“全部 GPU”表示不限制；选择会保存在本机浏览器。',
            'btn-load-config': '重新读取当前方法、变体和预设合并后的配置；不会启动训练，也不会保存当前未保存修改。',
            'btn-start-from-config': '先做训练前预检测，再用当前左侧表单/当前训练配置启动训练。若数据缺缓存，会提示先预处理。',
            'btn-stop-training': '停止当前正在运行的训练或预处理任务；已经写出的日志、样张和权重文件会保留。',
            'btn-refresh-resume-options': '重新扫描这个历史任务输出目录里的训练状态目录，例如 output_name-checkpoint-state。',
            'btn-resume-training': '从选中的训练状态目录恢复训练。它不是加载普通权重热启动，而是恢复 optimizer、scheduler、随机状态和步数。',
            'resume-checkpoint-select': '只有包含 train_state.json 的状态目录才会出现在这里；普通 safetensors 权重不能完整恢复训练进度。',
            'btn-import-toml': '从本地选择 TOML 文件导入到 WebUI 管理区；导入后仍需要加载或保存为配置才能训练。',
            'btn-export-toml': '下载/导出当前选中的 TOML 内容，适合训练前备份或分享配置。',
            'btn-save-toml': '保存左侧表单或当前 TOML 的未保存修改。保存后，“开始训练”才会使用这些新值。',
            'btn-toggle-toml-editor': '展开二级界面，查看、复制或直接编辑当前 TOML；适合批量改字段。',
            'btn-save-as-toml': '把当前配置另存为新 TOML，适合从系统预设复制出自己的可编辑版本。',
            'btn-apply-toml': '把右侧选中的 TOML 加载到左侧表单，并设为当前训练入口。',
            'btn-move-toml-group': '移动右侧配置文件所在分组，只改变列表归类，不改变 TOML 内容。',
            'btn-reload-toml': '从磁盘重新读取当前 TOML；会丢弃未保存编辑。',
            'btn-copy-toml': '复制当前编辑器里的 TOML 内容，方便备份或排查。',
            'btn-save-toml-direct': '保存直接编辑器里的 TOML 文本。需要连续点击两次确认写入。',
            'btn-lock-toml': '锁定当前配置文件，防止误改；系统预设或分组锁定的文件可能无法手动解锁。',
            'btn-delete-toml': '删除当前选中的可编辑 TOML。需要二次确认；不会删除训练输出目录。',
            'btn-restore-system-toml': '把项目内置系统预设恢复到项目版本。会自动备份，但不影响用户导入配置。',
            'btn-live-training': '从历史任务视图回到当前正在监控的训练/预处理状态。',
            'btn-refresh-history': '重新读取训练任务历史列表，包括日志、loss、输出目录和样张目录记录。',
            'btn-refresh-history-view': '重新读取当前正在查看的历史日志和 Loss；适合训练仍在写日志时手动更新。',
            'btn-merge-config-group-history': '按同一个配置文件分组合并查看训练日志和 Loss 曲线；预处理任务不会参与合并。',
            'btn-clear-log': '清空当前页面显示的日志文本；不会删除磁盘上的历史日志。',
            'history-show-archived': '显示已归档任务。归档只是隐藏列表项，不会删除训练记录。',
            'btn-refresh-preview': '重新扫描当前预览来源目录，读取最新生成的样张图片。',
            'btn-refresh-weights': '重新扫描选中训练任务的权重文件，显示保存轮次和步数。',
            'btn-sort-weights': '按 Epoch/Step 切换权重文件正序或反序排列。',
            'btn-save-preview-settings': '保存预览图路径设置，只影响预览图页面读取目录，不会改训练配置。',
            'btn-reset-preview-settings': '恢复预览图目录默认值，例如训练样张默认 output/ckpt/sample。',
            'preview-training-task': '选择一个历史训练任务后，预览图会读取该任务记录的 sample_dir，而不是只看默认目录。',
            'preview-training-dir': '未选择历史任务时，训练中采样预览默认从这个目录读取。',
            'preview-inference-dir': '推理预览来源目录，通常存放手动推理或测试生成的图片。',
            'preview-custom-dir': '自定义预览目录。填任意项目内或绝对路径后，可在“自定义路径”来源中查看图片。',
        };
        for (const [id, title] of Object.entries(tips)) {
            const el = document.getElementById(id);
            if (el && !el.title) el.title = title;
        }
        document.querySelectorAll('.tab-btn').forEach((btn) => {
            const labels = {
                config: '配置页：选择方法/变体/预设，编辑训练参数、数据集路径和 TOML 文件。',
                training: '训练页：查看当前任务、历史任务、loss 曲线、日志和显存状态。',
                preview: '预览图页：查看训练中样张、推理输出或自定义目录图片。',
            };
            const key = btn.dataset.tab;
            if (labels[key]) btn.title = labels[key];
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            const labels = {
                training: '读取训练任务的 sample_dir 或训练样张默认目录。',
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

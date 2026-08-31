/* Section grouping definitions for config sub-pages.
 * Maps sub-page IDs to arrays of section objects, each with an
 * eyebrow (small label), title, description, and list of field keys.
 * Sub-pages without an entry here fall back to a flat field list.
 */

import { FORM_CATEGORY_DEFS, FORM_SECTION_DEFS } from '../../config/catalog/form-layout.js?v=dragon-ui-20260830v2';

const TRAINING_CATEGORY_SECTION_GROUPS = Object.fromEntries(FORM_CATEGORY_DEFS.map((category) => [
    category.id,
    category.sections.map((title) => FORM_SECTION_DEFS.find((section) => section.title === title))
        .filter(Boolean)
        .map((section) => ({
            eyebrow: category.title,
            title: section.title,
            desc: section.description || '',
            keys: [...section.keys],
            collapsible: true,
            open: section.open !== false || (category.id === 'advanced' && section.title === '缓存与预处理'),
        })),
]));

export const SECTION_GROUPS = {
    ...TRAINING_CATEGORY_SECTION_GROUPS,
    'data-behavior': [
        {
            eyebrow: '标注',
            title: '标注变体与丢弃率',
            desc: '控制标题变体、遮罩损失和标题随机丢弃行为。',
            keys: ['use_shuffled_caption_variants', 'masked_loss', 'caption_dropout_rate'],
        },
        {
            eyebrow: '筛选规则',
            title: '路径与分辨率筛选',
            desc: '在预处理前过滤不符合路径、分辨率或像素阈值的源图。',
            keys: ['path_pattern', 'drop_lowres_images', 'min_pixels'],
        },
    ],
    'dataset-filter': [
        {
            eyebrow: '筛选规则',
            title: '路径与分辨率筛选',
            desc: '在预处理前过滤不符合路径、分辨率或像素阈值的源图。',
            keys: ['path_pattern', 'drop_lowres_images', 'min_pixels'],
        },
    ],
    'base-models': [
        {
            eyebrow: '底模',
            title: '扩散模型路径',
            desc: '训练所用的底模权重文件路径。',
            keys: ['pretrained_model_name_or_path'],
        },
        {
            eyebrow: '文本编码器',
            title: 'Qwen3 模型路径',
            desc: '文本编码器模型路径，用于 caption tokenize 和 TE 缓存。',
            keys: ['qwen3'],
        },
        {
            eyebrow: 'VAE',
            title: 'VAE 模型路径',
            desc: '图像编解码器路径，用于 latent cache 和采样出图。',
            keys: ['vae'],
        },
    ],
    'output-save': [
        {
            eyebrow: '输出',
            title: '输出命名',
            desc: '设置训练产物的输出名称。',
            keys: ['output_name'],
        },
        {
            eyebrow: '保存策略',
            title: '权重与续训点周期',
            desc: '设置模型权重、训练状态的保存频率和保留数量。',
            keys: ['save_every_n_epochs', 'save_last_n_epochs', 'checkpointing_epochs', 'checkpointing_last_n_epochs', 'network_train_unet_only'],
        },
        {
            eyebrow: '保存格式',
            title: '模型保存与精度',
            desc: '选择模型保存格式和保存精度。',
            keys: ['save_model_as', 'save_precision'],
        },
        {
            eyebrow: '训练参数',
            title: '正则化与诊断',
            desc: '权重衰减和诊断相关参数。',
            keys: ['weight_decay', 'use_cmmd', 'ip_diagnostics_epochs'],
        },
        {
            eyebrow: '日志',
            title: '日志记录',
            desc: '设置日志频率、输出目录和记录后端。',
            keys: ['log_every_n_steps', 'logging_dir', 'log_with'],
        },
        {
            eyebrow: '预览',
            title: '采样提示词与频率',
            desc: '配置训练中预览图的提示词、生成频率和首次采样。',
            keys: ['sample_prompts', 'sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first'],
        },
        {
            eyebrow: '预览',
            title: '采样器与种子',
            desc: '选择预览采样器并设置可复现的随机种子。',
            keys: ['sample_sampler', 'seed'],
        },
    ],
    'steps-volume': [
        {
            eyebrow: '训练步数',
            title: '最大训练步数',
            desc: '设置训练总步数上限，与 epochs 二选一。',
            keys: ['max_train_steps'],
        },
        {
            eyebrow: '批量',
            title: '批大小与梯度累积',
            desc: '控制每步处理的图片数量和梯度累积步数。',
            keys: ['train_batch_size', 'gradient_accumulation_steps'],
        },
        {
            eyebrow: '采样',
            title: '采样比例',
            desc: '训练数据中采样的比例。',
            keys: ['sample_ratio'],
        },
    ],
    'adapter-basics': [
        {
            eyebrow: '适配器容量',
            title: 'Rank 与 Alpha',
            desc: '设置适配器维度和缩放系数。',
            keys: ['network_dim', 'network_alpha', 'lora_adapter_kind', 'vera_projection_prng_key', 'vera_d_initial', 'vera_save_projection'],
        },
        {
            eyebrow: '高级',
            title: 'DoRA 与 LoKr',
            desc: 'DoRA 权重衰减和 LoKr 因子。',
            keys: ['dora_wd', 'lokr_factor'],
        },
        {
            eyebrow: '权重',
            title: '预训练权重加载',
            desc: '从已有权重恢复训练或初始化。',
            keys: ['network_weights', 'dim_from_weights'],
        },
        {
            eyebrow: 'LoKr',
            title: 'Kronecker 分解选项',
            desc: '当适配器类型为 LoKr 时，控制分解方式和兼容行为。',
            keys: ['lokr_use_einsum', 'lokr_decompose_w2', 'lokr_full_factor', 'lokr_allow_legacy_dim'],
        },
        {
            eyebrow: 'LoKr',
            title: '分组与投影显存',
            desc: '调整 LoKr 分组大小和投影分块字节数。',
            keys: ['lokr_factor_group_size', 'lokr_project_chunk_bytes'],
        },
    ],
    'lokr': [
        {
            eyebrow: '分解',
            title: 'Kronecker 分解选项',
            desc: '控制 LoKr 的 einsum 分解和全因子模式。',
            keys: ['lokr_use_einsum', 'lokr_decompose_w2', 'lokr_full_factor', 'lokr_allow_legacy_dim'],
        },
        {
            eyebrow: '显存',
            title: '分组与投影显存',
            desc: '控制分组大小和投影分块字节数。',
            keys: ['lokr_factor_group_size', 'lokr_project_chunk_bytes'],
        },
    ],
    'train-sampling': [
        {
            eyebrow: '提示词',
            title: '采样提示词文件',
            desc: '训练中出图所用的提示词文件路径。',
            keys: ['sample_prompts'],
        },
        {
            eyebrow: '频率',
            title: '采样频率',
            desc: '按 epoch 或步数间隔生成样张。',
            keys: ['sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first'],
        },
        {
            eyebrow: '采样器',
            title: '采样器类型',
            desc: '选择训练中出图的采样器。',
            keys: ['sample_sampler'],
        },
        {
            eyebrow: '随机性',
            title: '预览随机种子',
            desc: '未在提示词行单独指定种子时，使用该种子生成稳定、可比较的预览图。',
            keys: ['seed'],
        },
    ],
    'timestep': [
        {
            eyebrow: '采样',
            title: '时间步采样方式',
            desc: '选择时间步采样策略。',
            keys: ['timestep_sampling'],
        },
        {
            eyebrow: '偏移',
            title: 'Flow Shift',
            desc: '离散 flow 匹配的偏移参数。',
            keys: ['discrete_flow_shift'],
        },
    ],
    'logging': [
        {
            eyebrow: '频率',
            title: '日志记录频率',
            desc: '每隔多少步记录一次日志。',
            keys: ['log_every_n_steps'],
        },
        {
            eyebrow: '后端',
            title: '日志目录与后端',
            desc: '日志输出目录和记录后端。',
            keys: ['logging_dir', 'log_with'],
        },
    ],
    'gradient-checkpoint': [
        {
            eyebrow: '基础',
            title: '梯度检查点开关',
            desc: '启用或关闭梯度检查点以节省显存。',
            keys: ['gradient_checkpointing', 'unsloth_offload_checkpointing'],
        },
        {
            eyebrow: '选择性',
            title: '选择性检查点',
            desc: '仅对部分 block 启用检查点，兼顾速度和显存。',
            keys: ['selective_checkpoint', 'selective_checkpoint_blocks'],
        },
    ],
    'precision': [
        {
            eyebrow: '精度',
            title: '精度偏好',
            desc: '选择训练精度。',
            keys: ['precision_preference'],
        },
        {
            eyebrow: '计算',
            title: '基础计算精度',
            desc: 'DiT base 层的计算精度。',
            keys: ['base_compute'],
        },
    ],
    'compile': [
        {
            eyebrow: '开关',
            title: '编译启用',
            desc: '启用 torch.compile 加速训练。',
            keys: ['torch_compile'],
        },
        {
            eyebrow: '参数',
            title: '编译范围与模式',
            desc: '控制编译的 block 范围和 Inductor 模式。',
            keys: ['compile_block_scope', 'compile_inductor_mode', 'compile_dynamic_seq', 'use_custom_down_autograd', 'debug_finite_checks'],
        },
    ],
    'preprocess-batch': [
        {
            eyebrow: '批大小',
            title: '缓存批大小',
            desc: 'VAE 和文本编码器缓存的批处理大小。',
            keys: ['preprocess_vae_cache_batch_size', 'preprocess_text_cache_batch_size'],
        },
        {
            eyebrow: '预处理',
            title: '预处理配置',
            desc: '预处理显存分析和精度偏好。',
            keys: ['preprocess_memory_profile', 'preprocess_precision_preference'],
        },
    ],
    'cache-reuse': [
        {
            eyebrow: '复用',
            title: '缓存复用开关',
            desc: '控制是否复用已有的数据集、VAE、文本编码器和视觉特征缓存。',
            keys: ['use_vae_cache', 'use_text_cache', 'cache_llm_adapter_outputs', 'ip_features_cache_to_disk', 'reuse_dataset_cache_copy', 'reuse_vae_latents', 'reuse_text_encoder_cache'],
        },
        {
            eyebrow: '指纹',
            title: '指纹与重建',
            desc: '缓存指纹、检查行为和强制重建选项。',
            keys: ['cache_fingerprint_mode', 'skip_cache_check', 'force_rebuild_preprocess_cache'],
        },
    ],
    'reft': [
        {
            eyebrow: '启用',
            title: 'ReFT 开关',
            desc: '启用 ReFT 适配器。',
            keys: ['add_reft'],
        },
        {
            eyebrow: '参数',
            title: '维度与层范围',
            desc: '设置 ReFT 维度、alpha 和层范围。',
            keys: ['reft_dim', 'reft_alpha', 'reft_layers'],
        },
    ],
    'fei': [
        {
            eyebrow: '特征',
            title: 'FEI 特征维度',
            desc: 'FEI 路由器特征维度和 sigma 分桶参数。',
            keys: ['fei_feature_dim', 'fei_sigma_low_div'],
        },
        {
            eyebrow: 'FeRA',
            title: 'FeRA 损失',
            desc: 'FeRA 对比损失权重和频带数。',
            keys: ['fera_fecl_weight', 'fera_num_bands'],
        },
    ],
    'easycontrol': [
        {
            eyebrow: '开关与增强',
            title: 'EasyControl 条件行为',
            desc: '启用条件分支，并控制条件丢弃和噪声增强。',
            keys: ['use_easycontrol', 'easycontrol_drop_p', 'easycontrol_cond_noise_max'],
        },
        {
            eyebrow: '条件流',
            title: '条件流门控',
            desc: '条件流初始化和缩放系数。',
            keys: ['b_cond_init', 'cond_scale'],
        },
        {
            eyebrow: 'FFN',
            title: 'FFN LoRA 与 Token',
            desc: 'FFN LoRA 开关和条件 token 数量。',
            keys: ['apply_ffn_lora', 'cond_token_count'],
        },
    ],
    'spd': [
        {
            eyebrow: '路径',
            title: '模型与数据路径',
            desc: 'SPD 蒸馏的底模路径和数据目录。',
            keys: ['dit_path', 'data_dir'],
        },
        {
            eyebrow: '训练',
            title: '迭代与随机种子',
            desc: '迭代次数、随机种子和通道缩放。',
            keys: ['iterations', 'seed', 'channel_scaling_alpha'],
        },
    ],
    'output-format': [
        {
            eyebrow: '保存格式',
            title: '模型保存与精度',
            desc: '选择模型保存格式和保存精度。',
            keys: ['save_model_as', 'save_precision'],
        },
        {
            eyebrow: '训练参数',
            title: '正则化与诊断',
            desc: '权重衰减和诊断相关参数。',
            keys: ['weight_decay', 'use_cmmd', 'ip_diagnostics_epochs'],
        },
    ],
    'data-loading': [
        {
            eyebrow: '工作线程',
            title: 'DataLoader 配置',
            desc: 'DataLoader 工作线程数和持久化选项。',
            keys: ['max_data_loader_n_workers', 'persistent_data_loader_workers'],
        },
        {
            eyebrow: '内存',
            title: '锁定内存',
            desc: '是否启用锁定内存加速数据传输。',
            keys: ['dataloader_pin_memory'],
        },
    ],
    'vae-resource': [
        {
            eyebrow: '分块',
            title: 'VAE 分块大小',
            desc: 'VAE 编解码的分块大小。',
            keys: ['vae_chunk_size'],
        },
        {
            eyebrow: '缓存',
            title: 'VAE 缓存开关',
            desc: '是否禁用 VAE 缓存。',
            keys: ['vae_disable_cache'],
        },
    ],
    'optimizer': [
        {
            eyebrow: '训练时长',
            title: '轮数与最大步数',
            desc: '设置训练轮数或总步数上限。',
            keys: ['max_train_epochs', 'max_train_steps'],
        },
        {
            eyebrow: '批量',
            title: '批大小与梯度累积',
            desc: '控制每步图片数、梯度累积和数据采样比例。',
            keys: ['train_batch_size', 'gradient_accumulation_steps', 'sample_ratio'],
        },
        {
            eyebrow: '优化器',
            title: '优化器类型与参数',
            desc: '选择优化器类型并配置额外参数。',
            keys: ['optimizer_type', 'optimizer_args'],
        },
        {
            eyebrow: '学习率',
            title: '学习率与调度',
            desc: '设置学习率、调度器和预热步数。',
            keys: ['lr_scheduler', 'lr_warmup_steps', 'learning_rate'],
        },
        {
            eyebrow: '时间步',
            title: '采样方式与 Flow Shift',
            desc: '选择时间步采样策略并调整离散 flow 偏移。',
            keys: ['timestep_sampling', 'discrete_flow_shift'],
        },
    ],
    'lora-basics': [
        {
            eyebrow: 'LoRA 模块',
            title: '模块与正交化',
            desc: '选择网络模块类型和正交化选项。',
            keys: ['network_module', 'network_args', 'use_ortho', 'use_timestep_mask', 'channel_scaling_alpha'],
        },
        {
            eyebrow: 'Rank 与分层',
            title: '维度配置与层范围',
            desc: '设置 rank、alpha 缩放和起始层。',
            keys: ['min_rank', 'alpha_rank_scale', 'layer_start'],
        },
    ],
    'moe-routing': [
        {
            eyebrow: '混合专家',
            title: '路由风格与来源',
            desc: '配置 MoE 风格、每层路由和路由来源。',
            keys: ['use_moe_style', 'route_per_layer', 'router_source'],
        },
        {
            eyebrow: '专家',
            title: '专家数量与平衡',
            desc: '设置专家数量和平衡损失权重。',
            keys: ['num_experts', 'balance_loss_weight', 'balance_loss_warmup_ratio', 'network_router_lr_scale', 'router_targets'],
        },
        {
            eyebrow: 'Sigma 分桶',
            title: 'Sigma 分桶与特征维度',
            desc: '配置 sigma 特征维度和分桶参数。',
            keys: ['sigma_feature_dim', 'per_bucket_balance_weight', 'num_sigma_buckets', 'specialize_experts_by_sigma_buckets', 'sigma_bucket_boundaries'],
        },
        {
            eyebrow: '高级',
            title: '路由器隐藏层与温度',
            desc: '微调路由器隐藏维度和温度系数。',
            keys: ['router_hidden_dim', 'router_tau'],
        },
    ],
    'chimera-hydra': [
        {
            eyebrow: '双路由',
            title: '内容与频率双路由',
            desc: '启用 ChimeraHydra 并配置内容路由。',
            keys: ['use_chimera_hydra', 'num_experts_content', 'content_router_source', 'content_router_init_std', 'content_router_layer_norm', 'network_content_router_lr_scale', 'balance_w_content'],
        },
        {
            eyebrow: '频率',
            title: '频率路由专家',
            desc: '配置频率路由专家和平衡。',
            keys: ['num_experts_freq', 'freq_router_init_std', 'freq_router_layer_norm', 'network_freq_router_lr_scale', 'balance_w_freq'],
        },
    ],
    'soft-tokens': [
        {
            eyebrow: 'Token 结构',
            title: '层与分桶配置',
            desc: '设置 soft token 层数和 token 分桶。',
            keys: ['n_layers', 'n_t_buckets', 'init_std', 'splice_position'],
        },
        {
            eyebrow: '对比学习',
            title: '对比损失参数',
            desc: '配置对比学习的权重和采样策略。',
            keys: ['contrastive_weight', 'contrastive_k', 'contrastive_every_n', 'contrastive_negative_mode', 'contrastive_objective', 'contrastive_jaccard_alpha', 'contrastive_tau', 'contrastive_warmup_ratio'],
        },
        {
            eyebrow: '软排名',
            title: '软排名方法',
            desc: '设置软排名的平滑度和方法。',
            keys: ['softrank_softness', 'softrank_method', 'dual_bank'],
        },
    ],
    'ip-adapter': [
        {
            eyebrow: '开关与数据',
            title: 'IP 条件与身份配对',
            desc: '启用 IP-Adapter，并配置图像丢弃、身份配对和验证基线。',
            keys: ['use_ip_adapter', 'ip_image_drop_p', 'validation_baselines', 'ip_pair_mode', 'ip_pair_prob', 'ip_pair_min_level', 'ip_pair_caption_strip_p'],
        },
        {
            eyebrow: '编码器',
            title: '图像编码器配置',
            desc: '设置编码器类型和维度。',
            keys: ['encoder', 'encoder_dim', 'resampler_layers', 'resampler_heads'],
        },
        {
            eyebrow: 'IP 注入',
            title: 'IP 缩放与门控',
            desc: '配置 IP 缩放系数和门控学习率。',
            keys: ['ip_scale', 'gate_lr'],
        },
        {
            eyebrow: '视觉适配',
            title: 'PE-LoRA 适配',
            desc: 'PE-Core 视觉特征 LoRA 适配参数。',
            keys: ['pe_lora_enabled', 'pe_lora_rank', 'pe_lora_alpha', 'pe_lora_layer_from'],
        },
    ],
    'convrot': [
        {
            eyebrow: '旋转卷积',
            title: '分组与范围',
            desc: '设置旋转卷积的分组大小和作用范围。',
            keys: ['convrot_group_size', 'convrot_scope', 'convrot_hadamard'],
        },
        {
            eyebrow: '特征维度',
            title: '输入特征阈值',
            desc: '控制旋转卷积应用的输入特征维度。',
            keys: ['convrot_min_in_features', 'convrot_largest_in_features_only', 'convrot_large_min_in_features', 'convrot_large_layer_mode'],
        },
    ],
    'snr-weighting': [
        {
            eyebrow: '加权方案',
            title: 'SNR 加权选择',
            desc: '选择损失加权方案和参数。',
            keys: ['weighting_scheme', 'min_snr_gamma', 'p2_gamma', 'p2_k'],
        },
        {
            eyebrow: 'Sigmoid 加权',
            title: 'Sigmoid 参数',
            desc: 'Sigmoid 加权的尺度和偏移。',
            keys: ['sigmoid_scale', 'sigmoid_bias'],
        },
        {
            eyebrow: '速度',
            title: '速度方向损失',
            desc: '速度方向损失的权重系数。',
            keys: ['velocity_direction_loss_weight'],
        },
    ],
    'block-swap': [
        {
            eyebrow: '核心',
            title: '块交换基础',
            desc: '设置交换块数和传输精度。',
            keys: ['blocks_to_swap', 'block_swap_transfer_dtype', 'block_swap_restore_mode'],
        },
        {
            eyebrow: '高级',
            title: 'Profile 与评估',
            desc: '块交换 profile 和评估时的行为。',
            keys: ['block_swap_profile_jsonl', 'disable_block_swap_for_eval'],
        },
    ],
    'memory-probe': [
        {
            eyebrow: '显存监测',
            title: '显存探针',
            desc: '设置显存探针输出和步数。',
            keys: ['memory_probe_jsonl', 'memory_probe_max_steps'],
        },
        {
            eyebrow: '峰值监测',
            title: '峰值探针',
            desc: '峰值显存探针级别和步数。',
            keys: ['peak_probe_jsonl', 'peak_probe_max_steps', 'peak_probe_level'],
        },
    ],
    'no-dataset-reg': [
        {
            eyebrow: '先验保留',
            title: '先验保留权重',
            desc: '无数据集正则化的先验保留参数。',
            keys: ['prior_preservation_weight', 'blank_prompt_preservation'],
        },
        {
            eyebrow: '差异输出',
            title: '差异输出保留',
            desc: '差异输出保留触发器和类。',
            keys: ['diff_output_preservation_trigger', 'diff_output_preservation_class', 'inverted_mask_prior_weight'],
        },
    ],
    'attention-backend': [
        {
            eyebrow: '注意力',
            title: '注意力后端模式',
            desc: '选择注意力计算的后端。',
            keys: ['attn_mode', 'v100_flash_stability'],
        },
    ],
};

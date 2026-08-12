/* Navigation category map for the Dragon trainer UI.
 * Defines 5 broad categories with grouped sub-pages.
 * Each group has a header label and a list of fine-grained config sub-pages.
 * Structure: grouped flyout columns with section headers.
 */

export const DRAGON_NAV_CATEGORIES = [
    {
        id: 'training-config',
        label: '训练配置',
        layout: 'config',
        groups: [
            {
                header: '基础设置',
                items: [
                    { id: 'base-models', label: '基础模型', desc: '底模、文本编码器、VAE 路径', sections: ['基础模型路径'] },
                    { id: 'output-save', label: '输出与保存', desc: '输出命名、保存频率与检查点', sections: ['输出格式与训练范围'] },
                    { id: 'steps-volume', label: '步数与训练量', desc: '训练轮数、批大小与梯度累积', sections: ['步数与训练量'] },
                    { id: 'data-behavior', label: '数据与标注', desc: '标注变体、遮罩损失与丢弃率', sections: [] },
                    { id: 'dataset-filter', label: '数据筛选', desc: '路径匹配、分辨率筛选与像素阈值', sections: [] },
                ],
            },
            {
                header: '适配器',
                items: [
                    { id: 'adapter-basics', label: '适配器基础', desc: '秩、缩放系数、适配器类型与权重', sections: ['常用训练设置'] },
                    { id: 'lokr', label: 'LoKr 专用', desc: '克罗内克因子与显存优化', sections: ['LoKr 专用优化'] },
                ],
            },
            {
                header: '训练参数',
                items: [
                    { id: 'optimizer', label: '优化器与学习率', desc: '优化器、调度器、预热', sections: [] },
                    { id: 'timestep', label: '时间步采样', desc: '时间步采样与流偏移', sections: [] },
                    { id: 'logging', label: '日志设置', desc: '日志频率、目录、后端', sections: [] },
                ],
            },
            {
                header: '预览',
                items: [
                    { id: 'train-sampling', label: '训练中采样预览', desc: '样张提示词、频率、采样器', sections: ['训练中预览图'] },
                ],
            },
        ],
    },
    {
        id: 'memory-optimization',
        label: '显存与优化',
        layout: 'config',
        groups: [
            {
                header: '显存管理',
                items: [
                    { id: 'block-swap', label: '块交换', desc: '块交换、传输精度、恢复模式', sections: ['显存与速度优化'] },
                    { id: 'gradient-checkpoint', label: '梯度检查点', desc: '检查点模式与选择性策略', sections: [] },
                    { id: 'precision', label: '精度与计算', desc: '训练精度与基础计算类型', sections: [] },
                ],
            },
            {
                header: '编译加速',
                items: [
                    { id: 'compile', label: '编译优化', desc: '编译、Inductor 模式、范围', sections: [] },
                    { id: 'attention-backend', label: '注意力后端', desc: '注意力模式与可变长加速', sections: [] },
                ],
            },
            {
                header: '数据与缓存',
                items: [
                    { id: 'data-loading', label: '数据加载', desc: '数据线程数、锁定内存与持久化', sections: ['数据加载与 VAE 资源'] },
                    { id: 'vae-resource', label: 'VAE 资源', desc: 'VAE 分块大小、缓存开关', sections: [] },
                    { id: 'preprocess-batch', label: '预处理批大小', desc: 'VAE 与文本缓存批大小', sections: [] },
                    { id: 'cache-reuse', label: '缓存复用', desc: '复用缓存、指纹、强制重建', sections: [] },
                ],
            },
            {
                header: '实验参数',
                items: [
                    { id: 'convrot', label: 'ConvRot', desc: '旋转卷积实验参数', sections: [] },
                    { id: 'memory-probe', label: '显存探针', desc: '显存与峰值探针、输出', sections: [] },
                ],
            },
        ],
    },
    {
        id: 'advanced-methods',
        label: '高级方法',
        layout: 'config',
        groups: [
            {
                header: 'LoRA 系列',
                items: [
                    { id: 'lora-basics', label: 'LoRA 基础与正交', desc: '网络模块、正交化与秩缩放', sections: ['方法内部与实验架构'] },
                    { id: 'reft', label: 'ReFT', desc: 'ReFT 维度、缩放系数与层范围', sections: [] },
                    { id: 'moe-routing', label: 'MoE 路由', desc: 'MoE 风格、路由源、专家数', sections: [] },
                    { id: 'fei', label: 'FEI 特征', desc: 'FEI 维度、sigma 分桶', sections: [] },
                    { id: 'chimera-hydra', label: 'ChimeraHydra', desc: '内容/频率双路由专家', sections: [] },
                ],
            },
            {
                header: '条件注入',
                items: [
                    { id: 'ip-adapter', label: 'IP-Adapter', desc: '编码器、resampler、PE-LoRA', sections: ['IP-Adapter 高级参数'] },
                    { id: 'easycontrol', label: 'EasyControl', desc: '条件流门控、FFN LoRA', sections: ['EasyControl 高级参数'] },
                    { id: 'soft-tokens', label: 'Soft Tokens', desc: 'Soft Token 层与对比学习', sections: ['Soft Tokens 参数'] },
                ],
            },
            {
                header: '损失加权',
                items: [
                    { id: 'snr-weighting', label: 'SNR 与加权', desc: 'Sigmoid、Min-SNR、P2 与速度方向', sections: ['实验性功能'] },
                    { id: 'no-dataset-reg', label: '无数据集正则化', desc: '先验保留、DOP、反转遮罩', sections: ['无数据集正则化'] },
                ],
            },
            {
                header: '实验工具',
                items: [
                    { id: 'spd', label: 'SPD', desc: 'SPD 蒸馏实验', sections: ['SPD CLI 实验'] },
                    { id: 'output-format', label: '输出格式', desc: '保存格式、精度、权重衰减', sections: [] },
                ],
            },
        ],
    },
    {
        id: 'training-monitor',
        label: '训练监控',
        layout: 'quick',
        groups: [
            {
                header: '训练状态',
                elevated: true,
                items: [
                    { id: 'dashboard', label: '训练仪表盘', desc: '首页总览', isPage: 'dashboard' },
                    { id: 'live-training', label: '实时训练', desc: '损失曲线与步数进度', isPage: 'live-training' },
                ],
            },
            {
                header: '记录与任务',
                items: [
                    { id: 'history', label: '训练历史', desc: '历史任务列表与详情', isPage: 'history' },
                    { id: 'queue', label: '训练队列', desc: '排队任务管理', isPage: 'queue' },
                ],
            },
        ],
    },
    {
        id: 'model-system',
        label: '模型与系统',
        layout: 'tools',
        groups: [
            {
                header: '模型',
                items: [
                    { id: 'model-config', label: '全局模型配置', desc: '模型组件路径与模型族', isPage: 'model-config' },
                    { id: 'weight-analysis', label: '权重分析', desc: 'ΔW 结构分析', isPage: 'weight-analysis' },
                    { id: 'image-test', label: '生图测试', desc: '推理测试与验证', isPage: 'image-test' },
                ],
            },
            {
                header: '数据',
                items: [
                    { id: 'dataset-editor', label: '数据集蓝图', desc: '数据集预设编辑', isPage: 'dataset-editor' },
                    { id: 'preview-workspace', label: '预览工作区', desc: '训练样张预览', isPage: 'preview-workspace' },
                ],
            },
            {
                header: '系统',
                items: [
                    { id: 'environment', label: '环境检测', desc: '依赖与 GPU 检查', isPage: 'environment' },
                    { id: 'global-settings', label: '全局设置', desc: '输出根目录、配置路径', isPage: 'global-settings' },
                ],
            },
        ],
    },
];

/* Flatten groups for backward-compatible lookups */
export const DRAGON_ALL_SUB_ITEMS = DRAGON_NAV_CATEGORIES.flatMap((cat) =>
    (cat.groups || []).flatMap((group) =>
        group.items.map((sub) => ({ ...sub, categoryId: cat.id }))
    )
);

export const DRAGON_CONFIG_CATEGORY_IDS = new Set([
    'training-config',
    'memory-optimization',
    'advanced-methods',
]);

export function isConfigCategory(categoryId) {
    return DRAGON_CONFIG_CATEGORY_IDS.has(categoryId);
}

export function findSubItem(id) {
    return DRAGON_ALL_SUB_ITEMS.find((item) => item.id === id);
}

export function findCategory(categoryId) {
    return DRAGON_NAV_CATEGORIES.find((cat) => cat.id === categoryId);
}

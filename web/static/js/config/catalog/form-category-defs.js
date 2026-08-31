export const FORM_CATEGORY_DEFS = [
    {
        id: 'required',
        title: '必填',
        description: '模型路径和数据集。',
        sections: ['基础模型路径', '数据集设置'],
    },
    {
        id: 'common',
        title: '常用',
        description: '训练时长、学习率和输出。',
        sections: ['常用训练设置', '步数与训练量', 'LoKr 专用优化'],
    },
    {
        id: 'preview',
        title: '预览',
        description: '训练中样张。',
        sections: ['训练中预览图'],
    },
    {
        id: 'optimization',
        title: '优化',
        description: '显存、速度、诊断和编译。',
        sections: ['显存与速度优化', '数据加载与 VAE 资源', '实验性功能', '无数据集正则化'],
    },
    {
        id: 'advanced',
        title: '高级',
        description: '缓存、输出和方法实验参数。',
        advanced: true,
        sections: [
            '缓存与预处理',
            '更多数据集配置',
            'SPD CLI 实验',
            '输出格式与训练范围',
            '方法内部与实验架构',
            'Soft Tokens 参数',
            'IP-Adapter 高级参数',
            'EasyControl 高级参数',
            '其他高级选项',
        ],
    },
];

import { choiceHelp } from './help-builder.js?v=module-bootstrap-20260625-9';

export const METHOD_GUIDE_ZH = {
    lora: choiceHelp(
        'LoRA 家族',
        '最基础、兼容性最好的低秩微调方法，适合大多数角色、画风和概念训练。',
        '好处是稳定、可合并、推理链路最简单；代价是表达力主要靠 rank 和训练轮数。',
        '新手优先选它。'
    ),
    dora: choiceHelp(
        'DoRA',
        '在普通 LoRA 上额外训练输出通道幅度，把方向和幅度拆开学习。',
        '通常比同 rank 的普通 LoRA 更能贴近全量微调；代价是训练前向更重，且只适合普通 LoRA 路线。',
        '想增强普通 LoRA 拟合能力时选。'
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
    glora: choiceHelp(
        'GLoRA',
        '使用 Generalized LoRA 的 A/B 双低秩路径，导出 a1/a2/b1/b2 权重。',
        '表达方式不同于普通 LoRA；代价是推理、继续训练和合并路径都必须识别 GLoRA。',
        '明确需要 LyCORIS/GLoRA 格式时选。'
    ),
    vera: choiceHelp(
        'VeRA',
        '共享冻结随机投影，只训练向量缩放的极低参数 adapter。',
        '适合短期 rank/种子消融；代价是推理和继续训练需要 VeRA 兼容加载路径。',
        '想快速比较极低参数 adapter 时选。'
    ),
    loha: choiceHelp(
        'LoHa',
        '使用 Hadamard product 分解 LoRA 更新，输出 hada_w1/hada_w2 权重。',
        '适合需要 PEFT/LyCORIS LoHa 兼容权重的训练；代价是推理侧也需要 LoHa 兼容加载器。',
        '明确要 LoHa 格式时选，普通训练仍优先 LoRA。'
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

export const VARIANT_GUIDE_ZH = {
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
    glora: choiceHelp(
        'GLoRA',
        '输出 LyCORIS/ComfyUI 可识别的 a1/a2/b1/b2 权重，默认 rank=32。',
        'A 路径依赖底模 Linear 权重，不能当普通 LoRA up/down 无损转换。',
        '需要 GLoRA 格式或做兼容性实验时选；继续训练请使用同类 GLoRA 权重。'
    ),
    loha: choiceHelp(
        'LoHa',
        '输出 PEFT/LyCORIS 兼容的 hada_w1/hada_w2 权重，默认 rank=32。',
        '可合并进 DiT Linear 权重；推理或继续训练时需要 LoHa 权重识别支持。',
        '只有需要 LoHa 兼容格式时选。'
    ),
    vera: choiceHelp(
        'VeRA',
        '启用 VeRA，默认 rank 256，冻结共享随机投影 A/B，只训练缩放向量。',
        '参数量极低，适合短跑消融；保存时可选择是否写入随机投影矩阵。',
        '建议固定 projection_prng_key 后比较 rank、T-LoRA mask 和学习率。'
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

export const PRESET_GUIDE_ZH = {
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
    low_vram_blockswap: choiceHelp(
        '低显存交换块',
        '使用 block swap 作为主要省显存手段，默认交换 8 个 DiT 块。',
        '比 Unsloth 保命模式更轻；显存压力仍高时再提高交换块数。',
        '想手动对比交换块效率时选。'
    ),
    balanced_16g: choiceHelp(
        'Balanced 16G',
        '预测式 DiT block swap 默认交换 12 个 frozen base blocks，并保持 LoRA/router/trainable adapter 常驻 GPU。',
        '目标是省出约 4GB，同时把速度损失控制在较低范围；默认不开 Unsloth 和选择性重算。',
        '16GB 显卡优先选它；仍 OOM 时再试 mlp_only 选择性重算。'
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

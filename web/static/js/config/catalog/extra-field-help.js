import { help } from './help-builder.js?v=module-bootstrap-20260703-7';

export const EXTRA_FIELD_HELP_ZH = {
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
        '默认 10；小显存或快速对照时再下调。'
    ),
    n_t_buckets: help(
        'Soft Tokens 的时间桶数量。',
        '把 sigma/timestep 分桶，每桶学习一组时间偏移。',
        ['更多桶能表达更细的时间步差异。'],
        ['数据不足时许多桶训练很少。'],
        ['桶太多会增加参数并拖慢收敛。'],
        '默认 100；数据量较小时可先降低做对照。'
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
        '默认 end_of_sequence；需要复现旧实验时再确认原配置写法。'
    ),
    contrastive_weight: help(
        'Soft Tokens 对比目标权重。',
        '0 表示关闭额外对比前向；大于 0 会启用负样本目标。',
        ['可能增强提示词区分能力。'],
        ['每次触发会增加额外 DiT forward，训练变慢。'],
        ['权重过高会压过 FM 主损失。'],
        '默认 0；需要对比目标实验时再小步调高。'
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
        '默认 1；如果训练太慢，可增大间隔降低成本。'
    ),
    contrastive_negative_mode: help(
        'Soft Tokens 负样本来源。',
        'shuffled 为随机负样本，jaccard 会按 tag 重叠降权，hard 尝试同画师不同角色等困难负样本。',
        ['hard/jaccard 能提供更有针对性的区分信号。'],
        ['依赖 caption-index 质量。'],
        ['索引缺失或标签差时会退化或引入噪声。'],
        '默认 shuffled；有高质量 caption-index 后再尝试 hard/jaccard。'
    ),
    contrastive_objective: help(
        'Soft Tokens 对比目标函数。',
        'infonce 使用传统对比分类；softrank 使用可微排序目标。',
        ['softrank 更直接优化匹配 caption 的排序位置。'],
        ['softrank 依赖 softtorch 排序松弛，成本和调参复杂度更高。'],
        ['目标函数切换后历史经验不可直接套用。'],
        '默认 infonce；做排序目标实验时再切到 softrank。'
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
        ['softrank 目标不使用它。'],
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
    softrank_softness: help(
        'SoftRank 排序松弛的 softness。',
        '只在 contrastive_objective=softrank 且 contrastive_weight>0 时生效；数值越小排序越接近硬排序。',
        ['能调节排序目标的锐利程度。'],
        ['太小可能梯度不稳，太大排序信号会变钝。'],
        ['和 contrastive_k、softrank_method 一起影响显存和速度。'],
        '默认 0.1。'
    ),
    softrank_method: help(
        'SoftRank 使用的可微排序实现。',
        'neuralsort 是默认路径；softsort 可用于对照，但两者曲线和显存成本可能不同。',
        ['便于比较不同排序松弛的训练表现。'],
        ['切换后历史曲线不能直接横向比较。'],
        ['值写错会在训练启动时报错。'],
        '默认 neuralsort。'
    ),
    dual_bank: help(
        '是否启用正负两套 Soft Tokens bank。',
        '开启后会分别学习 ψ+ / ψ- token bank，用于 Soft Tokens 对比路径实验。',
        ['给正负分支更大的表达空间。'],
        ['参数量和调参复杂度增加。'],
        ['旧 checkpoint 不一定能和该结构直接对齐。'],
        '默认关闭。'
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

import { help } from './help-builder.js?v=module-bootstrap-20260704-2';

export const FIELD_HELP_TRAINING_ZH = {    learning_rate: help(
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
    save_last_n_epochs: help(
        "普通模型权重最多保留多少份。",
        "默认 -1 表示不清理旧权重，保存所有按“模型保存间隔”生成的 .safetensors。设置为 2、3 等正数时，只保留最近 N 个轮次权重；最终权重仍会单独保存。",
        ["可以保留多个阶段的 LoRA 权重，方便回看效果或挑选不过拟合的版本。"],
        ["数值越大，磁盘占用越多；-1 会一直累积权重文件。"],
        ["只影响普通权重文件，不影响完整续训点；续训点数量由“续训点保留数量”控制。"],
        "想省磁盘就填 2-5；想保留所有中间权重就保持 -1。"
    ),
    checkpointing_epochs: help(
        "每隔多少轮保存一次可恢复训练状态。",
        "它会写出完整续训点；重新开始同一配置时会自动从最新可用续训点继续。",
        ["中断后可恢复 adapter 权重、当前 step/epoch、optimizer、scheduler、随机状态等。"],
        ["续训点保留数量由下一个字段控制；checkpoint-state/ 体积可能比普通权重大。"],
        ["如果设得太大，中断时只能回到上一次续训点；如果只剩普通 .safetensors 而没有 checkpoint-state/，不能完整续训。"],
        "新手建议设 1。想减少中断损失时，让它小于或等于 save_every_n_epochs。"
    ),
    checkpointing_last_n_epochs: help(
        "自动续训点最多保留多少份。",
        "默认 1 表示只保留最近 1 个完整续训点；设置为 2、3 等正数会保留最近 N 个续训点。设置为 -1 表示不清理旧续训点，配合保存间隔 1 时每轮都可恢复。",
        ["保留多个续训点后，历史任务里可以选择不同轮数的 checkpoint-state 继续训练。"],
        ["数值越大，optimizer、scheduler 和随机状态文件占用的磁盘越多。"],
        ["设为 0 或其他非法值时，训练端会按 1 处理。"],
        "新手保持 1；想保留每轮完整现场时，把训练状态保存间隔设 1，并把这里设 -1。"
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
        "默认 AdamW；可选 CAME、Automagic、AdamW8bit、Lion、Prodigy、ProdigyPlusScheduleFree 等。AdamW8bit 使用 bitsandbytes，适合显存特别紧张时降低优化器状态占用。",
        ["不同优化器适合不同内存和收敛偏好。"],
        ["CAME 是内存友好的自适应优化器，但通常需要重新确认学习率。"],
        ["Automagic 属于实验优化器，会在优化器内部自适应调整实际学习率；建议从较小 learning_rate 开始。"],
        ["AdamW8bit 会省显存，但通常比 fused AdamW 慢；切换到 AdamW8bit 时不要保留 optimizer_args 里的 fused=True。"],
        ["ProdigyPlusScheduleFree 属于实验优化器；推荐 learning_rate=1.0、lr_scheduler=constant、max_grad_norm=0。"],
        ["随意切换会让历史经验不再适用。"],
        "先用 AdamW；只有显存不够时再试 AdamW8bit。"
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
        "constant 表示固定学习率；constant_with_warmup 表示先线性热身再固定；也可用 cosine、lulu_loss_gated_cosine 等调度。",
        ["调度可以让训练后期更平滑。"],
        ["多一个超参维度，需要搭配总步数理解。"],
        ["constant_with_warmup 只需要 lr_warmup_steps，热身结束后不会继续衰减。"],
        ["不合适的调度可能过早降低学习率；lulu 调度器的细调参数走 lr_scheduler_args。"],
        "默认 constant，先保持。"
    ),
    lr_warmup_steps: help(
        "学习率预热步数。",
        "可填整数步数，也可填 0 到 1 之间的比例；小于 1 时按总训练步数折算，例如 0.05 表示前 5% 的训练步数逐步升到目标学习率。",
        ["配合 constant_with_warmup 时，训练开头会更平滑。"],
        ["预热太短可能稳定效果不明显，太长会拖慢有效学习。"],
        ["短跑实验总步数很少时，过长的预热会让你只看到热身阶段。"],
        "先沿用当前配置；需要开头更稳时，再按总步数小幅调整。"
    ),
    timestep_sampling: help(
        "训练时如何采样去噪时间步。",
        "flow matching 训练推荐 sigmoid。",
        ["让训练更关注有效时间步区间。"],
        ["改变后会影响模型学到的噪声阶段分布。"],
        ["不匹配方法假设时可能降低质量。"],
        "推荐 sigmoid。"
    ),
    sigmoid_scale: help(
        "sigmoid/logit-normal 时间步采样的缩放系数。",
        "值越大，采样越集中在接近 0 或 1 的 sigma 两端；值越小，分布越靠中间。",
        ["可做 FasterDiT 类时间步分布消融。"],
        ["会改变模型看到的噪声阶段比例，和已有训练经验不可直接对比。"],
        ["不要和 sigmoid_bias、Min-SNR、P2 同时大幅调整。"],
        "短跑实验用 0.75 / 1.0 / 1.5 对照；正式训练先保持 1.0。"
    ),
    sigmoid_bias: help(
        "sigmoid/logit-normal 时间步采样的 logit 偏置。",
        "正值把采样推向高 sigma 的结构阶段；负值推向低 sigma 的细节阶段。",
        ["适合检查模型是否欠结构或欠细节。"],
        ["偏置过大可能让训练分布过窄，泛化变差。"],
        ["不同数据集最佳值可能不同。"],
        "短跑实验可试 -0.5 / 0 / 0.5；正式训练先保持 0。"
    ),
    weighting_scheme: help(
        "按 sigma/SNR 给基础 flow-matching loss 加权。",
        "uniform/none 基本等价于不额外加权；min_snr 和 p2 用 SNR 思路缓解时间步梯度冲突。",
        ["可减少低效时间步对训练的拖累。"],
        ["loss 数值会和 baseline 不再完全可比，要看样张和验证指标。"],
        ["配合时间步采样一起调时，变量太多，难判断原因。"],
        "先用 uniform 做基线；再单独试 min_snr=5 或 p2_gamma=0.5。"
    ),
    min_snr_gamma: help(
        "Min-SNR 权重方案的 gamma 上限。",
        "只在 weighting_scheme=min_snr 时生效；常见起点是 5。",
        ["能压低高 SNR/低 sigma 阶段的损失权重。"],
        ["gamma 越低干预越强，可能牺牲细节阶段学习。"],
        ["设了但 weighting_scheme 不是 min_snr 时不会生效。"],
        "短跑用 3 / 5 / 7 对照；默认先用 5。"
    ),
    p2_gamma: help(
        "P2 权重方案的指数强度。",
        "只在 weighting_scheme=p2 时生效；值越大，对高 SNR 阶段降权越强。",
        ["适合测试感知优先的训练权重。"],
        ["过强时可能低 sigma 细节不足。"],
        ["设了但 weighting_scheme 不是 p2 时不会生效。"],
        "短跑先试 0.5，再和 1.0 对照。"
    ),
    p2_k: help(
        "P2 权重方案的 SNR 偏移项。",
        "只在 weighting_scheme=p2 时生效；一般保持 1.0。",
        ["可以控制极端 SNR 区间的权重形状。"],
        ["不是优先调参项，乱改会增加实验维度。"],
        ["设了但 weighting_scheme 不是 p2 时不会生效。"],
        "保持 1.0。"
    ),
    velocity_direction_loss_weight: help(
        "FasterDiT 风格的速度方向辅助损失权重。",
        "在基础 velocity MSE 外，额外约束预测速度和目标速度的方向一致；验证 FM-MSE 不计入此项。",
        ["给每个像素位置增加方向监督，可能加快早期收敛。"],
        ["权重过大会压过 MSE 幅值学习，导致 loss/样张异常。"],
        ["属于实验项，不建议和多个 SNR/时间步改动一起开。"],
        "短跑先试 0.01 / 0.03 / 0.05；正式训练默认 0。"
    ),
    prior_preservation_weight: help(
        "无额外数据集的先验保留辅助损失权重。",
        "训练时用同一批 latent/noise/timestep，临时关闭 adapter 得到 base 预测，再让当前预测靠近这个先验。",
        ["当前支持 blank_prompt_preservation 或 DOP/class prompt 二选一。"],
        ["每步会多一次 DiT no-grad forward，训练会变慢并增加峰值显存压力。"],
        ["和有数据集正则化的 prior_loss_weight 是两条不同路线。"],
        "默认 0；设置为正数时必须同时选择空提示先验，或填写 DOP 类提示。短跑可先试 0.05 / 0.1。"
    ),
    blank_prompt_preservation: help(
        "使用空提示 T5(\"\") 作为先验保留条件。",
        "开启后复用预处理阶段生成的 max-padded unconditional crossattn sidecar，不在训练 loop 里重新跑 text encoder。",
        ["适合先验证“无额外正则化数据集”的基线方案。"],
        ["必须同时设置 prior_preservation_weight > 0 才会生效。"],
        ["不能和 DOP 类提示同时开启。"],
        "需要无数据集先验保留时开启；否则保持关闭。"
    ),
    diff_output_preservation_trigger: help(
        "DOP/class prompt 先验保留的触发词。",
        "填写 caption 中代表训练目标的那段文本，例如 sks、角色名、产品名或风格触发词。预处理文本缓存时，会把它替换成 DOP 类提示并额外写入 prior_crossattn_emb。",
        ["适合角色/物体 LoRA：例如 trigger=sks，class=woman；trigger=red_jacket，class=outfit。"],
        ["没有固定触发词时留空，prior caption 会直接使用 DOP 类提示本身。"],
        ["只填触发词不会启用 DOP；真正必填的是 DOP 类提示。"],
        ["修改后需要重新运行文本缓存/预处理，否则训练会找不到 prior_crossattn_emb。"],
        "有明确触发词时填 caption 里的原词；不确定时先检查训练 caption 里哪个词代表目标概念。"
    ),
    diff_output_preservation_class: help(
        "DOP/class prompt 先验保留的类提示。",
        "填写后启用 DOP 模式：class prompt 是 prior caption 的目标文本，训练用原 caption 跑带 adapter 的预测，用类提示 prior caption 跑关闭 adapter 的 base 预测。",
        ["必须同时设置 prior_preservation_weight > 0、use_text_cache=true、cache_llm_adapter_outputs=true。"],
        ["不能和 blank_prompt_preservation 同时使用。"],
        ["这里填比触发词更泛化的类别：人物/角色用 woman、man、girl、boy、character；物体用 object、outfit、weapon、vehicle；风格用 anime style、illustration style、painting style。"],
        ["不要填完整角色名、唯一专名或触发词；否则 prior caption 仍然太像训练目标，DOP 的正则化意义会变弱。"],
        ["每次修改该值后都要重新生成文本缓存。"],
        "选择规则：问自己“如果去掉专名，这批图大体属于什么类别？”就把那个类别写在这里。"
    ),
    inverted_mask_prior_weight: help(
        "只在遮罩外区域做先验保留的辅助损失权重。",
        "有 alpha mask 时，训练会用相同 prompt/latent/noise/timestep 临时关闭 adapter 跑 base 预测，并只约束 1-mask 区域。",
        ["适合局部编辑或带遮罩训练：目标区域继续学习，非目标区域尽量别被 LoRA 带偏。"],
        ["每步会多一次 DiT no-grad forward，训练会变慢。"],
        ["需要 use_text_cache=true、cache_llm_adapter_outputs=true；没有 mask 的 batch 此项为 0。"],
        "默认 0；局部训练短跑可先试 0.05 / 0.1。"
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
        "当前训练预览支持 euler、er_sde、lcm。旧配置里的 ddim、euler_a、dpmsolver++ 等 Diffusers 采样器名会按 euler 兼容处理。",
        ["会影响样张风格和速度。"],
        ["和最终推理采样器不同，样张观感会有差异。"],
        ["频繁切换会让训练过程对比不直观。"],
        "默认 euler；需要更随机的预览可试 er_sde，需要少步数蒸馏类预览可试 lcm。"
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
    block_swap_transfer_dtype: help(
        "块交换 frozen base 权重在 CPU 侧保存和传输时使用的精度。",
        "这里不是显卡训练精度开关。bf16 表示当前默认传输路径；即使显卡本身不支持 bf16 训练，也可以继续使用这个默认值。fp8_e4m3 会压缩 PCIe 传输，再在 GPU 上还原为执行精度。",
        ["可能降低 H2D 等待时间。"],
        ["fp8_e4m3 会引入 frozen base 权重量化误差。"],
        ["只影响 frozen base block，不会量化 LoRA、router 或优化器状态。"],
        ["旧卡没有 bf16 训练支持时，不需要因为这个字段改成 fp16；训练精度请看上面的“精度倾向”。"],
        "保持 bf16；只有做 FP8 交换传输消融时再改为 fp8_e4m3。"
    ),
    block_swap_restore_mode: help(
        "块交换 restore 阶段如何把 frozen base 权重恢复回 GPU。",
        "foreach 是当前默认正式路径；slab 会把同一 slot 的多个小 weight 恢复合并成更少的大 H2D，减少小 kernel / 小 copy 调度。",
        ["slab 在当前 LoKr 热测里能进一步降低 enqueue 和 H2D restore 开销。"],
        ["slab 会额外引入少量 GPU slab storage，占用一点显存余量。"],
        ["这仍属于更激进的 block swap 存储布局优化，推荐先在热测或对照实验中启用。"],
        "默认 foreach；做 block swap 性能验证时再试 slab。"
    ),
    selective_checkpoint: help(
        "只对部分 DiT 计算做 activation 重算。",
        "off 是最快默认；adapter_aware 整块重算大激活但缓存 LoRA/router 小中间值；peak_blocks_* 只对指定高峰 block 生效。",
        ["能在 block swap 仍然接近 OOM 时补出一些显存余量。"],
        ["会增加 backward 重算成本，速度会下降。"],
        ["不要和 full gradient_checkpointing 或 Unsloth offload 叠加。"],
        "LoKr/高 rank LoRA 可先试 adapter_aware 或 peak_blocks_adapter_aware；显存充足保持 off。"
    ),
    selective_checkpoint_blocks: help(
        "定点重算的 DiT block 编号列表。",
        "只在 peak_blocks_adapter_aware / peak_blocks_mlp_layer1 / peak_blocks_mlp 模式下使用。支持 25-27 或 24,25,26,27；留空/auto 表示最后 3 个 block。",
        ["可以只重算峰值最高的后段 block，减少速度损失。"],
        ["填错范围会启动失败，Anima 当前有效 block 是 0-27。"],
        ["对 off、adapter_aware、mlp_layer1_only、mlp_only、every_other 没有效果。"],
        "当前 LoKr 16G 消融优先填 25-27。"
    ),
    block_swap_profile_jsonl: help(
        "记录每个交换块的搬运和等待耗时。",
        "off 关闭；auto 在 WebUI 训练时写入当前任务目录的 block_swap_profile.jsonl。",
        ["能判断 block swap 是否真的卡在 H2D/D2H 或等待同步。"],
        ["会增加少量 I/O，长训通常只在调参阶段开启。"],
        ["显式路径写错会让 profile 无法落盘，但不应影响训练。"],
        "Balanced 16G 和 LoKr 16G 排查时用 auto；稳定后可关掉。"
    ),
    disable_block_swap_for_eval: help(
        "验证和训练中预览图阶段临时暂停块交换。",
        "开启后会先把交换到 CPU 的 DiT 块恢复到 GPU，评估结束后再恢复训练时的 block swap 布局。",
        ["评估/预览可能更快，适合训练需要换块但评估显存够用的机器。"],
        ["如果评估阶段放不下完整 DiT，会直接 OOM。"],
        ["只影响验证和预览，不改变训练 step 的块交换数量。"],
        "不确定就保持 false；只有 eval 明显慢且显存有余量时再打开。"
    ),
    memory_probe_jsonl: help(
        "记录训练级 CUDA 显存和 adapter/optimizer 摘要。",
        "off 关闭；auto 在 WebUI 训练时写入当前任务目录的 memory_probe.jsonl。",
        ["能定位 before_forward、backward、optimizer 等阶段的显存峰值。"],
        ["详细快照会带来少量开销，不建议长期每步记录。"],
        ["它是诊断工具，不会直接降低显存。"],
        "OOM 排查时设 auto，并把探针步数设为 1~3。"
    ),
    memory_probe_max_steps: help(
        "显存探针记录详细 step 快照的步数上限。",
        "0 表示每步记录；setup 摘要不受这个限制。",
        ["短跑定位时可以减少日志体积。"],
        ["设为 0 会产生大量 JSONL，不适合长训。"],
        ["步数太少可能错过后续才出现的峰值。"],
        "一般填 1、2 或 3；长训稳定后关闭显存探针。"
    ),
    peak_probe_jsonl: help(
        "记录更细粒度的 DiT block / LoKr 峰值显存事件。",
        "off 关闭；auto 在 WebUI 训练时写入当前任务目录的 peak_probe.jsonl。",
        ["能定位具体 block、MLP 或 LoKr delta apply 附近的峰值。"],
        ["ops/lokr/full 粒度会扰动 compiled graph，只适合短跑定位。"],
        ["它只做观测，不改变训练数学结果。"],
        "常规 50-step 定位用 block；只在短跑深查时改 ops、lokr 或 full。"
    ),
    peak_probe_max_steps: help(
        "峰值探针记录详细事件的步数上限。",
        "0 表示每步记录；峰值探针通常比普通显存探针更细。",
        ["控制 JSONL 体积和额外观测开销。"],
        ["设太大可能明显干扰速度统计。"],
        ["设太小可能只看到 warmup，不代表稳定阶段。"],
        "建议短跑填 1~2；需要跨 warmup 再填 5。"
    ),
    peak_probe_level: help(
        "峰值探针的事件粒度。",
        "block 只记录 DiT block 边界；ops 加 block 内 attention/MLP；lokr 加 LoKr delta；full 全开。",
        ["block 对 torch.compile 扰动最小。"],
        ["full 信息最多，但最容易影响速度和编译缓存。"],
        ["不要把 full 粒度的速度当成真实性能。"],
        "默认 block；只有确认 LoKr/MLP 峰值时再临时提高粒度。"
    ),
    preprocess_memory_profile: help(
        "预处理阶段的显存/速度预设。",
        "只影响 WebUI/任务链触发的 VAE latent cache 和文本缓存批大小；不改变训练 batch size。",
        ["low_vram 会降低预处理峰值显存。"],
        ["batch 越小，预处理越慢。"],
        ["手动填写 VAE 或文本缓存批大小时，会覆盖这个预设对应的值。"],
        "显存峰值卡在预处理时选 low_vram；正常机器保持 auto。"
    ),
    preprocess_vae_cache_batch_size: help(
        "VAE latent cache 的批大小。",
        "auto 保持历史默认 4；填 1 会逐张过 VAE，通常能明显压低预处理峰值显存。",
        ["直接针对 Caching latents 阶段的显存峰值。"],
        ["值越小越慢，尤其是图片数量多时。"],
        ["这不是训练 batch size，不影响训练 step 的有效批量。"],
        "低显存优先填 1；显存够用保持 auto。"
    ),
    preprocess_text_cache_batch_size: help(
        "文本编码缓存的批大小。",
        "auto 保持历史默认 16；降低它可以减少 Qwen3 文本缓存阶段的显存峰值。",
        ["文本缓存阶段 OOM 时可单独调低。"],
        ["值越小，文本缓存越慢。"],
        ["不会改变 caption 内容或训练时的文本缓存读取方式。"],
        "只有文本缓存阶段显存高或 OOM 时再改；通常保持 auto。"
    ),
    preprocess_precision_preference: help(
        "预处理阶段优先采用哪种计算精度。",
        "只影响 WebUI/任务链触发的 VAE latent cache 和文本缓存计算精度；不会改训练时的 mixed_precision。",
        ["bf16 适合支持 bf16 的新卡，通常兼顾速度、显存和稳定性。"],
        ["fp16 适合旧卡无 bf16 支持时继续跑预处理，但数值稳定性通常不如 bf16。"],
        ["fp32 最稳，但更慢、也更占显存。"],
        "默认先用 bf16；旧卡不支持 bf16 时改成 fp16；只有排查精度问题时再考虑 fp32。"
    ),
    torch_compile: help(
        "是否让 PyTorch 先编译模型计算图再训练。",
        "开启后会使用上游新的 native flatten + compile_blocks 路径。第一次启动会花时间编译；编译完成后通常更快。遇到 torch.compile/inductor 报错时可以关闭。",
        ["长时间训练时可能提高速度。"],
        ["首次启动更慢，还会在缓存目录写入编译缓存。"],
        ["block swap、梯度检查点和不同显卡驱动组合仍可能触发编译问题。"],
        "新手保持默认；如果报 torch.compile/inductor/triton 相关错误，再关闭排查。"
    ),
    compile_block_scope: help(
        "block swap 开启时，哪些 DiT block 参与 torch.compile。",
        "resident 只编译常驻 GPU 的头部 block，交换到 CPU 的尾部 block 走 eager；all 会把交换 block 也编译，接近旧版全量编译行为。",
        ["all 有时能借助 Inductor 降低第一步前向激活峰值，适合 10GB 这类极限显存排查。"],
        ["all 可能让交换 block 因 CPU/GPU 权重迁移触发更多重编译，速度可能变慢或启动更久。"],
        ["只在 torch_compile=true 且 blocks_to_swap>0 时有明显意义。"],
        "默认 resident；遇到 block swap 后第一步仍 OOM，可临时改 all 做对照。"
    ),
    compile_inductor_mode: help(
        "Inductor 编译器优化模式。",
        "default 最稳；reduce-overhead 更偏减少运行开销。",
        ["可影响 compile 后性能。"],
        ["不同环境收益不稳定。"],
        ["模式不兼容时会导致编译失败。"],
        "保持变体默认。"
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
    precision_preference: help(
        "训练时优先采用哪种数值精度方案。",
        "bf16 是默认推荐；fp16 表示 fp16/32 混合精度；fp32 表示关闭混合精度、全程使用 fp32。",
        ["bf16 通常兼顾显存、速度和稳定性，适合大多数新卡。"],
        ["fp16 会进一步压显存，但数值稳定性通常不如 bf16。"],
        ["fp32 最稳、最直观，但显存占用最高，速度也往往更慢。"],
        "默认先用 bf16；旧卡不支持 bf16 时试 fp16；只有排查数值问题或显存充足时再考虑 fp32。"
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
    use_vae_cache: help(
        "使用 VAE latent 缓存。",
        "开启后训练读取预处理生成的 latent 缓存。",
        ["避免每轮重复编码图像。"],
        ["需要磁盘保存缓存。"],
        ["图像或预处理参数变化后必须重建缓存。"],
        "推荐 true。"
    ),
    use_text_cache: help(
        "使用文本编码器输出缓存。",
        "开启后训练读取预处理生成的文本缓存。",
        ["编码后可释放文本编码器，给 DiT 腾显存。"],
        ["caption 改动后需要重新缓存。"],
        ["缓存和 caption 不一致会导致训练内容不对。"],
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
};

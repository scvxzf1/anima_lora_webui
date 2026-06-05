import { help } from './help-builder.js?v=module-bootstrap-20260604-8';

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
        "constant 表示固定学习率；也可用 cosine、lulu_loss_gated_cosine 等调度。",
        ["调度可以让训练后期更平滑。"],
        ["多一个超参维度，需要搭配总步数理解。"],
        ["不合适的调度可能过早降低学习率；lulu 调度器的细调参数走 lr_scheduler_args。"],
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
    block_swap_transfer_dtype: help(
        "块交换 frozen base 权重在 CPU 侧保存和传输时使用的精度。",
        "bf16 是当前稳定路径；fp8_e4m3 会压缩 PCIe 传输，再在 GPU 上还原为执行精度。",
        ["可能降低 H2D 等待时间。"],
        ["fp8_e4m3 会引入 frozen base 权重量化误差。"],
        ["只影响 frozen base block，不会量化 LoRA、router 或优化器状态。"],
        "保持 bf16；只有做 FP8 交换传输消融时再改为 fp8_e4m3。"
    ),
    selective_checkpoint: help(
        "只对部分 DiT 计算做 activation 重算。",
        "off 是最快默认；mlp_layer1_only 只重算每个 block 的 MLP 第一层；peak_blocks_* 只对指定高峰 block 生效。",
        ["能在 block swap 仍然接近 OOM 时补出一些显存余量。"],
        ["会增加 backward 重算成本，速度会下降。"],
        ["不要和 full gradient_checkpointing 或 Unsloth offload 叠加。"],
        "LoKr 16G 优先试 mlp_layer1_only 或 peak_blocks_mlp_layer1；普通 LoRA 保持 off。"
    ),
    selective_checkpoint_blocks: help(
        "定点重算的 DiT block 编号列表。",
        "只在 peak_blocks_mlp_layer1 / peak_blocks_mlp 模式下使用。支持 25-27 或 24,25,26,27；留空/auto 表示最后 3 个 block。",
        ["可以只重算峰值最高的后段 block，减少速度损失。"],
        ["填错范围会启动失败，Anima 当前有效 block 是 0-27。"],
        ["对 off、mlp_layer1_only、mlp_only、every_other 没有效果。"],
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

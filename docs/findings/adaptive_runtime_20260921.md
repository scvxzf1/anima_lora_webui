# 自适应精度与 OOM 恢复：第一阶段证据

状态：实验，未完成最终验收。
日期：2026-09-21。
实现入口：[实验说明](../experimental/adaptive-runtime.md)。

## 目标与验收范围

最终目标是精度规划、自动块交换、OOM 自动调整重试联合工作，并完成
Anima / Z-Image / Krea-2 在 RTX 3080 20GB 与 Tesla T10 16GB 的真实热测试。
以下单层验证不是六组合热测试的替代品。

## 环境核验

- PyTorch：2.12.0+cu130；驱动：610.43.03。
- Tesla T10：SM 7.5，nvidia-smi 标称 16384MiB。
- 用户标识为 RTX 3080 20GB 的设备：驱动名称 NVIDIA Graphics Device，
  SM 8.6，nvidia-smi 标称 20480MiB。
- 系统内存约 62GiB，测试前 available 约 51GiB。
- PyTorch 默认设备顺序与 nvidia-smi 相反，实验使用 CUDA_VISIBLE_DEVICES
  的 GPU UUID，不依赖枚举序号。
- 未修改功耗、风扇、驱动配置，未终止既有任务，未使用提权操作。

## 已落地代码与验证

| 模块 | 实际行为 | 仍缺少 |
| --- | --- | --- |
| precision.py | FP32 参考与候选输出/梯度比较、非有限值拒绝、保留 RoPE 输入精度 | 真实前向激活采集、全模型计划搜索 |
| islands.py | 冻结 Linear 独立精度域、FP32 输出边界、保持 state-dict 键 | adapter / swap / compile 联合验证 |
| retry.py | 按失败阶段生成有界调整策略，授权后才改变 micro-batch | 训练状态和 checkpoint 接线 |
| process.py | 每次新进程、超时回收、主机内存保护、结构化失败 | 真实训练 worker 与完整恢复验证 |

验证命令：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_adaptive_runtime.py tests/test_adaptive_runtime_process.py -q
.venv/bin/ruff check library/training/adaptive_runtime tests/test_adaptive_runtime.py tests/test_adaptive_runtime_process.py bench/adaptive_runtime/probe_krea_block.py
```

第一轮结果：14 passed；Ruff 通过。子进程测试使用模拟的结构化 CUDA OOM，验证
新的 PID、重试边界、普通错误不重试、超时及主机内存保护，不是实际 CUDA OOM。

## 双卡真实权重单层探针

使用本仓 `models/diffusion_models/krea2_raw_bf16.safetensors`，文件大小
26,283,332,608 bytes；严格加载 `blocks.0`，冻结参数。
序列长 128，激活幅度 0.02 / 1.0 / 8.0，modulation 为合成输入，
RoPE 保留 FP32 且不计入输入梯度指标。TF32 关闭。
输出和输入梯度通过相同随机 cotangent 比较；未训练 LoRA，未执行 optimizer。

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> timeout 180 .venv/bin/python -m bench.adaptive_runtime.probe_krea_block \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --output output/adaptive-runtime-20260921/<device>-krea-block0-rope-fp32.json
```

| 设备 / 候选 | 全部输出与梯度有限 | 最坏输出 rel-L2 | 最坏输入梯度 rel-L2 | 最低梯度 cosine | 校准耗时 | peak allocated |
| --- | --- | --- | --- | --- | --- | --- |
| T10 / FP16 | 是 | 5.9815% | 27.9479% | 0.96027 | 14.684s | 1,853,614,592 bytes |
| 3080 / BF16 | 是 | 10.9273% | 399.0136% | 0.23380 | 11.114s | 1,853,614,592 bytes |

当前实验门槛为输出 rel-L2 <= 1%、梯度 rel-L2 <= 2%、梯度 cosine >= 0.999，
两者都被校准器退回 FP32。**这些门槛是实验配置，不是已经验证的质量标准。**
输入来自合成分布，且 FP32 与低精度可能使用不同 attention 内核；这些数字
不能外推为实际训练质量下降，更不能据此禁用现有 BF16 训练。
校准耗时包含复制、前后向和比较，不是训练步时或速度收益。

原始 JSON 保存在独立输出目录：

- `output/adaptive-runtime-20260921/t10-krea-block0-rope-fp32.json`
- `output/adaptive-runtime-20260921/3080-krea-block0-rope-fp32.json`

同目录下不带 `rope-fp32` 后缀的首轮记录将位置编码错误地纳入统一降精度和
梯度比较，已被上述记录取代，保留用于追溯，不作为正式数值判定依据。

## 第一阶段结束时的矩阵与下一步

| 模型 | RTX 3080 20GB | Tesla T10 16GB |
| --- | --- | --- |
| Anima | 未测：权重历史路径失效 | 未测：同左 |
| Z-Image | 未测：权重历史路径失效 | 未测：另需建立非 BF16 训练支持 |
| Krea-2 | 仅首层真实权重探针；全模型训练未测 | 仅首层真实权重探针；全模型训练未测 |

1. 定位 Anima 和 Z-Image 权重；若需下载，取得用户授权和版本选择。
2. 在生产前向边界采集真实激活，验证 Linear 精度域和整块 FP32 两类方案。
3. 连接精度计划、现有自动 swap 和训练 worker，保留 NF4 BF16 强制边界。
4. 完成真实 CUDA OOM 注入、参数调整后新进程重试、完整 checkpoint 恢复。
5. 六组合完成前后向、optimizer 更新、多步数值及步时/显存报告。

当前不具备可发布的自动混合精度训练能力，也不宣称任意模型无 OOM。

## 第二阶段：真实 CUDA OOM 恢复

补充权重检索发现 `/home/scv/nvme0n1p1/Z-Image` 是完整 Diffusers 模型目录，
DiT 两个分片共 12,309,873,872 bytes，并有 TE、VAE、tokenizer。
此前“Z-Image 权重缺失”是搜索范围不足，已更正。Anima 底模仍未找到。

新增 `bench/adaptive_runtime/probe_oom.py`，通过每进程 CUDA allocator 128MiB
上限触发真实 `torch.cuda.OutOfMemoryError`，不通过填满显卡制造压力，
不修改设备全局配置。工作负载为普通 Linear 的 AdamW 训练，不是 DiT。

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> timeout 240 .venv/bin/python -m bench.adaptive_runtime.probe_oom \
  --output output/adaptive-runtime-20260921/<device>-real-oom
```

两张卡均得到相同决策序列：

| 尝试 | micro-batch | accumulation | 实测结果 |
| --- | --- | --- | --- |
| 0 | 8 | 1 | forward 真实 CUDA OOM |
| 1 | 4 | 2 | backward 真实 CUDA OOM |
| 2 | 2 | 4 | 三次 optimizer 更新成功，loss/梯度有限 |

有效 batch 始终为 8。T10 子进程 PID 为 73958 / 74080 / 74202；
3080 为 74687 / 74802 / 75958，确认每次为新进程。
成功尝试的 PyTorch peak allocated 均为 73,665,536 bytes。
原始每次请求结果、日志和 supervisor 记录分别位于：

- `output/adaptive-runtime-20260921/t10-real-oom/`
- `output/adaptive-runtime-20260921/3080-real-oom/`

这证明真实 CUDA OOM 分类和新进程降档链路，但不证明 DiT 的自动 swap
决策、训练中途 checkpoint 恢复或训练质量等价。
新增生产 `LoRAModule` 与 FP16 Linear 精度域组合回归后，定向测试为
15 passed，Ruff 通过。

## Z-Image 全量 DiT 与生产 LoRA 探针

新增 `bench/adaptive_runtime/probe_z_image_train.py`，加载实际完整 Diffusers
transformer，挂载生产 LoRANetwork（rank4，136 个模块），FP32 adapter 参数，
AdamW、full checkpoint、swap8、原生 torch attention、256x256、3 次更新。
输入为合成 latent 和合成 text embedding，因此只验证真实模型训练执行链，
不验证图像语义、数据预处理或收敛质量。

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> HF_HUB_OFFLINE=1 timeout 300 .venv/bin/python \
  -m bench.adaptive_runtime.probe_z_image_train \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --output output/adaptive-runtime-20260921/<device>-zimage-full.json \
  --precision bf16
```

3080 BF16 对照：

| 更新 | loss | 裁剪前梯度范数 | 步时 | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 5.4591856 | 0.0382607 | 2.5509s | 9,879,087,616 bytes |
| 2 | 2.9993889 | 0.0159623 | 1.4631s | 同上 |
| 3 | 1.9152492 | 0.0261438 | 1.7145s | 同上 |

adapter 最大更新量 0.0003004477，已保存 safetensors；loss 与梯度均有限。
各步 timestep 不同，不能把 loss 下降当作收敛证据。
记录为 `output/adaptive-runtime-20260921/3080-zimage-full-bf16.json`。

T10 使用显式实验候选 `--precision fp16-islands`，276 个 Linear 使用 FP16，
其余状态及残差边界保留 FP32。`t10-zimage-full-candidate-v2.json` 记录
首次 forward 的 loss 非有限，尚未执行 optimizer。此候选不准入生产，
也没有修改 Z-Image 的 BF16-only 训练兼容约束。
首轮 `t10-zimage-full-candidate.json` 是探针恢复模式参数错误，修正为
`foreach` 后才进入模型前向，不混同为精度失败。

已增加可选 `--trace-numerics` 记录首个非有限 leaf module，以定位真正需要
FP32 的区域。包含该工具测试后为 16 passed；完整精度校准仍待完成。

### 自动敏感区域发现

`library/training/adaptive_runtime/numerics.py` 和
`bench/adaptive_runtime/search_z_image.py` 已实现有限次数的独立进程搜索。
只对“输入有限、输出非有限”的首个普通 Linear 添加 FP32 例外；不会把 CUDA
OOM 当作数值失败，也不会把输入已经非有限的下游模块误当作根因。
搜索通过时只标记 `finite_only`，不标记 `precision_calibrated`。

实际搜索目录：`output/adaptive-runtime-20260921/t10-zimage-auto-islands/`。
已依次定位：

1. `noise_refiner.0.attention.to_out.0`
2. `noise_refiner.0.feed_forward.w2`
3. `noise_refiner.1.feed_forward.w2`
4. `layers.0.feed_forward.w2`，输入最大绝对值 226812，超出 FP16 有限范围。

搜索的最终状态以目录中 `summary.json` 为准，以上仅为阶段性诊断轨迹。
输入/模块 alias 冲突、损坏 worker 结果，以及 optimizer 进度未知时的拒绝
路径已补测试：24 passed，Ruff 通过。

本轮搜索最终在六次尝试后 `failed`，没有完成 optimizer 更新。第四次晋升
将主干 `layers.0.feed_forward.w2` 改成 FP32 后，出现
`expected mat1 and mat2 to have the same dtype, but got: c10::Half != float`。
旧搜索器被之前的 `layers.1.feed_forward.w2` 非有限追踪误导，又尝试一次
晋升，仍以同类 dtype 错误退出。已增加明确 `failure_kind=nonfinite`
判定和回归，当前搜索器不会用旧非有限追踪掩盖普通运行时错误。
最终定向测试：25 passed；Ruff 通过。

共享 offloader 的 `foreach` 路径会复用换出权重的 GPU storage，并将其绑定
给换入层；混合 dtype 的层间交换需要单独复现与修复。尚未修改此共享路径，
不得宣称 FP16/FP32 islands 与现有自动块交换已经完成兼容。

本轮全部探针已退出；下一步优先级：混合 dtype 交换回归与修复、重新运行
T10 Z-Image 搜索、真实激活/梯度校准、Krea-2 完整训练链，以及 Anima 权重定位。

## 第三阶段：异构精度块交换修复

`library/runtime/offloading.py` 的 CUDA foreach 恢复路径此前直接把换出块的
GPU storage 绑定给换入块，即使两者执行 dtype 不同也只做数值 copy。
新增六行修复：使用 CPU master 捕获的 source/target dtype 判断，只有 dtype
不同时分配目标 dtype 的 storage，同 dtype 仍保留原有复用快路径。

`tests/test_block_swap_mixed_precision.py` 验证六层 FP16/FP32 混排、swap4、
连续三次轮转，分别覆盖 foreach/slab、forward-only/backward，并将输出及
输入梯度与不交换基线逐值比较。默认 GPU 四项通过；T10 上前两项
forward-only 用例通过，完整 backward 双卡补测待记录。

合并测试曾有一次既有 `test_bf16_slab_forward_only_wraparound_matches_baseline`
逐值比较失败（绝对差 1.80006e-5），单独复测通过，未修改或放宽断言。
这个非稳定结果不能隐去，需要继续核验。

修复后 T10 的 `t10-zimage-w2-fp32-swapfix.json` 完成第一次真实 LoRA 更新：
loss 5.4359283，gradient norm 0.0305354，步时 4.5275s（含数值追踪），
peak allocated 8,463,991,296 bytes。候选为所有 `*.feed_forward.w2` 和
首个 noise-refiner attention 输出投影 FP32、其他 Linear FP16、swap16。
第二次更新前向在 `layers.6.attention.to_out.0` 溢出，所以本候选仍失败。
这证实单 timestep 有限值不足以建立完整精度计划。

### T10 保守候选三步通过

在上述证据基础上，将 `*.feed_forward.w2` 与 `*.attention.to_out.0` 共
68 个 Linear 保留 FP32，其余 Linear FP16，外部残差流 FP32，swap20。
完整 Z-Image DiT、136 个生产 LoRA 模块、三次 AdamW 更新均通过，adapter
最大变化 0.0003004476，并保存 safetensors。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 HF_HUB_OFFLINE=1 \
  timeout 360 .venv/bin/python -m bench.adaptive_runtime.probe_z_image_train \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --output output/adaptive-runtime-20260921/t10-zimage-w2-out-fp32.json \
  --precision fp16-islands --trace-numerics \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' --swap 20
```

| 更新 | loss | 梯度范数 | 步时（含追踪） | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 5.4360676 | 0.0376172 | 9.6852s | 7,106,621,952 bytes |
| 2 | 2.9969468 | 0.0152068 | 6.8597s | 同上 |
| 3 | 1.8944099 | 0.0244939 | 8.2802s | 同上 |

该候选是依据已观察到的敏感算子类型手动扩展的保守 profile，尚不是已经
完成误差校准的自动生成计划。BF16 与 FP32 输入生成路径的随机张量也不
保证逐值相同，所以不能直接用前述 BF16 loss 做精度比较；需要统一保存
输入、固定相同 timestep 和 adapter 初始状态后执行同卡参考对照。

第三次定向组合验证：95 passed、14 个既有 TorchScript 弃用警告；未放宽
之前偶现失败的 BF16 slab 断言。覆盖原 block-swap、混合 dtype 新增前后向
及 adaptive runtime 单测。

T10 单独执行混合 dtype 交换测试最终为 4 passed，前后向、foreach/slab
均覆盖；两卡都与各自无交换基线逐值一致。此次共享 runtime 修复尚未执行
`python tasks.py vendor-sync`，ComfyUI vendor 副本暂不包含该修复。

## 第四阶段：真实图片与 caption 缓存

新增 `bench/adaptive_runtime/prepare_z_image_inputs.py`，从现有
`image_dataset/style_8_4` 选取一对图片/caption，经官方 Qwen3 模板、
倒数第二层文本特征和 Z-Image VAE affine 生成共享测试缓存。
严格按 TE -> free -> VAE -> free 顺序，不同时加载 DiT。
参考预处理在 3080 执行：TE BF16，VAE FP32；不宣称 T10 预处理已验证。
图片使用 EXIF 方向纠正及中心裁剪到 256x256，未修改源文件。

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> HF_HUB_OFFLINE=1 timeout 240 .venv/bin/python \
  -m bench.adaptive_runtime.prepare_z_image_inputs \
  --weights /home/scv/nvme0n1p1/Z-Image --dataset image_dataset/style_8_4 \
  --output output/adaptive-runtime-20260921/zimage-real-inputs.safetensors
```

缓存包含 FP32 latent `(1,16,1,32,32)`、83 个有效 token 的真实文本特征、
固定 CPU seed 生成的 FP32 noise。SHA256：
`7afeef5cee39cc6d2cc82958af6b2c10d91a2189adb0981ca548527aa218b6b9`。
来源图片/caption 的哈希和路径保存在同名 JSON，未写入源数据缓存目录。
`bench/adaptive_runtime/inputs.py` 校验 schema、tensor key、形状、FP32 和
有限值；新增五项测试通过，Ruff 通过。

训练探针增加 `--inputs <cache>`，不再生成随机文本或随机 latent。
3080 BF16/swap8 三步通过，记录
`output/adaptive-runtime-20260921/3080-zimage-real-bf16.json`：

| 更新 | loss | 梯度范数 | 步时 | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 0.5361994 | 0.00149171 | 2.1883s | 9,879,087,616 bytes |
| 2 | 0.5092447 | 0.00192023 | 1.1829s | 同上 |
| 3 | 0.4412697 | 0.00368507 | 1.1958s | 同上 |

真实图像和文本均经过实际预训练组件，LoRA 最大更新量 0.0003003085。
此为单样本、三个 timestep 热测，不是数据集收敛或多 bucket 验收。

### 真实输入推翻合成输入的通过结论

同一缓存用于 T10：此前合成输入三步通过的 68 个 FP32 Linear profile，
在真实输入第一步于 `layers.28.attention.to_k` 出现非有限输出；输入仍有限，
最大绝对值为 19593.4609。记录为 `t10-zimage-real-islands.json`。

额外将该投影提升 FP32 后，`t10-zimage-real-islands-k28.json` 完成两次
optimizer 更新，loss 为 0.5360658 / 0.5090686，峰值 7,136,082,432 bytes；
第三步在 `layers.27.attention.to_k` 再次溢出，其有限输入最大绝对值
8311.4160。最终状态仍是 error，不能算三步通过或精度校准通过。

这说明敏感层依赖真实激活及 timestep，不能凭合成输入或单步有限性固化
通用 FP32 白名单。搜索入口现支持 `--inputs` 和初始 `--fp32-pattern`，
后续每次尝试均用固定缓存、新进程、相同初始化重新执行全部三步；这不是
从失败步恢复训练。新增输入与既有 adaptive runtime 定向单测合计 30 passed。

加入混合 dtype GPU 回归后为 34 passed（14 个既有 TorchScript 警告）。
Krea 消融探针导入的 `scripts/krea2/probe_train.py` 原先在 import 时强制
`CUDA_VISIBLE_DEVICES=1`，会覆盖 supervisor 指定的 UUID；现改为 setdefault，
功耗查询也使用同一外部设备选择。新增 subprocess 导入回归 1 passed。
发现该问题后的首个 Krea 进程已主动终止，不计为任何设备的有效测试。
新增 adaptive runtime 和测试文件 Ruff 通过；旧 `probe_train.py` 仍有两个
既有未使用 import 和一个未使用局部变量，本次未作无关清理。

## Krea-2 完整组件探针尝试：超时，未通过

修复设备选择后，在明确的 3080 UUID 上复用现有 NF4 消融脚本：

```bash
CUDA_VISIBLE_DEVICES=GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086 \
  HF_HUB_OFFLINE=1 K2_ABL_NF4=1 \
  K2_ABL_NF4_PATH=models/diffusion_models/krea2_raw_nf4_self_contained.safetensors \
  K2_ABL_SWAP=20 K2_ABL_CKPT=0 K2_ABL_IMG=256 K2_ABL_STEPS=6 \
  K2_ABL_LORA_DIM=4 K2_ABL_LORA_ALPHA=4 K2_ABL_ATTN_MODE=torch \
  K2_ABL_COMPILE=0 K2_ABL_TAG=adaptive_3080_nf4_256 \
  K2_ABL_OUT=output/adaptive-runtime-20260921/3080-krea-full-nf4.jsonl \
  timeout 360 .venv/bin/python -m scripts.krea2.probe_nf4_ablation
```

实际退出码 124。完成 TE/VAE 编码（212.7s）、完整自包含 NF4 DiT 加载、
196 个 LoRA 模块挂载和 5.84GiB CPU masters 准备，但未返回完整训练指标，
不能声称六次更新完成。脚本只在末尾写 JSONL，本次没有最终 JSONL 结果。
输入是原探针生成的图案和固定 prompt，不是用户真实图像。
TE 日志还报告 missing=1/unexpected=0，后续应核对缺失 key，不能忽略。

此次与 T10 搜索共享主机，存在主机资源竞争；即使获得步时也不能用于独立
吞吐比较。后续应拆开缓存准备与训练 worker、限制 CPU 线程，串行完成短测，
并逐步持久化状态，避免末尾汇总式探针在超时后丢失中间证据。

## 真实输入 T10 搜索：worker 通过，外层超时

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  HF_HUB_OFFLINE=1 timeout 900 .venv/bin/python \
  -m bench.adaptive_runtime.search_z_image \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --output output/adaptive-runtime-20260921/t10-zimage-real-search \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --swap 20 --max-attempts 5
```

前四次独立 worker 的首个溢出模块依次为：
`layers.28.attention.to_k`、`layers.27.attention.to_k`、
`layers.28.attention.to_q`、`layers.28.attention.to_v`。
它们分别完成 0、2、2、2 次 optimizer 更新，每次都重新初始化，不是续训。

第五次 worker 将以上四个 Linear 加上初始 68 个 Linear 保留 FP32，
共 72/276 个，其余 204 个 Linear 使用 FP16，残差流 FP32，swap20。
`attempt-004/result.json` 的实际状态为 ok，三次更新如下：

| 更新 | loss | 梯度范数 | 步时（含追踪） | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 0.5360630 | 0.00144233 | 8.4905s | 7,192,967,680 bytes |
| 2 | 0.5090636 | 0.00163933 | 7.6118s | 同上 |
| 3 | 0.4407076 | 0.00333898 | 8.9870s | 同上 |

adapter 最大更新量为 0.0003002942，worker 已保存 safetensors。
但是外层命令退出码为 124，`summary.json` 仍为 searching、仅收录前四次，
第五次没有 supervisor.json。原始文件保持原样，不将其改写为成功。
最终检查没有遗留 probe worker。结论是**单个真实输入候选三步通过，
搜索编排未正常收尾**，而非自动搜索端到端验收通过。

本轮外层总超时 900s 小于五次单 worker 300s 的总预算；后续需统一预算和
信号清理，并提供可验证的中断恢复。此次仍然 precision_calibrated=false，
没有同卡 FP32 全模型参考、多样本/多 bucket 校准、完整 checkpoint 续训或
长期训练质量证据，也不据此宣称比 FP32 更快。

## 第五阶段：独立 Krea 缓存与真实输入 worker

新增 `prepare_krea_inputs.py`，从与 Z-Image 相同的真实图片/caption 生成
独立 Krea cache；Qwen3-VL BF16、Qwen VAE FP32，在 3080 上按顺序加载并释放。
Krea 的文本保留 `(1,512,12,2560)`、boolean mask，有效 token 数为 80，
latent 是 FP32 `(1,16,1,32,32)`，没有借用 Z-Image 的 text/latent 语义。
缓存 `output/adaptive-runtime-20260921/krea-real-inputs.safetensors` 的 SHA256：
`e058dd82a7cf3c668a8898b2fdefd6265fefa88b15484999109f8007f83126a2`。
TE 的 missing=1 日志仍存在，尚未核对具体缺失 key；不宣称严格 TE 权重校验已完成。

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  timeout -k 10 360 .venv/bin/python -m bench.adaptive_runtime.prepare_krea_inputs \
  --text-encoder models/text_encoders/qwen3vl_4b_bf16.safetensors \
  --vae models/vae/qwen_image_vae.safetensors --dataset image_dataset/style_8_4 \
  --output output/adaptive-runtime-20260921/krea-real-inputs.safetensors

CUDA_VISIBLE_DEVICES=<3080-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/3080-krea-real-nf4 --timeout 360 \
  --swap 20 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_nf4_self_contained.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors --precision bf16
```

独立 `probe_krea_train.py` 完整加载 Krea DiT，生产 LoRA rank4/alpha4、196 个模块，
FP32 adapter/AdamW，full checkpoint、torch attention、固定真实输入缓存、三个 sigma。
3080 NF4/BF16 三次更新通过，worker 与 supervisor 都正常完成，总耗时 47.06s，
adapter 最大更新量 0.0003002162，保存了 safetensors。

| 更新 | loss | 梯度范数 | 步时（含追踪） | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 0.2407372 | 0.00152096 | 4.7038s | 3,127,939,584 bytes |
| 2 | 0.1406889 | 0.00180864 | 3.4014s | 3,272,446,976 bytes |
| 3 | 0.1368245 | 0.00113242 | 3.4599s | 同上 |

原始结果位于 `3080-krea-real-nf4/attempt-000/result.json`、`supervisor.json`
以及目录根的 `summary.json`。这补齐了 3080 Krea 的三步真实输入热测，
但不是精度校准、长期训练或 checkpoint 恢复验收。

独立 runner 使用单 worker 超时，不再叠加小于搜索总预算的外层 timeout。
`IsolatedRunner` 新增主线程 SIGTERM/SIGINT 的延迟取消：先回收所属进程组，
再写 cancelled 和清理 request；取消不会触发 OOM 重试。真实子进程信号回归
已通过。该机制不能抵御 SIGKILL 或主机崩溃，尚未实现中断后自动续接历史搜索。

### TE missing key 核验

禁用 CUDA、使用 `accelerate.init_empty_weights()` 的独立只读核验显示：模型
714 keys、checkpoint 713 keys，补 `model.` 前缀后的唯一差集是
`lm_head.weight`，unexpected 为 0。checkpoint 包含文本 embedding，配置开启
tie_word_embeddings。`strategy.py::encode_tokens()` 只选取 hidden states，
不读取 logits，所以该输出 head key 缺失不影响本次文本特征缓存。

### 主机内存失败与 CPU master 复用

T10 首个非量化候选 `t10-krea-real-islands` 在 244.39s 触发 host_limit，
没有训练更新。先修复逐块捕获后绑定 native CPU master，再测
`t10-krea-real-islands-master-binding`，虽建立 22.64GiB masters，仍在 252.86s
触发 host_limit。两次均由 supervisor 正常回收，不是 CUDA OOM。

进一步发现初始 tail placement 的 GPU -> CPU 拷贝又建立一份重复权重。
`_bind_captured_cpu_weights()` 现在分两处复用已存在的 native master：逐块
捕获后释放原 CPU payload，以及初始尾部 GPU 同步放置后直接绑定 master，
不再为相同冻结权重执行一次冗余 D2H 分配。仅处理同形同 dtype 的普通冻结
Parameter；量化容器、可训练参数和 dtype 不匹配的传输路径保持原样。

相关既有交换、稀疏 masters、CPU 指针与混合精度前后向回归：85 passed。
3080 UUID 上另外跑混合 dtype 四项，4 passed；覆盖 foreach/slab、连续轮转、
前后向与无交换基线逐值比较，并断言初始尾部 CPU 参数与 master 共享 storage。
此修改尚未 vendor-sync，不声明 ComfyUI 发布副本已同步。

### 真实 Krea OOM 与原子 checkpoint 续训

首轮 `3080-krea-real-oom`，固定进程 allocator 上限 4GiB，swap0/4/8 在
model_load OOM、swap12 在 forward OOM，swap16 完成一次更新后第二步
backward OOM。因当时没有 checkpoint，策略正确以 no_committed_checkpoint
停止，没有丢弃已更新进度后从头假装续训。

补充固定输入 worker 专用 checkpoint 后，新的命令为：

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> .venv/bin/python -m bench.adaptive_runtime.recover_krea \
  --weights models/diffusion_models/krea2_raw_nf4_self_contained.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/3080-krea-real-oom-resume \
  --memory-limit-gib 4 --initial-swap 12 --max-attempts 4 --worker-timeout 240
```

最终 status=ok，选定 swap20，三次独立 worker：

| 尝试 | swap | 结果 |
| --- | --- | --- |
| 0 | 12 | 第一次 forward 真实 CUDA OOM，无 optimizer 更新 |
| 1 | 16 | 第一次更新已保存；第二次 backward OOM |
| 2 | 20 | 加载第 1 步完整状态，从第 2 步继续，完成第 2、3 步 |

三个提交 step 的 loss 为 0.2407372、0.1407222、0.1368609；恢复进程峰值
3,272,446,976 bytes。精度始终为 NF4/BF16，micro-batch=1、accumulation=1，
未重做已提交更新、未降低精度、未改分辨率或量化。
第 1 步在 swap16 的峰值为 3,997,209,088 bytes。

`bench/adaptive_runtime/checkpoint.py` 使用单文件临时写入、flush/fsync 和
os.replace 提交，保存网络、AdamW、PyTorch CPU/CUDA RNG、原始 adapter
基线和已提交 step 记录。恢复校验输入 hash、底模路径/大小/mtime、精度例外、
训练长度、学习率等签名，以 weights_only 模式加载。底模指纹不是内容哈希，
不声明能检测保持大小/mtime 的权重篡改；也不声明抵御主机崩溃后的目录持久性问题。

最终 `attempt-002/checkpoints/step-000003.pt` 的 392 份 optimizer state
步数全部为 3；step/swap 记录是 `(1,16),(2,20),(3,20)`，其网络 state 与
`attempt-002/result.safetensors` 键集合相同，最大逐值差为 0。
CPU 回归另验证保存/恢复后的下一次随机输入、AdamW 更新与未中断基线逐值一致，
并验证写失败不发布半成品、签名不匹配/缺字段拒绝加载；含 Krea precision
前置拒绝测试共 5 passed。运行中的 GPU loss 不据此宣称跨执行完全 bitwise 一致。

这是**固定输入、无 scheduler/动态数据加载器的真实 DiT worker 续训证据**，
还不是生产 train.py/WebUI 的完整恢复。生产另有 Accelerate state、scheduler、
epoch/sampler/worker RNG 与梯度累积语义，必须沿既有 checkpoint hooks 接入并
独立验证，不能用这个探针 checkpoint 替代。precision_calibrated 仍为 false。

### T10 完整 Krea 真实输入三步通过

完成两处 native CPU master 复用后，重新执行：

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-krea-real-islands-master-reuse \
  --timeout 420 --swap 20 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --precision fp16-islands
```

worker 与 supervisor 均为 ok，总耗时 275.83s。264 个普通 Linear 使用 FP16，
非 Linear 状态和残差边界保留 FP32，196 个 FP32 LoRA 模块。此候选没有
额外 FP32 Linear 例外；只证明所测输入/三个 sigma 的有限性，不代表所有
Linear 已经通过误差校准或在别的输入上不会溢出。attention 的 FP32 Q/K/V
在 T10 上回退可用 SDPA 路径，不能据此宣称低精度 fused attention 已启用。

| 更新 | loss | 梯度范数 | 步时（含追踪） | peak allocated |
| --- | --- | --- | --- | --- |
| 1 | 0.2357628 | 0.00054420 | 12.3353s | 9,587,277,824 bytes |
| 2 | 0.1368086 | 0.00024616 | 10.1850s | 9,731,785,216 bytes |
| 3 | 0.1346926 | 0.00050313 | 11.5897s | 同上 |

adapter 最大更新量 0.0003000671，已保存 safetensors。GPU 峰值约 9.06GiB；
进入 forward 时主机 available 观察值约 25GiB。这里不是完整 RSS 峰值追踪。
对照 3080 用 NF4、此处用非量化底模，禁止用两者 loss/步时直接归因精度差异
或宣称加速比例。双卡 Krea 三步热测已补齐，但精度校准、T10 自动 OOM 续训、
生产训练接线和长窗口质量验证仍未完成。Anima 双卡仍待完整底模路径。

### BF16 slab 异步轮转回归追查

最后一次完整组合最初为 125 passed / 1 failed：既有
`test_bf16_slab_forward_only_wraparound_matches_baseline` 再次出现精确比较失败，
最大绝对差 1.80006e-5、最大相对差约 43%。缩小到 16 项时通过，但同进程
重复调用原测试仍复现，另一次绝对差 1.66893e-6、相对差约 4.5%。未放宽断言。
开启 CUDA_LAUNCH_BLOCKING=1 的十次对照通过，指向异步存储/流依赖问题。

实际修复把 `_get_cached_restore_slab()` 的首次 GPU allocation 移到真正执行
H2D 的 copy stream 内。此前在默认流分配后跨流使用，ready_event 又早于
allocation，不能覆盖 caching allocator 复用的、其后入队的默认流工作。
没有通过每次全局 synchronize 或禁用异步交换来掩盖问题。

修复后默认 GPU 正常异步模式连续 30 次原测试通过；明确指定 3080 UUID
后另做 30 次也通过。混合精度回归增加 current_stream 与对应 copy stream
一致性断言。最终完整组合 **126 passed，24 个既有警告**；覆盖 adaptive
策略/进程/缓存/checkpoint、Krea 拒绝边界、既有交换/稀疏 master 与混合 dtype。
历史失败记录保留；有限次 stress 不是无限运行稳定性的证明。

本轮新增模块和测试 Ruff 通过，git diff --check 通过。对完整
`library/runtime/offloading.py` 的 Ruff 检查仍有 8 个既有未使用 import/局部变量
告警；与前述 probe_train.py 的既有告警一样未做无关清理，不宣称全仓 lint 通过。

### 真实激活局部校准：Z-Image 双卡

新增有界 frozen Linear 捕获器及离线校准命令。每层每步最多 64 token 行，
包含最大幅值行；3 个 sigma 为 0.2/0.5/0.8。只在初次 forward 采集，
排除 backward 重算；训练更新数用于去重，同一步额外 forward 不作为新案例。
快照预估 tensor payload 321,454,080 bytes，保留权重、输入和行索引。
路径逃逸、别名、trainable 模块、非有限输入、空/不完整 manifest 均拒绝。

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-real-capture \
  --timeout 420 --swap 20 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image --precision fp16-islands \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors --steps 3 \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-module layers.28.attention.to_k --fp32-module layers.27.attention.to_k \
  --fp32-module layers.28.attention.to_q --fp32-module layers.28.attention.to_v \
  --capture-linear layers.0.attention.to_q --capture-linear layers.28.attention.to_q \
  --capture-linear layers.27.attention.to_k --capture-linear noise_refiner.0.feed_forward.w2
```

本次 worker **和 supervisor 均为 ok**，独立补齐此前搜索外层超时留下的
监督进程证据；并未修改先前超时报告。总时间 94.70s；三步 loss 为
0.53606296/0.50907785/0.44068477，GPU peak 7,192,967,680 bytes，adapter
最大变化 0.0003002945。配置仍为 72 FP32 / 204 FP16 Linear。

对同一 capture 分别在 T10、3080 执行：

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> .venv/bin/python -m bench.adaptive_runtime.calibrate_capture \
  --capture output/adaptive-runtime-20260921/t10-zimage-real-capture/attempt-000/capture \
  --output output/adaptive-runtime-20260921/<t10-or-3080>-zimage-local-calibration.json
```

| Frozen Linear | T10 FP16 判定 | 3080 BF16 最大输出 rel-L2 | 3080 最大输入 VJP rel-L2 |
| --- | --- | --- | --- |
| layers.0.attention.to_q | 通过；最大输出误差 0.000236，VJP 0.000363 | 0.002142 | 0.002638 |
| layers.28.attention.to_q | sigma=0.8 输出非有限，选 FP32 | 0.002157 | 0.002625 |
| layers.27.attention.to_k | sigma=0.8 输出非有限，选 FP32 | 0.001936 | 0.002641 |
| noise_refiner.0.feed_forward.w2 | 三个 sigma 输出均非有限，选 FP32 | 0.001979 | 0.002362 |

3080 四层均通过当前实验阈值；最小输入 VJP cosine 为 0.9999965。
这提供真实激活上的局部精度选择证据，也说明只覆盖一个 timestep 会漏掉敏感点。
各设备结果记录相同输入/权重 SHA256，完整逐案例结果在上述 JSON。

**限定**：冻结 base Linear，不包含 LoRA delta/参数梯度或整个 block/DiT 的
误差传播；只有一张真实图片文本、三个 sigma、少量 token 行。FP32 reference
是采集权重值提升到 FP32 后计算，不是原始未舍入 FP32 底模。当前输出阈值
1%、VJP 阈值 2%、cosine 0.999 是实验门槛，不是经生成质量验证的标准。
不能把 4 层计划推广到全部 276 层，更不能删除原来的其他 FP32 例外。
报告保留 `full_model_calibrated=false`，不会写入生产默认配置。

新增采集器与 Krea 前置拒绝定向测试 17 passed（14 个既有警告）；新增文件
Ruff 通过。全仓 diff-check 此时发现用户已有 `configs/web-ui-settings.toml`
第 69 行末尾空行，未修改这个无关配置，不能声称全仓 diff-check 通过。

### 真实激活局部校准：Krea 双卡

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-krea-real-capture \
  --timeout 420 --swap 20 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors --precision fp16-islands \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors --steps 3 \
  --capture-linear blocks.0.attn.wq --capture-linear blocks.27.attn.wq \
  --capture-linear blocks.27.mlp.down
```

worker 与 supervisor 均 ok，总时间 277.90s；三步 loss 为
0.23576282/0.13680860/0.13469264，peak 9,731,785,216 bytes。步时包含
激活采样、权重落盘和 finite trace，不能用于无采集基线速度比较。
仍为 264 FP16 Linear、非 Linear/残差 FP32、196 FP32 LoRA，无 FP32 Linear
例外。三次更新不是长期数值稳定性或生成质量证明。

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> .venv/bin/python -m bench.adaptive_runtime.calibrate_capture \
  --capture output/adaptive-runtime-20260921/t10-krea-real-capture/attempt-000/capture \
  --output output/adaptive-runtime-20260921/<t10-or-3080>-krea-local-calibration.json
```

同一份三层、每层三个 sigma 的 capture 已在双卡校准，输入和权重 hash 一致。
以下均为三个案例中的最大 rel-L2；三层均通过当前实验阈值：

| Frozen Linear | T10 FP16 输出 | T10 输入 VJP | 3080 BF16 输出 | 3080 输入 VJP |
| --- | --- | --- | --- | --- |
| blocks.0.attn.wq | 0.0002122 | 0.0003258 | 0.0017020 | 0.0023943 |
| blocks.27.attn.wq | 0.0002350 | 0.0003274 | 0.0018778 | 0.0023395 |
| blocks.27.mlp.down | 0.0002471 | 0.0003604 | 0.0018942 | 0.0023846 |

T10 最小输入 VJP cosine 0.999999935，3080 为 0.999997146。
本次快照已经过 BF16 -> FP16 权重转换；FP32 参考也是这些快照值的提升，
不评估最初权重转换丢失的信息。不是 NF4 校准，不放开生产 NF4 BF16 限制。
三个 Linear 的通过不能推导其整个 block、LoRA 参数梯度或整网都通过；
`full_model_calibrated=false` 保留。

本阶段 adaptive 策略、数值追踪、进程、输入、checkpoint、Krea 前置拒绝、
采集器组合回归 **55 passed，14 个既有警告**；bench/adaptive_runtime 与
library/training/adaptive_runtime 及新增采集测试 Ruff 通过。涉及共享 offloader
的既有 126 项回归结果见上节，本阶段没有再次修改共享 offloader。

### T10 Krea 真实 OOM 后自动增加 swap

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.recover_krea \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-krea-real-oom-recovery \
  --memory-limit-gib 9.05 --initial-swap 20 --max-attempts 3 --worker-timeout 420
```

只对 worker 设置 PyTorch allocator 9.05GiB 上限，没有修改设备/system 设置或
用外部占位进程挤占显存。硬件候选 auto 选择 fp16-islands，所有尝试保持相同
264 个 FP16 Linear、FP32 残差/非 Linear/LoRA、输入、micro-batch=1 和完整 checkpointing。

| 尝试 | swap | 结果 | worker 总耗时 |
| --- | --- | --- | --- |
| 0 | 20 | 首次 backward 申请 48MiB 真实 CUDA OOM；optimizer_started=false | 226.25s |
| 1 | 24 | 独立进程三次更新通过，保存完整实验检查点 | 296.67s |

最终 peak allocated 6,258,901,504 bytes（约 5.83GiB），loss 为
0.23576282/0.13680860/0.13469264。该组 sigma/输入下，与此前 swap20 成功的
无 allocator 限制实验 loss 一致。最终 392 份 AdamW state 的 step 均为 3；
进度 `(1,24),(2,24),(3,24)`，step-3 checkpoint 网络与最终 safetensors
同键，逐值最大差 0。

这证明 **T10 混合精度 + 真实 OOM + 自动 swap 重试**，但第一次失败没有
已提交更新，所以这里不是中途 checkpoint 恢复证据；不要把 3080 Krea 的
resume 结论直接外推到此处。检查点格式仍是固定输入实验 worker 专用。

### Z-Image 实验恢复接口

新增 `bench.adaptive_runtime.recover_z_image`，Krea 旧命令保留为兼容入口，
两个入口共用 `recovery.py` 的独立进程、有界重试、精度固定与失败分类逻辑。
Z-Image worker 在 optimizer 前记录已开始标志，每次更新原子提交检查点；
恢复前核对真实输入 hash、模型族、精度例外、训练参数及 transformer 文件清单。
文件清单使用 path/bytes/mtime，不宣称能发现保留这些信息的内容篡改。
禁止随机输入 checkpoint 恢复、恢复时采集激活、Turing BF16 候选。

新增 family dispatch、命令传递、Z-Image 底模文件清单/输入限制测试。组合回归
**61 passed，14 个既有警告**；新增 bench 模块与测试 Ruff 通过。

### 3080 Z-Image 真实 OOM 与独立 checkpoint 恢复

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> .venv/bin/python -m bench.adaptive_runtime.recover_z_image \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/3080-zimage-real-oom-recovery \
  --memory-limit-gib 5.5 --initial-swap 16 --max-attempts 4 --worker-timeout 360
```

最终 status=ok，auto 选 BF16。swap16 在 model_load 申请 76MiB 时 CUDA OOM
（39.71s）；swap20 独立 worker 完成三次更新（61.02s），peak allocated
5,499,853,312 bytes。loss 为 0.53619939/0.50917959/0.44122672。
没有改变输入、分辨率、BF16、LoRA rank、优化器或 batch。

为单独验证真实 Z-Image 的状态恢复，使用上述成功尝试已提交的 step-1：

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/3080-zimage-checkpoint-resume \
  --timeout 360 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision bf16 --checkpoint-every-step \
  --resume output/adaptive-runtime-20260921/3080-zimage-real-oom-recovery/attempt-001/checkpoints/step-000001.pt
```

worker/supervisor ok，48.83s。恢复后的 step-2/3 loss 为
0.50917959/0.44126666，当前 worker peak 4,040,108,544 bytes。
第 3 步与未中断 swap20 基线差约 3.99e-5；梯度范数也不是逐位一致，
不把状态完整性验证等同于跨执行/交换布局的 bitwise 数值等价。

最终 step-3 checkpoint 有 272 份 AdamW state，step 全部为 3，历史
`(1,20),(2,24),(3,24)`。恢复输出目录只新增 step-2/3 checkpoint，没有重存
step-1。最终 checkpoint 网络与本次结果 safetensors 同键且最大逐值差 0。
这是 **自动 OOM 重试成功 + 单独显式 checkpoint 恢复成功** 两项证据，
不是同一自动恢复任务中途发生 OOM 后续训的证据。

另补充 checkpoint 预校验，拒绝 initial/optimizer 张量中的 NaN/Inf；新增损坏
状态测试与恢复接口共 11 passed。独立只读审查未发现固定输入实验恢复的步号、
随机状态或命令传递错误，但不外推生产 dataloader/scheduler 的恢复正确性。

### T10 Z-Image 真实 OOM 后自动增加 swap

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.recover_z_image \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-zimage-real-oom-recovery \
  --memory-limit-gib 6.3 --initial-swap 20 --max-attempts 3 --worker-timeout 360 \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern layers.28.attention.to_k --fp32-pattern layers.27.attention.to_k \
  --fp32-pattern layers.28.attention.to_q --fp32-pattern layers.28.attention.to_v
```

status=ok；swap20 在 model_load 申请 76MiB 时 CUDA OOM（84.26s），
swap24 新 worker 成功完成三步（136.87s）。peak allocated 5,304,756,736
bytes（约 4.94GiB），loss 为 0.53606296/0.50907600/0.44071683。
成功报告确认 72 个 FP32 Linear，精度、输入、micro-batch=1 始终固定。

历史失败 `attempt-000/result.json` 的 `fp32_modules=[]` 是记录时机缺陷：
当时只有 build 返回后才把模式解析成精确模块名单，加载 OOM 导致未写出。
此空列表不能作为该尝试使用全 FP16 的证据；supervisor 的
`initial_fp32_patterns` 和相同命令传递可核对配置。后续代码将解析名单的记录
提前到 attention/LoRA/设备放置之前，并增加 `precision_profile_resolved` 标志及
`requested_fp32_patterns`；定向回归覆盖后续 OOM 仍保存精度配置。未修改历史结果。

checkpoint initial/optimizer 非有限值检查加入后，完整 adaptive 组合回归
63 passed，14 个既有警告；精度配置提前记录的单独回归另列后续结果。

### T10 Z-Image 独立 checkpoint 恢复

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-checkpoint-resume \
  --timeout 360 --swap 26 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --checkpoint-every-step \
  --resume output/adaptive-runtime-20260921/t10-zimage-real-oom-recovery/attempt-001/checkpoints/step-000001.pt \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern layers.28.attention.to_k --fp32-pattern layers.27.attention.to_k \
  --fp32-pattern layers.28.attention.to_q --fp32-pattern layers.28.attention.to_v
```

worker/supervisor ok，总时间 113.41s。只执行 step-2/3，保留来自源检查点的
step-1；当前 worker peak allocated 4,360,651,264 bytes（约 4.06GiB），
step-2/3 loss 为 0.50907600/0.44071251。第 3 步相对不中断 swap24 基线
差约 4.32e-6，梯度范数存在差异，不声明 bitwise 等价。
72 FP32 Linear 保持不变。

最终 272 份 AdamW state 的 step 全部为 3，历史为
`(1,24),(2,26),(3,26)`，恢复目录只生成 step-2/3 检查点。
step-3 checkpoint 通过新增非有限值校验，网络与最终 safetensors 同键且
最大逐值差 0。仍是独立显式恢复实验，不是同一自动 OOM 任务的中途恢复。

精度配置提前记录的单独回归为 7 passed；新增 bench/核心 adaptive/测试 Ruff
通过。本轮没有改动生产训练配置、用户设置、底模或 vendor 副本。
最终将提前记录测试纳入完整 adaptive 组合，结果 **64 passed，14 个既有警告**。
结束前检查未发现残留 `bench.adaptive_runtime.probe_*` / `recover_*` 实验进程。

### T10 真实精度与 OOM 联合预检

新增 `library/training/adaptive_runtime/joint.py`，由
`bench.adaptive_runtime.search_dit` 驱动同一个有界实验任务。已有单独
`search_z_image` / `recover_krea` / `recover_z_image` 入口保留。
本轮精度提升候选判断抽成 `numerics.py::promotion_candidate` 供独立/联合搜索复用。

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.search_dit \
  --model-family z_image --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-zimage-joint-preflight \
  --memory-limit-gib 6.3 --initial-swap 20 --max-attempts 4 --max-promotions 2 \
  --worker-timeout 360 --fp32-pattern '*.attention.to_out.0' \
  --fp32-pattern '*.feed_forward.w2' --fp32-pattern layers.28.attention.to_k \
  --fp32-pattern layers.27.attention.to_k --fp32-pattern layers.28.attention.to_v
```

auto 选择 fp16-islands；只限制当前 worker allocator 为 6.3GiB。
该测试以已知 71 个 FP32 Linear 为种子，故意不包含此前确认敏感的
`layers.28.attention.to_q`，检验内存失败和数值失败在一个任务中的串联，
不是无先验从零识别全部敏感层的证据。

| 尝试 | swap | FP32 Linear | 结果 | worker 总耗时 |
| --- | --- | --- | --- | --- |
| 0 | 20 | 71 | model_load 真实 CUDA OOM，增加 swap | 78.53s |
| 1 | 24 | 71 | 两个实验更新后，sigma=0.8 的 layers.28.attention.to_q 首次非有限，自动提升该层 | 111.94s |
| 2 | 24 | 72 | 从初始状态重新执行三个实验更新，通过 | 147.95s |

最终 status=`finite_only`，peak allocated 5,304,756,736 bytes，三步 loss
0.53606296/0.50905746/0.44070929。每次进程精度名单已在设备放置前记录，
失败报告不再出现未解析名单被误当无 FP32 层的歧义。所提升层仍包含在最终
worker 的 72 层完整名单中。没有因为 OOM 把敏感层降回 FP16。

三次 worker 均标记 `disposable_probe=true`，所有更新都是预检消耗，不是
用户训练进度。第二次的两个实验更新明确丢弃，不从改变精度前的 checkpoint
继续；第三次成功报告 `adapter_saved=false`。实际目录内没有
`result.safetensors` 或 `.pt` checkpoint 文件。原有普通训练/恢复 worker
继续保存产物，只有显式 disposable 模式禁止 checkpoint/resume 和最终 adapter 保存。

当前联合 API 还会要求成功 worker 的完整精度名单包含所有新提升层，否则
fail closed；成功报告提供 `resolved_fp32_modules` 与 `production_ready=false`。
这两个字段是在本次 T10 父进程启动后补充的，因此历史 T10 summary 不补写
它们；最终 worker 名单已独立核对，当前 API 的新门槛另有单元测试。
所有结果仍然 `precision_calibrated=false`，不认证整网数值质量。

独立只读核验发现 disposable 成功路径原本还会保存普通 LoRA 文件，已修复，
本次最后一个 worker 已使用修复并验证没有权重产物。其他候选风险经过代码
核对：promotion helper 必须 status=error 且 failure_kind=nonfinite；OOM/
host_limit/timeout/cancelled 不会因残留 trace 触发提升；内置内存策略单调增加
swap/降低合法 micro-batch，提升名单去重且尝试有界。相关边界已补定向测试。

### 3080 原生 BF16 联合预检

```bash
CUDA_VISIBLE_DEVICES=<3080-UUID> .venv/bin/python -m bench.adaptive_runtime.search_dit \
  --model-family z_image --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/3080-zimage-joint-preflight \
  --memory-limit-gib 5.5 --initial-swap 16 --max-attempts 3 --worker-timeout 360
```

auto 选择 bf16，无 FP32 Linear 例外；不为当前 BF16 worker 开启未支持的
逐层提升。swap16 在 model_load OOM（36.73s），swap20 成功（57.46s），
最终 `finite_only`，`production_ready=false`、`resolved_fp32_modules=[]`。
三步 loss 为 0.53619939/0.50933188/0.44096085，peak allocated
5,499,853,312 bytes。adapter_saved=false，实际无 LoRA/checkpoint 文件。
跨执行的 loss/梯度范数有差异，不将三步有限值测试当作确定性或训练质量认证。

联合 API 当前确实让“硬件候选 + 数值失败提升 + OOM swap 调整”在同一个
有界预检任务中协作；但还未完成全模型误差校准、生产训练入口和恢复接线。
T10 的种子精度名单、有限输入/步数和非 Anima 模型范围仍是明确的覆盖限制。

本阶段完整 adaptive 组合回归 **86 passed，14 个既有警告**；bench/核心
adaptive 和新增联合测试 Ruff 通过。结束前未发现遗留 probe/recover/search
实验进程；本仓 diffusion_models 目录仍只有 Krea 权重，Anima 缺失状态未解除。

### 整网训练轨迹采集与 FP32 参考

新增 `bench/adaptive_runtime/training_capture.py`、`compare_training.py`，
Krea/Z-Image worker 增加 `fp32-reference` 与 `--capture-training`。记录
backward 之后、clip/optimizer 之前的预测、输入 loss 梯度与全部 LoRA 参数
loss 梯度。每步缺少任一可训练参数梯度即拒绝；快照带 SHA256、sigma、
连续 step、初始 adapter hash 与源输入/底模/训练签名，默认总 tensor 预算 1GiB。

比较器要求 reference/candidate 都是成功完整 worker，且初始 adapter、源缓存、
底模文件指纹、训练参数和 sigma 序列匹配，才报告三组聚合 rel-L2/cosine
及误差最大的 10 个参数。底模身份是路径/大小/mtime，不是完整权重内容哈希。
FP32 reference 是同一 BF16 checkpoint 提升后计算，不是原始未舍入 FP32 底模。
后两步比较的是各自更新后的训练轨迹，因此含 optimizer 放大效应；不把其
梯度差全部直接归因于某一个算子的舍入。通过门槛也不等于生成质量认证。

#### Krea FP32 首轮主机内存失败

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-krea-fp32-reference \
  --timeout 600 --swap 24 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --precision fp32-reference --capture-training --disposable-probe
```

supervisor 为 `host_limit`，92.40s，在 model_load/weight_placement 阶段退出，
没有训练更新、没有可比较的整网参考。worker 最后写出的 running 状态不是最终
成功证据，以 supervisor/summary 为准。初始主机 available 约 56GiB，保护
reserve 约 6.2GiB，未降低保护阈值、未改系统 swap 或占用其他 GPU 进程。

探针先将 4 个 resident block 以 FP32 搬到 GPU，再扩展剩余权重。只读核验
确认 `move_to_device_except_swap_blocks` 跳过 blocks，并未把这些块搬回 CPU；
但首次 offloader prepare 仍会为参与交换的所有块创建 CPU masters，resident
CUDA 权重也会回拷生成主机 master。swap24/28 blocks 时参与集合覆盖全部块，
不能简单依赖 sparse masters 跳过这些常驻初始块。FP32/pinned master 的总量
和初始化临时副本因此仍越过主机保护界限。未为让参考通过而改成低精度计算。

#### T10 Z-Image FP32 三步参考

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-fp32-reference \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp32-reference --capture-training --disposable-probe
```

worker/supervisor ok，113.22s，peak allocated 8,003,061,248 bytes（约 7.45GiB）。
三步 loss 为 0.53606927/0.50906795/0.44070688。无 autocast、TF32 关闭，
base 和 LoRA 均 FP32，仍用 full checkpoint + swap24 + torch attention。
三份快照覆盖 sigma 0.2/0.5/0.8，每份均包含 272 个 LoRA 参数梯度；总
tensor payload 50,528,256 bytes。初始 adapter hash 为
`82547bd987a8689149263e6ae8210345596083df44ed71b54936ec39d7c419bf`。
`adapter_saved=false`，快照是诊断张量，不是可恢复训练 checkpoint。

新增短训采集/比较、FP32 dtype 路径、NF4 拒绝边界纳入完整 adaptive 回归，
**98 passed，14 个既有警告**；新增/修改 bench 及相关测试 Ruff 通过。

#### T10 未缩放混合候选：预测接近但梯度门槛失败

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-training-candidate \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --capture-training --disposable-probe \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern layers.28.attention.to_k --fp32-pattern layers.27.attention.to_k \
  --fp32-pattern layers.28.attention.to_q --fp32-pattern layers.28.attention.to_v

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-zimage-fp32-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-zimage-training-candidate/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-zimage-training-comparison.json
```

候选 worker/supervisor ok，118.43s，72 FP32 / 204 FP16 Linear，未启用 loss
scaling；GPU peak 5,304,756,736 bytes。相同源缓存、底模与初始 adapter 的
全预测和全部 272 个 LoRA 参数梯度对照结果如下，rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA 梯度 cosine |
| --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08894% | 8.980% | 13.712% | 0.990668 |
| 2 / 0.5 | 0.06704% | 133.340% | 38.642% | 0.934781 |
| 3 / 0.8 | 0.06000% | 22.841% | 17.911% | 0.984525 |

三个案例均未通过当前实验门槛。此前有限值/局部 Linear 测试的成功并不矛盾：
它们没有覆盖全模型反向传播。**不能把这个未缩放 72 层方案发布为经过整网
精度验证的自动回退方案**。报告仍 `production_ready=false`。

#### FP32 重复运行的误差底线

用与第一份 FP32 参考完全相同命令重新运行，仅将 output 改为
`output/adaptive-runtime-20260921/t10-zimage-fp32-reference-repeat`。
worker/supervisor ok，112.64s，三步 loss 与第一份参考相同，峰值相同。
使用 compare_training 将这份重复运行作为 candidate，结果写入
`t10-zimage-fp32-repeat-comparison.json`，三个案例均通过实验门槛：

| step | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 |
| --- | --- | --- | --- |
| 1 | 0 | 1.0774e-6 | 1.9231e-6 |
| 2 | 1.0160e-6 | 6.7592e-4 | 1.7810e-4 |
| 3 | 7.4625e-7 | 3.5987e-5 | 3.6185e-5 |

此表 rel-L2 是比例而非百分比。参考自身不是严格 bitwise，但波动远小于混合
候选的梯度误差，因此不能用“所有 CUDA 运行都有误差”来忽略候选失败。

新增受限 `--loss-scale` 对照，使用 PyTorch `torch.amp.GradScaler`。只有
disposable FP16 + training capture 可启用，不能带 checkpoint/resume，避免
在现有实验检查点中漏掉 scaler 状态。参数梯度由 scaler unscale；诊断输入
叶子的梯度另显式除以同一 scale，否则会产生虚假的输入梯度差异。
当前实验使用极长 growth interval，不做自动 scale 搜索；非有限梯度仍明确
失败，不把跳过 optimizer 的步骤写成成功更新。默认 scale=1 保持此前行为。
CPU toy 回归证明小梯度经过 FP16 可归零，scaling 后参数及输入梯度可恢复，
但这只是机制验证，不提前断言真实 DiT 失败全由下溢造成。

#### T10 AMP scale=128 对照：改善但未完全通过

```bash
CUDA_VISIBLE_DEVICES=<T10-UUID> .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-training-scaled128 \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --capture-training --disposable-probe --loss-scale 128 \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern layers.28.attention.to_k --fp32-pattern layers.27.attention.to_k \
  --fp32-pattern layers.28.attention.to_q --fp32-pattern layers.28.attention.to_v

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-zimage-fp32-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-zimage-training-scaled128/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-zimage-training-scaled128-comparison.json
```

worker/supervisor ok，100.68s；GPU peak 仍为 5,304,756,736 bytes；72 个
FP32 Linear 和 swap24 均不变。三步 loss 为 0.53606296/0.50905949/0.44070628。
初始 adapter hash、源输入、底模签名均与参考匹配，272 个参数梯度均已 unscale
后采集，诊断输入梯度也单独 unscale。以下 rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08894% | 1.0796% | 1.3471% | 0.999910 | 通过 |
| 2 / 0.5 | 0.07698% | 60.450% | 15.5908% | 0.988415 | 失败 |
| 3 / 0.8 | 0.05718% | 1.3171% | 1.0496% | 0.999945 | 通过 |

相对未缩放候选，scaling 明显降低梯度误差，与 FP16 反向下溢机制相符；但
sigma=0.5 仍显著超阈值，不能宣称只需 GradScaler 就完成整网精度适配。
后续还要区分前向舍入/敏感块与先前 optimizer 更新造成的轨迹放大，并扩大
输入覆盖。此门槛本身也不是长期生成质量的已验证标准。

本阶段整网轨迹比较仅完成 T10 Z-Image；3080 同设备 FP32/候选成对对照仍
待补，Krea FP32 参考受主机内存限制，Anima 仍缺底模。目标没有完成。
最新完整 adaptive 回归 **106 passed，14 个既有警告**，新增/修改模块 Ruff
通过。未改生产精度默认值、NF4 限制、用户配置或系统内存设置。

### 补充：T10 scale=1024 与同设备比较契约

保持前述 scale=128 命令的输入、swap24、72 个 FP32 Linear 和三步 sigma
序列不变，将 `--loss-scale` 改为 `1024`，输出改为
`output/adaptive-runtime-20260921/t10-zimage-training-scaled1024`。比较命令
仍引用 `t10-zimage-fp32-reference/attempt-000/training-capture`，候选替换为
该新目录的 `attempt-000/training-capture`，比较文件为
`t10-zimage-training-scaled1024-comparison.json`。worker/supervisor 均 ok，
总耗时 95.74s，GPU peak 5,304,756,736 bytes；没有保存正式 adapter。

以下 rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08894% | 1.0570% | 1.3584% | 0.999908 | 通过 |
| 2 / 0.5 | 0.06595% | 40.3712% | 10.4177% | 0.994734 | 失败 |
| 3 / 0.8 | 0.05893% | 1.2701% | 1.1518% | 0.999934 | 通过 |

提高 scale 后中间步误差下降，但仍远超门槛，不构成已解决下溢或已完成自适应
精度的证据。下一步应在相同 pre-step adapter 上重放以分离直接精度偏差和
optimizer 轨迹差异，而不是仅依据最大相对梯度误差盲目提升整个 block。

比较器现在强制参考和候选提供相同的非空 GPU UUID，避免跨卡运行被误当成
同设备数值实验。新增缺失/空/不同 UUID 拒绝测试；该文件定向测试 13 passed，
修改的比较器和测试 Ruff 通过。

同一份 capture 另生成 `t10-zimage-training-scaled1024-absolute-comparison.json`，
新增 `absolute_l2` 与 `largest_absolute_adapter_errors`，保留原有相对误差榜。
第 2 步绝对偏差前五为 layers5 attention output、layers9 V、layers5 V、
layers3 K、layers6 attention output 的 LoRA up 梯度，L2 分别约
4.677e-5、4.151e-5、3.677e-5、3.455e-5、3.182e-5。
这些是梯度偏差位置，不代表相应 Linear 是误差源，不能直接转成提升名单。
新增 toy 测试确认 tiny-gradient 的相对误差榜和绝对偏差榜不会混淆。
最终比较器定向测试 14 passed，Ruff 通过；增加该排序之前完整 adaptive 套件
109 passed / 14 个既有警告。

### 补充：3080 同卡 FP32/BF16 整网轨迹

本轮补齐 Z-Image 的 3080 同设备成对对照，两次串行运行，避免共同加载造成
主机内存争用。使用相同 BF16 底模、真实缓存、初始 adapter、swap24 和三步
sigma；参考禁用 autocast/TF32，BF16 候选沿实验 worker 的 BF16 输入路径运行。

```bash
CUDA_VISIBLE_DEVICES=GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/3080-zimage-fp32-reference \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp32-reference --capture-training --disposable-probe

CUDA_VISIBLE_DEVICES=GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/3080-zimage-training-bf16 \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision bf16 --capture-training --disposable-probe

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/3080-zimage-fp32-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/3080-zimage-training-bf16/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/3080-zimage-training-bf16-comparison.json
```

两次 worker/supervisor 均 ok，均不保存正式 adapter。FP32 总耗时 130.75s，
peak 8,003,061,248 bytes；BF16 总耗时 65.00s，peak 4,040,108,544 bytes。
总耗时包含模型加载和 capture，不是稳态训练速度比较。三步均采集完整 272 个
LoRA 参数梯度。以下 rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 1.4358% | 12.4626% | 30.5656% | 0.955505 | 失败 |
| 2 / 0.5 | 0.8941% | 91.9119% | 26.9677% | 0.964521 | 失败 |
| 3 / 0.8 | 1.0501% | 36.1767% | 56.9504% | 0.863169 | 失败 |

第 1 步在初始 adapter 相同的条件下已失败，不能将全部误差归因于 optimizer
轨迹发散；同时比较包含 BF16 输入/target 舍入与计算路径差异，不是仅 Linear
算子的单因素实验。BF16 支持只决定硬件候选，不保证满足指定误差预算。
这些实验门槛未经过长期生成质量验证，失败不证明现有 BF16 生产训练不可用；
也不能因此放宽门槛宣称校准成功。下一步需要固定 pre-step adapter 和运行
输入，拆分输入舍入、残差/算子精度以及更新轨迹的贡献。

截至此处 Z-Image 两卡均有真实整网 FP32/候选对照，但没有候选通过全部
短训门槛。Krea FP32 参考主机内存限制、Anima 底模缺失、生产恢复接入仍未解决。
本轮最终完整 adaptive 回归 110 passed / 14 个既有弃用警告（15.99s），
修改模块 Ruff 通过；所有本轮 GPU worker 已退出，没有遗留实验进程。

### 固定 pre-step 状态重放：T10 Z-Image

新增 `bench/adaptive_runtime/replay.py`，目的不是换一个更宽松的误差门槛，
而是排除各自 AdamW 更新造成的轨迹差异。只用于固定 plain LoRA 配置的
disposable 实验，不是生产恢复或可泛化到任意 router/adapter 的检查点格式。
快照包括所有 network 参数与 buffers（含非持久化 timestep mask）、真实
运行 noisy/target/prompt/sigma 和 torch CPU/CUDA、Python RNG。候选每步恢复
后重新计算摘要，比较器要求每一步的快照、状态、输入、RNG 四项摘要匹配。

新 FP32 参考从头训练并记录三个 pre-forward 快照，worker/supervisor 均 ok，
101.67s，GPU peak 8,003,061,248 bytes；快照 tensor 总量 53,103,036 bytes。
manifest 实测 `status=captured`、`steps=3`。路径为
`t10-zimage-replay-reference/attempt-000/training-replay`，相应梯度仍存于
同级 `training-capture`。两种产物均在 `output/adaptive-runtime-20260921/` 下。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-replay-reference \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp32-reference --capture-training --disposable-probe --record-replay

CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-replay-scaled1024 \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --capture-training --disposable-probe --loss-scale 1024 \
  --fp32-pattern '*.attention.to_out.0' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern layers.28.attention.to_k --fp32-pattern layers.27.attention.to_k \
  --fp32-pattern layers.28.attention.to_q --fp32-pattern layers.28.attention.to_v \
  --replay-reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-replay

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-zimage-replay-scaled1024/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-zimage-replay-scaled1024-comparison.json
```

候选仍是 72 个 FP32 Linear / 204 个 FP16 Linear、FP32 外部残差，scale1024，
swap24。worker/supervisor 均 ok，107.98s，GPU peak 5,304,756,736 bytes。
三个案例摘要均匹配，scope 为 `fixed_pre_step_state_and_inputs_not_quality_certificate`。
以下 rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08894% | 1.0473% | 1.3620% | 0.999908 | 通过 |
| 2 / 0.5 | 0.07521% | 58.9451% | 15.1951% | 0.989009 | 失败 |
| 3 / 0.8 | 0.05801% | 1.0652% | 0.9108% | 0.999959 | 通过 |

相同 pre-step adapter/buffers、运行输入、RNG 下中间步仍失败，不能把此前
梯度偏差完全归因于候选上一步 optimizer 更新。候选仍进行 disposable
optimizer 更新，但下一步会被参考参数覆盖；本实验不评价候选独立训练轨迹。
这里还不能直接定位到某个 block 或断言具体算子机制。

新模块测试覆盖参数与非持久化 buffer 恢复、RNG 摘要、逐步参数变化、拒绝
错误设备/不完整 worker/错误签名/损坏快照/超预算、固定状态比较证据不匹配、
CLI 禁止混入正式 checkpoint。完整 adaptive 回归 127 passed、14 个既有警告；
bench/core adaptive 及新增测试 Ruff 通过。

#### FP32 重放控制组

把上述候选命令输出改为 `t10-zimage-replay-fp32-control`，precision 改为
`fp32-reference`，删除 `--loss-scale` 和全部 `--fp32-pattern`，保留相同
`--replay-reference`。比较文件为 `t10-zimage-replay-fp32-control-comparison.json`。
worker/supervisor 均 ok，134.49s，GPU peak 8,003,061,248 bytes，三个案例均
通过。以下 rel-L2 为比例，不是百分比：

| step | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 |
| --- | --- | --- | --- |
| 1 | 0 | 8.7797e-7 | 1.9215e-6 |
| 2 | 0 | 1.2049e-6 | 1.3812e-6 |
| 3 | 0 | 2.3709e-6 | 2.7282e-6 |

相同状态重放下 FP32 的误差远低于混合候选的中间步偏差，因此不能用重放
噪声或 optimizer 轨迹解释候选失败。重放的完整 backward 和 CUDA 同步完成，
未跳过 offloader backward 生命周期；没有生成正式 LoRA/checkpoint。

### 3080 Z-Image 固定状态 BF16 重放

沿前述录制命令，将 UUID 改为
`GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086`，输出改为
`3080-zimage-replay-reference`，得到同卡三个完整参考快照。
FP32 worker/supervisor ok，138.07s，GPU peak 8,003,061,248 bytes。

```bash
CUDA_VISIBLE_DEVICES=GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/3080-zimage-replay-bf16 \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision bf16 --capture-training --disposable-probe \
  --replay-reference output/adaptive-runtime-20260921/3080-zimage-replay-reference/attempt-000/training-replay

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/3080-zimage-replay-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/3080-zimage-replay-bf16/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/3080-zimage-replay-bf16-comparison.json
```

BF16 worker/supervisor ok，51.54s，GPU peak 4,040,108,544 bytes。此次传入
FP32 参考 noisy/prompt/target，内部仍为 BF16 权重/autocast，不是原生产 BF16
输入路径。逐步参数/buffer、输入和 RNG 摘要均匹配；以下 rel-L2 为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 1.4028% | 18.3181% | 19.4746% | 0.981200 | 失败 |
| 2 / 0.5 | 1.0384% | 140.9830% | 39.4172% | 0.933772 | 失败 |
| 3 / 0.8 | 1.0972% | 25.0964% | 27.5631% | 0.962925 | 失败 |

排除外部输入舍入及前序候选更新后，BF16 候选仍未通过当前实验预算；不能把
此前失败全归因于输入或训练轨迹。未做本轮 3080 FP32-to-FP32 重放重复，
T10 的重放控制组不能冒充 3080 重复证据。该结果仍不是长期质量结论，未改变
Z-Image 生产 BF16 限制、默认精度或实验误差阈值。

### T10 干预：全部 attention Linear 升 FP32 未改善

这是在相同 FP32 pre-step 状态上对整组算子做的受控干预，不是依据梯度误差
位置直接认定底层算子为误差源，也不是已经实现自动敏感层选择。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-replay-fp32-attention \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --capture-training --disposable-probe --loss-scale 1024 \
  --fp32-pattern '*.attention.*' --fp32-pattern '*.feed_forward.w2' \
  --replay-reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-replay

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-zimage-replay-fp32-attention/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-zimage-replay-fp32-attention-comparison.json
```

worker/supervisor 均 ok，125.04s；FP32 Linear 170 个，FP16 Linear 106 个，
GPU peak 6,191,620,608 bytes，比 72 层方案增加 886,863,872 bytes（约0.83GiB）。
除 precision profile 外保持 reference、swap24、scale1024 和三步输入相同；
各案例摘要匹配，272 个 LoRA 参数梯度完整。以下 rel-L2 单位为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08709% | 0.9401% | 2.7484% | 0.999622 | 失败 |
| 2 / 0.5 | 0.07821% | 65.2772% | 16.8172% | 0.986581 | 失败 |
| 3 / 0.8 | 0.05817% | 1.0473% | 1.0791% | 0.999942 | 通过 |

与 72 层方案相比增加显存却没有降低关键中间步误差，还令第 1 步超过梯度
rel-L2 门槛。当前不采用该提升方案；精度域之间的相互作用不保证误差随 FP32
层数单调下降。下一轮可对 MLP 投影做独立干预或增加 block 内部数值观测，
但必须继续用同状态完整梯度比较来验证，不能把更大的 FP32 名单当作改进。

本轮共完成六次三步 GPU worker：T10 参考、72 层重放、FP32 控制、170 层
attention 干预，以及 3080 参考和 BF16 重放。均是短诊断，不是长训稳定性或
生成质量验收。Anima 底模缺失、Krea FP32 参考内存问题、生产精度规划与完整
OOM 续训仍在待办，整体目标未完成。

### T10 干预：MLP Linear 升 FP32 仍未通过

使用已有 T10 replay reference、swap24、scale1024，把 72 层基础方案中的
`--fp32-pattern '*.feed_forward.w2'` 替换为 `--fp32-pattern '*.feed_forward.*'`，
其余 patterns 不变；输出为 `t10-zimage-replay-fp32-mlp`。
比较命令沿用前述 `compare_training`，candidate 换为该目录下的
`attempt-000/training-capture`，输出 `t10-zimage-replay-fp32-mlp-comparison.json`。

worker/supervisor ok，120.55s，GPU peak 7,013,935,616 bytes；140 个 FP32
Linear、136 个 FP16 Linear。各案例 pre-step 状态、输入和 RNG 摘要匹配。
以下 rel-L2 为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.08420% | 1.0451% | 3.0338% | 0.999543 | 失败 |
| 2 / 0.5 | 0.06949% | 53.2747% | 13.7935% | 0.990863 | 失败 |
| 3 / 0.8 | 0.05623% | 0.9313% | 0.6876% | 0.999976 | 通过 |

MLP 分组干预稍降低中间步误差，但仍超预算，第 1 步也失败，不采用为默认。
独立结构定位确认 attention/FFN 之外还有38个 Linear：时间嵌入2、latent输入1、
caption输入1、noise/main层adaLN调制32、final调制1、final输出1。
已把这些模块的 FP32 分组加入搜索器的 `conditioning` 候选；结构定位本身
不是敏感性或通过证据，实际结论必须来自固定状态对照。

### 数值配置搜索接线与初次启动失败

新增 `library/training/adaptive_runtime/profile_search.py` 和
`bench/adaptive_runtime/search_z_image_precision.py`，在每个完整候选后调用
固定状态比较器，不再只检查有限值。真实 OOM 只调 swap，数值超限换下一个
显式 profile，所有尝试计入同一上限。不得携带用户进度或 resume；没有通过
门槛的候选就明确失败，控制项不得伪装混合精度收益。

首次使用输出 `t10-zimage-numerical-profile-search` 时，在启动第一个 worker
之前写 summary 遇到 `FileNotFoundError: .../summary.tmp`：新目录尚未创建。
该失败没有运行 GPU worker，也没有生成有效任务目录。已修复为参数验证后、
初始 record 前创建独占输出目录，并增加 CLI 回归测试；重试使用新名称 `-v2`，
不伪造首次成功。记录 callback 自身 IO 异常直接终止；磁盘不可写时不承诺
错误报告还能被写入同一磁盘。

修复后运行命令：

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.search_z_image_precision \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-replay \
  --output output/adaptive-runtime-20260921/t10-zimage-numerical-profile-search-v2 \
  --profile seed --profile fp32-control --initial-swap 20 --max-swap 28 \
  --swap-increment 4 --max-attempts 6 --memory-limit-gib 6.3
```

该 allocator cap 只限制 worker 的 PyTorch 分配，不修改 GPU 或系统设置。
全 FP32 控制项是命令显式请求的诊断，不属于默认混合精度候选集合。
新增模块定向测试26 passed，完整 adaptive 回归153 passed / 14个既有警告，
bench/core adaptive及新增测试 Ruff通过。测试覆盖数值拒绝后换配置、OOM沿用
精度仅调swap、全局尝试上限、普通错误/host_limit/timeout/cancelled停止、
比较器异常终态、拒绝无效比较报告、显式控制项和CLI输出目录初始化。

`-v2` 最终为 `probe_validated`，真实尝试历史如下，未手工修改 summary：

| attempt | profile | swap | 结果 | 自动决策 | worker总秒数 |
| --- | --- | --- | --- | --- | --- |
| 0 | seed | 20 | CUDA OOM | 精度不变，swap升24 | 78.07 |
| 1 | seed | 24 | 三步运行ok，但数值门槛失败 | 换下个显式profile | 119.02 |
| 2 | fp32-control | 24 | CUDA OOM | 精度不变，swap升28 | 161.09 |
| 3 | fp32-control | 28 | 三步及全部数值门槛通过 | 选中短探针方案 | 199.96 |

最终276个 Linear 全部FP32，虽然实现复用名为 `fp16-islands` 的安装入口，
**实际没有FP16 Linear**，不属于混合精度加速成果。GPU peak 5,104,547,328
bytes（约4.75GiB），三步预测与参考逐位一致；输入梯度 rel-L2 分别为
7.3130e-7/1.4567e-6/1.3038e-6，LoRA梯度为
1.1583e-6/1.2081e-6/1.2693e-6（均为比例）。逐步重放摘要一致。
状态只表示所给三个案例通过，`precision_calibrated=false`、
`full_model_calibrated=false`、`production_ready=false` 保持不变。

此任务完成了真实“数值拒绝 -> 精度配置切换 -> OOM -> swap重试”的同命令
闭环，各worker均disposable，没有正式训练进度恢复，也没有正式adapter输出。
seed成功与控制项成功之间不得共享 optimizer/checkpoint；只重放同一参考状态。

### T10 conditioning 分组：显著改善，但仍拒绝

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.search_z_image_precision \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-replay \
  --output output/adaptive-runtime-20260921/t10-zimage-conditioning-profile-search \
  --profile conditioning --initial-swap 24 --max-swap 28 --swap-increment 4 --max-attempts 2
```

在seed基础上提升38个 conditioning/input/output Linear，总计110个FP32、
166个FP16 Linear，scale1024、swap24。worker/supervisor ok，125.77s，
GPU peak 5,409,084,928 bytes，比72层seed多104,328,192 bytes（约99.5MiB）。
全部参数/输入/RNG重放摘要一致，272个LoRA参数梯度完整。
`comparison-000.json` 的 rel-L2 以下为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.03993% | 0.6742% | 2.1722% | 0.999766 | 失败 |
| 2 / 0.5 | 0.02362% | 7.1512% | 1.9015% | 0.999820 | 失败 |
| 3 / 0.8 | 0.01916% | 0.8444% | 0.7297% | 0.999973 | 通过 |

相对72层seed的固定状态对照，中间步LoRA梯度从15.1951%降至1.9015%，
输入梯度从58.9451%降至7.1512%，且增加显存远小于整组attention/MLP。
这是分组干预的改善证据，不足以认定38个模块均敏感或某一模块为唯一原因。
第1步LoRA梯度仍超过2%，第2步输入梯度仍超限，因此搜索summary真实返回
`failed`、`next_reason=numerical_tolerance_failed`、`stop_reason=profiles_exhausted`，
没有选中配置，不放宽门槛。下一轮可验证conditioning与其他组组合并拆分
timestep/modulation/input/output，但不得假设组合一定单调改善。

本轮完成MLP干预、四次真实数值/OOM联合搜索worker与conditioning干预，
均已退出。新增搜索器仍是Z-Image限定候选实验，未接入正式训练/配置/WebUI；
Anima底模、Krea FP32参考内存和生产完整恢复仍未解决，整体交付未完成。

### Conditioning 与 MLP 组合：仍未通过

T10使用既有同卡FP32 replay reference、scale1024、swap24，将seed的
`*.feed_forward.w2`替换为`*.feed_forward.*`，另加conditioning五个模式
`t_embedder.*`、`all_x_embedder.*`、`cap_embedder.*`、`*.adaLN_modulation.*`、
`all_final_layer.*`。通过`run_probe --module probe_z_image_train`运行，其他
参数仍为256px、真实输入、`--capture-training --disposable-probe`和原参考
`--replay-reference`。结果目录为`t10-zimage-replay-conditioning-mlp`，
比较文件为`t10-zimage-replay-conditioning-mlp-comparison.json`，
均位于`output/adaptive-runtime-20260921/`，比较使用该目录的
`attempt-000/training-capture`与原T10参考。

worker/supervisor ok，132.25s，178个FP32 Linear、98个FP16 Linear，
GPU peak 7,118,263,808 bytes。以下rel-L2为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.02988% | 1.7588% | 6.3841% | 0.997975 | 失败 |
| 2 / 0.5 | 0.01460% | 6.4850% | 1.6858% | 0.999858 | 失败 |
| 3 / 0.8 | 0.01469% | 0.5390% | 0.4320% | 0.999991 | 通过 |

第2步有改善，但第1步LoRA梯度明显变差，因此不采用该组合。
新增`--profile conditioning-mlp`和`--profile conditioning-attention`显式配置，
默认候选顺序保持不变，组合并未自动成为生产默认。对应模式集合测试通过，
完整adaptive回归155 passed、14个既有警告，修改模块Ruff通过。

### T10 conditioning 与 attention 组合：首个混合候选短探针通过

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.search_z_image_precision \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --reference output/adaptive-runtime-20260921/t10-zimage-replay-reference/attempt-000/training-replay \
  --output output/adaptive-runtime-20260921/t10-zimage-conditioning-attention-search \
  --profile conditioning-attention --initial-swap 24 --max-swap 28 --max-attempts 2
```

worker/supervisor ok，130.03s，搜索器真实返回`probe_validated`。208个Linear
为FP32、68个为FP16（FFN的w1/w3），不再是全FP32控制项。仍使用FP32残差、
plain LoRA rank4/alpha4、scale1024、swap24、full checkpoint与256px真实输入。
峰值6,295,979,520 bytes（约5.86GiB），比同swap24 FP32参考的8,003,061,248
bytes少约21.3%。总耗时含加载/捕获，不能用于宣称稳态加速。

`comparison-000.json`逐步状态/输入/RNG/快照摘要均匹配，272个LoRA参数
梯度完整；以下rel-L2为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.03181% | 0.3156% | 0.9096% | 0.999959 | 通过 |
| 2 / 0.5 | 0.01767% | 0.2037% | 0.3767% | 0.999993 | 通过 |
| 3 / 0.8 | 0.01384% | 0.4942% | 0.3824% | 0.999993 | 通过 |

这是首个在这组固定状态案例上全部通过的T10混合Linear方案。结论只覆盖
现有单输入/三个sigma，不是独立训练轨迹、其他分辨率、长训或生成质量验收；
`precision_calibrated=false`、`full_model_calibrated=false`和`production_ready=false`
保持不变。组合必须显式选择，尚未改生产默认或自动搜索默认候选顺序。

### 3080 同配置交叉验证：未通过

沿上述`conditioning-attention`搜索命令，仅将UUID改为
`GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086`，reference改为
`output/adaptive-runtime-20260921/3080-zimage-replay-reference/attempt-000/training-replay`，
输出改为`output/adaptive-runtime-20260921/3080-zimage-conditioning-attention-search`。
使用3080自己的FP32参考，不跨卡比较梯度。候选仍为FP16/FP32组合，不是BF16。

worker/supervisor ok，123.99s，208个FP32与68个FP16 Linear，峰值仍为
6,295,979,520 bytes。三个案例摘要匹配，以下rel-L2为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.03391% | 0.7549% | 2.8962% | 0.999581 | 失败 |
| 2 / 0.5 | 0.01803% | 6.3701% | 1.6552% | 0.999864 | 失败 |
| 3 / 0.8 | 0.01282% | 0.4399% | 0.3038% | 0.999995 | 通过 |

搜索器返回`failed`，`stop_reason=profiles_exhausted`，没有选中配置。
同一组合在T10通过，不代表3080通过；两卡结果差异尚未通过同卡重复和算子
级对照定位原因，不直接归因于某个硬件指令。自动规划必须逐设备校准，不能
把一张卡的FP32例外列表当作其他卡的已验证策略。

本轮三次短GPU诊断均已完成：T10 conditioning-MLP失败、T10
conditioning-attention通过、3080 conditioning-attention失败。仍未验证T10
该候选独立更新轨迹、长训质量或稳态加速；三模型双卡完整目标未达成。

### T10 conditioning-attention 独立训练轨迹：中间步仍失败

去掉固定状态重放，每一步保留候选自己的AdamW更新；使用原始同卡FP32
三步训练参考`t10-zimage-fp32-reference`，不是带重放签名的参考。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-trajectory-conditioning-attention \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --capture-training --disposable-probe --loss-scale 1024 \
  --fp32-pattern '*.attention.*' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern 't_embedder.*' --fp32-pattern 'all_x_embedder.*' \
  --fp32-pattern 'cap_embedder.*' --fp32-pattern '*.adaLN_modulation.*' \
  --fp32-pattern 'all_final_layer.*'

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-zimage-fp32-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-zimage-trajectory-conditioning-attention/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-zimage-trajectory-conditioning-attention-comparison.json
```

worker/supervisor ok，113.52s，208个FP32/68个FP16 Linear，GPU峰值
6,295,979,520 bytes；三步均实际更新且没有保存正式adapter。比较确认同卡、
相同初始adapter、底模/输入/训练签名，272个LoRA梯度完整。scope为
`aggregate_short_training_trajectory_not_quality_certificate`，不是固定状态比较。
以下rel-L2为百分比：

| step / sigma | 预测 rel-L2 | 输入梯度 rel-L2 | LoRA 梯度 rel-L2 | LoRA cosine | 实验门槛 |
| --- | --- | --- | --- | --- | --- |
| 1 / 0.2 | 0.03181% | 0.3159% | 0.9143% | 0.999958 | 通过 |
| 2 / 0.5 | 0.01912% | 7.6097% | 2.3190% | 0.999732 | 失败 |
| 3 / 0.8 | 0.01272% | 0.3349% | 0.3256% | 0.999995 | 通过 |

固定状态全部通过不能替代候选独立训练轨迹验证；当前该候选仍未满足完整
三步轨迹门槛，不升级为生产可用。此结果包含前序更新差异，不能仅凭这一项
实验认定哪个优化器参数或具体算子造成放大。

### 实验检查点 scaler 状态格式

为后续带loss scaling的OOM恢复补齐状态基础：`checkpoint.py`新增可选
`scaler`参数；提供时使用`adaptive_fixed_input_probe_v2`，保存完整AMP
state_dict（scale、增长/回退系数、增长间隔、增长计数）。未提供且未缩放时
继续写v1，保持已有实验检查点兼容。带`loss_scale!=1`的v1或缺失scaler状态
显式拒绝，不静默以新scaler重新开始；恢复还核对目标scaler启用状态。

`scaler_state.py`检查字段完整性、有限数值、正scale和合法增长计数/策略。
CPU GradScaler测试证明第1步保存、恢复后第2步与不中断路径的network、
AdamW、RNG和scaler完全一致，包括scale128->256的增长计数转换。
另覆盖缺失scaler、伪造旧scaled checkpoint、NaN/非法policy/progress拒绝。
检查点相关定向测试15 passed，修改模块Ruff通过。

这次只实现状态格式与恢复API，尚未放开`validate_scaling()`的disposable限制，
尚未把scaler接入Krea/Z-Image worker的checkpoint调用；因此不能宣称scaled
OOM续训已验证。生产Accelerate检查点完全未改。旧未缩放恢复路径继续按v1运行。
本轮最终完整adaptive回归165 passed、14个既有弃用警告（16.95s），
bench/core adaptive及新增测试Ruff通过；GPU worker已退出，无遗留实验进程。

### Z-Image scaled checkpoint 接线与T10显式恢复

Z-Image实验worker新增`--scaled-checkpoint`。非默认loss scale保存/恢复需显式
该开关、`--checkpoint-every-step`、真实输入、FP16-islands，拒绝disposable、
capture和replay混用。保存发生在scaler.step/update、CUDA同步及更新记录之后，
恢复把同一scaler传入v2恢复API。未缩放路径仍传None并保持v1格式。
Krea worker、自动recovery CLI、生产Accelerate尚未接线。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-zimage-scaled-checkpoint \
  --timeout 420 --swap 24 --module probe_z_image_train -- \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --precision fp16-islands --checkpoint-every-step --scaled-checkpoint --loss-scale 1024 \
  --fp32-pattern '*.attention.*' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern 't_embedder.*' --fp32-pattern 'all_x_embedder.*' \
  --fp32-pattern 'cap_embedder.*' --fp32-pattern '*.adaLN_modulation.*' \
  --fp32-pattern 'all_final_layer.*'
```

首轮worker/supervisor ok，109.83s，三步真实更新，峰值6,295,979,520 bytes。
逐步生成v2检查点与实验adapter。第1步scaler scale1024、growth tracker1，
272组AdamW状态完整。随后运行同一命令，output改为
`output/adaptive-runtime-20260921/t10-zimage-scaled-resume26`、swap改26，追加：

```text
--resume output/adaptive-runtime-20260921/t10-zimage-scaled-checkpoint/attempt-000/checkpoints/step-000001.pt
```

新worker/supervisor ok，113.43s，只继续执行第2/3步，恢复后的峰值为
5,157,101,056 bytes。独立审计文件
`output/adaptive-runtime-20260921/t10-zimage-scaled-resume26-audit.json`确认：

- 历史为`(step1,swap24),(step2,swap26),(step3,swap26)`。
- 最终scale1024、growth tracker3，与不中断路径scaler状态一致。
- 全部272组AdamW step为3。
- 恢复运行最终检查点network与输出adapter逐位一致。
- 与不中断swap24运行相比，最终network最大参数差为3.9287086e-5，非逐位一致。

最后一项差异原因尚未通过同swap重复/确定性实验定位，不能将本测试写成
逐位恢复等价认证。两次均未发生OOM，这只是带GradScaler的真实GPU显式
检查点恢复，不是中途OOM自动恢复；精度策略质量也未因此认证。
实验产物只写独立output目录，不改变用户正式训练配置或生产默认。

本轮检查点/缩放接线定向测试42 passed；完整adaptive回归176 passed、
14个既有弃用警告（28.04s），bench及修改测试Ruff通过。GPU进程均已结束。

### Scaled Z-Image 自动OOM入口

`recovery.py`新增`--loss-scale`和`--scaled-checkpoint`，仅允许Z-Image
FP16-islands、非默认scale的显式组合。未显式开启却指定非默认scale、Krea
scaled恢复、BF16 scaled恢复均在worker启动前拒绝。每个重试worker保持
相同precision、FP32 patterns和scale；只由既有memory planner调整swap。
默认未缩放调用与v1实验检查点保持不变，生产入口未修改。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.recover_z_image \
  --weights /home/scv/nvme0n1p1/Z-Image \
  --inputs output/adaptive-runtime-20260921/zimage-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-zimage-scaled-auto-recovery \
  --precision fp16-islands --scaled-checkpoint --loss-scale 1024 \
  --initial-swap 24 --max-swap 28 --swap-increment 2 --max-attempts 3 \
  --worker-timeout 420 --memory-limit-gib 5.5 \
  --fp32-pattern '*.attention.*' --fp32-pattern '*.feed_forward.w2' \
  --fp32-pattern 't_embedder.*' --fp32-pattern 'all_x_embedder.*' \
  --fp32-pattern 'cap_embedder.*' --fp32-pattern '*.adaLN_modulation.*' \
  --fp32-pattern 'all_final_layer.*'
```

分配器上限只限制该worker，不修改显卡/系统内存设置。首个swap24 worker
在model_load阶段真实CUDA OOM，`optimizer_started=false`，没有已提交
checkpoint。这一分支是初始更新前自动重启，不是中途训练进度恢复。
自动恢复与前一节显式resume是不同任务，不可合并成同一次中途OOM续训证据。

新增测试覆盖首启/带resume时传递相同scale与开关，并拒绝不支持组合。
定向44 passed，完整adaptive183 passed、14个既有警告（32.66s），Ruff通过。

最终summary为`ok`，两次attempt：swap24加载OOM（86.59s）后，swap26
新进程三步成功（137.47s），不需要第3次尝试。成功worker峰值
5,157,101,056 bytes（约4.80GiB）。`t10-zimage-scaled-auto-recovery/audit.json`
验证最终schema为v2，scale1024、growth tracker3，272组AdamW step均为3，
最终检查点network与adapter输出逐位一致，明确`mid_training_resume=false`。
精度名单在两次尝试中未改变，没有因为OOM降低敏感区域精度。
本轮worker及审计进程均已结束；3080 scaled恢复、scaled中途OOM恢复、Krea
scaled接线和生产完整恢复仍未完成，不把本实验外推到这些要求。

### 3080 scaled自动OOM重试与统一产物审计

复用前述`recover_z_image`完整命令，仅将UUID换成
`GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086`，output换成
`output/adaptive-runtime-20260921/3080-zimage-scaled-auto-recovery`。
显式FP16-islands、208个FP32/68个FP16 Linear、scale1024、5.5GiB分配器
上限、swap24起步、增量2、最多3次均保持不变。不是BF16模式，也没有改
系统设置或生产精度默认。

实际两次尝试：swap24在model_load CUDA OOM、optimizer未开始（80.14s）；
新进程swap26完成三步（155.98s），最终任务ok，GPU峰值5,157,101,056 bytes。
本次没有中途checkpoint resume；恢复成功不改变该组合在3080数值门槛失败
的既有结论，也不构成长训质量认证。

新增可复用审计命令`bench.adaptive_runtime.audit_recovery`，读取真实summary、
逐attempt的worker/supervisor、最终检查点及adapter。检查attempt连续性、
precision/scale/swap和FP32名单一致性、状态记录一致性、检查点输入签名与
更新历史、AdamW step、最终输出逐位一致性。声明resume时还检查源检查点
历史前缀和新增进度，不只看`resumed_from`字段。只适用固定输入实验产物，
不是生产状态或不中断轨迹等价的验证器。

```bash
.venv/bin/python -m bench.adaptive_runtime.audit_recovery \
  --directory output/adaptive-runtime-20260921/3080-zimage-scaled-auto-recovery \
  --output output/adaptive-runtime-20260921/3080-zimage-scaled-auto-recovery/artifact-audit.json

.venv/bin/python -m bench.adaptive_runtime.audit_recovery \
  --directory output/adaptive-runtime-20260921/t10-zimage-scaled-auto-recovery \
  --output output/adaptive-runtime-20260921/t10-zimage-scaled-auto-recovery/artifact-audit.json
```

两卡均`audited`：v2，step3，scale1024、growth tracker3，272组AdamW
step均为3，checkpoint与adapter逐位一致，`checkpoint_resume_observed=false`，
`production_ready=false`。审计不覆盖scheduler、sampler、多worker RNG等
生产状态，不能据此把三模型双卡完整目标标为完成。

新增7项toy测试覆盖正常产物、worker/summary不一致、错误adapter、错误
optimizer进度、输入签名变化、声称resume但没有新增进度的拒绝。完整adaptive
回归190 passed、14个既有弃用警告（16.96s），新模块和测试Ruff通过。
本轮GPU worker和审计进程已退出。

### Krea T10 scaled检查点接线及真实OOM重试

Krea实验worker新增显式`--scaled-checkpoint`，save/restore均传递scaler；
共享recovery允许Krea非量化FP16-islands缩放路径，仍拒绝NF4。
非缩放默认及v1检查点保持不变，生产Accelerate未接线。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.recover_krea \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-auto-recovery \
  --precision fp16-islands --scaled-checkpoint --loss-scale 1024 \
  --initial-swap 20 --max-swap 24 --swap-increment 4 --max-attempts 2 \
  --worker-timeout 420 --memory-limit-gib 9.05

.venv/bin/python -m bench.adaptive_runtime.audit_recovery \
  --directory output/adaptive-runtime-20260921/t10-krea-scaled-auto-recovery \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-auto-recovery/artifact-audit.json
```

实测264个普通Linear为FP16、外部残差FP32，未添加额外FP32 Linear名单。
swap20在首个backward真实CUDA OOM，optimizer未开始，耗时214.95s；
自动新进程swap24完成三步，整个worker292.80s。步时16.18/16.50/15.20s，
峰值6,258,903,040 bytes（约5.83GiB）。该步时不是隔离吞吐基准，未提供
同条件FP32速度对照，不宣称加速。3080当时有GPU活动，本轮未占用该卡。

统一审计结果audited：schema v2，step3，scale1024，growth tracker3，
392组AdamW状态step均为3，checkpoint与输出adapter逐位一致。
`checkpoint_resume_observed=false`，是首次更新前重启，不是中途OOM续训。
`precision_calibrated=false`、`production_ready=false`；Krea完整FP32参考
仍受主机内存限制，有限值与恢复通过不能替代数值校准。

定向52项测试通过；全adaptive回归195 passed、14个既有弃用警告。
首次60秒timeout运行虽打印195 passed，退出阶段返回124，未算成功退出；
随后180秒上限重跑195 passed（42.82s），进程退出码0。修改文件Ruff通过，
独立只读核验未发现接线漏洞。GPU worker及审计进程均正常退出。

### Krea scaled中途OOM：5.75GiB定位失败记录

从前轮swap24的第1/2步allocated峰值约5.69/5.83GiB推算，将分配器
上限设为5.75GiB，初始swap24、最大26、增量2、最多2次、timeout420s，
其余参数与上一节相同，output为`t10-krea-scaled-midstep-recovery`。
目录名描述测试意图，不代表已经观察到中途续训。

实际swap24首个backward就OOM，optimizer未开始，211.39s；自动swap26
三步成功，280.99s，峰值4,522,461,184 bytes（约4.21GiB）。统一审计
audited、v2、scale1024、tracker3、392组AdamW step3、输出与检查点一致，
但`checkpoint_resume_observed=false`。这是首次更新前重启，未达到中途
恢复目标。分配器限制不能仅靠allocated峰值推算，还需考虑reserved内存。
Krea probe因此新增每步`peak_reserved`观测，不改变训练计算或恢复策略。

### Krea T10带缩放真实中途OOM恢复成功

第二次尝试将上限改为6.0GiB，其余精度及训练设置不变：

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.recover_krea \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-midstep-recovery-6g \
  --precision fp16-islands --scaled-checkpoint --loss-scale 1024 \
  --initial-swap 24 --max-swap 26 --swap-increment 2 --max-attempts 2 \
  --worker-timeout 420 --memory-limit-gib 6.0

.venv/bin/python -m bench.adaptive_runtime.audit_recovery \
  --directory output/adaptive-runtime-20260921/t10-krea-scaled-midstep-recovery-6g \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-midstep-recovery-6g/artifact-audit.json
```

- attempt000：swap24提交step1（sigma0.2），allocated峰值6,114,395,648，
  reserved峰值6,442,450,944 bytes。sigma0.5的第2步backward真实CUDA OOM，
  请求108MiB失败；optimizer_started=true，保留已提交step1，worker241.55s。
- attempt001：自动新进程swap26从attempt000/checkpoints/step-000001.pt
  恢复，第2/3步成功，worker286.84s。历史为(1,24),(2,26),(3,26)，新增步时
  16.07/16.35s；恢复worker allocated峰值4,522,461,184（约4.21GiB），
  reserved峰值4,966,055,936 bytes。未改变precision、FP32名单或scale。
- artifact-audit.json：audited、schema v2、step3、scale1024、tracker3，
  392组AdamW状态step均为3，最终checkpoint/output逐位一致，
  checkpoint_resume_observed=true。审计新增源/最终训练签名一致性检查。
- final-adapter-comparison.json：与先前`t10-krea-scaled-auto-recovery`
  的swap24不中断三步adapter比较，588个张量逐位一致，max_abs_delta=0。
  只比较最终adapter，不宣称完整中间轨迹或生产全部状态等价。

该实验首次补齐Krea T10带scaler的真实中途OOM自动续训证据。它仍是256px、
固定真实缓存、batch1、rank4、三步实验；不认证Krea FP16数值精度、长训质量
或速度，不替代Anima和其他模型/设备的缺失测试。3080本轮仍未占用。

新增审计成功恢复/篡改源签名拒绝测试，及reserved字段断言；定向14 passed，
全adaptive回归197 passed、14个既有弃用警告（55.98s），退出码0。
修改Python文件Ruff通过。两次GPU实验、测试、审计与对照进程均正常结束。

### Krea T10完整FP32参考及缩放FP16整网对照

新增实验`reference_storage.py`及`--reference-bf16-storage`。普通BF16初次
placement后，保留冻结CPU masters的BF16源值，将GPU浮点参数/buffer提升
FP32，设置后续交换目标FP32并清除旧plan cache。运行时pre-hook检查模块
参数/buffer的CUDA FP32状态并拒绝CUDA autocast，覆盖checkpoint重算。
仅允许foreach、swap>0、disposable training capture；不接入生产或resume。
公共offloader未修改，原主机内存保护未放宽。BF16源值可精确提升FP32，
这里是同一BF16来源权重的FP32计算参考，不是原始FP32预训练权重。

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-krea-fp32-bf16-storage-reference \
  --timeout 420 --swap 26 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --precision fp32-reference --reference-bf16-storage --disposable-probe --capture-training

CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  .venv/bin/python -m bench.adaptive_runtime.run_probe \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-training-comparison \
  --timeout 420 --swap 26 --module probe_krea_train -- \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors \
  --inputs output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --precision fp16-islands --loss-scale 1024 --disposable-probe --capture-training

.venv/bin/python -m bench.adaptive_runtime.compare_training \
  --reference output/adaptive-runtime-20260921/t10-krea-fp32-bf16-storage-reference/attempt-000/training-capture \
  --candidate output/adaptive-runtime-20260921/t10-krea-scaled-training-comparison/attempt-000/training-capture \
  --output output/adaptive-runtime-20260921/t10-krea-scaled-training-comparison/comparison.json
```

两次worker均ok，三步完整预测、输入梯度和全部LoRA梯度已落盘。参考耗时
116.86s，3027次模块检查通过，CPU masters24,310,185,984 bytes（22.64GiB），
GPU allocated峰值7,587,844,608（7.07GiB）。参考步时13.42/12.65/13.97s。
候选耗时242.94s，264个FP16 Linear、外部FP32残差、scale1024，GPU峰值
4,522,461,184（4.21GiB），较参考低约40.4%；步时16.70/13.94/14.47s。
这组短测没有显示加速，不能只根据省显存宣称性能收益。

比较器确认同GPU UUID、源缓存/底模/初始adapter/训练签名一致，结果如下：

| step | prediction rel-L2 | input gradient rel-L2 | LoRA gradient rel-L2 |
| --- | --- | --- | --- |
| 1 | 0.0007912 | 0.0439710 | 0.0816669 |
| 2 | 0.0004930 | 0.0141977 | 0.0220139 |
| 3 | 0.0004575 | 0.0111855 | 0.0176277 |

within_experimental_tolerances=false：前两步LoRA梯度超2%门槛，第1步输入
梯度也超门槛，LoRA cosine=0.996674低于0.999。不能将有限值/OOM恢复通过
升级为精度认证。这是独立短训轨迹比较，后两步含各自参数更新差异；Krea
固定状态重放及敏感层定位仍待实现，门槛亦非已验证的长期质量标准。

新增11项reference storage测试在T10全部通过，包括真实CUDA三轮交换及
checkpoint重算，输出/输入梯度/可训练梯度与FP32基线逐位一致、BF16 masters
不变、执行guard和模式拒绝。CPU完整adaptive回归207 passed、1 CUDA项跳过，
14个既有警告；该CUDA项已单独真机通过。Ruff通过。所有测试与worker已结束。

## 下一阶段生产边界

- 精度计划需要在 `library/training/model_loading.py` 的权重加载之后、adapter
  apply 与 offloader master/compile 之前固定，且必须尊重各 model family 的限制。
- 既有 `auto_block_swap/coordinator.py::calibrate_if_requested()` 只负责 swap；
  不能把硬件候选选择或有限值探针直接当成数值校准通过。
- 生产恢复要沿 `checkpoints.py` 的 Accelerate save/load hooks 和
  `train_session.py` 的注册/恢复顺序接入，核验 scheduler、epoch、数据顺序、
  worker RNG 与梯度累积。固定输入 worker 的成功不能证明这些状态已恢复。
- Anima 完整底模仍缺路径；已向用户询问路径或具体版本的下载授权，没有擅自下载。

## 实际训练桥接与启动 OOM 重试（本轮）

按用户调整的方向，已从独立探针推进到真实训练入口。新增
`adaptive_precision=fp16_fp32`、显式 `adaptive_fp32_modules` 与
`adaptive_loss_scale`。仅非量化 Krea plain LoRA、torch attention、full checkpoint
等受限组合；默认关闭，不改变生产 NF4 BF16 限制，也不认证自动敏感名单。
精度域在 adapter/offloader 建立前安装，训练保留 FP32 残差和 LoRA 参数，
由 Accelerate 原生 scaler 执行缩放、unscale 和 clip。

前序小模型 CUDA 验证覆盖生产 LoRAModule、真实 Accelerate、unscaled clip，
以及保存第 1 步后重放第 2 步：adapter/optimizer 逐位相同、scheduler/scaler
状态相同。证据为 `output/adaptive-runtime-20260921/training-precision-bridge-tests.xml`；
这不是整模型 Krea 训练或数据游标恢复的证据。

本轮补齐 CLI supervisor -> 独立 training_worker -> 实际 session/loop 的接线。
冻结已合并配置，OOM 只调整 swap；只识别异常类型而非日志正则。首次 optimizer
更新前允许加载/搬运、forward、backward OOM 有界重试，普通错误、主机内存、
超时、取消停止。私有参数文件使用 0600，并在 worker 读取后或 supervisor
清理时删除。没有放宽主机内存保留量，没有启动长训练或修改显卡设置。

实际 loop 在 optimizer 前标记可能更新，scaler 跳步立即停止；完整更新后保存
Accelerate 状态到临时目录，检查必要文件和步数再原子发布，保留最近一份。
保存中失败不会发布临时目录或覆盖前一份已发布快照。仅以 `saved_state` 展示，
不会伪称可供自动数据恢复的 `committed_checkpoint`。

本轮明确发现的数据边界：bucket_indices 和 bucket 内容会原地 shuffle，
getitem 使用随机状态，Accelerate 数据预取与训练随机数交错。普通 save_state +
skip_first_batches 不足以证明下一批一致。因此更新后 OOM 会停在最近快照，
`data_cursor_resume_supported=false`；数据状态协议完成前不开放自动续训。

当前 nvidia-smi 只看到 RTX 3080 20GB（UUID 仍为 `GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086`），
约 1.6GiB 显存、88% 利用率；T10 未枚举。这轮使用 CPU/模拟异常及小模型状态保存
回归，没有整模 GPU 热测，不把历史双卡固定输入结果升级为实际训练通过。

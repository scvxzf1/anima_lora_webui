# Qwen Image 2.1 Flash CMP 90HX 热测

日期：2026-09-27
状态：A/B/A 与 60-step 复测完成；阶段 2 的真实形状 kernel 与 GPU wait 仍待补测。metadata cache 缺乏足够的整步收益证据，未采纳。Flash 仍为显式 opt-in，非默认后端。
结论：在 CMP 90HX 的同机 A/B/A 和 60-step 复测中，Flash 稳态步时比 torch 后端快约 7.1%。早先一次长测的慢值没有复现，原因未确定；此结果仅适用于当前硬件和配置。

## Mask 契约

Qwen 文本编码器使用左填充（`library/models/qwen_image_2_1/strategy.py`）。Diffusers 通用
`flash_varlen` backend 将 mask 中有效 key 的数量当作有效前缀长度；左填充时会选中 padding，
漏掉有效文本 token。初次直接映射的 GPU 探针与原生路径 output rel-L2 为 17.86%、输入梯度
rel-L2 为 14.49%，因此该通用 varlen 路径判为不兼容。

最终实现位于 `library/models/qwen_image_2_1/attention_backend.py`：FlashAttention 2 varlen
按完整 bool key-valid mask 实际 gather K/V；文本前缀的因果三角 attention 保持 native SDPA，
图像查询使用 packed FlashAttention。动态 packing wrapper 在编译 checkpoint block 外执行，
且不改变 Diffusers 的全局 attention backend。

## 数值探针

- GPU：CMP 90HX，BF16；单层 Qwen 2.1 attention，batch 1、64 个联合 token、前 16 个 key 为左 padding。
- 与相同权重/输入的 native SDPA 对照：output max abs `0.0009766`、rel-L2 `0.001058`；输入梯度 max abs `3.81e-6`、rel-L2 `0.002323`、cosine `0.9999973`。
- Flash 前向与反向均调用真实 `flash_attn_varlen_func`，非模拟 kernel。

## 训练热测

- GPU：NVIDIA CMP 90HX，SM 8.6；训练设备 `cuda:0`。运行环境：PyTorch 2.12.0+cu130、Diffusers 0.41.0.dev0、FlashAttention 2.8.3。
- Qwen Image 2.1 T2I、BF16、plain LoRA rank 16、batch size 1、fused AdamW；full gradient checkpointing、`blocks_to_swap=24/32`、32/32 block compile（Inductor、dynamic sequence）。唯一差异后端为 `attn_mode="flash"`。
- 同一张 `672x1536` 图片重复 60 次，60 optimizer steps。`progress.jsonl` 结束状态 `ok`、`final_step=60`、`error=null`；训练进程退出码 0，LoRA 检查点已保存（32.1 MB）。
- PyTorch 峰值 allocated/reserved 为 5.51/7.61 GiB；设备级显存采样约 8088 MiB。训练期间约每 10 秒采样一次：GPU 利用率基本 100%，温度 47–53°C，采样功耗最高约 240.6 W。
- 这次首轮长测（`output/runs/qwen21-hot-flash-90hx-20260926-60step/`）的 30 个有效 `recent_s_per_step` 记录中位数为 `10.051 s`、均值 `9.689 s`（原记录中的中位数 `10.16 s` 有误）。step 2–10 为约 7.45–7.56 秒，step 12 后升至约 9.1–10.8 秒，末段回落；该慢值在下述复测中没有重现，原因未确定，不据此断言是热降频。
- Inductor 有一条无法 trace Python `list.append` 的非致命警告；训练仍完成全部步骤并保存。应将 compile/Flash 性能视作当前硬件实测，不外推其他 GPU。

### A/B/A 与 60-step 复测

- 同一数据、seed、模型和训练配置依序运行 torch A1、Flash B、torch A2，每组 20 steps。每组 10 个有效 `recent_s_per_step` 样本的中位数分别为 `7.948 s`、`7.381 s`、`7.953 s`；两次 torch 控制相差约 0.06%，Flash B 比控制组均值快约 7.2%。
- 随后以相同 Flash 配置运行 60 steps：`output/runs/qwen21-flash-phase1-flash-b2-20260927/`。进度记录为 `ok`、`final_step=60`，进程退出码 0，checkpoint 已保存。30 个有效步时样本中位数 `7.384 s`，范围 `7.380–7.391 s`；没有复现 step 12 后的长窗漂移。
- B2 完整进度条约 7 分 59 秒（含编译/预热）。PyTorch peak allocated/reserved 为 `5.51/7.61 GiB`。本轮 2 秒遥测共 231 个 90HX 样本；其中 GPU 利用率至少 90% 的 199 个样本平均利用率 `99.8%`，温度 `48–54°C`，该活动样本平均/峰值功耗 `174.5/243.5 W`，设备显存采样约 `8088 MiB`。遥测文件为该 run 目录下的 `gpu_telemetry.csv`。

## Attention 与换块剖析

- Nsight/CUPTI 在本机返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，没有得到完整训练 CUDA kernel trace；硬件性能计数器也受系统权限限制。本轮未改驱动或系统权限，因此不能把整步训练时间完整分摊到 attention、GEMM、换块等部分。
- 未编译的单层 attention eager 探针使用 BF16、batch 1、75 个文本 token + 4032 个图像 token、24 heads × 128，前向加反向；真实缓存 mask 全有效。torch 外层中位数约 `68.6 ms`，两次 dispatch 的 CUDA Event 合计约 `7.47 ms`。Flash 外层中位数约 `29.4 ms`；每次 packing CPU wall 约 `4.84 ms`，其中 FA2 调用约 `0.28 ms` CPU / `3.05 ms` CUDA Event，剩余 packing 约 `4.56 ms` CPU。后续源码及权重 header 审阅确认真实模型为 **32 heads × 128**；该探针既非真实 head 数，也非编译后的整步分项测量。
- torch 模式的单步 block-swap profile（`output/runs/qwen21-flash-phase2-torch-20260927/block_swap.jsonl`）有 forward/backward wait 各 144 条记录。`wait_ms` 中位数分别约 `0.052/0.072 ms`，p95 约 `0.307/0.224 ms`；这是 wait 事件的 host 阻塞时间，不包含或代表完整传输成本。异步 H2D 传输之间存在重叠，不能把各事件传输时长简单相加当作训练关键路径。

### 阶段 2A–2C 复核

- 旧慢测和当前快测 snapshot 均记录 Git `5b60d1113`、BF16、full checkpoint、swap24、32-block dynamic compile、同一数据路径。当前快测显式 `seed=1234`，旧慢测没有记录 seed；两次运行时工作树未被 Git hash 完整指纹化。旧慢测也没有同粒度 GPU 遥测。因此旧的约 `10.051 s/step` 中位数尚不能归因于 Flash kernel、热降频或单一配置差异。
- 对称的阶段 2B 短窗分别写入 `output/runs/qwen21-phase2b-flash-r1-20260927/` 和 `output/runs/qwen21-phase2b-torch-r1-20260927/`，两侧均保存 block-swap profile。`--profile_steps 3-3` 在第 3 步后正常停止；其单步值受启动/编译状态影响，不能替代 A/B/A 稳态基线。
- 本机 `torch.profiler` 没有 CUDA kernel events；Nsight/CUPTI 返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`。因此本轮使用 CUDA Event、强制 SDPA backend 对照与 swap JSONL；没有可审计的完整训练 kernel 时间占比。
- 独立 attention probe 固定 BF16、B=1、Q=4032、K=4107、24 heads、head_dim=128、全有效 bool key mask。`attention_microbench_r3.json` 中 default native 为 `53.33 ms`，强制 mem-efficient 为 `52.96 ms`，强制 cuDNN 中位数 `16.30 ms` 但有严重长尾；关闭 mem-efficient 后约 `88.7 ms`，关闭 cuDNN 后仍约 `52.3 ms`。这些 backend toggle 与强制对照支持 **24-head 探针**选用 PyTorch mem-efficient SDPA 的判断，不能确认真实 32-head 训练的分派；也不能把 cuDNN 的较快单次中位数外推为默认训练行为。阶段 2C 的真实模型 kernel 确认尚未完成。
- 同一 24-head probe 的 raw FA2 varlen 为 `11.96 ms`，当时 production wrapper 为 `14.31 ms`。该形状下 FA2 本身比 native 路径快，额外开销主要在 wrapper 的 mask 元数据和 K/V packing；这仍不是完整训练中的 attention 占比。

### 阶段 3 缓存候选

- 临时原型缓存 mask-derived lengths、`cu_seqlens` 与 valid indices，对全有效 mask 跳过 K/V gather。优化后的 `attention_microbench_r4.json`（同目录）测得 raw FA2 `12.13 ms`、production wrapper `12.61 ms`、固定全有效 prototype `12.15 ms`（均为 forward+backward 中位数）。wrapper 对比 r3 下降约 `1.7 ms`，但 r3/r4 是不同时间的短 probe，不能据此声明整步加速。
- 真实 CMP 90HX BF16 GPU 对照覆盖全有效 mask、batch=2 不同有效 K 长度、无效 K/V 梯度为零及 non-reentrant checkpoint 重算。临时原型与独立 raw-varlen reference 的输出 max abs 和 Q/K/V 梯度 rel-L2 均为 `0`（本次小形状输入）。mask 设备不匹配现在会在 backend 入口明确拒绝。
- 临时原型的 20-step 训练产物位于 `output/runs/qwen21-phase3-flash-cache-r2-20260927/`：结束状态 `ok`、20/20 步、检查点已保存，peak allocated/reserved `5.44/7.55 GiB`。10 个稳态记录的步时中位数 `7.3756 s`，旧 Flash B 为 `7.3811 s`、B2 60-step 为 `7.3845 s`；相对 B 的差异仅约 `0.07%`。该轮 snapshot 的 Git hash 为 `4df001242`（旧对照为 `5b60d1113`），还额外开启 swap profile 且无同粒度 GPU 温频遥测，不是严格的单因素速度证明。
- Flash swap profile 有 960 条 wait 事件，host wait 中位数 `0.053 ms`、最大 `0.480 ms`。记录的 `gpu_wait_ms=0` 是未启用 `ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT` 时的默认填充值，不能据此排除 GPU 等待传输；host wait 也不代表 GPU wait。这批事件的 `step` 全为 1（计数在 prepare 时递增），不能直接按它划分 optimizer 步。H2D 与计算异步重叠，不应累加每条 `h2d_ms` 作为整步传输开销。没有证据表明 metadata cache 改善了换块关键路径。
- 按“完整步时稳定收益至少约 1%”的预设门槛，缓存/all-valid 特化已从生产 backend 撤下，仅保留 mask 同设备校验。阶段 4 不再为该候选改 compile wrapper、checkpoint 或 swap overlap，也不为该候选重复阶段 5 长测；已有未优化 Flash 的 60-step 复测仍是当前长窗证据。若未来取得完整 kernel trace 或新的性能异常，应重新建立同条件对照后再开启。

## 后续结构性审阅

同日硬件只读查询显示 90HX 的 current/max PCIe link 均为 1.0 ×16，sysfs 也报告 `2.5 GT/s PCIe`。上述 20-step Flash swap profile 记录每块约 416 MiB、每步共 48 次整块 H2D，约 **19.50 GiB/step**。按 PCIe 1.0 ×16 理想单向 4 GB/s 计算，传输下限约 **5.23 秒/步**；它与计算有重叠，不能作为可直接从步时扣除的独立耗时。这使压缩权重传输及低比特驻留成为比 metadata 小修更值得优先验证的方向。

完整证据、INT8 首次加载/重算一致性风险和高风险方案排序见[结构性优化审阅](../proposal/qwen_image_2_1_90hx_optimization_20260927.md)。以上仅为源码、已有产物及硬件查询审阅，没有新增训练性能结果。

## 验收边界

早期定向回归 `111 passed`；本轮撤下缓存候选后，attention/family/block-swap 定向回归 `38 passed`。这验证了当前 BF16 + full checkpoint + swap24 + compile + Flash 组合可以训练和保存；没有验证收敛、样图质量、保存后重载/续训、其他分辨率/批量、其他 swap 数或其他硬件。Flash 在本机受控测试中更快，但由于证据仅覆盖一台 GPU 和一组配置，方法配置仍默认 torch；使用 Flash 需显式设置 `attn_mode="flash"`。

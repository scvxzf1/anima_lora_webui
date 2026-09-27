# Qwen Image 2.1 Flash CMP 90HX 热测

日期：2026-09-27
状态：正确性、训练保存与定向性能复测通过；Flash 仍为显式 opt-in，非默认后端。
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
- 未编译的单层 attention eager 探针使用 BF16、batch 1、75 个文本 token + 4032 个图像 token、24 heads × 128，前向加反向；真实缓存 mask 全有效。torch 外层中位数约 `68.6 ms`，两次 dispatch 的 CUDA Event 合计约 `7.47 ms`。Flash 外层中位数约 `29.4 ms`；每次 packing CPU wall 约 `4.84 ms`，其中 FA2 调用约 `0.28 ms` CPU / `3.05 ms` CUDA Event，剩余 packing 约 `4.56 ms` CPU。该 eager 单层探针不能外推为编译后整步的分项耗时。
- torch 模式的单步 block-swap profile（`output/runs/qwen21-flash-phase2-torch-20260927/block_swap.jsonl`）有 forward/backward wait 各 144 条记录。`wait_ms` 中位数分别约 `0.052/0.072 ms`，p95 约 `0.307/0.224 ms`；这是 wait 事件的 host 阻塞时间，不包含或代表完整传输成本。异步 H2D 传输之间存在重叠，不能把各事件传输时长简单相加当作训练关键路径。

## 验收边界

定向回归 `111 passed`。这验证了当前 BF16 + full checkpoint + swap24 + compile + Flash 组合可以训练和保存；没有验证收敛、样图质量、保存后重载/续训、其他分辨率/批量、其他 swap 数或其他硬件。Flash 在本机受控测试中更快，但由于证据仅覆盖一台 GPU 和一组配置，方法配置仍默认 torch；使用 Flash 需显式设置 `attn_mode="flash"`。

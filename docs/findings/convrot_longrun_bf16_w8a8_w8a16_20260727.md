# ConvRot BF16 / W8A8 / W8A16 长训审计

状态：已完成 / 单次长训证据 / 最终 epoch 保存回归已修复
适用版本：Git `af19e8dc`–`bf66104c`
设备：NVIDIA GeForce RTX 3080 10GB（SM86）
日期：2026-07-27 至 2026-07-28

## 结论

三组任务都正常跑满 6 epoch / 1710 steps，没有 OOM、NaN、重试或中断。W8A8 和
W8A16 的 loss 分布与 BF16 接近，逐 epoch 样图均保持稳定且没有观察到量化导致的
明显崩坏；但因为没有固定训练全局 seed、验证集被自动禁用且只有一个唯一 sample
prompt，本次结果只能证明“可稳定长训”，不能证明严格质量等价。

在本次 `blocks_to_swap=26` 的真实配置下，稳态性能为：

- W8A16 比 BF16 慢约 **8.3%**。
- W8A8 比 BF16 慢约 **15.5%**，比 W8A16 慢约 **6.7%**。
- 两种 ConvRot 的峰值 allocated 都约 **4.10 GB**，而 BF16 仅约 **2.03 GB**。

因此，本次长训不支持“当时的 ConvRot 在重度 block swap 下节省显存”的说法。后续
residency probe 已确认根因：ConvRot INT8 权重 buffer 挂在 LoRA patch 上并常驻 GPU，
没有随 DiT block 一起交换，因而抵消了冻结 base weight 的节省。该结构性问题已于
2026-08-04 修复，见“Residency 修复与消融”。

三组还共同暴露出一个产物完整性问题：虽然训练到 epoch 6 / step 1710，但普通权重、
checkpoint state 和 sample 都只保存到 epoch 5 / step 1425。最后 285 steps 没有可用
产物，当前能公平比较的是 epoch 5 权重和样图。该共享保存回归已于
2026-08-04 在当前工作树修复，但历史运行的 epoch 6 权重无法事后重建。

## 运行目录

| 模式 | 运行目录 | Git |
| --- | --- | --- |
| BF16 | `anima缓存/okkotsu_goddess_725_75_tag-20260728-041344` | `bf66104c` |
| W8A8 | `anima缓存/okkotsu_goddess_725_75_tag-20260727-203533` | `bf66104c` |
| W8A16 | `anima缓存/okkotsu_goddess_725_75_tag-20260727-165411` | `af19e8dc` |

`af19e8dc..bf66104c` 之间只有 history WebUI 性能改动，没有训练/runtime 改动，因此
版本差异不影响本实验的训练内核比较。

## 可比性

逐字段比较 `config.runtime.toml` 后，除运行目录派生路径和 `base_compute` 外，三组配置
一致：

- LoRA dim/alpha：32/32。
- BF16 mixed/save precision，AdamW，LR `2e-5`，warmup 500。
- batch 1，gradient accumulation 1，6 epoch。
- torch compile + dynamic sequence，compile scope `all`。
- Flash attention，gradient checkpointing。
- `blocks_to_swap=26`，BF16 transfer，foreach restore。
- ConvRot scope `all`。
- 57 张源图，repeats 5，285 steps/epoch。
- 三组共享 cache fingerprint `5b3aa9085fa22a57`，A/B/C cache 全部复用。

限制：

- 配置没有固定训练全局 seed，三条 loss 轨迹不是逐 step 配对实验。
- 训练池只有 57 张图，小于 100，`validation_split=0.025` 被自动禁用。
- 没有 regularization images。
- sample prompt 文件的两行完全相同，且 seed 都是 114。每个 epoch 的 `_00` 和 `_01`
  PNG SHA-256 相同，所以所谓 10 张样图实际只有 5 张唯一图片。

## 性能和显存

稳态 step 时间取 `progress.jsonl` 相邻有效 step 的间隔，排除保存、采样等大间隔。总
任务时间取 `run_start.ts` 到 `run_end.ts`。

| 模式 | 稳态 s/step | 相对 BF16 | 任务总时长 | 峰值 allocated | 峰值 reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1.2662 | 1.000x | 2558.6 s | 2.0297 GB | 2.2773 GB |
| W8A16 | 1.3708 | 1.083x | 2737.4 s | 4.1103 GB | 4.3770 GB |
| W8A8 | 1.4626 | 1.155x | 2944.0 s | 4.1009 GB | 4.3965 GB |

这里不能沿用早期 no/low-block-swap microbench 的 `5.00 / 4.34 / 4.49 GB` 结论。
本次任务组合改变了 residency：BF16 的 26 个 blocks 能被交换出去，而 ConvRot 的全量
INT8 base buffers 看起来仍然常驻。

W8A8 相对 W8A16 多出的约 6.7% 稳态时间，与当前实现中的在线 activation RHT、动态
absmax INT8 量化、`torch._int_mm`、FP32 post-scale 以及 FP32 STE backward 一致。
SM86 不能使用 `torch._scaled_mm`，所以本结果不能直接类比 SM89 RTX 4070 的融合实现。

## Residency 修复与 `0/12/26` 消融（2026-08-04）

量化 weight/scale 现由实际 base `Linear` 下的 runtime-only carrier 持有，归属对应
DiT block，并复用 offloader 的 frozen-weight protocol 随 block 在 CPU/GPU 间迁移。
carrier 不进入 optimizer 或 `state_dict`。8-step 独立进程复测如下：

| 模式 | swap | sec/step | peak VRAM | apply 后 allocated | payload CPU/GPU tensors |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0 | 1.954 | 5.200 GB | 4.031 GB | 0 / 0 |
| W8A16 | 0 | 2.039 | 3.634 GB | 2.402 GB | 0 / 392 |
| W8A8 | 0 | 3.019 | 3.679 GB | 2.466 GB | 0 / 392 |
| BF16 | 12 | 2.849 | 3.669 GB | 2.464 GB | 0 / 0 |
| W8A16 | 12 | 2.748 | 2.776 GB | 1.541 GB | 168 / 224 |
| W8A8 | 12 | 3.988 | 3.093 GB | 1.606 GB | 168 / 224 |
| BF16 | 26 | 3.613 | 1.839 GB | 0.635 GB | 0 / 0 |
| W8A16 | 26 | 3.148 | 1.771 GB | 0.535 GB | 364 / 28 |
| W8A8 | 26 | 4.884 | 2.085 GB | 0.599 GB | 364 / 28 |

每个 ConvRot case 有 196 个 patched Linear × weight/scale = 392 payload tensors；swap12
和 swap26 的 CPU/GPU 计数精确匹配 `12/16` 与 `26/2` 个 block。apply 后与 step 后
residency 不变，所有 case 均 `status=ok`、`math_ok=true`。这证明 residency 修复有效。

显存和速度是明确交换关系：W8A16 swap0→26 的 peak 减少约 1.86 GB，step 慢约 54%；
W8A8 peak 减少约 1.59 GB，step 慢约 62%。因此不能用旧长训的 4.10 GB 判断当前实现；
显存允许时选 swap0，受限时用 swap12/26 换取更低 residency。

原始结果：`output/tests/convrot_residency_ablation_swap{0,12,26}_20260804.json`。

## Loss

| 模式 | 全程 mean | median | min | max | 相对 BF16 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.114626 | 0.100727 | 0.033604 | 0.505327 | 基线 |
| W8A16 | 0.116033 | 0.101359 | 0.033605 | 0.557423 | +1.23% |
| W8A8 | 0.116930 | 0.100633 | 0.035195 | 0.717167 | +2.01% |

三组都存在孤立 loss spike，但随后恢复，没有非有限值。W8A8 各 epoch mean 为
`0.117287, 0.117831, 0.118256, 0.121239, 0.114008, 0.112956`；W8A16 为
`0.117697, 0.114631, 0.113729, 0.117387, 0.117384, 0.115370`。W8A8 后两 epoch
没有继续恶化，说明在本任务中未观察到累积量化不稳定。

mean 差距只有 1%–2%，小于当前未固定 seed、无 validation 的实验不确定性，不应据此
给 W8A16/W8A8 排质量名次。

## 样图

下图每行依次是 BF16、W8A8、W8A16，每列是 epoch 1–5；只展示重复 prompt 中的第一份。

![ConvRot long-run samples](assets/convrot_longrun_epochs_20260727.jpg)

三组都从较弱的 epoch 1 逐步稳定到短发、紫眼、白蓝礼服、发光蓝花和夜间森林主体。
epoch 5 均保持完整构图和一致角色特征。模式间构图差异明显，但没有固定训练 seed，不能
把这种差异直接归因于量化精度。手部和局部衣饰问题在三组中都可见，并非 W8A8 独有。

## 产物完整性

三组共同状态：

- `run_end.status=ok`，`final_step=1710`，返回码 0。
- 普通 LoRA 只有 `-000001` 至 `-000005.safetensors`。
- checkpoint state 的 `train_state.json` 是 `current_epoch=5,current_step=1425`。
- sample 只有 epoch 1–5。
- 配置明确为 `save_every_n_epochs=1`、`checkpointing_epochs=1`、
  `sample_every_n_epochs=1`。

这不是单个量化模式异常，而是共享保存/采样生命周期问题。不能把日志里的 6 epoch
完成状态等同于“epoch 6 权重已经落盘”。

### 2026-08-04 修复

根因是两个回归叠加：

1. `library/training/loop.py::run_training_loop` 在 `global_step >= max_train_steps`
   时于 epoch-end save/resumable/sample 之前直接 `break`。
2. 训练模块拆分时，`library/training/train_session.py` 漏接了已有的
   `CheckpointSaver.save_final()`，因此连最终无编号模型也没有兜底写出。

当前工作树的修复保证达到最后一步后依次完成：

- 最终 epoch 编号权重，例如 `-000006.safetensors`。
- 最终可恢复 checkpoint 及 `train_state.json`，其 `current_step == final_step`。
- 最终 epoch sample。
- 最终无编号 `<output_name>.safetensors`。

定向验证：保存/训练尾部集合 `49 passed`；resume/checkpoint 扩展集合
`41 passed`；新增行为测试 Ruff 通过，`git diff --check` 通过。

## 决策

1. **稳定性：通过。** W8A8/W8A16 都完成真实 1710-step 长训，无数值崩溃。
2. **质量：暂定不劣化，尚未严格验收。** loss 和样图没有明显退化，但证据设计不足。
3. **速度：W8A16 可接受，W8A8 仍非速度方案。** 本配置分别比 BF16 慢 8.3% 和 15.5%。
4. **显存：旧实现失败，residency 已修复。** 旧长训与 26-block swap 组合时 ConvRot
   峰值约为 BF16 的 2 倍；修复后独立复测 W8A16/W8A8 swap26 为 1.77/2.09 GB，
   payload CPU/GPU 计数与 block 布局严格一致。
5. **历史产物：不完整；保存回归：已修复。** 三组旧运行实际可用的最终
   LoRA 都停在 epoch 5；修复仅对后续运行生效。

## 后续实验与记忆更新引导

后续 Agent 或实验不得只引用早期 microbench，应同时记住以下边界：

- 旧长训的 quant payload 全量常驻 GPU 问题已修复；当前 payload 随实际 DiT block
  迁移，swap0/12/26 的 CPU/GPU 计数已实证。
- W8A8 在 RTX 3080 SM86 上使用 `_int_mm + FP32 post-scale`，不是 RTX 4070 SM89 的
  `_scaled_mm` 路径。
- 当前最可靠的长训速度结论是 BF16 `1.2662`、W8A16 `1.3708`、W8A8 `1.4626` s/step。
- 当前质量结论仅为“无明显不稳定”，不是严格等价；下一轮必须固定训练 seed，至少 3 seeds，
  并提供真实 held-out validation。
- sample prompt 文件必须去重并扩展到多个 prompt/seed；本次每 epoch 只有 1 张唯一图。
- 下一次长训需实机复核已修复的 epoch 6 保存/采样，验收条件必须包含最终
  checkpoint 的 `current_step == final_step`。
- 后续性能引用应同时给出 `blocks_to_swap`：当前 8-step 复测已覆盖 `0/12/26`，证明
  residency 正确，也证明更重 swap 会显著增加 step latency。

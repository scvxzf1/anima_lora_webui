# Krea-2 全程动态 AUTO 块交换验收

日期：2026-09-08。状态：96步加压动态、24步固定对照及24步累积2动态验收通过；
仍为显式开启的Krea-2实验模式，未证明稳定加速。

## 实现范围

`auto_block_swap=true`、`auto_block_swap_mode="dynamic"` 开启 Krea-2 单卡
运行时调整。启动盘点实际块数和主机预算，从允许的最大交换数开始，完整 optimizer
更新边界检查资源与执行迁移；不改变 batch、累积、精度、checkpoint、学习率。

性能搜索使用 shape-matched A/B/A，逐负载排除两次预热，至少 3% 收益、参考漂移
不超过 10%，且剩余收益足以摊销首次编译超额成本与迁移成本才接受。步幅为 2/4/8，
短距离平台可扩大探索距离，失败后冷却，外部显存压力优先退避且释放后可再探索。

只回滚尚未执行 optimizer 更新的迁移 OOM，不重放部分训练更新。主机可用物理内存和
换页 IO 越界时停止，不将磁盘 swap 视为可安全占满的 RAM。训练采样、验证、阶段调度、
续训及非 plain LoRA 等限制见 [配置说明](../configuration/auto-block-swap.md)。

## 实验条件

- 20 GiB SM86 单 GPU，82 SM；Torch 2.12 + CUDA 13.0。
- 主机约 62.69 GiB RAM、8 GiB swap，不改频率、风扇或功率。
- Krea-2 self-contained NF4，rank16/alpha8，full checkpoint，Flash varlen，resident compile。
- 两个真实缓存 bucket：896x1200、768x1344；batch1，固定 LR 2e-4、warmup0。
- 隔离 fixture：`output/runs/auto-block-swap-hot-20260908`。
- 本次产物：`output/runs/auto-block-swap-dynamic-20260908`。

## 运行时自动热测

`live-pressure-v6` 完成 96 次正式 optimizer 更新。压力来自单独 CUDA 进程，先用
1 字节张量完成上下文预热再握手，避免把延迟创建上下文的开销漏出预算。
注入大小按当时实测可用显存、预测训练峰值、余量及一块退避计算，不修改生产保护阈值。

自动事件如下，均由控制器决定，不是预设固定 A/B 块数序列：

| 更新边界 | 交换数 | 原因 |
| --- | --- | --- |
| 9 | 26 -> 24 | 资源有余量，探索 |
| 10 | 24 -> 25 | 外部进程持有 11,352,587,020 bytes，显存压力退避 |
| 14 | 25 | 外部压力释放；资源检查继续，性能搜索冷却 |
| 49 | 25 -> 23 | 冷却后重新探索 |
| 58 | 23 -> 25 | A/B/A 参考配置回测 |

一次完整配对判断：参考 18.3862 s/update、候选 18.2190 s/update，收益约 0.91%，
参考漂移 1.10%。未达到 3% 门槛，所以不接受 23 块，保留 25 块。
不同 shape 等权汇总用于策略判定，不是整个训练工作负载加权吞吐。

96 次更新全部完成，最终 checkpoint 的 `ss_steps=96`，全部保存张量为有限值。
allocator peak 为 5.5098 GiB，编译图数量在覆盖两种 shape 后保持8；最终25块。
累计四次迁移耗时0.2298 s，控制器埋点累计0.4662 s（包含迁移），约占30分钟
训练循环的0.026%。埋点不包含全部有限值检查/同步等待影响或额外编译成本，
不能直接把0.026%当成端到端性能开销；已测切换没有增加图数。
包含盘点、加载和最终保存的实验总耗时1862.27 s。
训练时段859次遥测：GPU 65–70°C，时钟保持1920 MHz；主机可用 RAM 最低30.39 GiB。
包含外部压力进程的驱动显存占用最高18422 MiB，不能与本进程 allocator peak 混用。

## 固定配置对照

同一实现、相同 fixture/seed、固定 LR，关闭动态控制器并固定26块，完成24次更新，
checkpoint step/finite 检查通过；allocator peak 4.903 GiB，编译图数8。

下表均为实验外层同步后整次更新的 wall time，不混用控制器内部 CUDA event 计时。
固定运行取第9–24次更新，动态运行取完成 A/B/A 后第67–96次更新。

| bucket | 固定26块（每形状8次） | 动态最终25块（每形状15次） |
| --- | ---: | ---: |
| 896x1200 | 19.0241 s | 19.0072 s |
| 768x1344 | 17.9147 s | 17.8385 s |

两者基本持平。它们是顺序运行，不能消除所有时间漂移，因此不宣称动态模式稳定加速，
也不将0.09%/0.43%的差异解释为因果收益。当前硬件更偏向大矩阵计算瓶颈，
更多显存驻留未必带来足够收益。

前24次更新按相同 step/shape 配对：loss 最大绝对差0.00013441、平均绝对差0.00001910，
最大相对差0.03847%。没有观察到明显短程数值异常，但这不是严格全模型 bit-exact
证明，也不是收敛/图像质量验收。小模型的精确前向/梯度契约仍独立测试。
原始汇总见 [对照 JSON](assets/auto_block_swap_dynamic_comparison_20260908.json)。

## 梯度累积动态验收

`live-accumulation2` 使用最终形状清单回退修复，完成24次optimizer更新、48个
microstep，每次更新包含两个已知bucket。第6次完整更新后26 -> 24块探索，
第12次更新后24 -> 26块参考回测；没有在累积中途迁移。

配对参考37.8117 s/update、候选36.8748 s/update，收益约2.48%，参考漂移4.17%，
仍低于3%收益门槛，最终保留26块。两次迁移合计0.17085 s，控制器埋点0.23218 s；
allocator peak 5.4871 GiB，全程8张图。最后checkpoint的 `ss_steps=24`，
全部张量有限；含盘点/加载/保存总耗时972.70 s。

首轮96步测试后修复两项边界并补回归：训练session释放dataset group后从
DataLoader读取形状清单；资源下限已收紧时不把不可执行候选报告为accepted。
本累积测试验证修复后的正式训练入口。

运行摘要及开发失败记录见 [完整验收 JSON](assets/auto_block_swap_dynamic_20260908.json)。

## NF4 存储修复

旧代码传递 `weight.data` 绕过 Params4bit master 识别，并在恢复时丢失 Parameter
身份或反复分配存储。另一个关键点是 `torch.device("cuda") != torch.device("cuda:0")`，
未规范化设备索引导致 GPU storage reuse 分支一直失配。

修复保留 NF4 Parameter 身份、独立 CPU masters 和只读 GPU quantization state，
普通交换将打包码复制到退役块的 GPU 存储；动态降驻留只停车打包码。
显式恢复到 CPU 则连量化状态一起移动。NF4 不混入普通 tensor slab 打包。

同一 fixture 最大交换数 26 的开发诊断中，allocator peak 从约 15.5 GiB 降到
约 4.9 GiB；这属于存储正确性修复，不应归因为在线搜索算法的性能收益。
小型真实 CUDA NF4 模型验证 eager/compiled、`cuda`/`cuda:0`、多次双向切换、
前向/梯度精确一致、Parameter 身份不变、CPU master 独立和编译 callable 复用。

## 策略消融

以下是确定性合成策略实验，**不是 GPU 性能或训练质量测试**。
脚本仅在实验 subclass 中禁用单项机制，没有把不安全开关暴露到生产配置。

| 场景 | 完整策略 | 消融 |
| --- | --- | --- |
| 时间漂移 10 -> 8 -> 7 s | A/B/A 拒绝错误收益 | 去掉参考回测后误接受一次 |
| 候选首次编译额外约 9992 s | 拒绝无法摊销成本 | 忽略预热成本后接受一次 |
| 候选 24 块后，资源要求至少 25 块 | 0 次越界更新 | 去掉压力保护后 4 次越界 |
| 26..20 块为小收益平台，18 块有收益 | 扩大步幅后到 18 块并接受 | 固定步幅2停留26，未接受 |

复现命令：

```bash
python -m scripts.experiments.dynamic_block_swap_ablation --output /tmp/dynamic-policy-ablation.json
python -m pytest tests/test_dynamic_block_swap_ablation.py
```

结果见 [策略消融 JSON](assets/auto_block_swap_dynamic_policy_20260908.json)。

## 失败与开发中止记录

- v1/v2/v3：为诊断高显存峰值主动中止，不计成功。
- diagnostic-fixed / diagnostic-memory：短固定诊断，不是全程动态验收。
- v4：开发期间改了 lazy-import 模块签名，进程内新旧代码混用导致首次迁移失败；
  后续热测冻结运行时代码直到进程退出。
- v5：完成 26 -> 24 自动探索，但压力 worker 延迟创建 CUDA 上下文，实际压力超过
  测量预算；生产余量保护正常停止。v6 修正压力进程预热协议，而不是降低安全余量。
- 早期组合回归中 BF16 forward-only slab wraparound 的零容差断言曾失败两次。
  最终原顺序85项组合通过，加入controller/policy的117项组合通过；compiled NF4
  预热后重复三次原断言及逐层权重/输出检查均通过。未降低容差或修改 slab 算法，
  早期偶发差异尚无确定根因，不能称其已被修复。

## 证据边界

完成训练和保存 checkpoint 不等于证明稳定加速，更不等于模型质量无损。
单种模型、两个 bucket、有限次迁移只能支持已测配置；异常突发显存占用仍可 OOM。
编译图复用并不代表从未见过的更大常驻范围或新 shape 无编译成本。
不宣称获得全局最优交换数，也不自动推广为所有模型/显卡的默认配置。

报告生成器同时检查连续更新序列、唯一 runtime、循环结束步数、有限值 checkpoint，
以及要求加压时的退避/释放/再次探索事件。运行完成与因果性能证明是独立字段。

## 验证入口

本轮定向回归：237项配置、策略、实验脚本、Web预检/字段测试通过；85项真实CUDA
小模型、块交换与编译组合测试通过，共322项。相关新增模块 Ruff 通过。

此前扩大前端套件时仍有6项与本功能无关的既有 help-summary 文案不一致：
`dim_from_weights`、`max_train_epochs`、`max_train_steps`、`save_last_n_epochs`、
`checkpointing_last_n_epochs`、`sample_ratio`。没有改写这些用户已有文案。
全工作树 diff-check 还包含既有 `configs/web-ui-settings.toml` 空白EOF；本次定向
路径检查通过。没有运行整个仓库 test-all，不宣称全量测试无失败。

新 WebUI 进程为 `http://127.0.0.1:20206/?ui=dragon#config/training-config`，
没有重启原20203/20205实例。桌面1440x900、移动390x900完成Playwright检查：
无页面异常、无横向溢出；动态模式启用评估窗口、禁用启动候选次数。
仅操作草稿，拦截写API，没有覆盖用户训练配置。

真实热测复现使用现有隔离fixture，不下载模型：

```bash
python -m scripts.experiments.dynamic_block_swap_probe \
  --fixture output/runs/auto-block-swap-hot-20260908 \
  --output /tmp/dynamic-auto-new-run --steps 96 --interval 4 --pressure
python -m scripts.experiments.dynamic_block_swap_probe \
  --fixture output/runs/auto-block-swap-hot-20260908 \
  --output /tmp/dynamic-auto-fixed-new-run --mode fixed --steps 24 --blocks 26
```

输出目录必须不存在；不清空或覆盖既有训练数据。压力进程只由该实验创建并清理，
不终止其他GPU进程。所有本次训练/压力进程已结束，两个自建遥测进程已正常停止；
20206 WebUI仍保留可访问，没有提交、推送或清理用户运行数据。

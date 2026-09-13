# AUTO 块交换真实 GPU 热测与消融

状态：实验。适用范围为本机 Krea-2 NF4 plain LoRA；不代表三模型族验收。

结论：真实资源探测、OOM 子进程回退、安全余量过滤、长窗口更新及 16 步正式保存均通过；
**速度最优选择未验收通过**，不能据本轮宣称 AUTO 稳定加速或默认推广。

机器可读证据：[auto_block_swap_hot_20260908.json](assets/auto_block_swap_hot_20260908.json)。

## 环境与协议

- GPU 报告名称为 `NVIDIA Graphics Device`，SM 8.6、82 SM；NVML 20,480 MiB，
  PyTorch 可见容量 20,048 MiB。不能仅凭该名称断言具体零售型号。
- Torch `2.12.0+cu130`、CUDA 13.0、driver 580.178.04；物理 RAM 62.69 GiB，
  swap 8 GiB。整个实验不修改功耗、风扇、时钟或用户桌面进程。
- 自包含 Krea-2 NF4，BF16 compute，plain LoRA rank 16 / alpha 8，完整 checkpoint、
  Flash varlen、固定 resident compile、fused AdamW，学习率 `2e-4`，seed 114。
- 从现有数据复制两组真实图片/caption/latent/text cache 到独立实验目录：
  896x1200 / 4200 latent tokens 与 768x1344 / 4032 latent tokens，batch 1，梯度累积 1。
  原始数据、缓存、配置和训练产物不变；不下载模型。
- 盘点实际 28 块；模型和主机预算均允许最多 26 块。每块盘点 223,952,928 bytes，
  运行时全部块 CPU masters 共 5.660 GiB，不随交换尾部数量线性缩小。
- 每候选独立进程，执行正式 `_run_step` 的 forward、backward、optimizer、scheduler。
  每个 case 三次完整更新，第一步排除，剩余步时取中位数，再对两个 case 等权平均。
  六候选预算之外另有一次盘点和一次独立确认。共用实验内 Inductor 磁盘缓存，
  因此不是清空编译缓存后的首次安装启动测试。
- 受压组由协调进程实际持有并写入 8 GiB CUDA tensor。它是同卡显存竞争测试，
  不是另一张 10/12 GB 显卡的性能模拟；安全线仍按本卡总容量计算，约 1.958 GiB。
- 固定步长复用普通 AUTO 中相同候选；另外两项消融复用受压 AUTO 的公共候选。
  分叉候选以及所有最终确认都真实运行。复用避免把同一路径重复测量的时变噪声
  混入策略差异，但不等于每项都从头独立重跑过全部训练。

## 普通显存搜索

以下是搜索时的短窗口值，不是最终持续速度声明。

| 交换块数 | 步时 s | allocator peak GiB | 预算余量 GiB | 判定 |
| ---: | ---: | ---: | ---: | --- |
| 26 | 12.639 | 4.902 | 13.151 | 安全 |
| 24 | 12.728 | 5.307 | 12.631 | 安全 |
| 20 | 12.331 | 6.115 | 11.837 | 安全 |
| 12 | 11.804 | 7.732 | 10.251 | 安全 |
| 4 | 11.262 | 9.723 | 8.283 | 安全，选中 |
| 0 | 11.166 | 10.729 | 7.301 | 安全 |

4 与 0 的步时差约 0.87%，进入 3% 容差，因此优先选择更多交换的 4。
独立确认 11.380 s、余量 8.283 GiB，通过。盘点、六候选、确认的进程耗时合计
949.08 s，约 15.82 分钟；不含顶层 Python 初始化等开销。

固定步长 1 的六候选为 `26,25,24,23,22,21`，最终选 26。它在相同预算内
未探索到 20 以下；并非证明固定步长在更大预算下找不到同一个配置。
候选逻辑耗时 852.08 s，其中两项复用、四项新测实际耗时 568.08 s；确认另耗时
122.47 s。不能将复用后的实际耗时直接与完整冷启动校准比较。

## 显存受压边界

| 交换块数 | 步时 s | 预算余量 GiB | 判定 |
| ---: | ---: | ---: | --- |
| 26 | 8.733 | 4.874 | 安全 |
| 24 | 10.084 | 4.468 | 安全 |
| 20 | 9.752 | 3.657 | 安全 |
| 12 | 9.537 | 2.057 | 安全 |
| 4 | 无完整测量 | 不适用 | 真实 CUDA OOM |
| 8 | 9.318 | 1.020 | 更新完成，但低于安全线 |

4 块候选在申请 152 MiB 时发生真实 CUDA OOM，独立进程退出后，搜索正确转向 8。
8 块完成全部六次更新，但没有被完整策略纳入安全候选。没有把 OOM 当作成功、
没有在已部分更新的 optimizer 上重试，也没有清空其他进程的显存。

受压组最终选 26，因为它在当时的测量中最快；确认资源检查通过，但步时变为
12.970 s，比候选时慢约 48.5%。这一变化大于不同交换配置的许多差异。

## 策略消融

| 策略 | 场景 | 逻辑候选数 | 新测候选数 | 新确认数 | 选中块数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 完整 AUTO | 普通 | 6 | 6 | 1 | 4 |
| 固定步长 1 | 普通 | 6 | 4 | 1 | 26 |
| 完整 AUTO | 额外占用 8 GiB | 6 | 6 | 1 | 26 |
| 不做失败二分 | 额外占用 8 GiB | 5 | 0 | 1 | 26 |
| 不保留 GPU 余量 | 额外占用 8 GiB | 6 | 0 | 1 | 26 |

- 固定步长的探索深度确实不同，但两者的持续速度优劣未被本轮时变数据证明。
- 不做二分在 4 块 OOM 后停止，不再探测 8 块；本次并未改变最终选中值。
- 不保留 GPU 余量会把 8 块纳入可选集合，而完整策略会拒绝它；但公共候选中
  26 块更快，因此本次也没有改变选中值。**不能声称这项硬件消融最终选中了危险值**。
- 两项受压消融的新确认均资源安全，步时分别为 16.393 / 16.551 s。这些差异受
  时变负载影响，不是二分或安全阈值本身的执行开销。
- 整个硬件实验只实际执行了一次上述 OOM 候选。消融中的同一失败为证据复用，
  不能计成三次独立 OOM 复现；没有把合成策略测试的数字混入硬件结果。

## 长窗口 A/B/A

按 `26 -> 4 -> 26` 顺序分别重新启动进程，每个分辨率 12 次更新，每组 24 次。
warm 排除各 case 第一次更新；严格 hot 排除前两次更新，均先按 case 取中位数，
再对两个 case 等权平均，不把两个分辨率的所有更新混池取中位数。

| 顺序 / 块数 | warm s | 严格 hot s | allocator peak GiB | 最低预算余量 GiB | 最低可用 RAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 / 26 | 16.341 | 16.355 | 4.902 | 12.638 | 28.927 |
| B / 4 | 17.331 | 17.921 | 9.723 | 8.094 | 35.297 |
| A2 / 26 | 19.987 | 19.980 | 4.902 | 12.996 | 28.896 |

三组 72 次更新全部完成、loss 有限、无 CUDA OOM、系统 swap IO 为 0。
两次 26 块严格 hot 差约 22.2%；4 块位于两次基线之间，因此不报告持续加速比。
4 块第一个 case 的早三次 hot 平均值 15.543 s、末三次 20.053 s，也存在明显漂移。

长窗口活动采样的 GPU 温度中位数依次 72 / 70 / 71 C，最高 77 / 74 / 74 C，
SM 时钟中位数均为 1920 MHz。重复基线变慢并未伴随同幅度的 SM 降频；本轮不能
复用其他 GPU 历史报告中的热降频结论来解释它。没有调整散热或设备功耗。

同 seed、同 case/update 对齐的 loss 差异：

| 对比 | 最大绝对差 | 平均绝对差 |
| --- | ---: | ---: |
| 26 vs 重复 26 | 3.7223e-5 | 1.1758e-5 |
| 26 vs 4 | 1.6347e-4 | 4.1518e-5 |

4 块差异大于同配置复测差异，不能称为位级一致。这里仅验证 loss 轨迹接近且有限；
没有完整参数/梯度逐元素对照、保存重载数值 round-trip 或生成质量评估。

## 正式 Epoch 与保存

首次 `formal-epoch-16` 在完成第 1 步后失败：实验 helper 把
`save_every_n_steps` / `save_every_n_epochs` 设为 `0`，但正式 saver 的关闭契约为
`None`，因此按步保存分支发生 `ZeroDivisionError`。该问题在私有探针退出位置之后，
不能靠前面的 forward/backward 探针发现。失败目录和
`formal-epoch-16.worker.log` 完整保留，没有把它计为成功，也没有覆盖重试。

仅修复实验 helper 的关闭值，未修改生产 saver；添加关闭值断言与提前停止时不得
生成成功标记的回归检查。新目录 `formal-epoch-16-v2` 使用普通 AUTO 已确认的
4 块值，保留 `_auto_swap_resolved` 和正式资源 guards，从新 adapter/optimizer
状态开始训练，并且不重新执行一次完整校准。

重跑完成 16 步 / 8 epoch，总耗时 332.49 s，退出码 0。最终文件为
`formal-epoch-16-v2/auto-swap-acceptance.safetensors`，192,747,496 bytes；
CPU 读回 588 个张量全部有限，metadata `ss_steps=16`、`ss_epoch=8`、rank 16。
这不是续训 round-trip 验收，正式训练的随机 batch 顺序和 scheduler 长度也不用于
前面的逐步 loss 对照。

探针进程共 28 个：3 次盘点、24 个成功探针、1 个真实 OOM。探针累计完成 199 次
optimizer 更新，其中 1 次属于随后 OOM 的失败候选；另有首次正式验收失败前的
1 步和成功正式验收的 16 步。所有探针报告的系统换页 IO 为 0，最低可用 RAM
28.896 GiB。不能将失败尝试或复用候选混入独立成功次数。

## 时间波动与测量边界

同样的 26 块、同样两个 case，普通搜索为 12.639 s，固定步长确认为 8.954 s，
受压搜索为 8.733 s，受压确认为 12.970 s。loss、更新数量和 allocator 峰值没有
对应的大幅变化。桌面与训练共用 GPU，期间仍存在图形进程活动；现有证据不足以
把全部波动归因于某一个后台程序或热降频。

因此资源可行性与速度排序必须分开判断。三步探针与一次确认可以验证本次真实
更新的资源边界，但不能保证时变负载下选出最快配置。当前确认只强制检查资源，
没有基线穿插或确认步时漂移门禁；这仍是后续速度选择策略的改进项。

allocator peak 是整个进程的累计水位，包含加载/编译/首个 backward/optimizer 状态，
不是某个更新独占的瞬时峰值。驱动显存还包含其他进程与非 PyTorch 分配。
进程树 RSS 会重复计入 fork/shared pages，只作为上界；物理内存安全使用系统
最低可用 RAM 和换页 IO 独立判断。2 秒 GPU 采样可能漏掉更短尖峰。

本机缺少可用的 Anima 和 Z-Image 权重，所以不启动下载，也不把 Krea-2 的通过结果
外推为三族验收。没有真实低 RAM/cgroup 压力、swap 耗尽、多 GPU、batch>1、全部
bucket、采样、续训或运行期热切换测试。系统 swap 仍只报告，不加入可用 RAM 预算。

## 复现入口

实验目录：`output/runs/auto-block-swap-hot-20260908`。以下命令会占用 GPU，目录中的
原有候选不会被覆盖；重跑整套时使用新的 `--output` 目录，并先核对本机模型/缓存。

```bash
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action calibrate --label full-auto --limit 6
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action ablation --mode fixed-stride --blocks 26 --limit 6 --reuse full-auto
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action calibrate --label pressure-auto --limit 6 --reserve-gib 8
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action ablation --mode no-bisection --blocks 26 --limit 6 --reserve-gib 8 --reuse pressure-auto
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action ablation --mode no-reserve --blocks 26 --limit 6 --reserve-gib 8 --reuse pressure-auto
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action probe --label hot-swap-26 --blocks 26 --updates 12
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action probe --label hot-swap-4 --blocks 4 --updates 12
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action probe --label hot-swap-26-repeat --blocks 26 --updates 12
python -m scripts.experiments.auto_block_swap_probe --output output/runs/auto-block-swap-hot-20260908 --action formal --label formal-epoch-16-v2 --reuse full-auto --updates 16
```

实验目录创建后，在另一终端启动只读遥测；每 2 秒采样，最多运行两小时：

```bash
python -m scripts.experiments.auto_block_swap_metrics --output output/runs/auto-block-swap-hot-20260908/hardware.jsonl --seconds 7200
```

所有测试结束后生成汇总，不会加载模型或启动 GPU 训练：

```bash
python -m scripts.experiments.auto_block_swap_report --root output/runs/auto-block-swap-hot-20260908 --output docs/findings/assets/auto_block_swap_hot_20260908.json
python -m pytest tests/test_auto_block_swap*.py -q
```

本轮 AUTO 定向测试 85 项通过，相关 Python Ruff 检查通过。未将这些结果声称为
全仓测试通过；也未启动额外的长训练。实验拥有的遥测进程已停止。

`auto_block_swap_probe` 是依赖本机实验 fixture 的研究入口，不是面向任意数据集的
第二套训练 CLI。正式用户入口仍见 [AUTO 块交换配置](../configuration/auto-block-swap.md)。

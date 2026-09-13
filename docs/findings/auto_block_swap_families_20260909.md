# Anima / Z-Image AUTO 块交换热测

状态：实验；以下为指定日期的短窗口测量，不代表所有配置已验证。

日期：2026-09-09。测试使用单 CUDA 进程、BF16、Flash、完整梯度检查点、关闭 compile、
25% 总显存保留目标、`vram` 倾向、单张 896x1200 真实缓存样本。候选走生产
`run_calibration`，并用独立确认后才允许正式训练；该次热测运行于 SWAP 上限参数加入前，
当时系统换页 IO 超过 64 MiB 一律拒绝。

## 结果

| 模型 | 实际块数 | AUTO 候选 | 正式步数 | 结果 | 热步中位数 | 峰值分配 | 最小 headroom | 最大换页 IO |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Anima | 28 | 26 | 16 | 通过 | 2.902 s | 2.01 GiB | 15.64 GiB | 19.3 MiB |
| Z-Image | 30 | 28（首候选） | 0 | 拒绝 | 7.06 s（3 更新探针） | 4.44 GiB | 13.24 GiB | 241.9 MiB |

Anima 的 26 块候选、24/20/12 探索和独立确认均通过，16 步 checkpoint finite 且保存
成功。Z-Image 的 GPU 显存余量和 3 次训练更新均通过，但主机进程峰值 RSS 约 38.2 GiB，
候选产生约 241.9 MiB 系统换页 IO，AUTO 按资源策略拒绝，未启动正式训练；没有绕过保护、
关闭系统 swap 或把 GPU 通过冒充整体通过。

## 边界

- 这是 startup AUTO 的真实单样本热测，不是全程 dynamic，也不是速度 A/B 或多 bucket 泛化证明。
- Z-Image 为快速重测，仅完成首个 28 块候选，未声称全局最优；首次运行和重测都因当时的 64 MiB 换页门槛拒绝。当前代码可通过配置提高该上限。
- `balanced`/`vram` 与 25% 保留参数已覆盖；Anima/Z-Image 当前不支持 dynamic 或 `ram` 倾向。
- 单次短训练不代表小 RAM 机器一定通过，也不代表长训练、其他分辨率、质量或速度收益。

原始 JSON：[`auto_block_swap_families_20260909.json`](assets/auto_block_swap_families_20260909.json)。

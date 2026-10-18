# Block Swap 宏观流水线基线（CMP 50HX）

状态：实验 / 第 1 轮基线
适用版本：`codex/block-swap-int8`（基于 dev `d65f4a28`）
入口命令：`.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --label swap12`
相关代码：`library/runtime/offloading.py`、`bench/block_swap_pipeline/`

本文件记录 Block Swap 宏观调度方向的第一轮真机基线，目的是把"下一步该优化什么"从推断换成实测。

## 方法

同一份工作区（2 张 512×512 程序生成图、真实 Anima DiT/TE/VAE、plain LoRA rank8/alpha4、bf16、
gradient_checkpointing、`attn_mode="torch"`、`torch_compile=false`），只改 `blocks_to_swap` 做 A/B。
开启 `block_swap_profile_jsonl="auto"`，每个完成的 retire+restore 任务写一条事件；分析器按**相位周期**
切分训练步（offloader 的 `step` 字段实测恒为初始值，不可用于分组）并汇总。

run：
- `output/bench/block_swap_pipeline/20261019-005236-swap12`
- `output/bench/block_swap_pipeline/20261019-005555-swap0`

## 结果 1：swap 开销 ≈ 0.45~0.49 s/步

| 配置 | 步时样本（s/it，训练日志后 5 条） | 稳态步时 |
| --- | --- | --- |
| `blocks_to_swap=0` | 1.550 / 1.170 / 1.080 / 1.010 / 1.090 | ≈ 1.01~1.09 s |
| `blocks_to_swap=12` | 2.150 / 1.690 / 1.560 / 1.480 / 1.570 | ≈ 1.48~1.57 s |

差值 **+0.45~0.49 s/步**，与目标里 3.5 节的"固定截距 0.398 s"同一量级——**截距是真实的**。

## 结果 2：但这 0.45 s 不是"折返点重复搬运"造成的

swap=12 的 profiler 实测（第 2~4 步稳态）：

| 指标 | 实测值 | 说明 |
| --- | --- | --- |
| 任务数 | **24/步 = 2S**（前向 12 / 反向 12） | 与设计一致，没有多余搬运 |
| 重复恢复 | **0**（`duplicate_restores=[]`） | 同一步内没有任何 block 被恢复两次 |
| `exposed_wait` | **1.5~1.6 ms/步** | 等待点上几乎不阻塞：需要时块已就绪 |
| `worker_queue` | 17~24 ms/步 | 单线程拷贝池（`max_workers=1`）的排队量，稳态很小 |
| 前向→反向空窗 | 37~43 ms/步 | 折返点确实存在空窗，但只有几十毫秒 |
| Σ`h2d_ms` | 1064~1066 ms/步 | 12 个块的 H2D 合计 |
| Σ`d2h_ms` | **0** | retire 不复制回 CPU：冻结基座权重仍在 CPU master 上，只有 H2D 走线 |
| 单块 H2D | 44~77 ms（约 100 MB/块） | 有效带宽仅 **≈1.3~2.3 GB/s** |

**结论（与原始 P0 前提的偏离）**：
1. 顶部折返点的重复搬运在当前实现里**不存在**。`submit_move_blocks` 只在 `block_idx < blocks_to_swap` 时提交
   （`library/runtime/offloading.py` 的 `if not self.forward_only and (block_idx + step) >= self.blocks_to_swap: break`），
   所以训练态前向结束时最后一个换出块仍驻留，反向第一刻直接用，不会重传。
2. 反向的早块恢复由尾块 backward 钩子驱动（S=12 时 41→11、40→10……），已形成 lead≈1 job 的反向流水。
3. 因此"U-Turn Pinning 省 24 次搬运"没有可回收对象；实测 24 任务/步已是理论下限。
4. 真正的瓶颈是**串行 DMA 链长度**：每步 12 次 H2D 合计 ~1.07 s，而 swap=0 的计算只要 ~1.05 s，
   两者同量级；单线程拷贝池把 12 次 H2D 排成一条链，且单块有效带宽只有 1.3~2.3 GB/s（远低于链路理论上限），
   所以这 0.45 s 表现为"步时被 DMA 顶住"，而不是"等待点上被卡住"（`exposed_wait` 仅 1.6 ms）。

## 结果 3：据此修正的优先级

| 原优先级 | 修正后 | 依据 |
| --- | --- | --- |
| P0 U-Turn Pinning（省重复搬运） | **改为 P0'：压缩串行 DMA 链本身**（提高单块 H2D 有效带宽 / 走 packing 与 restore_mode 对比，量化每档 GB/s） | 重复搬运=0，无处可省；单块 1.3~2.3 GB/s 明显低于链路能力 |
| P1 `blocks_to_swap="auto"` | **保持 P1（价值上升）** | 少换块 = 直接少 H2D 字节；本机 swap 代价 ≈ 每块 ~40 ms |
| P2 int8 编码信噪比 | **保持 P2（价值上升）** | int8 master 把每块 H2D 字节减半，直接砍 DMA 链长度 |
| P3 交错换出/跨步预取 | **降级为 P3（收益受限）** | 折返点空窗仅 37~43 ms/步，且无暴露等待，调度类优化头部空间有限 |

## 边界与未证明项

- 本基线是 2 图 / 512×512 / 4 步的**调度观测**，不是吞吐基准；步时含 dataloader 与首批预热差异。
- 单块 H2D 有效带宽由事件时长推算（Σh2d ÷ 块字节数），未做纯拷贝微基准；"packing 有空间"是待验证假设。
- `d2h_ms≡0` 只说明该路径没有计时到的 D2H 拷贝，不代表 retire 零成本（释放/分配 pinned buffer 的开销未单独计量）。
- 折返点空窗只测了前向→反向方向；反向→下一步前向的空窗未单独统计。
- 只在 CMP 50HX（Turing sm_75，窄链路）上测过；不同卡型的链路带宽会改变结论权重。

## 复现

```bash
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --label swap12
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 0  --label swap0
```

工作区（`output/bench/block_swap_pipeline/workspace/`）持久复用：图像、权重副本、latent/TE 缓存只建一次，
同分辨率下的 A/B 只花一次 DiT 加载 + N 步训练。

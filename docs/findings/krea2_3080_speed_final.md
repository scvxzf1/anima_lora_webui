状态：长期速度研究完成审计
日期：2026-08-09
覆盖：`krea2_3080_speed_stage1.md` 至 `stage12.md`

# Krea-2 RTX 3080 约 12s/it 最终结论

## 根因

`12s/it` 不是单个 Python、H2D 或 padding bug：

1. 同机算子消融中，RTX 3080 的代表性 BF16 Linear 比 PG199 慢 4.4-5.2 倍，
   NF4 Linear 慢 4.1-4.6 倍，attention 慢 2.83 倍。
2. compiled PG199 单步中 GEMM 约 1593ms、attention 约 847ms，两者合计约占墙钟
   89%；3080 恰好在这两类大算子上差距最大。
3. 短窗口的主因是硬件矩阵吞吐；长窗口还叠加散热曲线。3080 纯 GEMM 120 秒达到
   84°C，吞吐退化 7-9%，训练 compile 路径也从约 12.06 漂到 12.65s。
4. swap H2D 在 PG199 双口径均约 0%，理论优化上限约 2%；生产训练不会每步重复
   `prepare_block_swap_before_forward`。文本尾部 padding 裁剪也无可测收益。

## 已落地生产配置

| 配置/能力 | 结论 | 证据 |
| --- | --- | --- |
| NF4 预量化 | 29.45→10.49GB，仅慢约 3.6% | NF4 五格消融 |
| full gradient checkpoint | 10GB 卡必需 | 3080 放开单 block 即 OOM |
| fixed resident compile | PG199 3.370→2.726s；3080 保证显存可运行 | 阶段 3/5/9 |
| multi-bucket | 24 buckets 只形成 4608/4864 两张复用图 | 阶段 9 |
| checkpoint/resume | LoRA/forward delta=0，reload 无重编译 | 阶段 10 |
| block swap | 3080 建议 swap24，约 5.91GB peak，速度代价约 2% | NF4 block-swap 提案 |

当前 3080 的稳妥训练组合是磁盘 NF4 + full checkpoint + fixed/default/resident compile，
再按实际可用显存选择 swap20-24。swap20 是临界速度点；桌面进程或其他 CUDA 占用不
稳定时优先 swap24。

## 实验候选

packed valid-token FlashAttention varlen 是唯一在 3080 长窗口仍测得正收益的新增软件
attention 候选：PG199 全模型/双 token family 快 11-13%，3080 swap20 末五步均值
`12.145s`，相对历史 cuDNN compile `12.65s` 快约 4%，GPU peak `6.09GB`。
PG199 50 步复核除一个 `2.804s` 抖动外保持 `2.417-2.439s`，末步 `2.429s`，
loss 和梯度均正常，证明收益不是短窗口偶然值。

它尚未生产化，因为仍缺显式 Krea backend、可选依赖与 V100 BF16 fallback、batch>1/
CFG、训练/推理 multi-bucket 和 Dynamo 动态输出配置边界。默认继续使用 cuDNN SDPA。

## 已否决或限定的方向

- `every_other` checkpoint：PG199 快 13.9%，但 3080 不可行；PG199 16/28 checkpoint
  可达 2.408s，但 31.55GB 无安全余量，只作实验档。
- FP16 NF4：输入梯度 rel-L2 35.6%，拒绝以训练轨迹换速度。
- `reduce-overhead`：CUDA Graph 与 non-reentrant checkpoint recompute 冲突。
- LoRA rank16→8：步时持平，只省约 145MB。
- H2D slab 重写、padding 裁剪、每步 prepare 清理：收益不存在或上限不足以覆盖风险。
- 风扇、功耗、降压：可能缓解长训热漂移，但属于硬件策略，未获明确许可时不修改。

## 完成审计

阶段 1-12 均有独立 findings 和阶段提交；生产代码变更已由定向单测覆盖，实验路径保留
可复现探针。最终审计修正了阶段 3 遗留的“3080 持续快 3.3%、约 60 步回本”旧表述：
该数字仅是冷态短窗口，阶段 5 的长窗口结果优先。当前结论没有把 Flash varlen 的实验
收益写成默认能力，也没有把 PG199 的结果外推成 3080 的两位数收益。

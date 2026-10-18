# Block Swap INT8 编码误差探针

回答"同样传 1 Byte/元素，换一种编码还能不能更准"。不训练、不加载完整模型。

```bash
.venv/bin/python -m bench.block_swap_int8.encode_error --real-limit 3
```

## 比较对象（字节数完全相同：1 Byte/元素 + 每行 1 个 FP32 scale）

| 键 | 编码 |
| --- | --- |
| `baseline_rel_error` | 现役 `scale = amax/127` 均匀对称 |
| `clip_grid_rel_error` | per-row 裁剪搜索（0.85~1.00 amax + 99.95 分位，行内 L2 最优） |
| `wide_bound_rel_error` | per-row 在 0.30~1.00 amax 穷举的上限 |
| `nf8_rel_error` | 固定 NF8 码本（N(0,1) 等概率 256 级），每行按 amax 缩放 |
| `lloyd_max_rel_error` | per-row 1D Lloyd-Max 256 级（拟合码本收益的下界） |

样本 = 4 组合成分布 + 真实 DiT 的 `mlp.layer1.weight`（`safe_open` 抽样，按 key 间隔取）。
产物写入 `output/bench/block_swap_int8/<stamp>-encode-error.json`。

结论与边界见 [INT8 单字节编码的误差上限](../../docs/findings/int8_encoding_floor_50hx_20261019.md)。

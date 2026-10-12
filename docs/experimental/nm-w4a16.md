# N:M INT4/BF16 实验

状态：实验 / **静态实现，未进行真机验收**  
适用版本：当前工作树  
入口：`python tasks.py exp-nm-w4a16 plan`

## 目标与对照

候选路径是 **2:4 结构化稀疏 + uniform INT4 权重 + BF16 激活**。首期唯一
性能对手是 **dense BF16**，不是 dense W4A16。NF4 不属于此格式；本实验也不
改变现有 Krea-2 NF4 路径。

模块在 `library/runtime/nm_w4a16/`，默认不接训练 bootstrap、不改配置、不替换
生产模型。`FrozenNMLinear` 是冻结基座的一阶参考算子，能为 LoRA 残差提供输入梯度；
`patch_plain_lora_mlp` 只接受已 apply 的、明确命名的 plain-LoRA MLP，并且保留原
BF16 权重，当前不能宣称省下完整模型显存。

## 数学论证（原始权重数为 `N×K`）

每四个权重保留两个：

```text
dense BF16                       = 16 bit / 原始权重
2:4 INT4 values                  = 4 bit × 1/2 = 2 bit / 原始权重
2:4 metadata（两位位置 × 2）     = 4 bit / 原始权重组 = 1 bit / 原始权重
```

因此 payload（不含 scale 与对齐）是 `3 bit / 原始权重`。若每 `G` 个权重一个
FP32 scale，平均为 `3 + 32/G bit`；相对 BF16 的理论存储比为：

```text
R_storage = 16 / (3 + 32/G)
```

例如 `G=128` 时约为 `16/3.25 = 4.92×`。这不是 2:4 的 8×，因为非零位置元数据
和 scale 仍然存在。

对输入批量 `X∈R^(T×K)`、权重 `W∈R^(N×K)`：

```text
Y = X Wᵀ
dX = dY W
```

dense BF16 前向和输入梯度各为 `2TKN` FLOPs；若稀疏乘法真正跳过零，则各为
`TKN`，理论算术减少 2×。但这只是 FLOP 上界：解码、scale、metadata、padding、
workspace、LoRA 分支和 kernel launch 可能令实际速度低于 BF16。

为了支持反向输入梯度，mask 必须沿 `W` 和 `Wᵀ` 都满足 2:4。本实现为每个 4×4
tile 选择每行、每列各保留两个位置（90 个平衡模式中按加权误差选最优）。如果只做
普通行向 2:4，前向可能合法，但不能直接把同一压缩布局用于 `dX`。

## 当前实现边界

- `reference`：每次操作解码成 BF16，再调用 `F.linear`；用于正确性和数学对照。
- `cusparselt`：显式 `prepare_sparse()` 后把解码后的 BF16 矩阵交给 PyTorch
  cuSPARSELt。它不是 fused W4A16 内核，仍有解码/压缩成本，且要求 SM80+。
- `cache_sparse=True` 会缓存前向与转置两个 BF16 稀疏表示，静态显存论证必须把它们
  算入。理想连续存储按每个方向 `9/8 byte / 原始权重`（每个保留值 16 bit，加每
  4 个权重 4 bit metadata），两个方向合计 `9/4 byte / 权重`；比 dense BF16 的
  `2 byte / 权重` 还大约多 12.5%。payload 本身不等于端到端的常驻显存收益。**当前
  patch 会保留原 BF16 base 权重**，因此整模型显存反而额外叠加 payload 与临时/缓存。
  不缓存则不能把准备成本排除在单步训练之外。
- payload buffer 使用 `persistent=False`，不会偷偷进入普通模型或 LoRA checkpoint。
- `probe` 必须同时显式提供 `--execute --device cuda:<index>`；默认 `plan` 只输出
  BF16 基线的解析成本，不启动 GPU、热测试、风扇或功耗改变。
- synthetic probe 是孤立冻结 Linear + LoRA 残差，不包含真实 DiT 的 checkpoint
  重算、block-swap 生命周期、训练器分配策略、optimizer、量化质量恢复或稳态热行为；它
  也不统计 integrated patch 保留的 BF16 原权重，因此不是集成训练显存结论。

## 尚未完成

1. 真正 fused INT4 sparse Tensor Core kernel（当前不是）。
2. Krea-2 训练 bootstrap / WebUI 配置 / checkpoint schema 接入。
3. 真实 DiT + LoRA 多 seed、质量、恢复、compile、block-swap 和热态验收。
4. Sparse-Marlin 类 backward kernel；当前只提供参考 `dX` 路径或 staged cuSPARSELt。

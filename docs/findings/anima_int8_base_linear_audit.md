# Anima int8 base Linear 离线审计

日期：2026-07-03

## 结论

`int8 + per-output-channel fp32 scale` 值得继续推进。当前已经完成三层证据：

1. 真实 Anima checkpoint 的冻结 base Linear 离线反量化误差审计。
2. Anima-shaped toy bf16 small-batch probe，验证冻结 base Linear int8 存储后，trainable adapter
   的 output、loss、grad norm 偏差很小。
3. 真实 `library.anima.models.Anima` tiny forward bf16 small-batch probe，覆盖 self-attn、
   cross-attn、MLP、final unpatchify，并记录 DiT block output L2 / cosine。

随后又补了更靠近显存收益的原型路径：`block_swap_transfer_dtype=int8`。它只把
block swap CPU master 中命中的 frozen MLP/attention Linear 大矩阵存成
`int8 + per-output-channel fp32 scale`，H2D/restore 时反量化回执行 dtype；AdaLN、
adapter/trainable weight、bias、norm 等非候选权重仍保持原 dtype CPU master。当前还增加了
CPU small-batch offloader 对照：同一 Anima-shaped block swap forward/backward 下，对比
`bf16` CPU master 和 `int8` CPU master 的 output、loss、adapter grad norm 和 block output。

但这仍然没有证明完整 Anima checkpoint + 真实训练数据 batch 的 loss、grad norm 或 DiT block
输出等价，也没有证明真实 GPU H2D 性能收益。

已验证范围：

- 默认优先目标：`blocks.*.mlp.layer1.weight`、`blocks.*.mlp.layer2.weight`
- 扩展目标：`self_attn` / `cross_attn` 的 projection weight
- 显式排除：AdaLN/modulation、final layer、timestep embedding、LoRA/adapter、
  router/FEI/guidance、norm/scale/bias、小敏感参数

真实 Anima checkpoint 上，`scope=all` 的 280 个候选 Linear 权重：

- payload 约为 bf16 的 `50.08%`
- overall relative L2 p95 为 `1.5568%`
- attention relative L2 p95 为 `1.5069%`
- MLP relative L2 p95 为 `1.6668%`
- 最坏单层为 `net.blocks.0.mlp.layer2.weight`，relative L2 `2.7596%`

因此，它比之前 raw/scaled FP8 路线更有继续实验价值，但不能直接进入训练路径。

## 新增工具

脚本：

```bash
.venv/bin/python scripts/experiments/int8_base_linear_audit.py \
  --model /path/to/anima-preview3-base.safetensors \
  --out-dir /tmp/anima-int8-base-linear-audit \
  --scope mlp
```

可选 scope：

- `mlp`：只审计第一批软目标 MLP Linear
- `attention`：只审计 attention projection
- `all`：MLP + attention projection

输出：

- `int8_base_linear_audit.jsonl`：逐 tensor 误差明细
- `int8_base_linear_audit_summary.json`：汇总、分 family p95、最坏层列表

实验性 runtime wrapper：

- `library/runtime/int8_linear.py`
- `Int8FrozenLinear`：冻结 Linear 的 int8 weight + per-output-channel fp32 scale 存储。
- `replace_frozen_base_linears_with_int8()`：只替换 `blocks.*.mlp.layer1/2` 和可选
  attention projection；跳过 trainable weight、bias 和非 block 路径。
- 支持 `blocks.*` 与 checkpoint 常见的 `net.blocks.*` 命名匹配。

实验性 block swap CPU master：

- `library/runtime/offloading.py`
- `block_swap_transfer_dtype=int8`
- 只量化 block 内命中的 MLP Linear 和 attention projection frozen weight。
- 未命中的 frozen weight 仍以普通 CPU tensor master 保存。
- `block_swap_config` profile 事件会记录 `int8_master_bytes`、
  `int8_quantized_tensors`、逐 block int8 bytes 和量化误差统计。
- 当前 slab/foreach restore 对 int8 master 保守禁用，走逐权重恢复，避免 slab 丢失 scale。

small-batch probe：

```bash
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope mlp
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope all
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope mlp
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope all
```

这个 probe 支持两种模式：

- `--model-kind toy`：Anima-shaped 的冻结 block + trainable adapter 小模型，快速回归 wrapper。
- `--model-kind anima`：真实 tiny `Anima` forward 路径，使用 input adapter 产生可训练梯度。

对比指标：

- bf16 baseline loss
- int8 frozen base Linear loss
- trainable adapter grad norm
- output relative L2 / cosine
- DiT block output relative L2 / cosine（`--model-kind anima`）

block swap CPU master small-batch probe：

```bash
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py
```

可选真实 CUDA/profile 入口：

```bash
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
  --device cuda \
  --profile-dir /tmp/anima-int8-blockswap-profile
```

这个 probe 比较：

- `block_swap_transfer_dtype=bf16`
- `block_swap_transfer_dtype=int8`

覆盖内容：

- Anima-shaped block surface：self-attn、cross-attn、MLP、AdaLN-like frozen path、trainable adapter
- `blocks_to_swap=2` 的 forward/backward offloader hooks
- output relative L2 / cosine
- loss relative delta
- trainable adapter grad norm delta
- block output relative L2 / cosine
- int8 CPU master bytes、量化 tensor 数量和逐 block 量化误差
- `--profile-dir` 会写出 `bf16_block_swap_profile.jsonl` 与
  `int8_block_swap_profile.jsonl`，并在结果 JSON 中汇总 wait/H2D 统计。

## 实测记录

模型：

```text
/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-preview3-base.safetensors
```

### MLP only

命令：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_base_linear_audit.py \
  --model /home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-preview3-base.safetensors \
  --out-dir /tmp/anima-int8-base-linear-audit-mlp \
  --scope mlp
```

结果：

```text
tensors: 56
payload ratio vs bf16: 0.5006
relative L2: p50=1.2182%, p95=1.6668%, max=2.7596%
gate: PASS
```

最坏层：

```text
net.blocks.0.mlp.layer2.weight  rel=2.7596%
net.blocks.0.mlp.layer1.weight  rel=1.9211%
net.blocks.2.mlp.layer2.weight  rel=1.6668%
net.blocks.27.mlp.layer2.weight rel=1.6382%
net.blocks.1.mlp.layer1.weight  rel=1.5639%
```

### MLP + attention

命令：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_base_linear_audit.py \
  --model /home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-preview3-base.safetensors \
  --out-dir /tmp/anima-int8-base-linear-audit-all \
  --scope all
```

结果：

```text
tensors: 280
payload ratio vs bf16: 0.5008
overall relative L2: p50=0.9297%, p95=1.5568%, max=2.7596%
attention: n=224, p95=1.5069%, max=2.3097%
mlp: n=56, p95=1.6668%, max=2.7596%
gate: PASS
```

最坏层：

```text
net.blocks.0.mlp.layer2.weight         rel=2.7596%
net.blocks.0.self_attn.v_proj.weight   rel=2.3097%
net.blocks.1.self_attn.v_proj.weight   rel=1.9860%
net.blocks.0.mlp.layer1.weight         rel=1.9211%
net.blocks.0.cross_attn.q_proj.weight  rel=1.8604%
```

### Small-batch bf16 probe

命令：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope mlp
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope all
```

MLP only 结果：

```text
replacement_count: 4
payload ratio vs bf16: 0.5781
output relative L2: 0.1588%
output cosine: 0.9999988
loss relative delta: 0.0188%
adapter grad norm relative delta: 0.0049%
gate: PASS
```

MLP + attention 结果：

```text
replacement_count: 8
payload ratio vs bf16: 0.5875
output relative L2: 0.1586%
output cosine: 0.9999988
loss relative delta: 0.0182%
adapter grad norm relative delta: 0.0049%
gate: PASS
```

### Block swap int8 CPU master probe

命令：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py
```

结果：

```text
model_kind: blockswap_toy
device: cpu
blocks_to_swap: 2
baseline_transfer_dtype: bf16
candidate_transfer_dtype: int8
int8_quantized_tensors: 28
int8_master_ratio_vs_bf16: 0.5662
output relative L2: 0.3650%
max block output relative L2: 0.3549%
output cosine: 0.9999937
loss relative delta: 0.0446%
adapter grad norm relative delta: 0.0473%
gate: PASS
```

开启 `--profile-dir` 的 CPU smoke test 也能写出 profile JSONL，并验证两边 config：

```text
bf16 profile events: 5
int8 profile events: 5
bf16 wait events: 4
int8 wait events: 4
```

CPU profile 里的 `h2d_ms` 只是 no-CUDA restore 代码段耗时，不代表真实 PCIe H2D；它只用于
确认 profile plumbing 和事件字段。真实 H2D/等待时间必须用 `--device cuda --profile-dir ...`
在可用 GPU 上跑。

### Tiny Anima bf16 probe

命令：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope mlp
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope all
```

MLP only 结果：

```text
replacement_count: 4
payload ratio vs bf16: 0.5625
output relative L2: 0.5342%
max block output relative L2: 0.3812%
output cosine: 0.9999858
loss relative delta: 0.0546%
adapter grad norm relative delta: 0.1805%
gate: PASS
```

MLP + attention 结果：

```text
replacement_count: 14
payload ratio vs bf16: 0.5764
output relative L2: 0.6770%
max block output relative L2: 0.5501%
output cosine: 0.9999775
loss relative delta: 0.1597%
adapter grad norm relative delta: 0.1036%
gate: PASS
```

## 当前解释

和 FP8 e4m3 相比，int8 的均匀量化配合 per-channel scale 更适合这些 frozen Linear
权重。对大矩阵而言，scale 元数据开销很小，实际 payload 基本接近 `numel * 1 byte`。

需要注意：

- p95 过门槛不代表所有层安全；block 0 的 MLP 和 early attention 仍是最敏感区域。
- 当前审计只看权重反量化误差，不看激活分布放大效应。
- toy probe 只能证明 wrapper 和 trainable adapter 梯度对照机制跑通。
- tiny Anima probe 使用真实 DiT forward 代码，但不是完整 checkpoint、不是真实数据分布，
  也没有覆盖真实 LoRA monkey-patch 后的训练图。
- block swap probe 覆盖了 offloader forward/backward hooks、CPU master restore 逻辑和 profile
  JSONL plumbing，但默认仍是 CPU toy surface，不是完整 Anima checkpoint。
- 如果 forward 前临时反量化，PyTorch backward 仍可能保存 dequantized weight，显存收益不一定兑现。
- 真实 CUDA H2D 反量化、等待时间和显存峰值还没有跑；已准备 `--device cuda --profile-dir`
  小跑入口。
- 真正要进入训练路径，还需要完整 checkpoint + 真实训练 batch 的 loss、grad norm、output L2 对照。

## 下一步

1. 用完整 Anima checkpoint + 真实训练 batch 对比 bf16 baseline 与 int8-dequant base Linear：
   - loss
   - adapter grad norm
   - DiT block output L2 / cosine
   - peak allocated / reserved
2. 先只替换 MLP Linear，保留 attention 为对照。
3. 如果 MLP 小 batch 通过，再扩到 attention projection。
4. 对 `block_swap_transfer_dtype=int8` 跑真实 GPU profile：
   - H2D / enqueue / wait 分布
   - `int8_master_bytes` 与 bf16/fp8 对照
   - loss/grad/output 是否和 bf16 block swap baseline 一致
5. 若真实 batch 和 GPU profile 都通过，再考虑 custom autograd，避免 backward 保存整份反量化权重。

## 验证

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_int8_base_linear_audit.py \
  tests/test_int8_linear_runtime.py \
  tests/test_int8_linear_equivalence_probe.py \
  tests/test_int8_blockswap_equivalence_probe.py \
  tests/test_block_swapping.py
timeout 60 .venv/bin/python scripts/experiments/int8_base_linear_audit.py --help
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope mlp
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope all
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope mlp
timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope all
timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py
tmpdir=$(mktemp -d /tmp/int8-blockswap-profile.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
git diff --check -- \
  scripts/experiments/int8_base_linear_audit.py \
  scripts/experiments/int8_linear_equivalence_probe.py \
  scripts/experiments/int8_blockswap_equivalence_probe.py \
  library/runtime/int8_linear.py \
  library/runtime/offloading.py \
  tests/test_int8_base_linear_audit.py \
  tests/test_int8_linear_runtime.py \
  tests/test_int8_linear_equivalence_probe.py \
  tests/test_int8_blockswap_equivalence_probe.py \
  tests/test_block_swapping.py \
  docs/findings/anima_int8_base_linear_audit.md
```

结果：

```text
tests/test_int8_linear_runtime.py ....             4 passed
tests/test_int8_base_linear_audit.py ...           3 passed
tests/test_int8_linear_equivalence_probe.py ....   4 passed
tests/test_int8_blockswap_equivalence_probe.py ..  2 passed
```

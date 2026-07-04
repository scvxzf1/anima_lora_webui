# Anima int8 base Linear 离线审计

日期：2026-07-03

## 结论

`int8 + per-output-channel fp32 scale` 值得继续推进。当前已经完成四层证据：

1. 真实 Anima checkpoint 的冻结 base Linear 离线反量化误差审计。
2. Anima-shaped toy bf16 small-batch probe，验证冻结 base Linear int8 存储后，trainable adapter
   的 output、loss、grad norm 偏差很小。
3. 真实 `library.anima.models.Anima` tiny forward bf16 small-batch probe，覆盖 self-attn、
   cross-attn、MLP、final unpatchify，并记录 DiT block output L2 / cosine。
4. 完整 Anima checkpoint + 真实缓存 latent/TE batch 的 CUDA probe，顺序加载 bf16 baseline
   与 int8-stored frozen MLP Linear，对比 output、loss、probe adapter grad norm 和 block output。

随后又补了更靠近显存收益的原型路径：`block_swap_transfer_dtype=int8`。它只把
block swap CPU master 中命中的 frozen MLP/attention Linear 大矩阵存成
`int8 + per-output-channel fp32 scale`，H2D/restore 时反量化回执行 dtype；AdaLN、
adapter/trainable weight、bias、norm 等非候选权重仍保持原 dtype CPU master。当前还增加了
CPU small-batch offloader 对照：同一 Anima-shaped block swap forward/backward 下，对比
`bf16` CPU master 和 `int8` CPU master 的 output、loss、adapter grad norm 和 block output。

现在已经证明完整 checkpoint 的 MLP-only 路径在一个真实缓存 batch 上可跑通并过 gate；但这仍然
不是完整训练 loop，不覆盖真实 LoRA monkey-patch 后的 adapter 图，也没有证明真实 block swap
H2D 性能收益。

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
- `self_attn_qkv` / `self_attn_out`：只审计 self-attention 的 QKV 或 output projection
- `cross_attn_q` / `cross_attn_kv` / `cross_attn_out`：只审计 cross-attention 的 Q、KV 或 output projection
- `attention_out`：只审计 self/cross attention output projection
- `all`：MLP + attention projection

输出：

- `int8_base_linear_audit.jsonl`：逐 tensor 误差明细
- `int8_base_linear_audit_summary.json`：汇总、分 family p95、最坏层列表

实验性 runtime wrapper：

- `library/runtime/int8_linear.py`
- `Int8FrozenLinear`：冻结 Linear 的 int8 weight + per-output-channel fp32 scale 存储。
- `replace_frozen_base_linears_with_int8()`：只替换 `blocks.*.mlp.layer1/2` 和可选
  attention projection；跳过 trainable weight、bias 和非 block 路径。
- `scope` 支持按 projection 子集组合，例如 `mlp,cross_attn_q`、
  `mlp,self_attn_qkv`、`mlp,attention_out`。
- 支持 `blocks.*` 与 checkpoint 常见的 `net.blocks.*` 命名匹配。

实验性 block swap CPU master：

- `library/runtime/offloading.py`
- `block_swap_transfer_dtype=int8`
- 只量化 block 内命中的 MLP Linear 和 attention projection frozen weight。
- `int8_scope` 默认保持旧行为 `all`；可用 `ANIMA_BLOCK_SWAP_INT8_SCOPE=mlp`
  或 probe 的 `--int8-scope mlp,cross_attn_q` 收窄候选范围。
- 未命中的 frozen weight 仍以普通 CPU tensor master 保存。
- `block_swap_config` profile 事件会记录 `int8_master_bytes`、
  `int8_quantized_tensors`、`int8_scope`、逐 block int8 bytes 和量化误差统计。
- 当前 slab/foreach restore 对 int8 master 保守禁用，走逐权重恢复，避免 slab 丢失 scale。

small-batch probe：

```bash
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope mlp
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --scope all
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope mlp
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --scope all
.venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
  --model-kind checkpoint \
  --scope mlp \
  --dit-path models/diffusion_models/anima-preview3-base.safetensors \
  --data-dir post_image_dataset/rokkotsu_goddess \
  --cache-index 0 \
  --device cuda:0
```

这个 probe 支持三种模式：

- `--model-kind toy`：Anima-shaped 的冻结 block + trainable adapter 小模型，快速回归 wrapper。
- `--model-kind anima`：真实 tiny `Anima` forward 路径，使用 input adapter 产生可训练梯度。
- `--model-kind checkpoint`：完整 Anima checkpoint + 真实 latent/TE 缓存 batch；顺序加载 baseline
  和 int8 模型，避免同时持有两份 DiT。

对比指标：

- bf16 baseline loss
- int8 frozen base Linear loss
- trainable adapter grad norm
- output relative L2 / cosine
- DiT block output relative L2 / cosine（`--model-kind anima` / `checkpoint`）
- CUDA peak allocated / reserved（`--model-kind checkpoint`）
- `--repeat-seeds N`：从 `--seed` 开始连续跑 N 个 seed，输出逐 seed 结果和聚合摘要。
- `--repeat-caches N`：checkpoint probe 专用，从 `--cache-index` 开始连续跑 N 个缓存样本。

block swap CPU master small-batch probe：

```bash
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py --int8-scope mlp
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py --int8-scope mlp,cross_attn_q
```

可选真实 CUDA/profile 入口：

```bash
.venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
  --device cuda \
  --int8-scope mlp \
  --profile-dir /tmp/anima-int8-blockswap-profile
```

这个 probe 比较：

- `block_swap_transfer_dtype=bf16`
- `block_swap_transfer_dtype=int8`
- candidate 侧的 `int8_scope`，默认 `all`，可显式切到 `mlp` 或 projection 子集组合。

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

本轮 CPU scope 控制小跑：

```bash
timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
  --int8-scope mlp,cross_attn_q \
  --out /tmp/anima-int8-blockswap-scope-probe.json
```

结果：

```text
scope: mlp,cross_attn_q
gate: PASS
int8_quantized_tensors: 12
int8_master_ratio_vs_bf16: 0.3088
output relative L2: 0.3722%
loss relative delta: 0.0237%
adapter grad norm relative delta: 0.0219%
```

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

RTX 3080 CUDA 小跑：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-profile.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果：

```text
device: cuda
gate: PASS
output relative L2: 0.3650%
loss relative delta: 0.0446%
adapter grad norm relative delta: 0.0528%
int8_master_ratio_vs_bf16: 0.5662

bf16 h2d mean/p95/max: 0.4325 / 0.6514 / 0.6514 ms
int8 h2d mean/p95/max: 2.0226 / 2.5610 / 2.5610 ms
bf16 wait mean/p95/max: 0.0795 / 0.1731 / 0.1731 ms
int8 wait mean/p95/max: 0.4818 / 0.8373 / 0.8373 ms
```

RTX 3080 CUDA 4-step repeat 小跑：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-repeat.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda:0 \
    --repeat-steps 4 \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果：

```text
device: cuda:0
repeat_steps: 4
gate: PASS
output relative L2: 0.4577%
loss relative delta: 0.1880%
adapter grad norm relative delta: 0.1645%
int8_master_ratio_vs_bf16: 0.5662

bf16 h2d mean/p95/max: 1.3932 / 4.1114 / 4.1114 ms
int8 h2d mean/p95/max: 2.4149 / 4.3950 / 4.3950 ms
h2d ratio int8/bf16 mean/p95/max: 1.7333 / 1.0690 / 1.0690

bf16 wait mean/p95/max: 0.0605 / 0.1585 / 0.1585 ms
int8 wait mean/p95/max: 0.6772 / 1.2519 / 1.2519 ms
wait ratio int8/bf16 mean/p95/max: 11.1846 / 7.8962 / 7.8962

bf16 peak allocated/reserved: 17.12 / 23.07 MB
int8 peak allocated/reserved: 17.13 / 25.17 MB
memory ratio allocated/reserved: 1.0006 / 1.0909
```

RTX 3080 CUDA 大矩阵小跑：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-large.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda:0 \
    --dim 1024 \
    --hidden-dim 4096 \
    --num-blocks 4 \
    --blocks-to-swap 2 \
    --batch-size 2 \
    --repeat-steps 3 \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果：

```text
device: cuda:0
dim / hidden_dim: 1024 / 4096
repeat_steps: 3
gate: PASS
output relative L2: 0.6579%
max block output relative L2: 0.5710%
loss relative delta: 0.0477%
adapter grad norm relative delta: 0.0604%
int8_master_ratio_vs_bf16: 0.4721

bf16 h2d mean/p95/max: 3.5384 / 4.1545 / 4.1545 ms
int8 h2d mean/p95/max: 3.1088 / 4.1359 / 4.1359 ms
h2d ratio int8/bf16 mean/p95/max: 0.8786 / 0.9955 / 0.9955

bf16 wait mean/p95/max: 0.0541 / 0.2313 / 0.2313 ms
int8 wait mean/p95/max: 0.7626 / 1.5380 / 1.5380 ms
wait ratio int8/bf16 mean/p95/max: 14.1018 / 6.6505 / 6.6505

bf16 peak allocated/reserved: 340.07 / 358.61 MB
int8 peak allocated/reserved: 375.75 / 480.25 MB
memory ratio allocated/reserved: 1.1049 / 1.3392
```

RTX 3080 CUDA 大矩阵 direct-bind restore 小跑：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-direct.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda:0 \
    --dim 1024 \
    --hidden-dim 4096 \
    --num-blocks 4 \
    --blocks-to-swap 2 \
    --batch-size 2 \
    --repeat-steps 3 \
    --int8-restore-mode direct_bind \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果：

```text
device: cuda:0
int8_restore_mode: direct_bind
gate: PASS
output relative L2: 0.6579%
loss relative delta: 0.0477%
adapter grad norm relative delta: 0.0604%
int8_master_ratio_vs_bf16: 0.4721

bf16 h2d mean/p95/max: 3.5542 / 4.0686 / 4.0686 ms
int8 h2d mean/p95/max: 3.1381 / 4.4902 / 4.4902 ms
h2d ratio int8/bf16 mean/p95/max: 0.8829 / 1.1036 / 1.1036

bf16 wait mean/p95/max: 0.0487 / 0.1364 / 0.1364 ms
int8 wait mean/p95/max: 0.7247 / 1.5681 / 1.5681 ms
wait ratio int8/bf16 mean/p95/max: 14.8764 / 11.4983 / 11.4983

bf16 peak allocated/reserved: 340.07 / 358.61 MB
int8 peak allocated/reserved: 375.75 / 589.30 MB
memory ratio allocated/reserved: 1.1049 / 1.6433
```

RTX 3080 CUDA 大矩阵 reuse-storage restore 小跑：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-reuse.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda:0 \
    --dim 1024 \
    --hidden-dim 4096 \
    --num-blocks 4 \
    --blocks-to-swap 2 \
    --batch-size 2 \
    --repeat-steps 3 \
    --int8-restore-mode reuse_storage \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果：

```text
device: cuda:0
int8_restore_mode: reuse_storage
gate: PASS
output relative L2: 0.6884%
loss relative delta: 0.0590%
adapter grad norm relative delta: 0.0482%
int8_master_ratio_vs_bf16: 0.4721

bf16 h2d mean/p95/max: 3.6485 / 4.1773 / 4.1773 ms
int8 h2d mean/p95/max: 2.7175 / 3.6792 / 3.6792 ms
h2d ratio int8/bf16 mean/p95/max: 0.7448 / 0.8808 / 0.8808

bf16 wait mean/p95/max: 0.0387 / 0.0706 / 0.0706 ms
int8 wait mean/p95/max: 0.4861 / 1.3764 / 1.3764 ms
wait ratio int8/bf16 mean/p95/max: 12.5660 / 19.5034 / 19.5034

bf16 peak allocated/reserved: 340.07 / 358.61 MB
int8 peak allocated/reserved: 375.75 / 425.72 MB
memory ratio allocated/reserved: 1.1049 / 1.1871
```

RTX 3080 CUDA 大矩阵 reuse-storage row-chunk restore 对照：

```bash
tmpdir=$(mktemp -d /tmp/int8-blockswap-cuda-chunk.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_blockswap_equivalence_probe.py \
    --device cuda:0 \
    --dim 1024 \
    --hidden-dim 4096 \
    --num-blocks 4 \
    --blocks-to-swap 2 \
    --batch-size 2 \
    --repeat-steps 3 \
    --int8-restore-mode reuse_storage \
    --int8-restore-chunk-rows 512 \
    --profile-dir "$tmpdir" \
    --out "$tmpdir/result.json"
```

结果汇总：

```text
chunk_rows  gate  h2d mean ratio  wait mean ratio  peak reserved ratio
0           PASS  0.7060          4.2855           1.1871
256         PASS  2.4513          63.3670          1.1930
512         PASS  1.8984          47.5287          1.1930
2048        PASS  0.8971          14.0310          1.1871
```

解释：几十 KiB 的 toy surface 上，当前 int8 restore 会执行
`int8/scale H2D -> GPU 反量化 -> copy 回复用 storage`，固定开销压过 payload 下降，整体更慢。
放大到 `dim=1024, hidden=4096` 后，H2D mean 开始出现收益（当前复跑的 `reuse_storage`
约为 bf16 的 `70.6%`），说明 payload 下降确实有机会变成 copy 带宽收益；但 wait 仍更差，
peak allocated/reserved 也更高。这说明当前实现的问题更集中在 restore 调度、反量化临时
tensor 和额外 GPU copy，而不是 int8 per-channel 量化本身。

`direct_bind` 去掉“反量化后 copy 回复用 storage”，但会产生更多新的 GPU storage 生命周期压力；
在这次小跑里 wait 只略有改善，reserved peak 明显更差，因此不能直接作为训练路径优化。
`reuse_storage` 把反量化结果写回复用 weight storage，H2D mean 和 reserved peak 都明显好于
`direct_bind`，也好于默认 copy 的 reserved peak；但 wait 仍显著落后，说明还需要继续减少
per-weight restore 调度和反量化临时 tensor。row-chunk restore 能降低单个反量化临时 tensor
的行数，但会把一次 weight restore 拆成更多 H2D 和反量化 kernel；`256/512/2048` 三个点都没有
降低 peak reserved，且 wait 明显变差，因此当前不建议启用，默认继续保持 `0`。

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

### 完整 checkpoint + 真实缓存 batch probe

模型和缓存：

```text
DiT: models/diffusion_models/anima-preview3-base.safetensors
data_dir: post_image_dataset/rokkotsu_goddess
cache_index: 0
latent: 0.png_202604200135 (1)_0896x1200_anima.npz
TE: 0.png_202604200135 (1)_anima_te.safetensors
latent shape: (16, 144, 112)
context shape: (512, 1024)
device: cuda:0 / RTX 3080
```

MLP-only backward probe：

```bash
tmpdir=$(mktemp -d /tmp/int8-checkpoint-backward.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
```

结果：

```text
replacement_count: 56
payload ratio vs bf16: 0.5006
output relative L2: 0.6963%
output cosine: 0.9999568
loss relative delta: 0.0180%
probe adapter grad norm relative delta: 3.1976%
baseline grad norm: 0.0309218
int8 grad norm: 0.0319105
baseline peak allocated/reserved: 4.87 / 4.99 GiB
int8 peak allocated/reserved: 5.02 / 5.18 GiB
gate: PASS
```

MLP-only block-output probe：

```bash
tmpdir=$(mktemp -d /tmp/int8-checkpoint-blocks.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --forward-only \
    --out "$tmpdir/result.json"
```

结果：

```text
replacement_count: 56
block output deltas: 28 blocks
max block output relative L2: 3.2868%
output relative L2: 0.6963%
loss relative delta: 0.0180%
gate: PASS
```

解释：完整 checkpoint 的最终 output/loss/grad 都在当前 gate 内；但中层 block output 最大相对
L2 到 `3.29%`，说明 MLP int8 误差在中间层确实会放大，只是最终输出在这一 batch 上重新收敛。
另外，当前 checkpoint probe 是“GPU 上 int8 存储、forward 前反量化”，不是 block swap CPU
master 路径；int8 run 的峰值显存略高于 baseline，符合“临时反量化 tensor 仍会吃显存”的预期。

MLP + all attention backward probe：

```bash
tmpdir=$(mktemp -d /tmp/int8-checkpoint-all-backward.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope all \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
```

结果：

```text
replacement_count: 196
payload ratio vs bf16: 0.5008
output relative L2: 0.7648%
loss relative delta: 0.0473%
probe adapter grad norm relative delta: 6.1153%
gate: FAIL
```

解释：`scope=all` 的 output/loss 仍然很接近，但 probe adapter grad norm 超过当前 `5%`
阈值，所以不建议把全部 attention projection 直接纳入 int8。

Attention projection 子集 sweep（同一 cache，seed 0，backward，`--no-capture-blocks`）：

```text
scope                 replacements  gate  output L2  loss delta  grad norm delta
attention             140           FAIL  0.6365%    0.0751%     8.2660%
mlp,self_attn_out     84            FAIL  0.7179%    0.0560%     6.0501%
mlp,cross_attn_out    84            FAIL  0.6854%    0.0108%     31.2598%
mlp,cross_attn_kv     84            FAIL  0.6927%    0.0944%     7.5453%
mlp,self_attn_qkv     84            PASS  0.7393%    0.0353%     0.0306%
mlp,cross_attn_q      84            PASS  0.6945%    0.0546%     1.3760%
mlp,attention_out     112           PASS  0.7131%    0.0207%     2.6107%
mlp,cross_attn        140           FAIL  0.6967%    0.0289%     77.2767%
mlp,self_attn         112           FAIL  0.7456%    0.0381%     10.6366%
```

Seed 1 复核显示 `mlp,self_attn_qkv` 和 `mlp,attention_out` 并不稳定：

```text
scope                 seed  gate  grad norm delta
mlp,self_attn_qkv     1     FAIL  28.9405%
mlp,cross_attn_q      1     PASS  1.2664%
mlp,attention_out     1     FAIL  34.9110%
```

`--repeat-seeds 3` 聚合结果：

```bash
tmpdir=$(mktemp -d /tmp/int8-repeat-checkpoint.XXXXXX) && \
  timeout 120 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp,cross_attn_q \
    --repeat-seeds 3 \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
```

```text
scope             pass/total  output L2 p50/max  loss delta p50/max  grad delta p50/max
mlp               1/3         0.6963% / 1.8615%  0.0346% / 0.2870%   29.8166% / 39.5554%
mlp,cross_attn_q  2/3         0.6945% / 1.8645%  0.0546% / 0.2882%   1.6224% / 32.1574%
```

结论：在这一个真实 cache 上，`cross_attn_q` 是当前相对最软的 attention projection 候选；
但单 batch + probe input adapter 的 grad norm 对 seed 很敏感，`mlp,cross_attn_q` 也不是三
个 seed 全过。因此当前只能把 `cross_attn_q` 作为下一轮优先候选，不能把它视为已经可进训练。

`--repeat-caches 2` 聚合结果（seed 0，cache 0/1）：

```bash
tmpdir=$(mktemp -d /tmp/int8-repeat-caches.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp,cross_attn_q \
    --repeat-caches 2 \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
```

```text
scope             pass/total  output L2 p50/max  loss delta p50/max  grad delta p50/max
mlp               2/2         0.7073% / 0.7073%  0.0362% / 0.0362%   4.1470% / 4.1470%
mlp,cross_attn_q  1/2         0.7104% / 0.7104%  0.0546% / 0.0546%   13.8514% / 13.8514%
```

逐 cache：

```text
scope             cache  gate  grad delta  base_stem
mlp               0      PASS  3.3537%     0.png_202604200135 (1)
mlp               1      PASS  4.1470%     0.png_202604200135 (2)
mlp,cross_attn_q  0      PASS  1.4036%     0.png_202604200135 (1)
mlp,cross_attn_q  1      FAIL  13.8514%    0.png_202604200135 (2)
```

解释：`cross_attn_q` 在 seed 维度比其它 attention 子集更好，但跨 cache 仍会失败。它依然是
“部分 attention”优先候选，不是已验证稳定配置。

## 当前解释

和 FP8 e4m3 相比，int8 的均匀量化配合 per-channel scale 更适合这些 frozen Linear
权重。对大矩阵而言，scale 元数据开销很小，实际 payload 基本接近 `numel * 1 byte`。

需要注意：

- p95 过门槛不代表所有层安全；block 0 的 MLP 和 early attention 仍是最敏感区域。
- 当前审计只看权重反量化误差，不看激活分布放大效应。
- toy probe 只能证明 wrapper 和 trainable adapter 梯度对照机制跑通。
- tiny Anima probe 使用真实 DiT forward 代码，但不是完整 checkpoint、不是真实数据分布，
  也没有覆盖真实 LoRA monkey-patch 后的训练图。
- checkpoint probe 已覆盖完整 Anima checkpoint + 真实 latent/TE 缓存 batch，并支持 projection
  子集和多 seed 聚合；但它仍是 probe adapter，不是实际 LoRA/adapter monkey-patch 训练图。
- block swap probe 覆盖了 offloader forward/backward hooks、CPU master restore 逻辑和 profile
  JSONL plumbing，但默认仍是 CPU toy surface，不是完整 Anima checkpoint。
- 如果 forward 前临时反量化，PyTorch backward 仍可能保存 dequantized weight，显存收益不一定兑现。
- 真实 CUDA 小跑显示：小矩阵上 `int8` 全面更慢；大矩阵上 H2D mean 已有收益，但 wait 和
  peak memory 仍更差。`reuse_storage` 比 `direct_bind` 更接近正确方向，但瓶颈仍在 restore
  调度和反量化临时 tensor。
- `int8_restore_chunk_rows` 的 `256/512/2048` 对照都没有降低 peak reserved，且 wait 明显更差；
  当前只保留为实验开关，默认应继续使用 `0`。
- 完整 checkpoint 的 MLP-only seed0 CUDA probe 显示 output/loss/grad 可过 gate，但 block 中间输出
  最大相对 L2 达到 `3.29%`；`--repeat-seeds 3` 又显示 MLP-only probe adapter grad norm 并不稳，
  所以后续应使用多 seed / 多 batch 聚合，不能只看单次 PASS。
- attention projection 中，`cross_attn_q` 是目前相对最软的子集；但 `--repeat-caches 2`
  已显示它跨 cache 仍不稳定。`attention/all`、`cross_attn`、`cross_attn_out`、`self_attn`、
  `self_attn_qkv` 等在当前 probe 下也出现 grad norm 风险。
- 真实完整训练 batch 的 block swap CUDA H2D、等待时间和显存峰值还没有跑。
- 真正要进入训练路径，还需要实际 LoRA/adapter monkey-patch 图下的 loss、grad norm、output L2 对照。

## 下一步

1. 把完整 checkpoint probe 扩成更系统的多 seed x 多 cache 聚合，优先比较：
   - `mlp`
   - `mlp,cross_attn_q`
   - `mlp,cross_attn_q` + 更宽松/更稳定的梯度统计，例如 grad cosine 或多 batch mean。
2. 用实际 LoRA/adapter monkey-patch 图复测完整 checkpoint + 真实 batch：
   - loss
   - adapter/LoRA grad norm
   - DiT block output L2 / cosine
   - peak allocated / reserved
3. 扩展 `block_swap_transfer_dtype=int8` 的 GPU profile：
   - 更接近完整 Anima block 的大矩阵尺寸和多 step warmup/采样。
   - H2D / enqueue / wait 分布。
   - `int8_master_bytes` 与 bf16/fp8 对照。
   - loss/grad/output 是否和 bf16 block swap baseline 一致。
4. 当前大矩阵 profile 已显示 H2D 有收益但 wait/memory 更差，优先评估 restore 优化：
   - `reuse_storage` 优于 `direct_bind`，继续沿复用目标 storage 的方向推进。
   - 需要减少反量化临时 tensor，而不是换模块 weight storage。
   - 合并 per-weight restore 调度。
   - row-chunk restore 没有降低当前 probe 的 peak reserved，后续应优先减少 kernel/拷贝次数。
   - 后续才考虑 custom autograd，避免 backward 保存整份反量化权重。

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
tmpdir=$(mktemp -d /tmp/int8-checkpoint-backward.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
tmpdir=$(mktemp -d /tmp/int8-checkpoint-blocks.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --forward-only \
    --out "$tmpdir/result.json"
tmpdir=$(mktemp -d /tmp/int8-repeat-checkpoint.XXXXXX) && \
  timeout 120 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp,cross_attn_q \
    --repeat-seeds 3 \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
tmpdir=$(mktemp -d /tmp/int8-repeat-caches.XXXXXX) && \
  timeout 60 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py \
    --model-kind checkpoint \
    --scope mlp,cross_attn_q \
    --repeat-caches 2 \
    --dit-path models/diffusion_models/anima-preview3-base.safetensors \
    --data-dir post_image_dataset/rokkotsu_goddess \
    --cache-index 0 \
    --device cuda:0 \
    --no-capture-blocks \
    --out "$tmpdir/result.json"
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

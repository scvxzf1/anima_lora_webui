# Krea-2-Raw 迁移 阶段 4: 训练串通 (findings)

状态：历史阶段快照 / 阶段 4 已完成
适用版本：阶段 4 训练探针落地时点；不作为当前完整能力说明
入口命令：`.venv/bin/python scripts/krea2/probe_train.py`
相关代码：`library/models/krea2_raw/family.py`、`scripts/krea2/probe_train.py`

> “已知限制 / 后续”保留阶段 4 当时状态；正式 trainer dispatch 和
> `cache_batch_outputs` 后续均已接通。当前边界见
> [`../multi_model_support.md`](../multi_model_support.md)。

## 目标

承重接口 `forward_for_loss` 落地 + 单 prompt 过拟合训练热测:
flow-matching 训练 loop 跑通, loss 单调下降, LoRA 梯度流到参数, DiT 冻结不变,
记录显存/功耗/速度/loss 基线.

## 设计定论

### forward_for_loss 承重接口 (library/models/krea2_raw/family.py)

`ModelFamily` Protocol 的 Krea-2 训练侧承重接口. 阶段 4 仅在自包含探针里验证,
正式串通 train.py / noise_target.py 是阶段 6 配置收口的事 (反上帝守则: 不在一轮
里同时改架构和改行为, train.py / noise_target.py 是热点文件).

```python
def forward_for_loss(dit, latents_5d, text_emb, t, **kw) -> Tensor:
    # latents_5d: (B, C, T=1, H, W) anima 5D 不变量
    # text_emb: Krea2TextEmbedding(hiddens (B,L,12,2560), mask (B,L) bool)
    # t: (B,) timestep = σ ∈ [0, 1]
    # 返回 velocity (B, C, T=1, H, W) 与 latents_5d 同形
```

内部: 5D → squeeze(2) 4D → patchify (B, L_img, patch²·C) + 构造 3D pos / mask →
DiT forward → rearrange 还原 4D → unsqueeze(2) 5D. `prepare_img_tokens()` 把
krea-ai/krea-2 sampling.prepare 提成 family 模块级函数, 避免探针复制.

### flow-matching 训练数学 (子代理核实 anima noise_target.py:380-381 / noise.py:171)

anima 的 flow-matching 与 Krea-2 同构 (都是 rectified flow), 数学可直接复用:

- `target = noise - latents` (x1 - x0 velocity)
- `x_t = (1-σ)·latents + σ·noise`, σ = t ∈ [0,1] (t=0 clean, t=1 noise)
- `loss = MSE(dit_output, target)`, 默认无 weighting
- DiT timestep = σ ∈ [0,1] float, temb 内部 `t*tfactor(1e3)` sinusoidal embedding
  (dit.py:75-92)
- 训练默认 sigmoid 采样, 不做 mu shift (推理才做)

**关键差异**: anima DiT forward 签名 (crossattn_emb / llm_adapter / 融合投影)
vs Krea-2 (img / context / t / pos / mask). forward_for_loss 只调 Krea-2 签名,
不沾 anima cross-attn / AdaLN 假设.

### 过拟合测试方法论 (重要)

flow-matching loss 的**绝对值依赖 σ** (σ≈0/1 自然高/低). 不同 σ 的 loss 不可
直接比较 "下降". 所以过拟合测试用:
- **固定 σ=0.5**: target 量级一致
- **固定 noise seed**: target 完全固定, 纯过拟合单样本

真实训练用 sigmoid 采样 + 每步随机 noise; 这里只验 LoRA 能否调整 frozen DiT
输出拟合固定 target.

## 出口验证

### 验证: 30 步过拟合 (PG199 bf16, 256×256, lora_dim=16, lr=2e-3, 固定 σ=0.5)

`scripts/krea2/probe_train.py`:

```
--- D. 训练 30 步 (flow-matching, 固定 σ=0.5 + 固定 noise 过拟合) ---
  step   0: loss=0.0125, grad_norm=0.0008, step=1137ms
  step   4: loss=0.0031, grad_norm=0.0025, step=363ms
  step  10: loss=0.0010, grad_norm=0.0016, step=367ms
  step 20: loss=0.0005, grad_norm=0.0012, step=378ms
  step 28: loss=0.0003, grad_norm=0.0008, step=375ms
  step 29: loss=0.0003, grad_norm=0.0006, step=411ms

finite: True
first5 avg=0.0088, last5 avg=0.0003, 下降: True
grad_norm 范围 [0.0005, 0.0112], 全非零: True

阶段 4 训练串通通过: True
```

验证项全绿:
- forward_for_loss 训练模式跑通, 输出 5D velocity (1,16,1,32,32) 有限 ✓
- loss 单调下降: 0.0125 → 0.0003 (40 倍下降), first5=0.0088 → last5=0.0003 ✓
- LoRA grad 全非零 (梯度流到 48.17M LoRA 参数) ✓
- DiT frozen (requires_grad=False, 仅 LoRA 训练) ✓
- 三阶段显存调度 (TE→free→VAE→free→DiT+LoRA→train) lazy loading 不变量保持 ✓

## 基线 (PG199 bf16, 256×256, lora_dim=16, lr=2e-3)

| 指标 | 值 |
|---|---|
| DiT+LoRA 显存 peak | 32.62GB |
| LoRA 可训参数 | 48.17M |
| avg step 时间 | 400ms |
| 首 step 时间 | 1137ms (含 cudnn autotune) |
| 末 step 时间 | 411ms |
| loss first5 → last5 | 0.0088 → 0.0003 |
| GPU 功耗 (idle → train) | 44.8W → 221.9W |
| TE 加载耗时 | 147s (8.4GB safetensors) |
| DiT 加载耗时 | 12.77s |
| LoRA 创建耗时 | <1s (196 modules) |

显存 32.62GB 紧贴 PG199 32GB 上限 — block swap / offload 在阶段 5 推理 + 阶段 6
大分辨率训练时可能需要 (256×256 训练勉强够, 512×512 训练必爆).

## 阶段 4 截止时的已知限制 / 后续（历史快照）

- **阶段 4 当时未串通 train.py**：`forward_for_loss` 仅在探针验证；正式串通
  `train.py` / `noise_target.py` / `model_loading.py` 是阶段 6 配置收口工作。当前已由
  `library/training/batch_step.py`、`model_loading.py` 等 family dispatch 接通。
- **阶段 4 当时存在 flux_shift x2 不匹配**：训练与推理端点随后在 Krea-2 专用
  family/sampling 链路中对齐；本条保留原始排查记录。
- **阶段 4 当时 256×256 显存紧**：后续 block swap、NF4 与梯度检查点改变了可用配置；
  本条数值只适用于阶段 4 的 PG199 探针。
- **阶段 4 当时未测 `cache_batch_outputs`**：当前
  `Krea2TextEncoderOutputsCachingStrategy` 已支持 multi-variant + per-sample 写盘；
  原条目是阶段性验证缺口。
- **阶段 4 当时未测 checkpoint 保存/加载**：后续阶段 6 及续训 findings 已覆盖保存、
  加载和 round-trip；不要把本条当作当前能力缺失。

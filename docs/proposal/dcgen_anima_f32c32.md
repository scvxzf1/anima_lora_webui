# DC-Gen Anima f32c32 锻造方案（POC 起步）

状态：**进行中（POC 阶段 0/1 已跑通）**。最后更新：2026-08-19。

## 1. 目标

把本仓库的 Anima 基座（`anima-preview3-base`，2.091B，f8/c16/patch2）按 DC-Gen
方法迁移到更高压缩潜空间，产出可分发的新基座（“锻造基座”）。

首个迁移目标（用户已拍板）：

| 项 | 值 |
|---|---|
| 目标 DC-AE | `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` |
| 新潜空间 | f32 / c32 / patch1 |
| 起始分辨率 | 1024² |
| 本地权重 | `models/dc_ae/dc-ae-f32c32-sana-1.1-diffusers/`（diffusers 格式） |

## 2. 为什么是这个目标

Anima 原空间有效像素步长 = 8（VAE）× 2（patch）= 16。新空间 = 32 × 1 = 32。
同分辨率下视觉 token 数变为原来的 1/4：

| 分辨率 | 原 Anima tokens | DC-Gen f32c32 tokens |
|---|---:|---:|
| 1024² | 4096 | 1024 |
| 2048² | 16384 | 4096 |

权重形状契约（实测）：

```text
旧输入层 net.x_embedder.proj.1.weight = [2048, 68]  # (16+mask)*2*2
旧输出层 net.final_layer.linear.weight = [64, 2048]  # 16*2*2
新输入层 = [2048, 33]  # (32+mask)*1*1
新输出层 = [32, 2048]  # 32*1*1
```

DiT 主干 2.091B 参数可原样继承；必须重建的只有输入/输出层。Qwen3-0.6B 文本
编码器不需要迁移。Anima 使用标准 rectified-flow 目标（`target = noise - latents`，
见 `library/training/noise_target.py`），**不需要** FLUX.1-Krea 那套
guidance-distilled 修正目标。

## 3. 锻造三阶段（DC-Gen 方法）

1. **阶段 0 — Patch Embedding Alignment**：冻结旧模型，下采样旧 patch 特征
   到新网格，训练新 `x_embedder` 做 MSE 对齐。
2. **阶段 1 — Output-head Alignment**：冻结 DiT 主干，联合对齐新
   `x_embedder` 与 `final_layer`。
3. **阶段 2 — 端到端 LoRA 后训练**：rank/alpha=256 LoRA，在新潜空间上做
   短程 rectified-flow 训练。

## 4. 已落地

### 4.1 新模块

- `library/models/latent_space.py`：`LatentSpaceSpec`，参数化 VAE 压缩比、
  latent 通道、patch、缓存后缀与归一化方式。预置
  `ANIMA_F8C16_P2` 与 `DCGEN_F32C32_P1`（scaling_factor=0.41407）。
- `library/models/dc_ae.py`：diffusers 格式 DC-AE 加载 + 训练空间 encode
  （`raw_latent * scaling_factor`）。
- `library/io/cache_names.py`：新增 `latent_cache_suffix(space_name)`，原
  `_anima.npz` 命名不变，DC-Gen 空间为 `_dcgen_f32c32.npz`。

### 4.2 向后兼容的参数化

- `library/anima/models.py`：`Anima.__init__` 新增
  `vae_spatial_compression` 参数，默认 8，现有调用行为不变。
- `library/anima/weights.py`：`load_anima_model` 新增
  `in_channels/out_channels/patch_spatial/patch_temporal/vae_spatial_compression`
  可选参数，默认值保持原 Anima checkpoint 字节不变。

### 4.3 探针结果（GPU 0 / 170HX，bf16）

- `scripts/dcgen/probe_dual_latent_cache.py`：**通过**
  - 256²：anima `(1,16,32,32)`，dcgen `(1,32,8,8)`
  - 1024²：anima `(1,16,128,128)`，dcgen `(1,32,32,32)`
  - 两种命名 sidecar 各自落盘、读回无损。
  - 1024² token 比：anima 4096 vs dcgen 1024 = 0.25。
- `scripts/dcgen/probe_patch_align.py`：**通过**
  - 旧 x_embedder `[2048,68]` 读入教师，冻结；新 `[2048,33]` 随机初始化。
  - 旧特征 `(1,1,16,16,2048)` --avgpool2--> 目标 `(1,1,8,8,2048)`。
  - 200 步 Adam(1e-3)：MSE 1.734 → 0.080（-95.4%）。

## 5. 待办

- [ ] 阶段 1：output-head alignment trainer（冻结主干，联合训新输入/输出层）。
- [ ] 阶段 2：rank-256 LoRA 在新潜空间上端到端训练（复用现有 LoRA 路由）。
- [ ] 双 latent 缓存接入正式 preprocess 管线（当前只有探针级缓存）。
- [ ] 锻造基座 checkpoint 保存/加载 + `ss_model_family` / DC-Gen 指纹。
- [ ] 推理侧 DC-Gen loader（新输入/输出层 + DC-AE decode）。
- [ ] 数据池：500–2000 张合成图 + prompt（本机可用 `anima-preview3` 自举）。
- [ ] 迁移前 DC-AE 画质验收（参考 Lulynx 143 图探针：f32 中位 PSNR 33.15
      vs 原 f8 42.03，细节敏感，需按真实素材复验）。

## 6. 硬件与数据

- GPU 0 CMP 170HX 64GB：阶段 0/1/2 主训练卡，2.091B 模型余量充足。
- GPU 1 RTX 3080 10GB：独立做缓存/生成/验证。
- RAM 62GB、数据盘余 176GB：POC 够用；大规模合成池仍偏紧。
- DC-AE encode/decode 实测（170HX，bf16）：1024² encode 1.65s、decode 0.21s。

## 7. 分发形态（锻造完成后）

新基座 = 新 DiT（旧主干 + 新输入/输出层）+ DC-AE + 配置 + normalization +
采样器 + 加载代码。新基座 LoRA 绑定新基座，不能默认挂回原 Anima；分发前需
另行确认原模型与 DC-AE 的许可证。

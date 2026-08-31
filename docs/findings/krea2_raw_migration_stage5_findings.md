# Krea-2-Raw 迁移 阶段 5: 推理串通 (findings)

状态：历史阶段快照 / 阶段 5 已完成
适用版本：阶段 5 推理探针落地时点；不作为当前完整能力说明
入口命令：`.venv/bin/python scripts/krea2/probe_sample.py`
相关代码：`library/models/krea2_raw/sampling.py`、`scripts/krea2/probe_sample.py`

> “已知限制 / 后续”保留阶段 5 当时状态；独立 inference runner、LoRA attach 和
> block swap 后续均已落地。当前边界见
> [`../multi_model_support.md`](../multi_model_support.md)。

## 目标

flow-matching Euler ODE + mu shift + CFG 采样跑通, VAE decode 出图,
记录显存/速度/功耗基线. 验证 Krea-2 推理链路真实可用.

## 设计定论

### Krea-2 官方采样器 (子代理核实 krea-ai/krea-2 sampling.py)

与 anima 的 `library/inference/sampling.py` 不同, **不复用** anima 的
`get_timesteps_sigmas` (anima 用线性 mu shift `(shift*sigmas)/(1+(shift-1)*sigmas)`,
Krea-2 官方用 log-sigmoid shift `exp(mu)/(exp(mu)+(1/ts-1)**sigma)`). 新写
`library/models/krea2_raw/sampling.py`.

关键数值 (移植自官方 sampling.py):
- Euler ODE, 默认 28 步, 确定性无噪声注入
- 更新公式: `img = img + (tprev - tcurr) * v` (tprev < tcurr, ts 倒序反向积分)
- sigma 网格: `linspace(1.0, 0.0, steps+1)` 倒序
- **mu 端点: (x1=256, y1=0.5) / (x2=6400, y2=1.15)** (推理 x2=6400, 非训练 4096)
- mu 公式: `mu = (y2-y1)/(x2-x1) * seq_len + (y1 - slope*x1)`
  **seq_len = 纯图像 token 数** (patchify 后 h_*w_), 不含文本 token
- shift 应用: `ts = exp(mu) / (exp(mu) + (1/ts - 1)**sigma)`, sigma=1.0
- **CFG: `v = cond + guidance*(cond - uncond)`**, uncond=空字符串 `""`, 默认 cfg=4.5
- 初始 latent = `torch.randn(...)`, 无 sigma_max 缩放 (σ=1.0 隐含)
- 官方无 block swap / offload (13B bf16 全驻留)

### anima 推理路径不复用 (子代理核实 generation.py)

`library/inference/generation.py:748-892` denoise loop 硬编码 anima DiT 签名
(`anima(latents_5d, t, embed, padding_mask, h_offset, w_offset)`), 不可复用.
`sampling.py` 纯函数虽 model-agnostic 但 mu shift 公式不同, 也不直接复用.
阶段 5 用自包含探针 (反上帝守则, 不动 generation.py 热点文件), 正式串通是阶段 6.

### 承重接口复用

推理探针的 `dit_forward(latents_5d, text_emb, t)` 直接复用阶段 4 的
`forward_for_loss` (family.py), 它已做 5D↔序列转换. 训练和推理共用同一承重
接口, 保证一致性.

### uncond 路径

uncond 用空字符串走同一套 Qwen3-VL ChatML encode (system prompt 固定, 空 user
prompt pad 到 541). 与 anima 的 T5("")+LLM adapter uncond sidecar 不同构
(anima 走 uncond.py 的 sidecar, Krea-2 直接空 prompt encode).

### VAE decode 输出 [-1,1]

`AutoencoderKLQwenImage.decode_to_pixels` 返回 [-1,1] (anima 约定, 非推理侧的
[0,1]). 存图需 `(px + 1) / 2` 转 [0,1]. 这是 anima VAE 的统一约定, Krea-2 复用.

## 出口验证

### 验证: 28 步采样 (PG199 bf16, 256×256, cfg=4.5)

`scripts/krea2/probe_sample.py`:

```
--- C. 采样 28 步 (cfg=4.5, mu shift 自动) ---
采样耗时: 8.41s (301ms/step), peak 26.38GB
final latent: (1, 16, 1, 32, 32), 有限: True
final latent 范围: [-2.203, 2.500]

--- D. VAE decode -> 存 PNG ---
pixels: (1, 3, 256, 256), 范围 [-1.000, 1.000]
存图: output/tests/krea2_stage5/sample_cfg4.5.png

--- E. CFG=0 对比 (无 CFG, 同 seed) ---
CFG=0 采样耗时: 3.94s, 存图: output/tests/krea2_stage5/sample_cfg0.png
cfg vs no-cfg latent abs mean diff: 0.2637

=== F. 验证 ===
final latent 有限: True
pixels 范围合理 [-1,1]: True
CFG 起作用 (diff>0.01): True

阶段 5 推理串通通过: True
```

验证项全绿:
- 28 步 Euler ODE 采样跑通, latent 有限 ✓
- VAE decode 出像素 (1,3,256,256) 范围 [-1,1] 合理 ✓
- PNG 存到 `output/tests/krea2_stage5/` (受 resolve_output_root 边界约束) ✓
- CFG 起作用 (cfg4.5 vs cfg0 latent diff=0.2637, 明显不同) ✓
- mu shift 自动按 img_seq_len=256 算 (256×256 → latent 32×32 → patch 16×16=256 token) ✓
- 三阶段显存调度 (TE→free→VAE+DiT→sample) lazy loading 不变量保持 ✓

## 基线 (PG199 bf16, 256×256, 28 steps, cfg=4.5)

| 指标 | 值 |
|---|---|
| DiT+VAE 显存 peak | 26.38GB |
| 采样总耗时 (cfg=4.5) | 8.41s (301ms/step) |
| 采样总耗时 (cfg=0) | 3.94s (141ms/step) |
| GPU 功耗 (idle→sample) | 44.8W → 304.9W |
| TE 加载耗时 | 148s (8.4GB safetensors) |
| DiT 加载耗时 | 12.92s |
| 输出图 | `output/tests/krea2_stage5/sample_cfg4.5.png`, `sample_cfg0.png` |

显存 26.38GB (推理 no_grad) 比训练 32.62GB (含 backward 激活) 省 6GB. 256×256
推理在 PG199 32GB 富余; 1024×1024 推理 (latent 128×128, img_seq_len=4096) 激活
大很多, 需 block swap — 阶段 6 大分辨率推理时补.

## 阶段 5 截止时的已知限制 / 后续（历史快照）

- **阶段 5 当时未串通通用 Anima `generation.py`**：`sampling.py` 仅探针验证；
  后续由 `library/models/krea2_raw/inference_runner.py` 和 family runtime 接通独立
  single/euler 路径，而不是把 Krea-2 逻辑硬塞进 Anima generation facade。
- **256×256 出图质量**: base model 训练在 1024×1024, 256×256 是为 fit PG199 32GB
  的探针尺寸, 出图质量不作为阶段 5 验证点. 真实训练结果质量验证在阶段 6
  (训练过的 LoRA + 1024 推理对比).
- **阶段 5 当时未实现 block swap**：`SingleStreamDiT` 尚无
  `enable_block_swap`/`prepare_block_swap`/`_run_blocks`；后续阶段 6 已接入，当前入口
  见 `library/models/krea2_raw/dit.py`。
- **阶段 5 探针未挂 LoRA 推理**：当时只测 base model 采样；当前加载器已支持
  checkpoint attach，但端到端风格对比仍应以专门测试/运行记录为准。
- **tiled 采样未做**：官方无 tiled；当前大分辨率仍依靠 block swap/显存配置，不把
  tile 当作已支持能力。

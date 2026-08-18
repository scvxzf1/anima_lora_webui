# scripts/dcgen — DC-Gen Anima f32c32 工作目录

本目录存放 DC-Gen Anima 锻造（f32c32 + patch1 + 1024²）的探针与后续训练脚本。
方案文档见 `docs/proposal/dcgen_anima_f32c32.md`。

## 进度

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-08-19 | 双 latent 缓存探针（probe 0） | 通过 |
| 2026-08-19 | Patch embedding alignment dry-run（probe 1） | 通过 |
| 2026-08-19 | Output-head alignment dry-run（probe 2，阶段 1） | 通过 |
| 2026-08-19 | LoRA E2E dry-run（probe 3，阶段 2） | 通过 |

## 运行

```bash
# 双 latent 缓存（生成 _out/dual_latent_cache/*.npz）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_dual_latent_cache.py

# patch alignment dry-run（依赖 probe 0 产物）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_patch_align.py

# output-head alignment dry-run（阶段 1，需加载 4GB 旧 DiT）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_output_head_align.py

# LoRA E2E dry-run（阶段 2，需加载 4GB 旧 DiT）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_lora_e2e.py
```

## 目录

- `probe_dual_latent_cache.py`：probe 0，双 latent 形状/命名/读回契约。
- `probe_patch_align.py`：probe 1，新 x_embedder 对齐旧下采样特征（阶段 0）。
- `probe_output_head_align.py`：probe 2，冻结主干联合训输入/输出层（阶段 1）。
- `probe_lora_e2e.py`：probe 3，LoRA 端到端训练链路（阶段 2）。
- `_out/`：探针产物，不提交。

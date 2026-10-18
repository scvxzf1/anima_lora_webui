# cache_identity_hot

模型身份（`library/cache_identity.py`）驱动缓存重建的真机热测试。两个脚本都在真实 GPU 上用磁盘上
已有的真实权重跑真实预处理 CLI，再用产出物做消费端断言；不下载任何模型。

## 载体

| 脚本 | 覆盖 | 真实权重 |
| --- | --- | --- |
| `run_hot.py` | VAE latent 身份（所有模型族共享的 `.npz` 路径） | `models/vae/qwen_image_vae.safetensors` |
| `run_te_hot.py` | Anima 文本缓存身份（`_anima_te.safetensors`） | `<ComfyUI>/models/text_encoders/anima/anima_qwen_3_06b_base.safetensors` |
| `run_train_hot.py` | 真实训练启动：身份一致的缓存复用、身份变化 fail-closed、严格模式、重建恢复 | Anima VAE + Anima Qwen3 TE + `anima-base-v1.0-sha.safetensors` |

## 运行

```bash
.venv/bin/python -m bench.cache_identity_hot.run_hot --gpu 0
.venv/bin/python -m bench.cache_identity_hot.run_te_hot --gpu 0
.venv/bin/python -m bench.cache_identity_hot.run_train_hot --gpu 0 --steps 10
```

`run_train_hot.py` 会自己写 `config.runtime.toml` / `dataset.toml` 到 run 目录，训练用
`blocks_to_swap=0`（避开当前工作树的 int8 block-swap 参数回归）、`attn_mode=torch`（CMP 50HX 是
Turing，flash 不可用）。

两者都会自建 2 张图的一次性数据集与缓存目录，权重放在 `<run_dir>/workspace/`（VAE 副本 253 MB、
TE 副本 1.2 GB），结果写到 `output/bench/cache_identity_hot/<YYYYMMDD-HHMM>/`：
`result.json`（bench 信封）、`stages.json` / `te-stages.json`（逐阶段计数与断言）、
每个阶段一份 CLI 原始日志。

## 阶段与期望

| 阶段 | 期望 | 含义 |
| --- | --- | --- |
| `cold_encode` | written=2 | 冷启动建立缓存 |
| `rerun_same_*` | skipped=2 | 身份一致：复用 |
| `*_path_changed` | written=2 | 配置路径变化：判定失效并重编码 |
| `*_mtime_changed` | written=2 | 同路径就地换权重：mtime 变化即失效 |
| `legacy_stamp_default` | skipped=2 | 旧（无身份）缓存默认复用并告警一次 |
| `legacy_stamp_strict` | written=2 | `--require_cache_identity` 下旧缓存也重编码 |
| `stale_resolution_dropped`（latent） | written=2 | 换 VAE 时同文件其他分辨率 key 被丢弃 |

最后一步是消费端断言：用 `AnimaLatentsCachingStrategy` / `AnimaTextEncoderOutputsCachingStrategy`
以匹配身份接受缓存、以另一身份拒绝它。

## 最近一次结果（2026-10-18，CMP 50HX 10GB，torch 2.12.0+cu130）

- `run_hot.py`：7/7 阶段符合期望，`matching_vae_accepted=true`、`other_vae_rejected=true`。
- `run_te_hot.py`：见同目录最新 `result.json`。

## 注意

- 只使用 `CUDA_VISIBLE_DEVICES` 指定的一张卡；默认 GPU 0。跑之前先确认没有用户训练在占用。
- `run_train_hot.py` 需要真实 Anima DiT/TE/VAE；未传 `--dit` 或 `--te` 时使用本机 ComfyUI 路径。
- 工作区里会保留权重副本；确认结果后可以删除 `<run_dir>/workspace/` 释放空间。
- 这些是身份失效机制的验收，不是训练吞吐或画质结论。

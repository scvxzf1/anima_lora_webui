# 模型身份缓存重建 · CMP 50HX 真机热测试（2026-10-18）

状态：实验（真机验收证据）
适用版本：当前 dev 工作树（`dd0ea5fc` + 未提交的 cache_identity 改动）
入口命令：`.venv/bin/python -m bench.cache_identity_hot.run_hot --gpu 0` / `...run_te_hot --gpu 0`
相关代码：`library/cache_identity.py`、`library/anima/strategy.py`、`library/preprocess/latents.py`、`library/preprocess/text.py`
测试载体：`bench/cache_identity_hot/`

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA CMP 50HX 10240 MiB（`CUDA_VISIBLE_DEVICES=0`；GPU1 GTX 1050 全程未使用） |
| torch / CUDA | 2.12.0+cu130 / CUDA 13.0 |
| 权重（真实、本机已有） | VAE `models/vae/qwen_image_vae.safetensors`；TE `<ComfyUI>/models/text_encoders/anima/anima_qwen_3_06b_base.safetensors` |
| 数据集 | 一次性 2 张图（512×512、640×384），写在 run 目录内，未触碰用户数据 |

两条热测试都走真实预处理 CLI（`scripts.preprocess.cache_latents`、`cache_text_embeddings`），
每次都是一个独立子进程，计数来自 CLI 自身的 summary 行。

## 结果 A：latent / VAE 身份（`run_hot.py`）

run 目录：`output/bench/cache_identity_hot/20261018-1946/`

| 阶段 | 实测 written / skipped | 期望 |
| --- | --- | --- |
| cold_encode | 2 / 0 | 建缓存 |
| rerun_same_vae | 0 / 2 | 身份一致 → 复用 |
| vae_path_changed（复制到另一路径） | 2 / 0 | 路径身份变化 → 重建 |
| vae_mtime_changed（同路径 touch） | 2 / 0 | 就地换权重信号 → 重建 |
| legacy_stamp_default（剥离身份记录） | 0 / 2 | 旧缓存复用 + 告警 |
| legacy_stamp_strict（`--require_cache_identity`） | 2 / 0 | 严格模式重建旧缓存 |
| stale_resolution_dropped（注入旧 VAE 的其他分辨率 key） | 2 / 0 | 失效时丢弃陈旧 key |

- 消费端断言：`matching_vae_accepted=true`、`other_vae_rejected=true`。
- 产出物携带身份：`{"stage":"vae-latents","family":"anima","vae":{...,"mtime_ns":…}}`，注入的
  `latents_99x99` 在换 VAE 后被丢弃。

## 结果 B：Anima 文本缓存身份（`run_te_hot.py`）

run 目录：`output/bench/cache_identity_hot/20261018-1950/`

| 阶段 | 实测 written / skipped | 期望 |
| --- | --- | --- |
| cold_encode | 2 / 0 | 建缓存 |
| rerun_same_te | 0 / 2 | 身份一致 → 复用 |
| te_path_changed（symlink 到同一份权重） | 2 / 0 | 路径身份变化 → 重建 |
| te_mtime_changed（副本 touch） | 2 / 0 | 就地换权重信号 → 重建 |
| legacy_stamp_default | 0 / 2 | 旧缓存复用 + 告警 |
| legacy_stamp_strict | 2 / 0 | 严格模式重建旧缓存 |

- 消费端断言：`matching_te_accepted=true`、`other_te_rejected=true`。
- 该轮未加载 LLM Adapter（不传 `--dit`），因此身份记录里 DiT 为 `<unused>`，与
  “DiT 只在缓存含 `crossattn_emb` 时才被断言” 的实现一致。

## 结果 C：真实训练启动（`run_train_hot.py`）

run 目录：`output/bench/cache_identity_hot/20261018-2131/`

同一份 2 图 / 512×512 数据，先用真实 CLI 建好带身份的 latent + TE 缓存，再用真实
`train.py`（`--config_file` + `--dataset_config`，`blocks_to_swap=0`，`attn_mode=torch`，
`require_cache_identity=true`，`cache_llm_adapter_outputs=true`）反复启动：

| 阶段 | 实测 | 说明 |
| --- | --- | --- |
| preprocess_latents / preprocess_te | written=2 / written=2 | 建缓存并写身份（TE 侧含 `--dit`，故身份记录带 DiT 路径） |
| train_clean | exit 0，保存 LoRA | 10 步完成：0.81~0.88 s/步，`avr_loss` 0.00025→0.0040，max allocated 4.44 GiB / reserved 4.52 GiB |
| train_te_changed | exit 1 + `Text-encoder cache is incomplete` | 只改 TE 副本 mtime（同路径就地换权重） |
| train_vae_changed | exit 1 + `Latent cache is incomplete` | 只改 VAE 副本 mtime |
| train_strict_legacy_te | exit 1 + TE 报错 | 剥掉 TE 身份戳（严格模式） |
| train_strict_legacy_latent | exit 1 + latent 报错 | 剥掉 latent 身份戳（严格模式） |
| train_recovered | exit 0，保存 LoRA | 重跑预处理重建后恢复正常训练 |

干净启动的日志同时给出复用证据：`Skipping text encoder load`、`Skipping VAE load: all latents`、
`text_encoder is not needed for training` —— 身份一致的缓存被直接复用，懒加载顺序未被破坏。
两次失败都在模型加载之前 fail fast，不会静默用旧张量训练。

## 本轮暴露的两个无关阻塞（已定位，未在本分支修复）

1. `FlashAttention only supports Ampere GPUs or newer`：CMP 50HX 是 Turing(sm_75)，而
   `networks/attention_dispatch.py::flash_attn_available_for_dtype` 只按 dtype 判断可用性，
   于是 LLM Adapter 的注意力（`library/anima/models.py`）在 pre-Ampere 上选了 flash 并在 kernel 处崩溃。
   本轮已加设备能力守卫 `flash_attn_device_supported()`（Ampere+，或 V100 provider 下的 sm_70+）并补测试；
   Anima DiT 自身仍按用户显式 `attn_mode` 选择后端，本机需用 `torch`。
2. `Anima.enable_block_swap() got an unexpected keyword argument 'int8_restore_mode'`：
   当前工作树未提交改动给 4 个 DiT loader（anima/krea2_raw/z_image/qwen_image_2_1）都传了 int8 系列参数，
   但 `library/anima/models.py` 与 `library/models/krea2_raw/dit.py` 的 `enable_block_swap` 尚未接受。
   因此 **`blocks_to_swap>0` 的 Anima / Krea-2 训练在当前工作树会直接 TypeError**；本热测试用
   `blocks_to_swap=0` 绕开该路径。

## 覆盖与未覆盖

已覆盖：VAE 身份（族无关，覆盖全部 5 个模型族的 latent 路径）、Anima TE 身份（含 DiT/LLM Adapter 分支）
和真实 `train.py` 启动路径上的 fail-closed 拒绝与恢复。

未覆盖：

- **Krea-2 / Z-Image / Qwen-Image 2.1 / MOSS 的 TE 缓存尚未接入身份**（当前只有 latent 身份是全族覆盖）。
- 热测试只用 `blocks_to_swap=0`：Anima 块交换路径被上述第 2 条阻塞，未做身份 × 块交换的组合验收。
- vocab pack 身份在生产端每文件重算（内容 sha256）；配置 pack 时预处理大集会有可观开销，需后续优化。
- 身份基于 size+mtime（vocab pack 例外，走内容 sha256）；同尺寸同 mtime 的就地替换权重仍无法识别。
- 训练吞吐数字来自 2 图 / 512×512 / 10 步的单次短跑，不能当作吞吐基准。

## 复现

```bash
.venv/bin/python -m bench.cache_identity_hot.run_hot --gpu 0
.venv/bin/python -m bench.cache_identity_hot.run_te_hot --gpu 0
.venv/bin/python -m bench.cache_identity_hot.run_train_hot --gpu 0 --steps 10
```

run 目录里的 `workspace/*.safetensors` 是本轮临时权重副本，确认结果后已删除；日志、`result.json`、
`stages.json` 保留作为证据。

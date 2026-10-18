# MOSS-Image 分辨率 / 图像 token 预算档位

状态：实验。配置与静态契约已验收；真机热测试的**速度与显存部分已于 2026-10-18 在
CMP 50HX 执行**（7 行，10 步/行，见第 8 节）。质量对照（采样与人工判读）仍未做。
本文档同时是热测试的协议与结果落点。

- 适用版本：当前 dev 工作树（ai-toolkit 侧设计依据为 MOSS_IMAGE_TRAINING_DESIGN.md，2026-10-17）
- 入口配置：`configs/methods/moss_image_lora{,_t768,_t512,_tecap1024}.toml`
- 相关代码：`library/datasets/buckets.py`、`library/preprocess/bucket_geometry.py`、
  `library/preprocess/images.py`、`library/training/bootstrap.py`
- 契约测试：`tests/test_moss_image_token_tiers.py`
- 前置基线：[CMP 50HX 十步基准](moss_image_bf16_baseline10_50hx_20261017.md)
  （256×256 / 22 token / 11.34 秒每步 / 5.85 GiB）

## 1. 结论

1. 仓库默认的 `moss_image_lora.toml` 不是"672–1536 分辨率区间"，而是
   **24 个常量 token bucket、2 个 token family（4032 / 4200）**，面积恒定在 1.03–1.08 MPx。
2. 该档位下 `min_bucket_reso` / `max_bucket_reso` / `bucket_reso_steps` 是**空转参数**：
   `make_constant_token_bucket_resolutions()` 在 `max_reso == (1024, 1024)` 时提前返回，
   三个键不参与过滤。
3. 真正的显存/算力变量不是面积，而是**文本长度**：MOSS 用 per-batch 最长 caption 对齐 + mask，
   图像 token 恒定，联合序列长度 = 4032~4200 + 本 batch 最长 caption。
4. 因此对照矩阵必须是二维（面积档 × 文本长度）；单改面积得到的结论会随 caption 长度反转。

## 2. 事实核对（静态，可复现）

```python
from library.datasets.buckets import BucketManager
m = BucketManager(max_reso=(1024,1024), min_size=672, max_size=1536, reso_steps=64)
m.make_buckets(constant_token_buckets=True)
len(m.predefined_resos)                                   # 24
sorted({(w//16)*(h//16) for w, h in m.predefined_resos})   # [4032, 4200]
```

MOSS 有效 stride 为 16 px（FLUX.2 VAE /8 后再 2×2 packing，
`library/models/moss_image/latent.py` 强制 H,W % 16），所以图像 token 数 = 面积 / 256，
与常量 token 表的 4032 / 4200 完全对齐。

### 2.1 只改 resolution 会静默塌缩

`make_constant_token_bucket_resolutions()` 只在 `max_reso != (1024,1024)` 时应用
min/max 过滤。沿用基配置的 `min=672 / max=1536` 而只改 `resolution`：

| 只改 resolution | 实际 bucket 数 | 说明 |
|---|---:|---|
| 768 | 8 | aspect 覆盖被砍到 0.75–1.0 附近 |
| 512 | 1 | 只剩 512×512，平面 |

这不是等比缩放，是**阶梯破坏**。档位配置必须把 min/max 一起等比缩放。

## 3. 档位定义

| 档位 | 配置文件 | resolution | min/max | bucket 数 | 图像 token | aspect 覆盖 |
|---|---|---:|---|---:|---|---|
| 1024（默认） | moss_image_lora.toml | 1024 | 672 / 1536 | 24 | 4032–4200 | 0.25–3.94 |
| 768 | moss_image_lora_t768.toml | 768 | 504 / 1152 | 14 | 2256–2394 | 0.44–2.25 |
| 512 | moss_image_lora_t512.toml | 512 | 336 / 768 | 13 | 1008–1064 | 0.44–2.29 |
| 文本上限 1024 | moss_image_lora_tecap1024.toml | 1024 | 672 / 1536 | 24 | 4032–4200 | 0.25–3.94 |

生效路径：档位写在方法配置的 `[[datasets]]` 里，由
`library/config/io.py::load_dataset_config_from_base(method=...)` 从 base.toml 的
dataset blueprint 出发做浅覆盖（`library/training/bootstrap.py:437` 调用）。
也就是说，**不带 `--dataset_config` 的运行**（WebUI 方法选择、`tasks.py train --method ...`）
会拿到档位；一旦显式传了 `--dataset_config`，那个文件赢，方法里的 `[[datasets]]` 被忽略。
这一点已由 `test_tier_dataset_blueprint_resolves_via_method_override` 固定。

三个变体都自包含，与基配置的差异被白名单测试锁死：

- t768 / t512：仅 `output_name`、`resolution`、`min_bucket_reso`、`max_bucket_reso`
- tecap1024：仅 `output_name`、`max_token_length`

其余键（swap20、bf16、torch attention、rank/alpha 32、lr 5e-5、AdamW 等）与基配置完全一致，
热测试才能把差异归因到面积或文本长度。

## 4. 数据设计契约（重要）

`bucket_no_upscale=true` 只在"仍有预设 bucket 能装下源图"时生效；一旦没有候选，会回退到
`min_size` 兜底，**反而放大**小图。实测（1024 档）：

| 源图 | 训练 bucket | 图像 token | 备注 |
|---|---|---:|---|
| 512×512 | 672×672 | 1764 | 被 min_size 放大，违反 no_upscale 直感 |
| 640×960 | 672×960 | 2520 | 同样被放大 |
| 672×1400 | 672×1344 | 3528 | 不在 24 张表内，新增离群尺寸 |
| 900×1600 | 768×1344 | 4032 | 在带内 |
| 1200×1600 | 896×1200 | 4200 | 在带内 |
| 2048×2048 | 1008×1024 | 4032 | 在带内 |

结论：**1024 档要求源图短边 ≥ 768 px** 才会落在 4032/4200 带内；短边更小的图会带来
1764–3528 的额外 token family，使"面积恒定"前提失效，显存与耗时出现隐藏波动。
768 / 512 档同理按比例（约 576 / 384 px）设置数据下限。

## 5. 成本模型（attention ∝ seq²）

以每个档位的 band 上限（4200 / 2394 / 1064）计算，基准 = 1024 档 + 4096 文本（seq 8296）：

| caption tokens | 1024 档 seq | 768 档 | 512 档 |
|---:|---:|---|---|
| 128 | 4328 | 0.34× | 0.076× |
| 512 | 4712 | 0.380× | 0.112× |
| 1024 | 5224 | 0.428× | 0.160× |
| 4096 | 8296 | 0.612× | 0.387× |

caption 越长，降面积收益越小（0.34× → 0.61×）。在 1024 档不动分辨率、只压文本上限：

| max_token_length | seq | 相对 4096 的 attention |
|---:|---:|---:|
| 4096（默认） | 8296 | 1.00× |
| 2048 | 6248 | 0.57× |
| 1024 | 5224 | 0.40× |
| 512 | 4712 | 0.32× |

所以最短路径的实验顺序是：先 text cap 1024（只重建 TE 缓存），再比面积档。
上述比值由 `tests/test_moss_image_token_tiers.py` 锁定，改表必须同步本文档。

## 6. 静态验收（已执行）

```bash
.venv/bin/python -m pytest tests/test_moss_image_token_tiers.py -q    # 31 passed
.venv/bin/python -m pytest tests/test_smoke.py -q                    # 20 passed
.venv/bin/python -m pytest tests/test_config.py -v -k moss_image_lora  # 4 passed
.venv/bin/python -m pytest tests/test_config.py -q \
  -k "method_configs_clean or moss or bucket or resolution"           # 20 passed
```

覆盖的契约：

- 基档解析为 24 个 canonical bucket，token family 恰为 {4032, 4200}
- 基档 min/max/steps 空转（实现若改动会立即失败，逼同步注释与本文档）
- 只改 resolution 的塌缩（768→8，512→1）
- t768 / t512 的 bucket 数、token 带、aspect 覆盖
- 所有档位边长 %16 == 0（MOSS latent 硬要求）
- 变体差异白名单（防漂移）
- tecap1024 图像 bucket 与基档一致
- 档位键可穿过 ConfigSanitizer 到达 DatasetParams
- 档位经真实的 load_dataset_config_from_base(method=...) 覆盖路径生效
- 1200×1600 源图落在各档带内
- 短边 <768 源图脱离 token 带（已知限制，故意钉住）
- 本文档第 5 节的 attention 比值

## 7. 热测试协议（已执行）

固定条件（与 256×256 基线一致）：CMP 50HX、BF16 参数 + autocast、plain LoRA rank/alpha 32、
swap20、torch attention、full gradient checkpointing、AdamW lr 5e-5、seed 42、batch 1、
accumulation 1、关闭 EMA / 预览 / 中途保存、10 次有效更新（首步单列，后 9 步统计）。

矩阵（7 个 run）：

| # | 配置 | 数据档 | caption |
|---:|---|---|---|
| 1 | moss_image_lora | 1024 档（896×1200 / 4200 图像 token） | 短（35 token） |
| 2 | moss_image_lora | 1024 档 | 长（4096 token，实测触顶） |
| 3 | moss_image_lora_t768 | 768 档（672×896 / 2352） | 短 |
| 4 | moss_image_lora_t768 | 768 档 | 长 |
| 5 | moss_image_lora_t512 | 512 档（448×608 / 1064） | 短 |
| 6 | moss_image_lora_t512 | 512 档 | 长 |
| 7 | moss_image_lora_tecap1024 | 1024 档 | 长（截断到 1024） |

执行入口（本机维护用法）：

```bash
python tasks.py moss-tier-hot -- plan                       # 只打印矩阵与每行解析到的 bucket
python tasks.py moss-tier-hot -- all --steps 10             # 建缓存 + 训练 + 汇总
python tasks.py moss-tier-hot -- report --root output/bench/moss_tier_hot/matrix-<stamp>
```

实现见 `bench/moss_tier_hot/`（`matrix.py` 定义矩阵与数据/配置生成，`runner.py` 负责
resize+缓存与训练、`collect.py` 采集 nvidia-smi 与 progress 事件，`report.py` 汇总）。
每行产物落在 `output/bench/moss_tier_hot/matrix-<stamp>/<row>/`：`dataset.toml`、
`config.runtime.toml`、`cache.log`、`train.log`、`progress.jsonl`、`memory_probe.jsonl`、
`gpu.csv`、`result.json`。计时取自 `progress.jsonl` 的 `ev=step` 时间戳差值（首步单列），
显存同时记录 PyTorch allocator 峰值与 nvidia-smi 整卡峰值；功耗/温度为 1 秒采样的观测峰值。

执行中确认的两条协议前提：

1. **resize 必须按档位钉住**。`preprocess-config` 只把尾随参数转发给 resize 阶段，档位
   （`--resolution/--min_bucket_reso/--max_bucket_reso/--constant_token_buckets`）必须显式传入；
   否则所有档位都会按 base 默认 1024 生成 896×1200 的 latent，档位实验失效。
2. **文本上限属于缓存身份**。MOSS 文本缓存 schema 已升到 `moss_image_te_v2`，新增
   `max_text_tokens`：4096 档写入的缓存不能被 1024 档复用（反之亦然）。旧 `v1` 缓存仍可读，
   但无法证明截断上限；按档位比较前需重建。

结果落点：`output/bench/moss_tier_hot/matrix-20261018b/`（本地产物，不提交）。

## 8. 热测试结果

2026-10-18 已在 CMP 50HX 按第 7 节矩阵执行 7 行 × 10 步；**完整条件、总表、逐步耗时、OOM
证据、与 256×256 基线的可比性说明，以及被作废首轮矩阵的根因，全部记入独立实测报告：**

- [MOSS-Image 档位热测试：CMP 50HX 七行实测](moss_image_tier_hot_50hx_20261018.md)

摘要（数字以该报告为准，本节不再重复维护）：

| 行 | 图像 token | 文本 cap | 后 9 步均值 s | 退出 |
| --- | ---: | ---: | ---: | --- |
| 1024-short | 4200 | 4096 | 5.666 | 0 |
| 1024-long | 4200 | 4096 | – | 1（首步 OOM） |
| tecap1024-long | 4200 | 1024 | 7.234 | 0 |
| 768-short | 2352 | 4096 | 3.439 | 0 |
| 768-long | 2352 | 4096 | – | 1（第 3 步 OOM） |
| 512-short | 1064 | 4096 | 3.036 | 0 |
| 512-long | 1064 | 4096 | 7.210 | 0 |
| 1024-cap2048-long | 4200 | 2048 | 9.365 | 0 |
| 1024-cap3072-long | 4200 | 3072 | – | 1（首步 OOM） |
| 1024-swap28-long | 4200 | 4096（swap28） | 14.120 | 0 |
| 1024-short-int8 | 4200 | 4096（int8+fused） | 5.822 | 0 |
| 1024-cap2048-long-int8 | 4200 | 2048（int8+fused） | – | 1（第 2 步 OOM） |

两条与协议直接相关的结论：面积档位收益次线性（1024→768 1.65×、1024→512 1.87×，固定项约
2.1 s/step 占主导）；文本长度是显存主变量，4096-token 字幕在本卡只装得下 512 档，`tecap1024`
是 1024 档的可用性开关。2026-10-18 追加的三条 1024 档短测进一步定位边界：cap2048 可跑但余量
不足 8%（9.365 s/step、9777 MiB、61 °C），cap3072 首步 OOM，cap4096 在 swap28 下可跑完 10 步但
慢到 14.120 s/step（用 PCIe 换显存，不是提速）。int8 传输 + fused 还原的配对测试显示，在 1024 档
它既不提速（5.822 vs bf16 5.688 s）也不省显存（8927 vs 8972 MiB），长字幕配对仍在第 2 步 OOM；
与 256×256 下 1.222× 的结论相反，详见 [int8 诊断](moss_image_int8_blockswap_diagnosis_20261018.md)。
**质量对照仍未做。**

## 9. 未做与边界

- 默认 `moss_image_lora.toml` 的行为未改变，只补了空转参数注释；默认值不动。
- 分辨率 / token budget **没有**写进 `configs/presets.toml`：按仓库分层，presets 只放
  硬件/采样 profile，档位属于方法变体。
- `make_constant_token_bucket_resolutions()` 的 1024 早退逻辑未改。若要让它统一遵守
  min/max（语义修正），会改变既有 24-bucket 解析结果，需独立验收后再动。
- WebUI 暂未增加档位下拉入口；需要按档位建队列时另行排期。
- 本文档只覆盖 T2I plain LoRA；edit / reference 条件路径不在范围内。

# MOSS-Image 档位热测试：CMP 50HX 七行实测（2026-10-18）

状态：实测报告（速度与显存）。质量对照未做。
适用版本：当前 dev 工作树；执行于 2026-10-18 17:1x–17:4x（Asia/Shanghai）。
入口命令：`python tasks.py moss-tier-hot -- all --steps 10`（`-- plan` / `-- report` 同源）。
相关代码：`bench/moss_tier_hot/`、`configs/methods/moss_image_lora{,_t768,_t512,_tecap1024}.toml`、
`library/models/moss_image/`、`scripts/moss_image/preprocess_te_cache.py`。
协议与成本模型：[档位协议文档](moss_image_token_budget_tiers_20261017.md) 第 5、7 节。
前置基线：[CMP 50HX 十步基准](moss_image_bf16_baseline10_50hx_20261017.md)（256×256）。

本文档是**本次实测的落点**：有效矩阵（第 2–4 节）与被作废的首轮矩阵（第 5 节）都在这里，
避免作废数据被后人当成结论复用。

## 0. 实验条件

| 项目 | 本轮值 |
| --- | --- |
| GPU | NVIDIA CMP 50HX，10 GB 标称、9.56 GiB 可用，按 UUID 选择设备 |
| 精度 | DiT BF16 参数 + BF16 autocast；LoRA 参数 FP32；adaptive precision 关闭 |
| Adapter | plain LoRA rank/alpha 32，attention-only（`network_train_unet_only`） |
| 训练 | full gradient checkpointing，`blocks_to_swap=20`，BF16 transfer，`foreach` restore，`torch` attention，compile 关闭 |
| 优化器 | AdamW `fused=True`，lr 5e-5，constant scheduler，`max_grad_norm=1` |
| 数据 | 每行 1 张确定性合成图 + 1 条 caption，batch 1，accumulation 1，`num_repeats=1` |
| 步预算 | `max_train_steps=10`（显式删除 `max_train_epochs`，它优先级更高） |
| 保存/预览 | 中途保存关闭（`checkpointing_epochs=0`），采样与验证关闭；仅末尾保存一次 |
| 随机性 | seed 42，从全新 LoRA 初始化开始 |
| 显存/功耗采样 | 进程内 `memory_probe`（PyTorch allocator）+ 外部 `nvidia-smi` 1 秒采样整卡 |

图像与文本档位（由 resize 阶段按档位钉住，latent key 为 `H/16 x W/16`）：

| 行 | 方法配置 | 源图 | resize 后 | 图像 token | 文本 cap | caption 实测长度 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1024-short | `moss_image_lora` | 1200×1600 | 896×1200 | 4200 | 4096 | 35 |
| 1024-long | `moss_image_lora` | 1200×1600 | 896×1200 | 4200 | 4096 | 4096（触顶） |
| tecap1024-long | `moss_image_lora_tecap1024` | 1200×1600 | 896×1200 | 4200 | 1024 | 1024 |
| 768-short | `moss_image_lora_t768` | 900×1200 | 672×896 | 2352 | 4096 | 35 |
| 768-long | `moss_image_lora_t768` | 900×1200 | 672×896 | 2352 | 4096 | 4096 |
| 512-short | `moss_image_lora_t512` | 600×800 | 448×608 | 1064 | 4096 | 35 |
| 512-long | `moss_image_lora_t512` | 600×800 | 448×608 | 1064 | 4096 | 4096 |
| 1024-cap2048-long | `moss_image_lora` | 1200×1600 | 896×1200 | 4200 | 2048 | 2048 |
| 1024-cap3072-long | `moss_image_lora` | 1200×1600 | 896×1200 | 4200 | 3072 | 3072 |
| 1024-swap28-long | `moss_image_lora` | 1200×1600 | 896×1200 | 4200 | 4096 | 4096 |

后三行是 2026-10-18 的补充 1024 档短测（`matrix-20261018c/`）：图像侧与最重档完全相同
（4200 token、896×1200、swap20），只分别改文本上限或 `blocks_to_swap`，用来给
“cap1024 可跑 / cap4096 OOM”这条边界定位中间点。

## 1. 计时与显存口径

- 逐步耗时取 `progress.jsonl` 中 `ev=step` 行的 `ts` 差值（`ts` 为 run 起点起的秒数），
  首步单列，后 9 步统计。区间包含该 step 的 forward/backward/optimizer 与数据取用。
- PyTorch 峰值取每步 `cuda/max_memory_allocated_gb` / `cuda/max_memory_reserved_gb` 的最大值。
- 整卡峰值、温度、功耗、SM 时钟来自训练期间 1 秒间隔的 `nvidia-smi` 采样（`gpu.csv`）。
- **与 256×256 基线不可直接相减**：基线的计时在 `library.training.loop._run_step` 两端做 CUDA
  synchronize 且按 epoch 保存 checkpoint；本轮是 progress 时间戳差值并关闭了中途保存。
  轮内各档之间同法可比，跨报告对比需要按本协议重量一次 256×256 行。

## 2. 结果总表

| 行 | 图像 token | 文本 cap | 首步 s | 后 9 步均值 s | 中位 s | 后 9 步 min–max | PyTorch 峰值 alloc/reserved GiB | 整卡峰值 MiB | 温度 °C | 功耗峰值 W | SM MHz | 退出 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1024-short | 4200 | 4096 | 7.586 | **5.666** | 5.670 | 5.640–5.689 | 7.90 / 8.06 | 8619 | 46–58 | 240.93 | 1545–1875 | 0 |
| 1024-long | 4200 | 4096 | – | – | – | – | – | 9675 | 47–54 | 223.35 | 1590–1860 | 1（首步 OOM） |
| tecap1024-long | 4200 | 1024 | 9.089 | **7.234** | 7.242 | 7.186–7.281 | 8.44 / 8.63 | 9195 | 45–59 | 227.59 | 1530–1875 | 0 |
| 768-short | 2352 | 4096 | 5.409 | **3.439** | 3.438 | 3.434–3.446 | 6.91 / 7.06 | 7595 | 47–56 | 226.69 | 1605–1860 | 0 |
| 768-long | 2352 | 4096 | 11.682 | （2 步后 OOM） | – | – | 9.11 / 9.21 | 9787 | 47–56 | 239.01 | 1560–1860 | 1（第 3 步 OOM） |
| 512-short | 1064 | 4096 | 4.967 | **3.036** | 3.035 | 3.033–3.040 | 6.22 / 6.36 | 6877 | 47–53 | 241.25 | 1665–1860 | 0 |
| 512-long | 1064 | 4096 | 9.050 | **7.210** | 7.215 | 7.159–7.250 | 8.42 / 8.61 | 9173 | 45–59 | 230.58 | 1515–1875 | 0 |
| 1024-cap2048-long | 4200 | 2048 | 11.261 | **9.365** | 9.365 | 9.292–9.441 | 8.99 / 9.20 | 9777 | 47–61 | 236.46 | 1485–1860 | 0 |
| 1024-cap3072-long | 4200 | 3072 | – | – | – | – | – | 9695 | 50–55 | 221.84 | 1620–1860 | 1（首步 OOM） |
| 1024-swap28-long | 4200 | 4096 | 15.940 | **14.120** | 14.138 | 14.013–14.176 | 8.41 / 8.62 | 9187 | 47–62 | 241.25 | 1500–1860 | 0 |
| 1024-short-int8 | 4200 | 4096（实测 35） | 7.857 | **5.822** | 5.828 | 5.785–5.855 | 7.99 / 8.38 | 8927 | 47–61 | 225.36 | 1530–1875 | 0 |
| 1024-cap2048-long-int8 | 4200 | 2048 | 11.530 | – | – | – | 8.84 / 9.22 | 9788 | 46–56 | 238.10 | 1530–1875 | 1（第 2 步 OOM） |

后两行是 int8 传输 + fused 还原的配对行（`matrix-20261018e/`），除 `block_swap_transfer_dtype`
与 `block_swap_int8_restore_mode` 外与被配对行完全一致；详细对比见
[int8 块交换诊断](moss_image_int8_blockswap_diagnosis_20261018.md) 的「1024 档长序列复测」节。

后 9 步逐步耗时（秒）：

```text
1024-short     5.640 5.648 5.651 5.665 5.670 5.675 5.679 5.679 5.689
768-short      3.434 3.439 3.435 3.439 3.438 3.438 3.437 3.444 3.446
512-short      3.034 3.038 3.035 3.034 3.033 3.035 3.040 3.036 3.038
512-long       7.159 7.170 7.196 7.206 7.215 7.225 7.228 7.238 7.250
tecap1024-long 7.186 7.198 7.210 7.231 7.242 7.245 7.250 7.259 7.281
1024-cap2048  9.292 9.322 9.346 9.364 9.365 9.372 9.389 9.390 9.441
1024-swap28  14.013 14.061 14.104 14.138 14.161 14.165 14.176 14.171 14.092
```

单行缓存构建（resize + VAE + NF4 TE）耗时 43–54 秒，不计入上表。

## 3. 结论

1. **面积档位收益次线性**：1024→768 为 1.65×、1024→512 为 1.87×，而图像 token 少 1.8×/3.9×。
   用短字幕三行拟合约 `t ≈ 2.1 s + 0.00084 s/token`：固定项（swap20 的 block 搬运与框架开销）
   在本卡本链路上占主导，**只降分辨率换不到等比例时间**。
2. **文本长度是显存与时间的主变量**：512 档 4096-token 字幕把 3.036 s 拉到 7.210 s（+137%），
   整卡峰值 6877→9173 MiB；1024 档同字幕首步即 OOM。
3. **`tecap1024` 是 1024 档的可用性开关**：同为 4200 图像 token，4096 字幕 OOM，截断到 1024
   后可跑完 10 步（7.234 s/step，峰值 9195 MiB）。这与协议第 5 节“先压文本上限、再比面积”的
   建议一致。
4. **本卡 9.56 GiB + swap20 的显存天花板**：4096-token 字幕只有 512 档（1064 图像 token）
   装得下；768 档（2352）在第 3 步 OOM，1024 档（4200）首步 OOM。
5. **1024 档的文本天花板在 cap 2048 与 cap 3072 之间**：4200 图像 token + swap20 下，
   cap1024 = 7.234 s/step（9195 MiB）、cap2048 = 9.365 s/step（9777 MiB，温度升到 61 °C、
   SM 降到 1485 MHz）、cap3072 首步即 OOM、cap4096 首步 OOM。也就是说在同一图像面积上，
   文本侧能装下的上限是 2048 token，1024 仍有约 600 MiB 余量。
6. **提高 swap 预算能换来容量，但代价很大**：cap4096 在 swap28 下可以跑完 10 步
   （14.120 s/step、9187 MiB、alloc 8.41 GiB），比 cap1024/swap20 慢 95%、比 cap2048/swap20
   慢 51%。它把可用显存从 9777 MiB 压回 9187 MiB，说明这条路线是“用 PCIe 换显存”，不是提速。
7. **int8 传输 + fused 还原在 1024 档没有收益**：短字幕配对 5.822 s vs bf16 5.688 s（慢 2.4%，
   整卡 8927 vs 8972 MiB），长字幕配对第 2 步 OOM。这与 256×256 下 1.222× 的结论相反，原因是
   大 seq 时传输不再是瓶颈。详见 int8 诊断文档的 1024 档复测节。
8. **10 步窗口内没有热降频证据**：后 9 步极差 ≤ 0.049 s，温度 45–62 °C，SM 1485–1875 MHz，
   未见随步数单调变慢；功耗观测峰值 225–241 W（标称上限 225 W，1 秒采样值）。这不否定更长
   训练窗口的热降频结论，只是本轮 10 步太短、负载也未持续打满（util 均值 52–82%）。
   补充行里 cap2048（温度 61 °C）与 swap28（62 °C）已经比基础档更接近功耗/温度上限，
   仍未见逐步变慢，因此“短窗口内不降频”的结论对更高负载档同样成立。

## 4. 复现与产物

```bash
python tasks.py moss-tier-hot -- plan                       # 打印矩阵与每行解析到的 bucket
python tasks.py moss-tier-hot -- all --steps 10             # 建缓存 + 训练 + 汇总
python tasks.py moss-tier-hot -- report --root output/bench/moss_tier_hot/matrix-<stamp>
```

本轮有效矩阵落点（本地产物，不提交）：`output/bench/moss_tier_hot/matrix-20261018b/`，
每行含 `dataset.toml`、`config.runtime.toml`、`cache.log`、`train.log`、`progress.jsonl`、
`memory_probe.jsonl`、`gpu.csv`、`result.json`；根目录另有 `summary.json` / `summary.md`。

补充的 1024 档三行落在 `output/bench/moss_tier_hot/matrix-20261018c/`，目录结构与上表一致：

```bash
python tasks.py moss-tier-hot -- all \
  --rows 1024-cap2048-long,1024-cap3072-long,1024-swap28-long --steps 10
```

## 5. 被作废的首轮矩阵（尸体记录，勿复用）

首轮 `output/bench/moss_tier_hot/matrix-20261018/` 的短字幕数据**不可作为档位结论**，原因是
预处理没有按档位钉住 resize：七行全部生成了 `896×1200` 的 latent（npz key `latents_75x56`），
即每一行都跑了 1024 档的 4200 图像 token；因此 1024/768/512 三行给出几乎相同的
5.66/5.67/5.68 s 与完全相同的 7.8996 GiB / 8619 MiB 峰值——这个“三档同数”本身就是失效信号。

根因与证据：

- `preprocess-config` 只把尾随参数转发给 resize 阶段，而 resize 的调用行没有 `--resolution`：
  `resize_images.py --src … --dst … --no_copy_captions --min_pixels 0 --bucket_reso_steps 64 --recursive`；
  未传档位时 resize 走 base 默认 1024，于是所有档位都得到 896×1200。
- 训练侧本身是按档位解析的（512 行日志可见 `resolution: 512 / min 336 / max 768`），
  所以失效只发生在缓存侧，且表现为“缓存 key 与档位不匹配”。
- 有效矩阵已在 `runner.build_cache` 中显式传 `--resolution/--min_bucket_reso/--max_bucket_reso/`
  `--constant_token_buckets`，并记录 `resized_image` 与 `latent_keys` 作为 provenance；
  第 0 节的 resize 列即来自该字段。

同一轮调试中还作废过两个中间尝试，一并记录：

1. 首次 `all` 运行把 `--overwrite` 透传给了 resize，`resize_images.py` 不认该参数（exit 2）。
   有效版本改为由 harness 自行清理本行 `data/` 与 `cache/`，不再透传该 flag。
2. 曾用 `max_train_epochs = 0` 想“让位”给 `max_train_steps`，实际得到 “steps for 0 epochs is 0”，
   训练 0 步结束。正确做法是从 runtime 配置里**删除** `max_train_epochs`（它优先级高于 step 预算）。

### 5.1 失败与作废判定清单

| 尝试/现象 | 判定 | 不可复用的原因 | 后续防线 |
| --- | --- | --- | --- |
| resize 收到不支持的 `--overwrite` 参数 | 启动失败 | 预处理 exit 2，未进入训练 | 按阶段核对 CLI；不能把缓存阶段参数无差别透传给 resize |
| 保留方法配置的 `max_train_epochs=3` | 步预算作废 | 单图仅运行 3 步，不是协议要求的 10 步 | 从生成的 runtime 配置删除 epoch 上限，并核对实际 step 事件数 |
| 使用 `max_train_epochs=0` | 零步作废 | 0 被解释为零个 epoch，而不是未设置；进程 exit 0 不等于完成训练 | 同时检查退出码、run_start 总步数与实际完成步数 |
| 768/512 行缓存仍为 896×1200 | 面积对照作废 | 名义配置不同，但实际缓存 workload 未分档 | 在训练前核对 resize 尺寸、latent key、图像 token 和 TE 实际长度 |
| 有效矩阵的 1024-long 首步 OOM | 有效失败证据 | 未完成更新，不能给出稳态步时 | 保留 traceback、配置和遥测；只报告该配置的容量边界 |
| 有效矩阵的 768-long 完成 2 步后 OOM | 有效失败证据 | 第 2 步耗时不是“后 9 步均值”，也不能视为通过 | 保留部分步耗时但明确标为未完成，不与 10 步成功行混算 |
| `TierRow` 新增字段后旧行仍按位置传 `note` | 配置作废 | `note` 落进 `blocks_to_swap`，schema 报 “expects int, got ‘heaviest tier…’”，该轮 `1024-short` 未起训 | 新增 dataclass 字段一律关键字传参；训练前看 `result.json` 的 `train_exit_code`，不只看目录是否存在 |
| `scripts/tasks/preprocess.py` 的 `_reuse_overwrite_args` 引用未定义 `policy` | 环境缺陷（已修） | `NameError` 打断所有 MOSS 预处理缓存，任何档位都无法建缓存 | 已补 `policy = parse_cache_reuse_policy(cfg)`；预处理失败先看 `cache.log` 末行 |

以上“作废”只针对对应实验结论，不删除原始文件，也不将修复后的重跑覆盖为首轮成功。
首轮目录与有效重跑目录必须分开。本次各轮分别为：`matrix-20261018/`（作废首轮）、
`matrix-20261018b/`（7 行有效协议矩阵）、`matrix-20261018c/`（1024 档三条补充短测）、
`matrix-20261018d/`（int8 单行冒烟）、`matrix-20261018e/`（bf16 / int8 配对 A/B）。

> 归因限制：次线性加速并不能单独证明固定耗时全部来自 block swap。现有少量档位点只支持
> “存在非 token 比例缩放的开销”这一观察；线性近似不是经独立 profiling 验证的性能模型。
> 同样，短窗口步时稳定不等于长训热稳态已验证；本轮没有改变功耗、风扇或电压设置。

## 6. 顺带修复的档位前提

首轮还暴露一个会静默作废 `tecap1024` 的问题：MOSS 文本缓存身份只含量化方式，不含 token 上限，
而 MOSS 保存的是真实长度（不套 Anima max-padding）。于是 1024 档会复用 4096 档缓存，
`tecap1024` 行的“文本截断”根本不生效。

修复：schema 升到 `moss_image_te_v2`，在 metadata 写入 `max_text_tokens`，训练侧与预处理侧都必须
匹配；旧 `v1` 缓存仍可读（不带该字段，无法证明截断上限）。实测证据：

```text
matrix-20261018/tecap1024-long/cache/*_moss_image_te_nf4.safetensors
  max_text_tokens=1024  hiddens=[1024, 4096]      # 截断生效
matrix-20261018/1024-long/cache/*_moss_image_te_nf4.safetensors
  max_text_tokens=4096  hiddens=[4096, 4096]      # 4096 触顶
matrix-20261018/1024-short/cache/*_moss_image_te_nf4.safetensors
  max_text_tokens=4096  hiddens=[35, 4096]        # 真实长度
```

## 7. 未做与边界

- **质量对照未做**：协议第 7 节要求每档“基座 / 短训 / 最终 checkpoint”各 3 组、固定 prompt/seed、
  50 步采样与人工判读；本轮只有速度与显存，不能据此评价画质。
- **1024 档“跑得动”不等于“可训练”**：cap2048 虽然只是 9.365 s/step，但整卡峰值 9777 MiB /
  9560 MiB 可用，余量不足 8%，且温度已到 61 °C；真实数据集里更长的 longest-caption 或更高的
  分辨率抖动都可能直接 OOM。生产上 1024 档建议配 1024 上限（余量约 600 MiB）。
- `1024-swap28-long` 只证明了 cap4096 在 swap28 下能完成 10 步，没有测 swap 24/32 或 int8 传输；
  因此不能给出“cap4096 在 1024 档的最佳 swap”结论。
- 768/1024 档的 4096-token 字幕要跑完需要更低显存占用：提高 `blocks_to_swap`、改用 int8 传输
  （见 [int8 块交换诊断](moss_image_int8_blockswap_diagnosis_20261018.md)）、或进一步压
  `max_token_length`。本轮没有改协议参数去凑通过。
- 数据为 1 图 1 caption 的确定性合成图，caption 为确定性的重复填充句；这与真实数据集分布不同，
  只用于面积/文本两侧的受控对比。
- 只有 10 步，不含 warmup/长训收敛与热稳态漂移；没有测 TE/VAE 预处理在真实数据集规模下的吞吐。
- 本文档不改变任何默认配置或档位语义。

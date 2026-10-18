# MOSS-Image 档位热测试（bench/moss_tier_hot）

协议与结果落点：

- 协议、成本模型、数据短边契约：[docs/findings/moss_image_token_budget_tiers_20261017.md](../../docs/findings/moss_image_token_budget_tiers_20261017.md)
- 2026-10-18 CMP 50HX 实测报告：[docs/findings/moss_image_tier_hot_50hx_20261018.md](../../docs/findings/moss_image_tier_hot_50hx_20261018.md)

## 用途

按 `docs/findings/moss_image_token_budget_tiers_20261017.md` 第 7 节矩阵，跑 7 行
（1024 / 768 / 512 档 × 短/长 caption，加 `tecap1024` 长 caption）的 10 步热测试，
记录每步耗时、PyTorch allocator 峰值、整卡显存/温度/功耗/SM 时钟，并汇总落盘。

## 用法

```bash
python tasks.py moss-tier-hot -- plan                       # 只打印矩阵与每行解析到的 bucket
python tasks.py moss-tier-hot -- all --steps 10             # prepare + cache + run + report
python tasks.py moss-tier-hot -- cache --rows 512-short     # 只建缓存
python tasks.py moss-tier-hot -- run --rows 512-short       # 只训练（复用已有缓存）
python tasks.py moss-tier-hot -- report --root output/bench/moss_tier_hot/matrix-<stamp>
```

等价直接入口：`python -m bench.moss_tier_hot.run_matrix <action> [...]`。
默认 `--gpu-uuid` 是本机 CMP 50HX；换机器必须显式指定，否则会选错设备。

## 模块

| 文件 | 职责 |
| --- | --- |
| `matrix.py` | 7 行定义、短/长 caption 生成、源图与 `dataset.toml` / `config.runtime.toml` 生成 |
| `runner.py` | resize+缓存（显式钉住档位）、10 步训练、缓存 provenance 采集 |
| `collect.py` | `progress.jsonl` 步事件解析、`nvidia-smi` 1 秒采样与峰值统计 |
| `report.py` | 汇总 `summary.json` / `summary.md` |
| `run_matrix.py` | CLI（`plan` / `prepare` / `cache` / `run` / `report` / `all`） |

## 产物与契约

落点 `output/bench/moss_tier_hot/matrix-<stamp>/<row>/`（本地产物，不提交）：

```text
dataset.toml          # 该行的 bucket 档位 + image_dir/cache_dir/text_cache_dir
config.runtime.toml   # 由方法配置展开的运行配置（步预算、seed、无中途保存、telemetry 路径）
src/ data/ cache/     # 源图、resize 结果、latent + TE 缓存
cache.log train.log   # 两个子进程的完整日志
progress.jsonl memory_probe.jsonl gpu.csv   # 计时、allocator、整卡遥测原始数据
result.json           # 该行结构化结果（含 resized_image / latent_keys provenance）
```

两条必须保持的前提（见实测报告第 5、6 节）：

1. **档位必须传给 resize 阶段**：`preprocess-config` 只把尾随参数转发给 resize，
   缺 `--resolution/--min/max/--constant_token_buckets` 时每行都会按 base 默认 1024 生成 latent，
   档位实验会全部失效（表面特征是“三档同数、峰值完全相同”）。
2. **文本上限属于缓存身份**：MOSS 文本缓存 schema `moss_image_te_v2` 带 `max_text_tokens`，
   4096 档缓存不能被 1024 档复用；旧 `v1` 可读但无法证明截断上限。

计时口径：`ev=step` 的 `ts` 差值（首步单列）。它与 256×256 基线的 `_run_step` 同步计时不同法，
跨报告对比前需要按本协议重量一次 256×256 行。

# Block Swap 各族分块载荷与晋升算术（CMP 50HX）

状态：实验（跨族算术 + Anima 真机锚点）
适用版本：`codex/block-swap-int8`（`block_swap_auto` 已实现）
相关代码：`bench/block_swap_pipeline/family_bytes.py`、`library/runtime/block_swap_auto.py`
复现：`.venv/bin/python -m bench.block_swap_pipeline.family_bytes --all`

## 问题

P1 的进程内水位调优（`--block_swap_auto`）目前只在 Anima 上跑过真机。其它族能不能照搬，
取决于两个量：**每块换出载荷**（决定一次晋升省下多少 PCIe 时间）和**块数上界**
（决定最多能晋升几次）。这两个量都能从权重文件直接算出来。

## 方法（不加载权重）

读 safetensors 头部（8 字节长度前缀 + JSON header），按块前缀分组张量，
换出载荷 = `numel × transfer_dtype 字节数`（bf16 即 2）。

与运行时口径对齐：`Offloader` 的 `h2d_bytes` 来自 `_estimate_swap_bytes()`，
即交换计划里各 master 的 `stored_nbytes()`；bf16 主拷贝下同为 `numel × 2`。

## 验证（先证工具，再下结论）

1. **Anima 锚点**：工具算出 28 块、每块恒为 132.0 MiB（138.4 MB）。1024 真机 profile 的
   104 个任务**逐个**都是 138.4 MB，逐字一致。
2. **工具自身抓到的错误**：`net.llm_adapter.blocks.N`（6 个适配器块）与 DiT 的
   `net.blocks.0..5` 同号，最初并入后块 0~5 变成 164 MiB。运行时只换 DiT 的块，
   因此工具改为「取下标数最多的块前缀」；修正后才与真机一致。这条错误说明
   「按名字里的 `blocks.N` 分组」在 Anima 上会静默多算 32 MiB/块。
3. 总字节与文件大小交叉核对（见下表末列）。

## 结果

| 族 | 块前缀 | 块数 | `swap` 上界 | 每块载荷 | 块总量 | 非块部分 | 文件大小（核对） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anima | `net.blocks` | 28 | 26 | 132.0 MiB（138.4 MB） | 3.61 GiB | 292.4 MiB | 4.18 GB ✓ |
| z_image | `layers` | 30 | 28 | 345.1 MiB（361.8 MB） | 10.11 GiB | 1387.8 MiB | 12.31 GB ✓ |
| krea2_raw | `blocks` | 28 | 26 | 828.1 MiB（868.3 MB） | 22.64 GiB | 1265.7 MiB | 26.28 GB ✓ |
| qwen_image_2_1 | `transformer_blocks` | 60 | 58 | 648.2 MiB（679.7 MB） | 37.98 GiB | 77.3 MiB | 20.48 GB（盘上 fp8） |

注：`krea2_raw` 的文件比「块 + 非块」略大，因为非块里的 norm 以 F32 存盘、传输仍按 bf16；
`qwen_image_2_1` 盘上是 fp8（1 字节/元素），传输按 bf16 计，所以块总量约为文件的 2 倍。

真机核对（后续轮次）：Krea-2-Raw 日志 `22.64 GiB across 28 blocks`、Z-Image 日志
`10.11 GiB across 30 blocks`（promotion 行报 `block=345.0 MiB`）——两者都与上表逐项一致，
见 [Krea-2 验收](block_swap_krea2_hot_50hx_20261019.md) 与
[Z-Image 验收](block_swap_zimage_hot_50hx_20261019.md)。

四族都是**均匀块**（`uniform=True`）：`plan_promotion` 用「块 `count - current` 的载荷」
作单块代表的口径成立，不需要按块差分。

## 推论（按本机实测 3.14 GB/s 链路）

每步 DMA = `2 × swap × 每块载荷`（每个换出块一步恢复两次，见基线 findings）。

| 族 | 满水位每步 DMA | 满水位每步 DMA 时间下界 | 省下 1 块 = 少多少 DMA |
| --- | --- | --- | --- |
| anima | 7197 MB | 2.29 s | 0.088 s/步 |
| z_image | 20262 MB | 6.45 s | 0.230 s/步 |
| krea2_raw | 45153 MB | 14.38 s | 0.553 s/步 |
| qwen_image_2_1 | 78841 MB | 25.11 s | 0.433 s/步 |

1. **非 Anima 族里「省一块」的价值是 Anima 的 2.6~6.3 倍**：`--block_swap_auto`
   默认的激进晋升方向对这些族更对；`--block_swap_auto_stop_ratio` 的阈值含义也应按族解读
   （在 krea2 上，一块的值就是 0.55 s）。
2. `krea2_raw`（块总量 22.64 GiB）与 `qwen_image_2_1`（37.98 GiB）在 10 GB 卡上
   **不可能全常驻**，必须长期换出；它们不存在 Anima 那种「晋升到 2 块」的终态。
3. 满水位每步 DMA 下界 6.5~25 s，说明这两族的可用配置区间被 PCIe 卡死在**很小的 `swap`**：
   例如 `krea2_raw` 在 `swap=4` 时每步 6.9 GB ≈ 2.2 s DMA。

## 边界与未证明项

- 这是**真实权重字节 + 算术**，不是其它族的真机训练：各族实际 free VRAM（激活占用）与
  步时未测，「该升到几块」没有实测答案，本表只给每块的价格。
- `qwen_image_2_1` 盘上是 fp8；若实现 fp8 直传（未实现），载荷会减半，本表不适用。
- 3.14 GB/s 是本机 pinned 实测链路，跨硬件不外推。
- 跨族真机的阻塞在**探针**，不在缓存流水线（本节初稿曾误判为后者，已按源码更正）：
  `scripts/tasks/preprocess.py::_run_preprocess_te`（同文件 669~692 行）已按运行时配置里的
  `model_family` 分派 anima / krea2_raw / z_image / qwen_image_2_1 / moss_image，
  latent 侧也通过 `--model_family` 透传；只有最底层的
  `scripts/preprocess/cache_text_embeddings.py` 是 Anima 专用。
  要把跨族真机跑起来，需要给 `run_probe.py` 加族预设：它当前硬编码 Anima 的
  dit/te/vae 路径，并直接调用 Anima 专有的 TE 缓存脚本。

## 复现

```bash
.venv/bin/python -m bench.block_swap_pipeline.family_bytes --all
.venv/bin/python -m bench.block_swap_pipeline.family_bytes --family krea2_raw --free-mib 6550 --margin-mib 400
```

`--free-mib` 会按 `plan_promotion` 的同一算术给出「若此刻空闲为 X，会晋升多少块」。
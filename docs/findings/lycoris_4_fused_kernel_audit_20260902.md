# LyCORIS 4.0.0 fused kernel 宣传审计与本项目借鉴建议

状态：完成审计；本地 LoKr backward stage 2 已通过 CMP 170HX 端到端热测，仍不建议直接集成整套 LyCORIS
日期：2026-09-02
目标仓库：`/home/scv/nvme0n1p1/训练器测试/LyCORIS`
目标版本：`v4.0.0` / `03270a3839102e63b48578c80e7c024036de74d7`
当前项目：`/home/scv/nvme0n1p1/训练器相关/krea2-webui`

## 最终结论

LyCORIS 4.0.0 是真实正式发布，Triton/TileLang kernel 也是真实的大规模实现，
不是只有接口或 README 占位。最值得肯定的是其按调用选择 backend、自定义 autograd、
不物化大中间量、FP64 oracle、shape planner 和 eager sentinel 设计。

但宣传只能在很窄的口径下成立：

1. `1.4x-7.3x` 是 RTX 4090、FP16、选定 shape sweep 的 **forward device-time
   几何平均**，不是完整训练 step，也不是所有算法、GPU、dtype 或配置的普遍提速。
2. 正式表中低于 `1.0x` 的项目不少，例如 LoKr merge 的 fwd+bwd wall 只有
   `0.73x eager`，TileLang LoKr merge 对 compile 的 forward device ratio 只有
   `0.53x`。
3. 正式表中标为 `lora` / `lora_bypass` 的 full-sweep 脚本实际运行 LoHa 四因子
   Hadamard 公式。当前源码不能把表中的 `4.87x` / `1.37x` 复现为普通 LoRA 数据。
4. “精度更好或相当”不能作为普遍结论。正式表已有 OFT/TileLang ULP 高于 eager，
   精度文档还承认 TileLang BOFT 为 `31.7 ULP`，eager 只有 `2.3 ULP`。
5. VRAM 下降的机制可信，但仓库只给出单算子 `max_memory_allocated` 峰值比，没有提交
   原始 4090 JSON，也没有完整模型训练峰值证据。
6. 本机 RTX 3080 复测中，FP16 只有 `3/13`、BF16 只有 `1/13` 路径满足仓库自己的
   速度/精度排序；普通 LoRA、LoKr、DoRA 多数反而慢于 eager。

因此，对本项目的建议不是引入整个 `lycoris_lora` 或照搬 kernel，而是：

- 优先参考 LoKr **backward** 的 tile 化和 factor-on-the-fly 思路，接到本项目已经验证
  有真实训练收益的 grouped-delta 路径。
- 第二优先级做 LoHa 不物化 `Delta W` 的本地 prototype。
- plain LoRA 暂不投入生产实现；当前完整训练中 adapter 不是主要瓶颈。
- 借鉴 backend capability、fallback、精度测试和 release workflow 的工程模式，但补齐
  LyCORIS 当前缺少的 GPU CI、cache 失效和端到端门禁。

后续实现已按上述优先级推进：本项目独立增加 LoKr `grad_w1` Triton reduction，组件级
3-seed backward 平均 `24.020 -> 19.958 ms`；CMP 170HX 端到端 3-seed x 50-step 平均
`2.497463 -> 1.870028s`，降低 `25.11%`，Nsight 确认新 kernel 真实执行。精度、resume、
compile/swap 组合与限制见
[LoKr fused backward stage 1-2](lokr_fused_backward_stage1_20260902.md)。后续项目策略已将
本地 LoKr 组合 `triton` + `triton_grad_w1_w2_grad_x` 设为默认；这不改变本报告对
LyCORIS 普遍宣传口径的裁决，也不能外推到其他 adapter 或硬件。

## 宣传逐项裁决

| 宣传项 | 裁决 | 证据边界 |
| --- | --- | --- |
| `4.0.0` 正式发布 | 成立 | tag、GitHub Release、PyPI wheel/sdist 均存在且未撤回 |
| Triton/TileLang fused kernel | 成立 | 两套 23-op backend、算法专属 autograd 和 planner 均有实现 |
| 覆盖 lora/loha/lokr/oft/boft/dora/doha/dokr 等 | 有条件成立 | 热路径覆盖广，但部分只有 merge、linear 或 forward；DoHa/DoKr 只是复用 DoRA epilogue |
| `1.4x-7.3x` 提速 | 仅指定 4090 FP16 forward 微基准成立 | 不是端到端训练；部分 wall/compile 对比低于 `1.0x`；普通 LoRA full sweep 标注错误 |
| VRAM 更少或相当 | 单算子口径基本成立 | 已列行约 `1.06x-1.18x`，总览称最高 `1.6x`；没有完整训练与原始结果文件 |
| 精度更好或相当 | 普遍说法不成立 | 多行 ULP 高于 eager；无训练质量证据；BF16 未进入 kernel correctness tests |
| release 与 KohakuTerrarium 相同 | 设计思想相近，不是相同实现 | 都用单一发布入口、OIDC、tag/nightly；KohakuTerrarium 另有桌面、Docker、manifest 多平台流水线 |

## 发布真实性

### 仓库和制品

- annotated tag `v4.0.0` 指向 `03270a3`，tag message 为 `LyCORIS 4.0.0`。
- kernel 主提交为 `7f235c3`，一次加入 87 个文件、12449 行；release 文案提交为
  `d79770d`。
- `pyproject.toml:5-16` 的项目名为 `lycoris_lora`、版本为 `4.0.0`、许可证为
  Apache-2.0。
- PyPI 于 2026-09-01 15:20 UTC 上传：
  - `lycoris_lora-4.0.0-py3-none-any.whl`，179544 bytes；
  - `lycoris_lora-4.0.0.tar.gz`，137664 bytes；
  - 两者均未 yanked。
- GitHub Release `v4.0.0` 于 2026-09-01 15:20:38 UTC 发布，`draft=false`、
  `prerelease=false`，发布者为 `github-actions[bot]`。

外部核验入口：

- <https://pypi.org/project/lycoris-lora/4.0.0/>
- <https://github.com/KohakuBlueleaf/LyCORIS/releases/tag/v4.0.0>

需要区分两层状态：包元数据标为 `Production/Stable`，但 fused kernel 文档在
`docs/kernels/README.md:1-7` 和 `README.md:182-205` 明确标为 **early experimental**。

### release 机制

LyCORIS 的 `release.yml` 是唯一构建和上传入口：

- release published、`v*` tag 或手动 dispatch 均可触发：
  `.github/workflows/release.yml:3-18`。
- tag 必须与 `pyproject.toml` 版本一致：同文件 `:60-68`。
- 构建 wheel/sdist 并运行 `twine check`：`:70-84`。
- PyPI 使用 OIDC trusted publishing，不保存 API token：`:86-100`。
- PyPI 成功后创建/更新 GitHub Release 并上传相同制品：`:102-137`。
- nightly 和 auto-patch 只计算版本/打 tag，再 dispatch 同一个 release workflow。

这与 KohakuTerrarium 的核心思路一致：单一发布真相源、PEP 440 版本、OIDC、正式和
nightly 共用构建路径。但截至本次审计，LyCORIS 源码和历史中没有
`KohakuTerrarium` 字样；KohakuTerrarium 当前 workflow 还包含 release tree、桌面包、
Docker、多平台 manifest 和签名，不能称为逐文件相同。

参考：<https://github.com/Kohaku-Lab/KohakuTerrarium/tree/main/.github/workflows>

## kernel 实现事实

### dispatch 和 autograd

- `lycoris/kernels/dispatch.py:1-31` 定义顺序
  `triton > tilelang > compile > torch`。
- `lycoris/kernels/select.py:34-81` 每次调用检查 CUDA device、floating dtype family
  和 caller layout；超出范围降级。
- `lycoris/kernels/select.py:93-110` 对单个 op 做缓存的 `torch.compile(dynamic=None)`，
  只捕获 Dynamo/Inductor 编译错误，业务 shape/dtype 错误继续抛出。
- Triton 与 TileLang 的 `ops.py` 暴露镜像的 23 个 op。
- 每个算法用 `torch.autograd.Function` 连接 fused forward/backward，而不是只换一个
  inference kernel。

“fused”的实际含义是把 `Delta W` rebuild、`Delta W @ x`、Cayley、Kronecker tile、
DoRA normalization 等链条压为每方向少量或单次 launch，并避免把完整中间量写回显存。
它不是把整个模型 block 或完整训练 step 融为一个 kernel。

### 覆盖边界

权威能力表位于 `docs/kernels/README.md:21-48`：

- LoRA/LoCon：linear merge 和 bypass forward/backward；conv bypass fallback。
- LoRA Tucker：只融合 merge forward，rank `>64` 和 backward fallback。
- LoHa：普通 merge 与 linear bypass forward/backward；rank `>128`、Tucker backward、
  conv bypass fallback。
- LoKr：full/factored merge 与 linear bypass forward/backward；conv、Tucker `w2`、
  factor `>128` fallback。
- OFT/BOFT：主要 merge/bypass forward/backward；OFT block size `>32` fallback。
- DoRA：融合 weight-decompose epilogue，不是完整 base Linear；DoHa/DoKr 共用这一段。
- GLoRA、DyLoRA、IA3、full/norm 也有对应热路径，但并非每种 layout 都覆盖。
- rank/module dropout 和动态 multiplier 保留；mask 必须放在两个 factor 中间时回退。

因此“every algorithm”可以理解为每个算法至少有热点实现，不能理解为所有 mode、
layout、forward/backward 都必定运行 fused kernel。

### planner 和 cache

`lycoris/kernels/plans/tune.py:26-124` 的做法值得参考：

- token 维度分桶；
- 解析设备能力并对候选 tile 建模；
- shortlist 做一次实测；
- 候选异常时跳过；
- 可以把 eager sentinel 选为该 shape 的胜者；
- 结果写入 `~/.cache/lycoris_kernels/tuning.json`。

但当前 cache key 只有 device name、op、bucketed shape 和 dtype，没有 driver、CUDA、
PyTorch、Triton/TileLang 版本、kernel code hash。JSON 写入也不是 lock + atomic rename。
这个实现不应原样移植到长生命周期 WebUI；升级依赖或多进程并发时可能复用陈旧 plan
或写坏 cache。

## 官方 benchmark 审计

### 正式数字到底测了什么

`docs/kernels/benchmarks.md:43-60` 的表是 RTX 4090、FP16、family shape sweep 的几何
平均。`scripts/bench/kernels/verdict.py:37-92` 显示：

- `dev/eager` 和 `dev/compile` 使用 `fwd_dev_ms`，即 forward-only；
- `fb/eager` 和 `fb/compile` 才是 forward+backward wall time；
- VRAM 是一次 forward+backward 的 `max_memory_allocated`；
- ULP 只对 forward output。

宣传取的是 `dev/*` 列的最好区间，而不是用户更关心的 `fb/*` 或完整训练 step。

正式表自身已经包含反例：

| family/backend | forward device | fwd+bwd wall | 精度/显存备注 |
| --- | --- | --- | --- |
| LoRA bypass Triton | `1.37x eager` | `0.91x eager` | forward 快，但 fwd+bwd wall 慢 |
| LoKr merge Triton | `2.98x eager`, `0.93x compile` | `0.73x eager` | 对 compile 和 wall 均无速度优势 |
| LoKr merge TileLang | `1.70x eager`, `0.53x compile` | `0.69x eager` | 明显慢于 compile/eager wall |
| OFT Triton | `5.70x eager` | `2.23x eager` | `5.0 ULP` 高于 eager `4.0` |
| OFT TileLang | `4.35x eager` | `2.16x eager` | `8.4 ULP` 高于 eager `4.0` |
| OFT bypass TileLang | `4.59x eager` | `1.93x eager` | `4.9 ULP` 高于 eager `2.2` |

### 普通 LoRA 数据标注错误

这是当前发布最严重的可复现性问题：

- `scripts/bench/kernels/run_all.py:19-42` 把 `lora.py` 作为正式 full-sweep stage。
- `scripts/bench/kernels/lora.py:1-6` 自己写的是 “LoHa rebuild and weight-free bypass”。
- 同文件 `:20-21` 导入 `functional.loha`、`loha_diff_weight`、
  `loha_bypass_diff`。
- `:39-59` 构造四个因子并运行 Hadamard product。
- 但 `:62-70` 把 family 写成 `lora`，`:74-105` 把 LoHa bypass 写成
  `lora_bypass`。
- 普通 `locon_diff_weight` 只在固定 1280 shape 的 `quick.py` 中出现。

所以正式表的 `lora 4.87x` 和 `lora_bypass 1.37x` 至少在当前源码中是 **LoHa 数据被
标成 LoRA**，不能用来证明普通 LoRA full-sweep 性能。

### 方法文档与代码不一致

`docs/kernels/benchmarks.md:23-39` 说 primary device time 来自 profiler、三窗口各
50 次、每次 flush L2。当前代码却是：

- `harness.py:61-81`：wall path 为 warmup 3、8 次、CUDA event、逐次 L2 flush；
- `harness.py:84-111`：device path 为 warmup 5、两轮各 15 次、无 L2 flush；
- `harness.py:127-144`：两轮 interleave 后再单独测 host/device。

这意味着文档描述不是 4.0.0 当前代码的真实协议。仓库也没有提交生成表格的
`out/bench/kernels/*.json`，无法核验 4090 的逐 case 原始记录、软件版本和统计波动。

此外，`family_common.py:104-110` 虽定义 `assert_compiled()`，正式 `measure_case()`
并未调用；compile arm 没有强制证明本次确实形成 graph。

## 精度和测试审计

### 已有优点

- `test/kernels/refs.py:1-6` 使用独立 FP64 plain-Torch oracle。
- `test_ops.py` 覆盖主要 op 的 forward、input grad 和 factor grad。
- `test_autograd.py` 对比 fused Function 与 eager functional，并检查非零梯度。
- FP16/FP32 代表 shape 的大多数 Triton 路径在本机通过。

### 不能支撑“精度更好”的缺口

- `test/kernels/test_ops.py:21-24` 虽定义 BF16 tolerance，实际 `DTYPES` 只有 FP16、
  FP32；`test_autograd.py:44-47` 同样没有 BF16。
- 相对误差是全张量 `max(abs(err))/max(abs(ref))`，阈值 FP16 `2e-2`、FP32
  `5e-3`，不是文档宣称的 ULP test gate。
- 没有 autocast、non-contiguous、极端 rank/shape、NaN/Inf/overflow、gradgrad。
- `.github/workflows/ci.yml:27-52` 明确 runner 无 GPU，只跑 import 和 CPU smoke；
  Triton/TileLang kernel 不进入 CI。
- `precision.md:37-42` 一边说 error “at or below eager”，一边给出 OFT
  `5.0 vs 4.0` 和 TileLang BOFT `31.7 vs 2.3`，结论与数字直接矛盾。
- ULP 只衡量单次算子输出，不是训练收敛、checkpoint round-trip 或图像质量证据。

## RTX 3080 本机复测

### 环境和口径

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 10GB, sm86 |
| driver | 610.43.02 |
| Python | 3.13.11 |
| PyTorch | 2.12.0+cu130 |
| Triton | 3.7.0 |
| TileLang | 未安装，未实跑 |
| shape | `o=i=1280, rank=16, tokens=512` |
| script | `scripts/bench/kernels/quick.py` |

使用隔离的 `/tmp` tuning cache；没有安装或修改目标仓库依赖。下表为
`eager time / triton-arm time`，小于 `1.0x` 表示更慢。注意 triton arm 内部可能被
planner 选成 eager sentinel，因此它表示 LyCORIS API 的实际选路成本，不保证真有
Triton kernel 执行。

| case | FP16 | BF16 |
| --- | ---: | ---: |
| LoRA merge | `0.31x` | `0.39x` |
| LoRA bypass | `0.49x` | `0.22x` |
| LoHa merge | `0.57x` | `1.39x` |
| LoHa bypass | `0.58x` | `0.60x` |
| LoKr merge, both full | `0.12x` | `0.09x` |
| LoKr merge, B factored | `0.57x` | `0.51x` |
| LoKr bypass | `0.62x` | `0.51x` |
| OFT merge | `3.39x` | `4.44x` |
| OFT bypass | `4.02x` | `3.80x` |
| BOFT merge | `2.08x` | `2.63x` |
| BOFT bypass | `2.59x` | `2.76x` |
| DoRA epilogue | `0.71x` | `0.83x` |
| IA3 bypass | `0.37x` | `0.40x` |

脚本自己的总评：FP16 `3/13`、BF16 `1/13` 满足完整速度/ULP target ordering。

该 quick 结果也只是 forward 微基准，不能替代端到端训练；它的意义是明确证明
4090 FP16 的宣传区间不能外推到本项目的 3080 BF16 环境。

### fallback 复测

关闭 tuner (`LYCORIS_KERNEL_TUNE=off`) 后运行 51 项 CUDA tests：

- `47 passed, 4 failed`；
- 失败均是 LoRA merge backward；
- FP16/FP32 都在 Triton 3.7 编译时报
  `Mismatched type for acc between then block ... and else block ...`；
- `lycoris/kernels/triton/lora/bypass.py:414-470` 的 role-split backward kernel
  是失败入口。

重新启用默认 tuner 后，失败的 4 项全部通过。tuning table 显示该 shape 的
`triton.lora.merge_bwd` 被记录为 `limiter=eager`。因此默认行为是正确 fallback，
不是该环境中成功运行 fused LoRA backward。

这也暴露两个边界：

1. 文档中的“fused forward+backward”是能力表，不代表每个环境实际选择 fused。
2. `LYCORIS_KERNEL_TUNE=off` 不只是关闭性能调优，也可能绕过保护性的 eager sentinel
   而直接触发编译失败。

冷 cache 的完整 default-tuned suite 在本次 300 秒预算内未完成；因此本报告不宣称
本机完整 51 项 default-tuned suite 一次性通过。已完成的是 47 项非失败路径和上述
4 项默认 fallback 的定向复核。

## 与本项目的实现对比

### 当前 plain LoRA

`networks/lora_modules/base.py:162-184` 的训练 forward 是：

```text
base(x)
  + up(dropout(rank_dropout(timestep_mask(down(rebalanced(x))))))
    * multiplier * scale
```

关键差异：

- 本项目在两个 factor 之间有 T-LoRA mask、dropout、rank dropout；LyCORIS 自己明确
  这种 mask 位置会让调用 fallback。
- 本项目支持 channel-scale rebalance 和 Linear/Conv2d；LyCORIS linear bypass kernel
  不能覆盖完整语义面。
- adapter 必须先 apply/load，再做 block-level `torch.compile`。额外引入 per-op compile
  或 opaque custom autograd 可能改变现有 resident graph，必须整 block A/B。
- Krea-2 rank16 降到 rank8 后 adapter 参数减半，但 step `2.726 -> 2.728-2.737s`
  持平，只省 145MB：`krea2_3080_speed_stage8.md:13-29`。
- compiled Krea-2 的 base GEMM + attention 已占约 89%：
  `krea2_3080_speed_stage11.md:37-45`。

结论：plain LoRA fused kernel 不是当前 Krea/Anima 的优先性能方向。即便局部快数倍，
整步 Amdahl 上限也很低；而本机 LyCORIS LoRA quick 本身还慢于 eager。

### 当前 LoHa

`networks/plugins/loha/autograd.py:1-72` 当前仍在 forward 构造完整 Hadamard delta weight，
backward 再重建一次，只是避免把完整 weight 保存在 autograd tape。这与 LyCORIS 的
tile-on-the-fly、绕过 `Delta W` 物化正好对应。

LoHa 是值得做本地 prototype 的对象，但不能直接采用 LyCORIS 的 4090 数字：本机
FP16 LoHa merge/bypass 为 `0.57x/0.58x`，BF16 只有 merge 为 `1.39x`、bypass 仍为
`0.60x`。必须使用本项目实际 shape、FP32 compute 语义和训练 step 验证。

### 当前 LoKr

本项目已经有独立的 `eager|triton` grouped-delta backend，并且证据比 LyCORIS 宣传更
贴近生产：

- `networks/plugins/lokr/module.py:187-203` 只在 full `w2`、无 dropout、scalar gate、
  square factor、contiguous layout 下走 fused path。
- 当前默认配置为 `triton` + `triton_grad_w1_w2_grad_x`；能力条件不满足时回退 eager，
  显式 `eager` 仍可用于对照：`configs/gui-methods/lokr.toml`。
- RTX 3080 50-step、3 seeds 的真实训练平均 step 提升为
  `18.58% / 19.42% / 22.65%`，loss delta 只有 `6e-5` 到 `8e-5`，reserved
  显存稳定下降约 0.0898GB：
  `training_profiling_hot_test_20260629.md:613-668`。
- forward launch 从 108723 降到 3807，elementwise-like kernel 从 91719 降到
  270：同文 `:437-470`。
- backward 仍保留 1151 次 launch，trace kernel time 基本不变，是明确下一热点：
  同文 `:693-758`。

LyCORIS 对本项目 LoKr 的真正参考价值是 fused backward、factorized `w2` 和统一 planner，
不是替换已经验证过的 grouped forward。任何新实现都必须与现有 Triton 路径同 shape
正面对比。

### 其他方法

| LyCORIS 能力 | 本项目状态 | 建议 |
| --- | --- | --- |
| DoRA epilogue | 有 DoRA，但 live forward 还含 base weight row norm 和 LoRA 语义 | 本机 quick 慢，暂不优先 |
| GLoRA | 有独立两路 adapter，依赖 base weight | 只参考 autograd/precision，不直接复用普通 LoRA kernel |
| OFT/BOFT/DoHa/DoKr | 当前 registry/config 未发现对应生产方法 | 没有业务入口前不引入 kernel |
| Hydra/Chimera/stacked experts | 本项目有动态 router/多 expert | LyCORIS 普通 fused op 不覆盖路由语义 |
| OrthoLoRA | 本项目已有 Cayley/SVD basis 结构 | 数学和 LyCORIS OFT 不同，不应按名字类比移植 |
| ReFT | block residual intervention | 不适用 Linear adapter kernel |

## 值得借鉴的内容

### P0：立即借鉴工程门禁，不引入 kernel

1. 为 adapter kernel 建立显式 capability contract：device、dtype、layout、rank、mask、
   dropout、router、compile scope，每次调用 fail closed。
2. 每个 kernel 同时测 forward、input grad、所有 trainable leaf grad，并用独立 FP64
   oracle；本项目主口径必须包含 BF16。
3. benchmark 同行记录 operator/device、wall、launch count、allocated/reserved/NVML、
   数值误差和实际选中的 backend，不能只打印请求 backend。
4. 保留 eager sentinel，但 cache key 必须包含 GPU UUID/CC、driver、CUDA、Torch、
   Triton/TileLang、kernel ABI/hash；使用 file lock、临时文件和 atomic rename。
5. 可选 backend 缺失时核心包仍可 import；显式请求不可用 backend 时前置报错。

许可证边界：LyCORIS 是 Apache-2.0，本项目是 MIT。算法和工程思路可以独立实现；若
直接复制或修改 LyCORIS kernel 源码，需要保留适用的 Apache-2.0 许可、版权归属和修改
说明，并继续核验 Triton/TileLang 自身的分发条款。不要把上游 kernel 当作无来源代码
并入本项目。

### P1：LoKr backward prototype

目标是补本项目已确认的剩余热点，而不是并行造另一套 LoKr forward：

- 优先 `grad_w2_reduce`、`grad_w1_reduce`，再看 `grad_x_writeback`。
- 参考 LyCORIS 在 tile 内生成 Kronecker factor、避免完整 `Delta W`/grad 的做法。
- 保留本项目 scalar timestep gate、group size、chunk bytes、block swap 和 BF16 契约。
- 第一阶段只支持当前 `_can_use_fused_grouped_delta()` 的窄集合，其余明确 fallback。

通过门槛：

- RTX 3080 与 PG199 同 shape correctness；
- BF16 forward/input-grad/factor-grad 相对误差、cosine 和 finite 检查；
- 3 seeds x 50-step，avg/median/p90 至少稳定快 `>5%`；
- backward launch 和 `<50us` kernel 数显著下降；
- loss、checkpoint round-trip、resume、compiled block、block swap 均无回归。

### P1：LoHa no-materialize prototype

只在本项目 LoHa hot shapes 上实现最小 Triton prototype，先比较：

```text
current HadaLinearFn
vs torch.compile(current op)
vs local Triton tiled bypass
```

必须覆盖 FP32 factor compute、BF16 activation/output、timestep scalar gate 和
rank-dropout fallback。若完整训练没有稳定 `>5%` 或显存收益，不进入配置面。

### P2：release 工程

可直接借鉴而与 kernel 无关的部分：

- `pyproject.toml` 单一版本源；
- tag/version 不一致即拒绝；
- wheel + sdist + `twine check`；
- clean venv wheel install smoke；
- PyPI OIDC trusted publishing；
- nightly/patch/release 共用唯一 build/publish workflow。

本项目依赖大、平台和 CUDA wheel 约束更复杂，不能照搬 LyCORIS 的纯 Python
`py3-none-any` 发布假设。

## 明确不建议

- 不把 `lycoris_lora` 作为本项目运行时依赖；两边 wrapper、state dict、dropout、routing、
  compile 生命周期不同，会形成双重 adapter 系统。
- 不复制整个 1.2 万行 kernel 树；先做单一热点 prototype，再按证据扩展。
- 不因安装 LyCORIS 的 Triton/TileLang 就对所有方法自动启用。当前默认只覆盖本项目自身
  在 CMP 170HX 上完成端到端 gate 的 LoKr 路径；其他方法和硬件仍需独立验证。
- 不宣传本项目可获得 `1.4x-7.3x` 训练加速。
- 不把 FP16 4090 微基准外推到 BF16/NF4、checkpoint、block-swap 的 Krea/Anima 训练。
- 不优先引入 TileLang。当前环境未安装，LyCORIS 正式表中它多数慢于 Triton，且增加
  平台、编译器和 cache 维护面。
- 不先做 plain LoRA；当前 profile 和 rank 消融都说明它不是主瓶颈。

## 建议实施顺序

1. 修订本项目 adapter benchmark contract，先固定真实 shape、BF16、compiled block 和
   50-step 端到端口径。
2. 在现有 LoKr grouped-delta 模块内做 backward prototype，不新建第二套网络实现。
3. 若 LoKr 通过，再做 LoHa tiled bypass prototype。
4. 只有两个 prototype 都证明通用 backend abstraction 能减少重复，才抽共享
   `networks/kernels/` registry/planner/cache。
5. plain LoRA、DoRA、GLoRA 后续按 profiler 占比决定，不按 LyCORIS 功能清单排期。

总体判断：LyCORIS 4.0.0 的 kernel 工程有真实技术含量，尤其适合作为 LoKr backward、
LoHa no-materialize 和 backend testing 的参考实现；其性能宣传则把特定 4090 FP16
forward 微基准过度概括成了普遍收益，当前证据不足以支持本项目直接采用或对外复述。

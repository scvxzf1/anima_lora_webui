# AdaLN LoRA

状态：能力已合入，默认关闭（`configs/base.toml` 里 `train_adaln = false`）
适用版本：当前 main
入口开关：顶层 TOML / CLI 的 `train_adaln` / `adaln_rank` / `adaln_alpha`
相关代码：`networks/lora_anima/config.py`（旋钮解糖）、`networks/lora_utils.py`
（键布局互转）、`networks/lora_save.py::_relayout_adaln_to_comfy`（保存端）、
`networks/lora_anima/factory.py` + `networks/lora_anima/persistence.py`（加载端）

## 这一层是什么

Anima 沿用 Cosmos-Predict2 的 per-block modulation：每个 `Block` 有三条分支的
调制 MLP（`adaln_modulation_{self_attn,cross_attn,mlp}`），仅由 timestep
embedding 产生 shift/scale/gate——不吃文本，也不吃空间条件。所有 Anima 权重都
是 **AdaLN-LoRA 瓶颈形式**（`use_adaln_lora=True, adaln_lora_dim=256`）：

```text
SiLU → Linear 2048→256 (".1", down) → Linear 256→6144 (".2", up)
```

“AdaLN-LoRA”是 NVIDIA 的命名，指的是**预训练架构**里调制 MLP 上的低秩瓶颈，
不是适配器。参数量是原因：全秩 adaln 为 2048×6144 ×3 分支 ×28 block ≈ 1.06B，
256 瓶颈约 176M。它不是逐权重可选项——`blocks.{b}.adaln_modulation_{br}.1/.2.weight`
在 base 和 turbo 里都存在。

本文说的“训练 adaln”指的是**在这些既有的 `.2` up-projection 上再挂 LoRA**。

## 键名契约（唯一真正的坑）

同一个 Linear 有两个名字，而 ComfyUI 只认其中一个：

| 场景 | 模块路径 | LoRA 键 |
| --- | --- | --- |
| 权重文件 / ComfyUI | `blocks.{b}.adaln_modulation_{br}.2` | `lora_unet_blocks_{b}_adaln_modulation_{br}_2` |
| 本仓运行时（经 `_dit_rename_hook`，`library/anima/weights.py:57`） | `blocks.{b}.adaln_up_{br}` | `lora_unet_blocks_{b}_adaln_up_{br}` |

运行时命名的 adaln 键会被 ComfyUI **静默丢弃**。所以：对外发布必须用 comfy
布局，仓内消费（热启动、`--lora_weight`、resume）要的是 runtime 布局。

`networks/lora_utils.py` 提供三个纯函数做互转，非 adaln 键原样透传：

| 函数 | 方向 |
| --- | --- |
| `relayout_adaln_runtime_to_comfy` | `adaln_up_{br}` → `adaln_modulation_{br}_2` |
| `relayout_adaln_comfy_to_runtime` | 反向 |
| `has_comfy_adaln_keys` | 判存在，用于给两个加载口做 presence gate |

## 保存端

`lora_save.py::save_network_weights` 的标准写路径上，`_relayout_adaln_to_comfy`
在 **qkv defuse 之后、算 hash 之前**执行（hash 必须覆盖真正写盘的键），并盖
`ss_adaln_layout = "comfy"`。presence-gated：没有 adaln 键的 checkpoint 完全不
受影响，metadata 也不会多出这一项。

插件 save handler（`SAVE_HANDLERS`：loha / lokr / vera / glora）会短路
`save_network_weights`，所以四个 handler 各自调用同一个函数——否则某个变体带
`train_adaln` 训出来的权重会漏出 runtime 键名。MoE / chimera 的 `_moe` 兄弟文件
本来就不是 ComfyUI 可加载的，不在覆盖范围内。

## 加载端

两个 chokepoint 都做反向重命名，都是 presence-gated：

| 入口 | 覆盖场景 |
| --- | --- |
| `lora_anima/factory.py::create_network_from_weights` | 推理 / merge / 脚本 |
| `lora_anima/persistence.py::load_lora_network_weights` | resume / soup / init（`LoRANetwork.load_weights` 委托到这里） |

**保存与加载必须成对合入**：只有保存端会让 resume 时每个 adaln 模块落进
`missing_keys` 并静默从头训；只有加载端则发布的权重在 ComfyUI 里不起作用。

## 旋钮

`train_adaln` / `adaln_rank` / `adaln_alpha` 由
`LoRANetworkCfg.from_kwargs` 读取，解糖成已有的原语：

```toml
# 等价的原语写法
include_patterns = [".*adaln_up_.*"]
```

- `train_adaln`：往 `include_patterns` 追加 `.*adaln_up_.*`。这是
  **exclude 覆盖，不是白名单**——`_DEFAULT_EXCLUDE`
  （`config.py:216`）里列了 `adaln_up_`，include 命中即翻案，默认的 attn+MLP
  目标集不受影响。
- `adaln_rank`：写进 `reg_dims`（同一 pattern）。0 / 缺省 = 跟网络同 rank。
- `adaln_alpha`：写进 `reg_alphas`。0 / 缺省时按 √r 律从网络自身 rank/alpha 推导。
- `adaln_rank` / `adaln_alpha` 不带 `train_adaln` 会直接 `ValueError`。

三个键都在 `networks/registry_api.py::SHARED_KWARG_FLAGS` 里注册过，因此可以
写成**顶层 TOML 键**而不必塞 `network_args`——`bootstrap.build_net_kwargs` 只
转发 allowlist 里有的键，没注册的话 `train_adaln = true` 会被静默忽略。

`network_reg_alphas`（`pattern=alpha` 的 kv 串）是本次一并补齐的原语，此前本
仓只有 `network_reg_dims` / `network_reg_lrs`。它独立于 `reg_dims` 生效：无论
rank 来自 reg_dims 命中还是网络默认值，都能单独覆盖 alpha
（`lora_anima/targeting.py`）。

### √r 律

最优 LoRA α 随 rank 亚线性增长（α\*(r) ≈ C·√r），跨 rank 保持尺度一致意味着
保持 **α/√r** 恒定：

```text
adaln_alpha = network_alpha · sqrt(adaln_rank / network_dim)
```

`from_kwargs` 在 `adaln_alpha` 为 0/未设时算这一项，所以在任意
`network_dim`/`network_alpha`（包括 CLI `--network_alpha` 覆盖）下 adaln 分支都
与 attn+MLP 保持同一 α/√r。`adaln_rank = 0` 时因子为 1，等于直接继承
`network_alpha`。硬编码某个常数（例如 90）只对某一套 dim/alpha 成立，换 preset
就会偏热，所以这里选择推导而不是写死。

### 为什么默认关

`adaln_rank = 16` 这类配置来自上游的实测结论（学到的 adaln ΔW 有效秩很低，
r16 已能保住绝大部分能量）。但打开它会扩大受训模块集合与显存占用，而本仓现有
recipe 都是在**没有** adaln 的前提下调出来的。因此 `configs/base.toml` 以
`train_adaln = false` 落能力，按需在单次运行或某个 method 配置里打开：

```toml
train_adaln = true
adaln_rank = 16
```

冻结 DiT 的方法（`soft_tokens` / `easycontrol`）对这组键是惰性的。

### 与 compile 的相互作用

在 `compile_dynamic_seq` 下训练 adaln 会在第一次带梯度的 forward 崩在
`ConstraintViolationError`：adaln LoRA 让 shift/scale/gate 需要梯度，反向因此
多出一条 seq 轴 reduction，inductor 的 mix-order-reduction 融合会按首次 trace
的 hint 记一条 4096 边界 guard，与严格的 `mark_dynamic` 区间矛盾。修法是在
dynamic-seq 标记生效时关掉 `triton.mix_order_reduction`，且必须用
`library/runtime/dynamo.py::pin_inductor_flag`——普通赋值只作用于当前上下文
（inductor config override 是 thread-local ContextVar），会在带梯度的编译上下文
里回滚。这条对任何挂在广播消费型 Linear 上的 LoRA 都成立，不止 adaln。

## 测试

| 文件 | 覆盖 |
| --- | --- |
| `tests/test_network_cfg.py` | 旋钮解糖、√r 律、缺 `train_adaln` 时报错、allowlist 注册 |
| `tests/test_network_registry.py` | 保存端改名 + `ss_adaln_layout` 盖章、无 adaln 时惰性、runtime↔comfy 往返 |
| `tests/test_pin_inductor_flag.py` | 上面那条 compile 修法的 ContextVar 语义 |
